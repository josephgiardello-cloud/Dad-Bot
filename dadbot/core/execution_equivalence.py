"""Phase 4.2 — Execution Equivalence Classes.

Formal equivalence classes for execution comparison.

Two executions are the "same computation" when they share the same:
    - ToolGraphClass:     DAG structure class (node types + edge relation types)
    - PlanClass:          Planning output class (intent + strategy + tool_count)
    - EventStructureClass: Event log structure class (event type sequence)

Combined into an ExecutionEquivalenceClass, which is the "fingerprint" of
the computation independent of LLM output, timestamps, and session IDs.

Formal definition:
    is_equivalent_execution(A, B) ↔
        ToolGraphClass(A) == ToolGraphClass(B)
        AND PlanClass(A) == PlanClass(B)
        AND EventStructureClass(A) == EventStructureClass(B)

Use cases:
    - "Did this run produce the same computation as the golden run?"
    - "Are two distinct sessions computing the same thing?"
    - "Has a code change altered the computation class?"
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from dadbot.core.tool_dag import ToolDAG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _sha256_short(payload: Any) -> str:
    return _sha256(payload)[:16]


# ---------------------------------------------------------------------------
# ToolGraphClass
# ---------------------------------------------------------------------------


def tool_graph_class(dag: "ToolDAG") -> str:
    """Compute the equivalence class of a ToolDAG.

    The class is determined by:
        - Sorted list of (tool_name, intent) pairs (node types)
        - Sorted list of edge types

    This is INDEPENDENT of:
        - Node IDs, sequences, deterministic_ids (instance details)
        - Edge source/target IDs (structural binding, not type)
        - Order of node registration
    """
    nodes = list(getattr(dag, "nodes", []))
    edges = list(getattr(dag, "edges", []))

    node_types = sorted((str(n.tool_name), str(n.intent)) for n in nodes)
    edge_types = sorted(str(getattr(e, "edge_type", "sequential")) for e in edges)

    return _sha256_short({
        "node_types": node_types,
        "edge_types": edge_types,
    })


def tool_graph_class_from_names(tool_names: list[str], intents: list[str] | None = None) -> str:
    """Convenience: compute graph class from tool_names + optional intents."""
    if intents is None:
        intents = [""] * len(tool_names)
    node_types = sorted(zip(tool_names, intents))
    return _sha256_short({
        "node_types": [(str(t), str(i)) for t, i in node_types],
        "edge_types": [],
    })


# ---------------------------------------------------------------------------
# PlanClass
# ---------------------------------------------------------------------------


def plan_class(intent_type: str, strategy: str, tool_count: int) -> str:
    """Compute the equivalence class of a planner output.

    The class is determined by:
        - intent_type (normalized to lowercase)
        - strategy (normalized to lowercase)
        - tool_count (number of tools in the plan)

    Independent of: specific tool names, order of tools, planner text.
    """
    return _sha256_short({
        "intent_type": str(intent_type or "").strip().lower(),
        "strategy": str(strategy or "").strip().lower(),
        "tool_count": int(tool_count),
    })


def plan_class_from_planner_output(planner_output: dict[str, Any]) -> str:
    """Convenience: compute plan class from a planner output dict."""
    return plan_class(
        intent_type=str(planner_output.get("intent_type", "") or ""),
        strategy=str(planner_output.get("strategy", "") or ""),
        tool_count=len(planner_output.get("tool_plan", []) or []),
    )


# ---------------------------------------------------------------------------
# EventStructureClass
# ---------------------------------------------------------------------------


def event_structure_class(event_log: list[dict[str, Any]]) -> str:
    """Compute the equivalence class of an event log.

    The class is determined by the SEQUENCE of event types only.
    Payloads, timestamps, session_ids, and trace_ids are excluded.

    Independent of: event payloads, when events occurred, session context.
    """
    type_sequence = [
        str(e.get("type") or e.get("event_type") or "unknown")
        for e in (event_log or [])
    ]
    return _sha256_short({"event_type_sequence": type_sequence})


# ---------------------------------------------------------------------------
# ExecutionEquivalenceClass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionEquivalenceClass:
    """Combined fingerprint of a computation.

    Attributes:
        graph_class:   ToolGraphClass hash.
        plan_class:    PlanClass hash.
        event_class:   EventStructureClass hash.
        equivalence_key: Combined hash of all three classes.
    """
    graph_class: str
    plan_class_str: str
    event_class: str
    equivalence_key: str

    @classmethod
    def build(
        cls,
        graph_class: str,
        plan_class_str: str,
        event_class: str,
    ) -> "ExecutionEquivalenceClass":
        equivalence_key = _sha256_short({
            "graph_class": graph_class,
            "plan_class": plan_class_str,
            "event_class": event_class,
        })
        return cls(
            graph_class=graph_class,
            plan_class_str=plan_class_str,
            event_class=event_class,
            equivalence_key=equivalence_key,
        )

    def matches(self, other: "ExecutionEquivalenceClass") -> bool:
        return self.equivalence_key == other.equivalence_key

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_class": self.graph_class,
            "plan_class": self.plan_class_str,
            "event_class": self.event_class,
            "equivalence_key": self.equivalence_key,
        }


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def classify_execution(
    planner_output: dict[str, Any],
    dag: "ToolDAG | None",
    event_log: list[dict[str, Any]],
) -> ExecutionEquivalenceClass:
    """Classify a full execution into its equivalence class.

    Args:
        planner_output: Dict with keys intent_type, strategy, tool_plan.
        dag:            ToolDAG instance (or None for empty graph class).
        event_log:      List of event dicts.

    Returns:
        ExecutionEquivalenceClass with a stable equivalence_key.
    """
    if dag is not None:
        gc = tool_graph_class(dag)
    else:
        gc = tool_graph_class_from_names([])

    pc = plan_class_from_planner_output(planner_output)
    ec = event_structure_class(event_log)

    return ExecutionEquivalenceClass.build(gc, pc, ec)


def is_equivalent_execution(
    exec_a: ExecutionEquivalenceClass,
    exec_b: ExecutionEquivalenceClass,
) -> bool:
    """True iff both executions share the same equivalence_key."""
    return exec_a.equivalence_key == exec_b.equivalence_key


def equivalence_proof(
    exec_a: ExecutionEquivalenceClass,
    exec_b: ExecutionEquivalenceClass,
) -> dict[str, Any]:
    """Return a structured proof of equivalence (or non-equivalence) between two executions."""
    return {
        "equivalent": exec_a.equivalence_key == exec_b.equivalence_key,
        "graph_class_match": exec_a.graph_class == exec_b.graph_class,
        "plan_class_match": exec_a.plan_class_str == exec_b.plan_class_str,
        "event_class_match": exec_a.event_class == exec_b.event_class,
        "exec_a_key": exec_a.equivalence_key,
        "exec_b_key": exec_b.equivalence_key,
    }


__all__ = [
    "ExecutionEquivalenceClass",
    "classify_execution",
    "equivalence_proof",
    "event_structure_class",
    "is_equivalent_execution",
    "plan_class",
    "plan_class_from_planner_output",
    "tool_graph_class",
    "tool_graph_class_from_names",
]
