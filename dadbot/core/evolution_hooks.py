"""L5 — Self-Evolvable Execution Substrate (Future Readiness Hooks).

SCAFFOLDING ONLY — no active execution logic implemented here.

These classes define the evolution-readiness boundaries of the system.
They are placeholders for a future policy-separation layer.

Design principle:
    "An evolvable system separates observation from mutation, and mutation
    from policy selection. The hooks exist before the policies do."

Layers:
- GraphIntrospectionAPI   — read-only DAG introspection + mutation hooks
- PolicySeparationLayer   — container for typed policy layers
- ExecutionTelemetryVector — structured feature vector for observability
- OptimizationBoundary    — declares which components are evolvable vs frozen
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Execution telemetry vector
# ---------------------------------------------------------------------------


@dataclass
class ExecutionTelemetryVector:
    """Structured feature vector for execution observability.

    This is NOT a log entry — it is a semantic summary of a single execution
    step, suitable for downstream policy-optimization systems to consume.

    Fields are stable: adding new fields must not break existing consumers.
    """
    event_count: int = 0
    tool_count: int = 0
    latency_ms: float = 0.0
    schedule_waves: int = 0
    state_transitions: int = 0
    dag_hash: str = ""
    session_id: str = ""

    def to_feature_vector(self) -> list[float]:
        """Return a fixed-width numeric feature vector.

        Order is stable.  New fields must be appended, never inserted.
        """
        return [
            float(self.event_count),
            float(self.tool_count),
            float(self.latency_ms),
            float(self.schedule_waves),
            float(self.state_transitions),
            float(len(self.dag_hash)),     # hash length as a proxy for presence
            float(len(self.session_id)),   # session id length as proxy
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_count": self.event_count,
            "tool_count": self.tool_count,
            "latency_ms": self.latency_ms,
            "schedule_waves": self.schedule_waves,
            "state_transitions": self.state_transitions,
            "dag_hash": self.dag_hash,
            "session_id": self.session_id,
        }


# ---------------------------------------------------------------------------
# Optimization boundary
# ---------------------------------------------------------------------------


_DEFAULT_EVOLVABLE = frozenset({
    "tool_selection_policy",
    "planning_policy",
    "memory_ranker",
    "critic_constraints",
    "schedule_confluence",
})

_DEFAULT_FROZEN = frozenset({
    "event_authority",
    "event_reducer",
    "dag_integrity",
    "tool_ir_boundary",
    "execution_ledger",
    "canonical_event",
})


@dataclass(frozen=True)
class OptimizationBoundary:
    """Declares which system components are evolvable vs. frozen.

    Frozen components have formal invariants that must not change.
    Evolvable components may be updated by the policy layer.

    boundary_hash is computed from the frozensets so it changes whenever
    the boundary declaration changes.
    """
    evolvable_components: frozenset[str]
    frozen_components: frozenset[str]
    boundary_hash: str

    @classmethod
    def default(cls) -> "OptimizationBoundary":
        """Return the default boundary separating evolvable from frozen."""
        evolvable = _DEFAULT_EVOLVABLE
        frozen = _DEFAULT_FROZEN
        boundary_hash = _sha256({
            "evolvable": sorted(evolvable),
            "frozen": sorted(frozen),
        })
        return cls(
            evolvable_components=evolvable,
            frozen_components=frozen,
            boundary_hash=boundary_hash,
        )

    def is_evolvable(self, component: str) -> bool:
        return component in self.evolvable_components

    def is_frozen(self, component: str) -> bool:
        return component in self.frozen_components

    def to_dict(self) -> dict[str, Any]:
        return {
            "evolvable_components": sorted(self.evolvable_components),
            "frozen_components": sorted(self.frozen_components),
            "boundary_hash": self.boundary_hash,
        }


# ---------------------------------------------------------------------------
# Policy separation layer
# ---------------------------------------------------------------------------


class ExecutionPolicyLayer:
    """Abstract boundary for execution policy.

    Scaffolding: no active implementation.  Override in subclasses.
    """
    def select_execution_order(self, dag: Any, context: dict[str, Any]) -> Any:
        return dag  # identity — not yet active


class PlanningPolicyLayer:
    """Abstract boundary for planning policy.

    Scaffolding: no active implementation.  Override in subclasses.
    """
    def select_plan(self, candidates: list[Any], context: dict[str, Any]) -> Any:
        return candidates[0] if candidates else None


class ToolSelectionPolicyLayer:
    """Abstract boundary for tool selection policy.

    Scaffolding: no active implementation.  Override in subclasses.
    """
    def select_tools(self, available: list[str], context: dict[str, Any]) -> list[str]:
        return list(available)  # identity


class PolicySeparationLayer:
    """Container for typed policy layers.

    Separates concerns: each policy layer owns exactly one decision type.
    Policy layers are swappable — the boundary_hash changes when they do.
    """

    def __init__(
        self,
        execution: ExecutionPolicyLayer | None = None,
        planning: PlanningPolicyLayer | None = None,
        tool_selection: ToolSelectionPolicyLayer | None = None,
    ) -> None:
        self.execution = execution or ExecutionPolicyLayer()
        self.planning = planning or PlanningPolicyLayer()
        self.tool_selection = tool_selection or ToolSelectionPolicyLayer()

    def layer_names(self) -> list[str]:
        return ["execution", "planning", "tool_selection"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_policy": type(self.execution).__name__,
            "planning_policy": type(self.planning).__name__,
            "tool_selection_policy": type(self.tool_selection).__name__,
        }


# ---------------------------------------------------------------------------
# Graph introspection API
# ---------------------------------------------------------------------------


class GraphIntrospectionAPI:
    """Read-only introspection layer + structural mutation hooks.

    Purpose:
    - Provides safe, read-only visibility into DAG structure.
    - Registers pre/post-mutation hooks for future mutation systems.
    - Does NOT mutate the DAG directly.

    Hooks fire when a future mutation is applied (not yet implemented).
    """

    def __init__(self) -> None:
        self._pre_hooks: list[Callable[[Any], None]] = []
        self._post_hooks: list[Callable[[Any], None]] = []

    def introspect(self, dag: Any) -> dict[str, Any]:
        """Return a read-only structural summary of the DAG.

        Does NOT modify dag.
        """
        try:
            nodes = list(dag.nodes)
            edges = list(dag.edges)
            dag_hash = dag.deterministic_hash()
        except Exception:
            nodes = []
            edges = []
            dag_hash = ""

        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "dag_hash": dag_hash,
            "node_ids": [str(getattr(n, "node_id", "?")) for n in nodes],
        }

    def hook_pre_mutation(self, callback: Callable[[Any], None]) -> None:
        """Register a callback fired before any future DAG mutation."""
        self._pre_hooks.append(callback)

    def hook_post_mutation(self, callback: Callable[[Any], None]) -> None:
        """Register a callback fired after any future DAG mutation."""
        self._post_hooks.append(callback)

    def clear_hooks(self) -> None:
        self._pre_hooks.clear()
        self._post_hooks.clear()

    def pre_hook_count(self) -> int:
        return len(self._pre_hooks)

    def post_hook_count(self) -> int:
        return len(self._post_hooks)


__all__ = [
    "ExecutionPolicyLayer",
    "ExecutionTelemetryVector",
    "GraphIntrospectionAPI",
    "OptimizationBoundary",
    "PlanningPolicyLayer",
    "PolicySeparationLayer",
    "ToolSelectionPolicyLayer",
]
