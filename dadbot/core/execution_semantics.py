from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from dadbot.core.execution_trace_context import canonicalize_execution_trace_context


SemanticMode = Literal["exact", "approximate"]


@dataclass(frozen=True)
class AllowedTransformations:
    allow_causal_reorder: bool = False
    allow_representation_variance: bool = False
    tool_output_tolerance: float = 0.0


@dataclass(frozen=True)
class ExecutionState:
    memory_state: dict[str, Any]
    trace_dag: dict[str, Any]
    tool_io_graph: dict[str, Any]
    model_outputs: list[str]
    embedding_state: dict[str, Any]
    semantic_actions: list[dict[str, Any]]


@dataclass(frozen=True)
class ExecutionEquivalenceDecision:
    equivalent: bool
    structural_equivalent: bool
    semantic_equivalent: bool
    invariants_preserved: bool
    violations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "equivalent": bool(self.equivalent),
            "structural_equivalent": bool(self.structural_equivalent),
            "semantic_equivalent": bool(self.semantic_equivalent),
            "invariants_preserved": bool(self.invariants_preserved),
            "violations": list(self.violations),
        }


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _normalize_value(value: Any, *, mode: SemanticMode) -> Any:
    if isinstance(value, str):
        return _normalize_text(value) if mode == "approximate" else value
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 3) if mode == "approximate" else value
    if isinstance(value, dict):
        normalized = {
            str(key): _normalize_value(val, mode=mode)
            for key, val in dict(value).items()
        }
        return dict(sorted(normalized.items(), key=lambda kv: kv[0]))
    if isinstance(value, list):
        normalized_list = [_normalize_value(item, mode=mode) for item in list(value)]
        if mode == "approximate":
            return sorted(normalized_list, key=lambda item: json.dumps(item, sort_keys=True, default=str))
        return normalized_list
    return str(value)


def _semantic_actions(trace_context: dict[str, Any], *, mode: SemanticMode) -> list[dict[str, Any]]:
    trace = canonicalize_execution_trace_context(trace_context)
    actions: list[dict[str, Any]] = []
    for step in list(trace.get("steps") or []):
        operation = str(step.get("operation") or "")
        if operation in {"iteration_start", "iteration_output", "critique_iteration"}:
            continue
        payload = dict(step.get("payload") or {})
        action = {
            "operation": operation,
            "purpose": str(payload.get("purpose") or ""),
            "system": str(payload.get("system") or ""),
            "status": str(payload.get("status") or ""),
        }
        actions.append(_normalize_value(action, mode=mode))
    return actions


def build_execution_state(
    *,
    terminal_state: dict[str, Any],
    execution_trace_context: dict[str, Any],
    semantic_mode: SemanticMode = "exact",
) -> ExecutionState:
    trace = canonicalize_execution_trace_context(execution_trace_context)
    view = dict(terminal_state.get("final_memory_view") or {})
    memory_state = {
        "memory_structured": dict(view.get("memory_structured") or {}),
        "memory_retrieval_set": list(view.get("memory_retrieval_set") or []),
        "memory_full_history_id": str(view.get("memory_full_history_id") or ""),
    }
    trace_dag = dict(trace.get("execution_dag") or {})
    tool_io_graph = dict(trace.get("external_system_calls") or {})
    model_outputs = [
        str((step.get("payload") or {}).get("output_hash") or "")
        for step in list(trace.get("steps") or [])
        if str(step.get("operation") or "") == "model_output"
    ]
    embedding_state = {
        "lock_hash": str((terminal_state.get("policy_snapshot") or {}).get("embedding_lock_hash") or ""),
        "lock_model": str((terminal_state.get("policy_snapshot") or {}).get("embedding_lock_model") or ""),
    }
    return ExecutionState(
        memory_state=_normalize_value(memory_state, mode=semantic_mode),
        trace_dag=_normalize_value(trace_dag, mode="exact"),
        tool_io_graph=_normalize_value(tool_io_graph, mode="exact"),
        model_outputs=_normalize_value(model_outputs, mode=semantic_mode),
        embedding_state=_normalize_value(embedding_state, mode="exact"),
        semantic_actions=_semantic_actions(trace, mode=semantic_mode),
    )


def _structural_equivalent(
    left: ExecutionState,
    right: ExecutionState,
    *,
    rules: AllowedTransformations,
) -> tuple[bool, list[str]]:
    violations: list[str] = []

    left_edges = list((left.trace_dag.get("edges") or []))
    right_edges = list((right.trace_dag.get("edges") or []))
    if left_edges != right_edges:
        violations.append("causal_structure")

    left_nodes = list((left.trace_dag.get("nodes") or []))
    right_nodes = list((right.trace_dag.get("nodes") or []))
    if rules.allow_causal_reorder:
        left_projection = sorted(
            [
                {
                    "operation": str(node.get("operation") or ""),
                    "payload_hash": str(node.get("payload_hash") or ""),
                }
                for node in left_nodes
            ],
            key=lambda item: json.dumps(item, sort_keys=True),
        )
        right_projection = sorted(
            [
                {
                    "operation": str(node.get("operation") or ""),
                    "payload_hash": str(node.get("payload_hash") or ""),
                }
                for node in right_nodes
            ],
            key=lambda item: json.dumps(item, sort_keys=True),
        )
        if left_projection != right_projection:
            violations.append("execution_dag_topology")
    else:
        if left_nodes != right_nodes:
            violations.append("execution_dag_topology")

    if left.tool_io_graph != right.tool_io_graph:
        violations.append("tool_io_graph")

    return (len(violations) == 0, violations)


def _semantic_equivalent(
    left: ExecutionState,
    right: ExecutionState,
    *,
    rules: AllowedTransformations,
) -> tuple[bool, list[str]]:
    violations: list[str] = []

    if left.semantic_actions != right.semantic_actions:
        violations.append("semantic_actions")
    if left.memory_state != right.memory_state:
        violations.append("memory_state")

    if rules.allow_representation_variance:
        left_tools = sorted(left.model_outputs)
        right_tools = sorted(right.model_outputs)
    else:
        left_tools = list(left.model_outputs)
        right_tools = list(right.model_outputs)
    if left_tools != right_tools:
        violations.append("model_outputs")

    return (len(violations) == 0, violations)


def _invariants_preserved(left: ExecutionState, right: ExecutionState) -> tuple[bool, list[str]]:
    violations: list[str] = []
    if str(left.trace_dag.get("dag_hash") or "") != str(right.trace_dag.get("dag_hash") or ""):
        violations.append("trace_dag_hash")
    if str(left.tool_io_graph.get("graph_hash") or "") != str(right.tool_io_graph.get("graph_hash") or ""):
        violations.append("tool_io_graph_hash")
    if left.embedding_state != right.embedding_state:
        violations.append("embedding_lock_state")
    return (len(violations) == 0, violations)


def execution_equivalence_relation(
    left: ExecutionState,
    right: ExecutionState,
    *,
    rules: AllowedTransformations | None = None,
) -> ExecutionEquivalenceDecision:
    active_rules = rules or AllowedTransformations()
    structural_ok, structural_violations = _structural_equivalent(left, right, rules=active_rules)
    semantic_ok, semantic_violations = _semantic_equivalent(left, right, rules=active_rules)
    invariants_ok, invariant_violations = _invariants_preserved(left, right)

    violations = list(structural_violations) + list(semantic_violations) + list(invariant_violations)
    return ExecutionEquivalenceDecision(
        equivalent=bool(structural_ok and semantic_ok and invariants_ok),
        structural_equivalent=bool(structural_ok),
        semantic_equivalent=bool(semantic_ok),
        invariants_preserved=bool(invariants_ok),
        violations=violations,
    )
