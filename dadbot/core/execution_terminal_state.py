from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from dadbot.core.execution_context import build_tool_invocation_projection
from dadbot.core.execution_memory_view import ExecutionMemoryView
from dadbot.core.graph_context import TurnContext

_SCHEMA_VERSION = "1.0"


def _stable_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


def _final_output_value(finalized_result: Any) -> str:
    if isinstance(finalized_result, tuple) and finalized_result:
        return str(finalized_result[0] or "")
    return str(finalized_result or "")


def _model_output_hashes(execution_trace_context: dict[str, Any]) -> list[str]:
    steps = list(execution_trace_context.get("steps") or [])
    hashes: list[str] = []
    for step in steps:
        if str(step.get("operation") or "") != "model_output":
            continue
        payload = dict(step.get("payload") or {})
        output_hash = str(payload.get("output_hash") or "").strip()
        if output_hash:
            hashes.append(output_hash)
    return hashes


def _execution_order_hash(execution_trace_context: dict[str, Any]) -> str:
    operations = list(execution_trace_context.get("operations") or [])
    if not operations:
        operations = [
            str(step.get("operation") or "")
            for step in list(execution_trace_context.get("steps") or [])
        ]
    return _stable_sha256(list(operations))


def _node_decision_sequence_hash(execution_trace_context: dict[str, Any]) -> str:
    sequence: list[dict[str, Any]] = []
    for step in list(execution_trace_context.get("steps") or []):
        payload = dict(step.get("payload") or {})
        sequence.append(
            {
                "operation": str(step.get("operation") or ""),
                "status": str(payload.get("status") or ""),
                "purpose": str(payload.get("purpose") or ""),
                "system": str(payload.get("system") or ""),
                "passed": bool(payload.get("passed", False)),
                "issue_count": int(payload.get("issue_count") or 0),
            },
        )
    return _stable_sha256(sequence)


def _failure_recovery_transition_hash(
    execution_trace_context: dict[str, Any],
    determinism_manifest: dict[str, Any],
) -> str:
    failure_replay = list(determinism_manifest.get("failure_replay") or [])
    if failure_replay:
        projection = [
            {
                "stage": str(item.get("stage") or ""),
                "error_type": str(item.get("error_type") or ""),
                "error_msg": str(item.get("error_msg") or ""),
            }
            for item in failure_replay
        ]
        return _stable_sha256(projection)

    transitions: list[dict[str, Any]] = []
    for step in list(execution_trace_context.get("steps") or []):
        payload = dict(step.get("payload") or {})
        status = str(payload.get("status") or "")
        if status.lower() in {"error", "failed", "retry", "recover", "recovered"}:
            transitions.append(
                {
                    "operation": str(step.get("operation") or ""),
                    "status": status,
                    "error": str(payload.get("error") or payload.get("error_type") or ""),
                },
            )
    return _stable_sha256(transitions)


def _tool_invocation_sequence_hash(execution_trace_context: dict[str, Any]) -> str:
    return _stable_sha256(build_tool_invocation_projection(execution_trace_context))


def _post_commit_mutation_effects_hash(
    context: TurnContext,
    execution_trace_context: dict[str, Any],
) -> str:
    snapshot = dict(execution_trace_context.get("execution_snapshot") or {})
    payload = {
        "outputs_per_step": list(snapshot.get("outputs_per_step") or []),
        "final_output": str(
            snapshot.get("final_output")
            or execution_trace_context.get("normalized_response")
            or ""
        ),
        "memory_write_intents": list(snapshot.get("memory_write_intents") or []),
        "memory_delta_summary": dict(snapshot.get("memory_delta_summary") or {}),
        "memory_influence_feedback": dict(context.state.get("memory_influence_feedback") or {}),
        "output_coherence": dict(context.state.get("output_coherence") or {}),
    }
    return _stable_sha256(payload)


def _policy_snapshot(context: TurnContext) -> dict[str, Any]:
    metadata = dict(context.metadata or {})
    state = dict(context.state or {})
    return {
        "kernel_policy": dict(metadata.get("kernel_policy") or {}),
        "kernel_rejections": list(metadata.get("kernel_rejections") or []),
        "capability_audit_report": dict(state.get("capability_audit_report") or {}),
        "safety_check_result": dict(state.get("safety_check_result") or {}),
        "tony_level": str(state.get("tony_level") or ""),
        "tony_score": int(state.get("tony_score") or 0),
    }


@dataclass(frozen=True)
class ExecutionTerminalState:
    schema_version: str
    final_output: str
    final_memory_view: dict[str, Any]
    memory_view_state_id: str
    final_trace_hash: str
    execution_dag_hash: str
    policy_snapshot: dict[str, Any]
    model_output_hashes: list[str]
    memory_retrieval_hash: str
    policy_hash: str
    tool_trace_hash: str
    execution_order_hash: str
    node_decision_sequence_hash: str
    failure_recovery_transition_hash: str
    tool_invocation_sequence_hash: str
    post_commit_mutation_effects_hash: str
    determinism_closure_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "final_output": self.final_output,
            "final_memory_view": self.final_memory_view,
            "memory_view_state_id": self.memory_view_state_id,
            "final_trace_hash": self.final_trace_hash,
            "execution_dag_hash": self.execution_dag_hash,
            "policy_snapshot": self.policy_snapshot,
            "model_output_hashes": self.model_output_hashes,
            "memory_retrieval_hash": self.memory_retrieval_hash,
            "policy_hash": self.policy_hash,
            "tool_trace_hash": self.tool_trace_hash,
            "execution_order_hash": self.execution_order_hash,
            "node_decision_sequence_hash": self.node_decision_sequence_hash,
            "failure_recovery_transition_hash": self.failure_recovery_transition_hash,
            "tool_invocation_sequence_hash": self.tool_invocation_sequence_hash,
            "post_commit_mutation_effects_hash": self.post_commit_mutation_effects_hash,
            "determinism_closure_hash": self.determinism_closure_hash,
        }


def build_execution_terminal_state(
    context: TurnContext,
    *,
    finalized_result: Any,
) -> ExecutionTerminalState:
    trace_context = dict(context.metadata.get("execution_trace_context") or {})
    final_trace_hash = str(trace_context.get("final_hash") or "")
    execution_dag = dict(trace_context.get("execution_dag") or {})
    execution_dag_hash = str(execution_dag.get("dag_hash") or "")

    model_hashes = _model_output_hashes(trace_context)
    memory_view = ExecutionMemoryView.from_context(context)
    final_memory_view = memory_view.to_dict()
    policy_snapshot = _policy_snapshot(context)

    memory_retrieval_hash = _stable_sha256(list(memory_view.memory_retrieval_set or []))
    policy_hash = _stable_sha256(policy_snapshot)
    execution_order_hash = _execution_order_hash(trace_context)
    node_decision_sequence_hash = _node_decision_sequence_hash(trace_context)
    failure_recovery_transition_hash = _failure_recovery_transition_hash(
        trace_context,
        dict(getattr(context, "determinism_manifest", {}) or {}),
    )
    tool_invocation_sequence_hash = _tool_invocation_sequence_hash(trace_context)
    post_commit_mutation_effects_hash = _post_commit_mutation_effects_hash(
        context,
        trace_context,
    )

    determinism = dict(context.metadata.get("determinism") or {})
    tool_trace_hash = str(determinism.get("tool_trace_hash") or "")

    closure_payload = {
        "model_output_hashes": model_hashes,
        "memory_retrieval_hash": memory_retrieval_hash,
        "policy_hash": policy_hash,
        "final_trace_hash": final_trace_hash,
        "execution_dag_hash": execution_dag_hash,
        "tool_trace_hash": tool_trace_hash,
        "execution_order_hash": execution_order_hash,
        "node_decision_sequence_hash": node_decision_sequence_hash,
        "failure_recovery_transition_hash": failure_recovery_transition_hash,
        "tool_invocation_sequence_hash": tool_invocation_sequence_hash,
        "post_commit_mutation_effects_hash": post_commit_mutation_effects_hash,
    }
    determinism_closure_hash = _stable_sha256(closure_payload)

    return ExecutionTerminalState(
        schema_version=_SCHEMA_VERSION,
        final_output=_final_output_value(finalized_result),
        final_memory_view=final_memory_view,
        memory_view_state_id=memory_view.state_id,
        final_trace_hash=final_trace_hash,
        execution_dag_hash=execution_dag_hash,
        policy_snapshot=policy_snapshot,
        model_output_hashes=model_hashes,
        memory_retrieval_hash=memory_retrieval_hash,
        policy_hash=policy_hash,
        tool_trace_hash=tool_trace_hash,
        execution_order_hash=execution_order_hash,
        node_decision_sequence_hash=node_decision_sequence_hash,
        failure_recovery_transition_hash=failure_recovery_transition_hash,
        tool_invocation_sequence_hash=tool_invocation_sequence_hash,
        post_commit_mutation_effects_hash=post_commit_mutation_effects_hash,
        determinism_closure_hash=determinism_closure_hash,
    )
