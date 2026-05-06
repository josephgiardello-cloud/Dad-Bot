from __future__ import annotations

import hashlib
import json
from typing import Any

from dadbot.core.execution_context import build_tool_invocation_projection


def _stable_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


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


def _failure_recovery_transition_hash(execution_trace_context: dict[str, Any]) -> str:
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


def _ledger_event_chain_hash(ledger_events: list[dict[str, Any]]) -> str:
    normalized = [
        {
            "sequence": int(evt.get("sequence") or 0),
            "type": str(evt.get("type") or evt.get("event_type") or ""),
            "stage": str(evt.get("stage") or ""),
            "job_id": str(evt.get("job_id") or ""),
        }
        for evt in list(ledger_events or [])
    ]
    return _stable_sha256(normalized)


def _invariant_decision_hash(invariant_decisions: list[dict[str, Any]]) -> str:
    normalized = [
        {
            "approved": bool(decision.get("approved", False)),
            "reason": str(decision.get("reason") or ""),
            "scope": str(decision.get("scope") or ""),
        }
        for decision in list(invariant_decisions or [])
    ]
    return _stable_sha256(normalized)


def reduce_official_execution_state(
    *,
    graph_output: str,
    execution_trace_context: dict[str, Any],
    memory_view: dict[str, Any],
    memory_view_state_id: str,
    policy_snapshot: dict[str, Any],
    tool_trace_hash: str = "",
    final_trace_hash_fallback: str = "",
    invariant_decisions: list[dict[str, Any]] | None = None,
    ledger_events: list[dict[str, Any]] | None = None,
    live_tool_mode: bool = False,
) -> dict[str, Any]:
    trace = dict(execution_trace_context or {})
    final_trace_hash = str(trace.get("final_hash") or final_trace_hash_fallback or "")
    execution_dag_hash = str((trace.get("execution_dag") or {}).get("dag_hash") or "")
    external_system_call_graph_hash = str((trace.get("external_system_calls") or {}).get("graph_hash") or "")

    model_output_hashes = _model_output_hashes(trace)
    memory_retrieval_hash = _stable_sha256(list(memory_view.get("memory_retrieval_set") or []))
    policy_hash = _stable_sha256(dict(policy_snapshot or {}))
    execution_order_hash = _execution_order_hash(trace)
    node_decision_sequence_hash = _node_decision_sequence_hash(trace)
    failure_recovery_transition_hash = _failure_recovery_transition_hash(trace)
    tool_invocation_sequence_hash = _stable_sha256(
        build_tool_invocation_projection(trace, live_tool_mode=live_tool_mode),
    )

    snapshot = dict(trace.get("execution_snapshot") or {})
    post_commit_mutation_effects_hash = _stable_sha256(
        {
            "outputs_per_step": list(snapshot.get("outputs_per_step") or []),
            "final_output": str(snapshot.get("final_output") or graph_output or ""),
            "memory_write_intents": list(snapshot.get("memory_write_intents") or []),
            "memory_delta_summary": dict(snapshot.get("memory_delta_summary") or {}),
        },
    )

    invariant_hash = _invariant_decision_hash(list(invariant_decisions or []))
    ledger_hash = _ledger_event_chain_hash(list(ledger_events or []))

    closure_payload = {
        "model_output_hashes": model_output_hashes,
        "memory_retrieval_hash": memory_retrieval_hash,
        "policy_hash": policy_hash,
        "final_trace_hash": final_trace_hash,
        "execution_dag_hash": execution_dag_hash,
        "external_system_call_graph_hash": external_system_call_graph_hash,
        "tool_trace_hash": str(tool_trace_hash or ""),
        "execution_order_hash": execution_order_hash,
        "node_decision_sequence_hash": node_decision_sequence_hash,
        "failure_recovery_transition_hash": failure_recovery_transition_hash,
        "tool_invocation_sequence_hash": tool_invocation_sequence_hash,
        "post_commit_mutation_effects_hash": post_commit_mutation_effects_hash,
        "invariant_decision_hash": invariant_hash,
        "ledger_event_chain_hash": ledger_hash,
    }

    return {
        "schema_version": "1.0",
        "official_state_reducer": "canonical_execution_reducer.v1",
        "final_output": str(graph_output or ""),
        "final_memory_view": dict(memory_view or {}),
        "memory_view_state_id": str(memory_view_state_id or ""),
        "final_trace_hash": final_trace_hash,
        "execution_dag_hash": execution_dag_hash,
        "external_system_call_graph_hash": external_system_call_graph_hash,
        "policy_snapshot": dict(policy_snapshot or {}),
        "model_output_hashes": model_output_hashes,
        "memory_retrieval_hash": memory_retrieval_hash,
        "policy_hash": policy_hash,
        "tool_trace_hash": str(tool_trace_hash or ""),
        "execution_order_hash": execution_order_hash,
        "node_decision_sequence_hash": node_decision_sequence_hash,
        "failure_recovery_transition_hash": failure_recovery_transition_hash,
        "tool_invocation_sequence_hash": tool_invocation_sequence_hash,
        "post_commit_mutation_effects_hash": post_commit_mutation_effects_hash,
        "invariant_decision_hash": invariant_hash,
        "ledger_event_chain_hash": ledger_hash,
        "determinism_closure_hash": _stable_sha256(closure_payload),
    }
