from __future__ import annotations

import hashlib
import json
from typing import Any

from dadbot.core.execution_memory_view import ExecutionMemoryView
from dadbot.core.execution_context import (
    build_tool_invocation_projection,
    canonicalize_execution_trace_context,
    derive_execution_trace_hash,
)


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


def _tool_invocation_sequence_hash(
    execution_trace_context: dict[str, Any],
    *,
    live_tool_mode: bool = False,
) -> str:
    return _stable_sha256(
        build_tool_invocation_projection(
            execution_trace_context,
            live_tool_mode=live_tool_mode,
        ),
    )


def _post_commit_mutation_effects_hash(execution_trace_context: dict[str, Any]) -> str:
    snapshot = dict(execution_trace_context.get("execution_snapshot") or {})
    payload = {
        "outputs_per_step": list(snapshot.get("outputs_per_step") or []),
        "final_output": str(snapshot.get("final_output") or execution_trace_context.get("normalized_response") or ""),
        "memory_write_intents": list(snapshot.get("memory_write_intents") or []),
        "memory_delta_summary": dict(snapshot.get("memory_delta_summary") or {}),
    }
    return _stable_sha256(payload)


def reconstruct_terminal_state_from_trace(
    *,
    terminal_state_seed: dict[str, Any],
    execution_trace_context: dict[str, Any],
    memory_view_override: dict[str, Any] | None = None,
    policy_snapshot_override: dict[str, Any] | None = None,
    live_tool_mode: bool = False,
) -> dict[str, Any]:
    seed = dict(terminal_state_seed or {})
    trace = dict(execution_trace_context or {})
    canonical_trace = canonicalize_execution_trace_context(trace)

    memory_view = ExecutionMemoryView.from_trace(
        canonical_trace,
        fallback_memory_view=memory_view_override or seed.get("final_memory_view") or {},
    )
    policy_snapshot = dict(
        policy_snapshot_override or seed.get("policy_snapshot") or {},
    )

    claimed_final_trace_hash = str(trace.get("final_hash") or "")
    final_trace_hash = derive_execution_trace_hash(canonical_trace)
    execution_dag_hash = str(
        (canonical_trace.get("execution_dag") or {}).get("dag_hash") or "",
    )
    external_system_call_graph_hash = str(
        (canonical_trace.get("external_system_calls") or {}).get("graph_hash") or "",
    )
    model_output_hashes = _model_output_hashes(canonical_trace)
    memory_retrieval_hash = _stable_sha256(list(memory_view.memory_retrieval_set or []))
    policy_hash = _stable_sha256(policy_snapshot)
    execution_order_hash = _execution_order_hash(canonical_trace)
    node_decision_sequence_hash = _node_decision_sequence_hash(canonical_trace)
    failure_recovery_transition_hash = _failure_recovery_transition_hash(canonical_trace)
    tool_invocation_sequence_hash = _tool_invocation_sequence_hash(
        canonical_trace,
        live_tool_mode=live_tool_mode,
    )
    post_commit_mutation_effects_hash = _post_commit_mutation_effects_hash(canonical_trace)
    tool_trace_hash = str(seed.get("tool_trace_hash") or "")

    closure_payload = {
        "model_output_hashes": model_output_hashes,
        "memory_retrieval_hash": memory_retrieval_hash,
        "policy_hash": policy_hash,
        "final_trace_hash": final_trace_hash,
        "execution_dag_hash": execution_dag_hash,
        "external_system_call_graph_hash": external_system_call_graph_hash,
        "tool_trace_hash": tool_trace_hash,
        "execution_order_hash": execution_order_hash,
        "node_decision_sequence_hash": node_decision_sequence_hash,
        "failure_recovery_transition_hash": failure_recovery_transition_hash,
        "tool_invocation_sequence_hash": tool_invocation_sequence_hash,
        "post_commit_mutation_effects_hash": post_commit_mutation_effects_hash,
    }

    return {
        "schema_version": str(seed.get("schema_version") or "1.0"),
        "final_output": str(
            canonical_trace.get("normalized_response") or seed.get("final_output") or "",
        ),
        "final_memory_view": memory_view.to_dict(),
        "memory_view_state_id": memory_view.state_id,
        "final_trace_hash": final_trace_hash,
        "claimed_final_trace_hash": claimed_final_trace_hash,
        "execution_dag_hash": execution_dag_hash,
        "external_system_call_graph_hash": external_system_call_graph_hash,
        "policy_snapshot": policy_snapshot,
        "model_output_hashes": model_output_hashes,
        "memory_retrieval_hash": memory_retrieval_hash,
        "policy_hash": policy_hash,
        "tool_trace_hash": tool_trace_hash,
        "execution_order_hash": execution_order_hash,
        "node_decision_sequence_hash": node_decision_sequence_hash,
        "failure_recovery_transition_hash": failure_recovery_transition_hash,
        "tool_invocation_sequence_hash": tool_invocation_sequence_hash,
        "post_commit_mutation_effects_hash": post_commit_mutation_effects_hash,
        "determinism_closure_hash": _stable_sha256(closure_payload),
    }


def verify_terminal_state_replay_equivalence(
    *,
    terminal_state_seed: dict[str, Any],
    execution_trace_context: dict[str, Any],
    memory_view_override: dict[str, Any] | None = None,
    policy_snapshot_override: dict[str, Any] | None = None,
    enforce_dag_equivalence: bool = True,
    live_tool_mode: bool = False,
) -> dict[str, Any]:
    expected = dict(terminal_state_seed or {})
    replayed = reconstruct_terminal_state_from_trace(
        terminal_state_seed=expected,
        execution_trace_context=execution_trace_context,
        memory_view_override=memory_view_override,
        policy_snapshot_override=policy_snapshot_override,
        live_tool_mode=live_tool_mode,
    )

    violations: list[str] = []

    def _assert_equal(field: str) -> None:
        if expected.get(field) != replayed.get(field):
            violations.append(field)

    _assert_equal("final_output")
    _assert_equal("final_trace_hash")
    _assert_equal("model_output_hashes")
    _assert_equal("memory_retrieval_hash")
    _assert_equal("policy_hash")
    _assert_equal("external_system_call_graph_hash")
    _assert_equal("execution_order_hash")
    _assert_equal("node_decision_sequence_hash")
    _assert_equal("failure_recovery_transition_hash")
    _assert_equal("tool_invocation_sequence_hash")
    _assert_equal("post_commit_mutation_effects_hash")
    _assert_equal("determinism_closure_hash")

    expected_state_id = str(expected.get("memory_view_state_id") or "")
    replayed_state_id = str(replayed.get("memory_view_state_id") or "")
    if expected_state_id and expected_state_id != replayed_state_id:
        violations.append("memory_view_state_id")

    if enforce_dag_equivalence and expected.get("execution_dag_hash") != replayed.get(
        "execution_dag_hash",
    ):
        violations.append("execution_dag_hash")

    return {
        "equivalent": len(violations) == 0,
        "violations": violations,
        "expected_terminal_state": expected,
        "replayed_terminal_state": replayed,
    }
