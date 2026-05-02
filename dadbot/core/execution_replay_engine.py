from __future__ import annotations

import hashlib
import json
from typing import Any

from dadbot.core.execution_memory_view import ExecutionMemoryView
from dadbot.core.execution_context import (
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


def reconstruct_terminal_state_from_trace(
    *,
    terminal_state_seed: dict[str, Any],
    execution_trace_context: dict[str, Any],
    memory_view_override: dict[str, Any] | None = None,
    policy_snapshot_override: dict[str, Any] | None = None,
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
    tool_trace_hash = str(seed.get("tool_trace_hash") or "")

    closure_payload = {
        "model_output_hashes": model_output_hashes,
        "memory_retrieval_hash": memory_retrieval_hash,
        "policy_hash": policy_hash,
        "final_trace_hash": final_trace_hash,
        "execution_dag_hash": execution_dag_hash,
        "external_system_call_graph_hash": external_system_call_graph_hash,
        "tool_trace_hash": tool_trace_hash,
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
        "determinism_closure_hash": _stable_sha256(closure_payload),
    }


def verify_terminal_state_replay_equivalence(
    *,
    terminal_state_seed: dict[str, Any],
    execution_trace_context: dict[str, Any],
    memory_view_override: dict[str, Any] | None = None,
    policy_snapshot_override: dict[str, Any] | None = None,
    enforce_dag_equivalence: bool = True,
) -> dict[str, Any]:
    expected = dict(terminal_state_seed or {})
    replayed = reconstruct_terminal_state_from_trace(
        terminal_state_seed=expected,
        execution_trace_context=execution_trace_context,
        memory_view_override=memory_view_override,
        policy_snapshot_override=policy_snapshot_override,
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
