from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

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

    determinism = dict(context.metadata.get("determinism") or {})
    tool_trace_hash = str(determinism.get("tool_trace_hash") or "")

    closure_payload = {
        "model_output_hashes": model_hashes,
        "memory_retrieval_hash": memory_retrieval_hash,
        "policy_hash": policy_hash,
        "final_trace_hash": final_trace_hash,
        "execution_dag_hash": execution_dag_hash,
        "tool_trace_hash": tool_trace_hash,
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
        determinism_closure_hash=determinism_closure_hash,
    )
