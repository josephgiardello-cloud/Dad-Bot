from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4


def _stable_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


@dataclass(frozen=True)
class ToolExecutionTraceNode:
    node_id: str
    sequence: int
    operation: str
    system: str
    request_hash: str
    response_hash: str
    status: str
    time_token: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExternalSystemCallGraph:
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    graph_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_time_token(
    *,
    seq: int,
    operation: str,
    system: str,
    request_hash: str,
) -> str:
    payload = {
        "seq": int(seq),
        "operation": str(operation or "").strip().lower(),
        "system": str(system or "").strip().lower(),
        "request_hash": str(request_hash or "").strip().lower(),
    }
    return _stable_sha256(payload)[:16]


def record_external_system_call(
    *,
    operation: str,
    system: str,
    request_payload: dict[str, Any] | None = None,
    response_payload: dict[str, Any] | None = None,
    status: str = "ok",
    source: str = "",
    deterministic_id: str = "",
    required: bool = False,
) -> dict[str, Any] | None:
    request = dict(request_payload or {})
    response = dict(response_payload or {})
    normalized_operation = str(operation or "external_system_call").strip().lower() or "external_system_call"
    normalized_system = str(system or "unknown").strip().lower() or "unknown"
    request_hash = _stable_sha256(request)
    response_hash = _stable_sha256(response)
    seq_hint = int(request.get("sequence") or request.get("seq") or 0)
    time_token = _normalize_time_token(
        seq=seq_hint,
        operation=normalized_operation,
        system=normalized_system,
        request_hash=request_hash,
    )
    payload = {
        "operation": normalized_operation,
        "system": normalized_system,
        "status": str(status or "ok").strip().lower() or "ok",
        "source": str(source or "").strip().lower(),
        "deterministic_id": str(deterministic_id or "").strip(),
        "request": request,
        "response": response,
        "request_hash": request_hash,
        "response_hash": response_hash,
        "time_token": time_token,
    }
    return record_execution_step(
        "external_system_call",
        payload=payload,
        required=required,
    )


def build_external_system_call_graph(steps: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_ids: list[str] = []
    for step in steps:
        if str(step.get("operation") or "").strip().lower() != "external_system_call":
            continue
        seq = int(step.get("seq") or 0)
        payload = dict(step.get("payload") or {})
        operation = str(payload.get("operation") or "external_system_call").strip().lower()
        system = str(payload.get("system") or "unknown").strip().lower()
        request_hash = str(payload.get("request_hash") or "")
        response_hash = str(payload.get("response_hash") or "")
        status = str(payload.get("status") or "ok").strip().lower() or "ok"
        time_token = str(payload.get("time_token") or "")
        node_id = f"ext:{seq}:{system}:{request_hash[:8] or 'na'}"
        node = ToolExecutionTraceNode(
            node_id=node_id,
            sequence=seq,
            operation=operation,
            system=system,
            request_hash=request_hash,
            response_hash=response_hash,
            status=status,
            time_token=time_token,
        )
        node_ids.append(node_id)
        nodes.append(node.to_dict())

    for idx in range(1, len(node_ids)):
        edges.append(
            {"from": node_ids[idx - 1], "to": node_ids[idx], "type": "sequence"},
        )

    graph_payload = {"nodes": nodes, "edges": edges}
    graph = ExternalSystemCallGraph(
        nodes=nodes,
        edges=edges,
        graph_hash=_stable_sha256(graph_payload),
    )
    return graph.to_dict()


class RuntimeTraceViolation(RuntimeError):
    """Raised when a trace-required operation executes without an active trace."""


class ExecutionTraceRecorder:
    """Turn-scoped recorder for authoritative execution operations."""

    def __init__(
        self,
        *,
        trace_id: str,
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ):
        self.trace_id = str(trace_id or "")
        self.prompt = str(prompt or "")
        self.metadata = dict(metadata or {})
        self._steps: list[dict[str, Any]] = []

    def record(
        self,
        operation: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        step = {
            "seq": len(self._steps),
            "operation": str(operation or "").strip().lower(),
            "payload": dict(payload or {}),
        }
        self._steps.append(step)
        return step

    @property
    def steps(self) -> list[dict[str, Any]]:
        return [dict(step) for step in self._steps]


_ACTIVE_TRACE: contextvars.ContextVar[ExecutionTraceRecorder | None] = contextvars.ContextVar(
    "dadbot_active_execution_trace",
    default=None,
)
_TRACE_REQUIRED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "dadbot_execution_trace_required",
    default=False,
)


def active_execution_trace() -> ExecutionTraceRecorder | None:
    return _ACTIVE_TRACE.get()


def require_execution_trace(*, operation: str) -> ExecutionTraceRecorder | None:
    recorder = active_execution_trace()
    if recorder is None and bool(_TRACE_REQUIRED.get()):
        raise RuntimeTraceViolation(
            f"Execution trace required for operation '{operation}' but no active trace recorder was bound",
        )
    return recorder


@contextlib.contextmanager
def bind_execution_trace(recorder: ExecutionTraceRecorder, *, required: bool = True):
    token_trace = _ACTIVE_TRACE.set(recorder)
    token_required = _TRACE_REQUIRED.set(bool(required))
    try:
        yield recorder
    finally:
        _TRACE_REQUIRED.reset(token_required)
        _ACTIVE_TRACE.reset(token_trace)


def record_execution_step(
    operation: str,
    *,
    payload: dict[str, Any] | None = None,
    required: bool = False,
) -> dict[str, Any] | None:
    recorder = active_execution_trace()
    must_exist = bool(required or _TRACE_REQUIRED.get())
    if recorder is None:
        if must_exist:
            raise RuntimeTraceViolation(
                f"Execution trace step '{operation}' emitted without active trace recorder",
            )
        return None
    return recorder.record(operation=operation, payload=payload)


@contextlib.contextmanager
def ensure_execution_trace_root(
    *,
    operation: str,
    prompt: str = "",
    metadata: dict[str, Any] | None = None,
    required: bool = True,
):
    """Ensure an active trace recorder exists for side-effecting entrypoints.

    If a recorder is already bound, this is a no-op passthrough. Otherwise a
    synthetic out-of-band root recorder is created and bound for the scope.
    """
    active = active_execution_trace()
    if active is not None:
        yield active
        return

    root_op = str(operation or "out_of_band_entry").strip().lower() or "out_of_band_entry"
    synthetic_trace_id = f"oob-{uuid4().hex}"
    recorder = ExecutionTraceRecorder(
        trace_id=synthetic_trace_id,
        prompt=str(prompt or f"[{root_op}]"),
        metadata={
            "out_of_band": True,
            "root_operation": root_op,
            **dict(metadata or {}),
        },
    )
    with bind_execution_trace(recorder, required=required):
        yield recorder


@dataclass(frozen=True)
class ExecutionTraceContext:
    schema_version: str
    prompt: str
    memory_snapshot_used: dict[str, Any]
    model_call_parameters: dict[str, Any]
    model_output: Any
    memory_retrieval_set: list[dict[str, Any]]
    tool_outputs: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    operations: list[str]
    execution_snapshot: dict[str, Any]
    execution_dag: dict[str, Any]
    external_system_calls: dict[str, Any]
    normalized_response: str
    final_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalized_steps(raw_steps: list[Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for index, item in enumerate(list(raw_steps or [])):
        if not isinstance(item, dict):
            continue
        steps.append(
            {
                "seq": int(
                    item.get("seq") if isinstance(item.get("seq"), int) else index,
                ),
                "operation": str(item.get("operation") or "").strip().lower(),
                "payload": dict(item.get("payload") or {}),
            },
        )
    return steps


def canonicalize_execution_trace_context(
    trace_context: dict[str, Any],
) -> dict[str, Any]:
    trace = dict(trace_context or {})
    steps = _normalized_steps(list(trace.get("steps") or []))
    external_system_calls = build_external_system_call_graph(steps)
    execution_dag = _build_execution_dag(
        steps,
        external_system_calls=external_system_calls,
    )
    operations = [str(step.get("operation") or "") for step in steps]

    return {
        "schema_version": str(trace.get("schema_version") or "2.0"),
        "prompt": str(trace.get("prompt") or ""),
        "memory_snapshot_used": dict(trace.get("memory_snapshot_used") or {}),
        "model_call_parameters": dict(trace.get("model_call_parameters") or {}),
        "model_output": trace.get("model_output"),
        "memory_retrieval_set": list(trace.get("memory_retrieval_set") or []),
        "tool_outputs": list(trace.get("tool_outputs") or []),
        "steps": steps,
        "operations": operations,
        "execution_snapshot": dict(trace.get("execution_snapshot") or {}),
        "execution_dag": execution_dag,
        "external_system_calls": external_system_calls,
        "normalized_response": str(trace.get("normalized_response") or ""),
    }


def derive_execution_trace_hash(trace_context: dict[str, Any]) -> str:
    canonical = canonicalize_execution_trace_context(trace_context)
    return _stable_sha256(canonical)


def build_execution_trace_context(
    *,
    context,
    result,
    recorder: ExecutionTraceRecorder | None = None,
) -> dict[str, Any]:
    state = dict(getattr(context, "state", {}) or {})
    metadata = dict(getattr(context, "metadata", {}) or {})
    determinism = dict(metadata.get("determinism") or {})

    normalized_response = str(
        (result[0] if isinstance(result, tuple) else result) or "",
    )

    retrieval_set = list(
        state.get("memory_retrieval_set") or state.get("retrieval_set") or state.get("memory_retrieval_results") or [],
    )
    retrieval_records = [item if isinstance(item, dict) else {"value": item} for item in retrieval_set]

    tool_outputs = list(state.get("tool_results") or [])
    tool_output_records = [item if isinstance(item, dict) else {"value": item} for item in tool_outputs]

    memory_snapshot_used = {
        "memory_fingerprint": str(determinism.get("memory_fingerprint") or ""),
        "memory_structured": dict(state.get("memory_structured") or {}),
        "memory_full_history_id": str(state.get("memory_full_history_id") or ""),
    }

    model_call_parameters = {
        "provider": str(determinism.get("llm_provider") or ""),
        "model": str(determinism.get("llm_model") or ""),
        "seed_policy": str(determinism.get("seed_policy") or ""),
        "temperature_policy": str(determinism.get("temperature_policy") or ""),
    }

    model_output = state.get("model_output") or state.get("inference_output") or normalized_response

    trace_recorder = recorder or active_execution_trace()
    recorded_steps = trace_recorder.steps if trace_recorder is not None else []
    operations = [str(step.get("operation") or "") for step in recorded_steps]
    execution_snapshot = _build_execution_snapshot(
        context=context,
        result=result,
        steps=recorded_steps,
        memory_snapshot_used=memory_snapshot_used,
        model_call_parameters=model_call_parameters,
    )
    external_system_calls = build_external_system_call_graph(recorded_steps)
    execution_dag = _build_execution_dag(
        recorded_steps,
        external_system_calls=external_system_calls,
    )

    payload = {
        "schema_version": "2.0",
        "prompt": str(getattr(context, "user_input", "") or ""),
        "memory_snapshot_used": memory_snapshot_used,
        "model_call_parameters": model_call_parameters,
        "model_output": model_output,
        "memory_retrieval_set": retrieval_records,
        "tool_outputs": tool_output_records,
        "steps": recorded_steps,
        "operations": operations,
        "execution_snapshot": execution_snapshot,
        "execution_dag": execution_dag,
        "external_system_calls": external_system_calls,
        "normalized_response": normalized_response,
    }
    final_hash = derive_execution_trace_hash(payload)

    trace = ExecutionTraceContext(
        schema_version=payload["schema_version"],
        prompt=payload["prompt"],
        memory_snapshot_used=memory_snapshot_used,
        model_call_parameters=model_call_parameters,
        model_output=model_output,
        memory_retrieval_set=retrieval_records,
        tool_outputs=tool_output_records,
        steps=recorded_steps,
        operations=operations,
        execution_snapshot=execution_snapshot,
        execution_dag=execution_dag,
        external_system_calls=external_system_calls,
        normalized_response=normalized_response,
        final_hash=final_hash,
    )
    return trace.to_dict()


def _step_input_payload(step: dict[str, Any]) -> dict[str, Any]:
    payload = dict(step.get("payload") or {})
    projection: dict[str, Any] = {}
    for key in (
        "iteration",
        "mode",
        "provider",
        "model",
        "purpose",
        "message_count",
        "input_hash",
    ):
        if key in payload:
            projection[key] = payload.get(key)
    return projection


def _step_output_payload(step: dict[str, Any]) -> dict[str, Any]:
    payload = dict(step.get("payload") or {})
    projection: dict[str, Any] = {}
    for key in (
        "passed",
        "issue_count",
        "reply_preview",
        "has_error",
        "output_hash",
        "output_length",
    ):
        if key in payload:
            projection[key] = payload.get(key)
    return projection


def _build_execution_snapshot(
    *,
    context: Any,
    result: Any,
    steps: list[dict[str, Any]],
    memory_snapshot_used: dict[str, Any],
    model_call_parameters: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(getattr(context, "metadata", {}) or {})
    state = dict(getattr(context, "state", {}) or {})
    control_plane = dict(metadata.get("control_plane") or {})
    normalized_response = str(
        (result[0] if isinstance(result, tuple) else result) or "",
    )

    outputs_per_step = [
        {
            "seq": int(step.get("seq") or 0),
            "operation": str(step.get("operation") or ""),
            "input": _step_input_payload(step),
            "output": _step_output_payload(step),
            "payload_hash": _stable_sha256(dict(step.get("payload") or {})),
        }
        for step in steps
    ]

    model_inputs = [_step_input_payload(step) for step in steps if str(step.get("operation") or "") == "model_call"]

    snapshot = {
        "inputs": {
            "prompt": str(getattr(context, "user_input", "") or ""),
            "trace_id": str(getattr(context, "trace_id", "") or ""),
            "session_id": str(control_plane.get("session_id") or ""),
        },
        "memory_snapshot": {
            **memory_snapshot_used,
            "retrieval_set": list(state.get("memory_retrieval_set") or []),
        },
        "model_inputs": {
            "parameters": dict(model_call_parameters),
            "calls": model_inputs,
        },
        "outputs_per_step": outputs_per_step,
        "final_output": normalized_response,
    }
    snapshot["snapshot_hash"] = _stable_sha256(snapshot)
    return snapshot


def _build_execution_dag(
    steps: list[dict[str, Any]],
    *,
    external_system_calls: dict[str, Any] | None = None,
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_ids: list[str] = []

    for step in steps:
        seq = int(step.get("seq") or 0)
        operation = str(step.get("operation") or "")
        node_id = f"step:{seq}:{operation or 'unknown'}"
        node_ids.append(node_id)
        nodes.append(
            {
                "id": node_id,
                "seq": seq,
                "operation": operation,
                "payload_hash": _stable_sha256(dict(step.get("payload") or {})),
            },
        )

    for idx in range(1, len(node_ids)):
        edges.append(
            {
                "from": node_ids[idx - 1],
                "to": node_ids[idx],
                "type": "sequence",
            },
        )

    iteration_starts: dict[int, str] = {}
    for step, node_id in zip(steps, node_ids):
        operation = str(step.get("operation") or "")
        payload = dict(step.get("payload") or {})
        if "iteration" not in payload:
            continue
        iteration = int(payload.get("iteration") or 0)
        if operation == "iteration_start":
            iteration_starts[iteration] = node_id
            continue
        source = iteration_starts.get(iteration)
        if source and source != node_id:
            edges.append(
                {
                    "from": source,
                    "to": node_id,
                    "type": "iteration",
                },
            )

    graph = {
        "nodes": nodes,
        "edges": edges,
        "entry": node_ids[0] if node_ids else "",
        "topological_order": node_ids,
        "external_system_call_graph_hash": str(
            (external_system_calls or {}).get("graph_hash") or "",
        ),
    }
    graph["dag_hash"] = _stable_sha256(graph)
    return graph
