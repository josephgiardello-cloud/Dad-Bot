from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from dadbot.core.mutation_entry_invariants import enforce_mutation_entry_invariants


logger = logging.getLogger(__name__)


def _stable_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8"),
    ).hexdigest()


def _json_clone(payload: Any) -> Any:
    return json.loads(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ),
    )


def _drop_ignored_fields(payload: Any, ignored_fields: list[str]) -> Any:
    cloned = _json_clone(payload)
    if not ignored_fields:
        return cloned

    for field_path in ignored_fields:
        path = [segment.strip() for segment in str(field_path or "").split(".") if segment.strip()]
        if not path:
            continue
        cursor: Any = cloned
        for idx, segment in enumerate(path):
            is_leaf = idx == len(path) - 1
            if not isinstance(cursor, dict):
                break
            if is_leaf:
                cursor.pop(segment, None)
                break
            cursor = cursor.get(segment)
    return cloned


def _normalize_determinism_contract(contract: dict[str, Any] | None) -> dict[str, Any]:
    base = dict(contract or {})
    ignore_request = [
        str(item).strip() for item in list(base.get("ignore_request_fields") or []) if str(item).strip()
    ]
    ignore_response = [
        str(item).strip() for item in list(base.get("ignore_response_fields") or []) if str(item).strip()
    ]
    return {
        "schema_version": str(base.get("schema_version") or "1.0"),
        "ignore_request_fields": sorted(set(ignore_request)),
        "ignore_response_fields": sorted(set(ignore_response)),
        "mode": str(base.get("mode") or "replay_strict"),
    }


def _coerce_memory_snapshot(snapshot: Any) -> tuple[dict[str, Any], str]:
    if not isinstance(snapshot, dict):
        return {
            "memory_structured": {},
            "memory_full_history_id": "",
        }, "missing_or_non_dict"

    structured = snapshot.get("memory_structured")
    history_id = snapshot.get("memory_full_history_id")
    issues: list[str] = []

    if not isinstance(structured, dict):
        issues.append("memory_structured_not_dict")
    if history_id is not None and not isinstance(history_id, str):
        issues.append("memory_full_history_id_not_str")

    normalized = {
        "memory_structured": dict(structured) if isinstance(structured, dict) else {},
        "memory_full_history_id": str(history_id or ""),
    }
    return normalized, ",".join(issues)


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


@dataclass(frozen=True)
class ToolCallRecord:
    canonicalized_input_payload: dict[str, Any]
    canonicalized_input_hash: str
    raw_output_payload: dict[str, Any]
    raw_output_hash: str
    response_schema_version: str
    determinism_contract: dict[str, Any] = field(default_factory=dict)
    stable_time_token: str = ""

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


def _normalize_external_call_inputs(
    *,
    operation: str,
    system: str,
    request_payload: dict[str, Any] | None,
    response_payload: dict[str, Any] | None,
    determinism_contract: dict[str, Any] | None,
) -> dict[str, Any]:
    request = dict(request_payload or {})
    response = dict(response_payload or {})
    normalized_contract = _normalize_determinism_contract(determinism_contract)
    canonicalized_request = _drop_ignored_fields(
        request,
        list(normalized_contract.get("ignore_request_fields") or []),
    )
    canonicalized_response = _drop_ignored_fields(
        response,
        list(normalized_contract.get("ignore_response_fields") or []),
    )
    normalized_operation = str(operation or "external_system_call").strip().lower() or "external_system_call"
    normalized_system = str(system or "unknown").strip().lower() or "unknown"
    request_hash = _stable_sha256(canonicalized_request)
    response_hash = _stable_sha256(canonicalized_response)
    seq_hint = int(request.get("sequence") or request.get("seq") or 0)
    time_token = _normalize_time_token(
        seq=seq_hint,
        operation=normalized_operation,
        system=normalized_system,
        request_hash=request_hash,
    )
    return {
        "request": request,
        "response": response,
        "normalized_contract": normalized_contract,
        "normalized_operation": normalized_operation,
        "normalized_system": normalized_system,
        "request_hash": request_hash,
        "response_hash": response_hash,
        "time_token": time_token,
    }


def _build_tool_call_record(
    *,
    canonicalized_request: dict[str, Any],
    response: dict[str, Any],
    request_hash: str,
    response_schema_version: str,
    normalized_contract: dict[str, Any],
    time_token: str,
) -> ToolCallRecord:
    return ToolCallRecord(
        canonicalized_input_payload=dict(canonicalized_request or {}),
        canonicalized_input_hash=request_hash,
        raw_output_payload=dict(response or {}),
        raw_output_hash=_stable_sha256(dict(response or {})),
        response_schema_version=str(response_schema_version or "1.0"),
        determinism_contract=normalized_contract,
        stable_time_token=time_token,
    )


def _tool_called_event_payload(
    *,
    normalized_operation: str,
    normalized_system: str,
    status: str,
    time_token: str,
    request_hash: str,
    response_hash: str,
    deterministic_id: str,
) -> dict[str, Any]:
    return {
        "operation": normalized_operation,
        "system": normalized_system,
        "status": str(status or "ok").strip().lower() or "ok",
        "time_token": time_token,
        "request_hash": request_hash,
        "response_hash": response_hash,
        "deterministic_id": str(deterministic_id or "").strip(),
    }


def record_external_system_call(
    *,
    operation: str,
    system: str,
    request_payload: dict[str, Any] | None = None,
    response_payload: dict[str, Any] | None = None,
    status: str = "ok",
    source: str = "",
    deterministic_id: str = "",
    response_schema_version: str = "1.0",
    determinism_contract: dict[str, Any] | None = None,
    required: bool = False,
) -> dict[str, Any] | None:
    normalized = _normalize_external_call_inputs(
        operation=operation,
        system=system,
        request_payload=request_payload,
        response_payload=response_payload,
        determinism_contract=determinism_contract,
    )
    request = dict(normalized.get("request") or {})
    response = dict(normalized.get("response") or {})
    normalized_contract = dict(normalized.get("normalized_contract") or {})
    normalized_operation = str(normalized.get("normalized_operation") or "external_system_call")
    normalized_system = str(normalized.get("normalized_system") or "unknown")
    request_hash = str(normalized.get("request_hash") or "")
    response_hash = str(normalized.get("response_hash") or "")
    time_token = str(normalized.get("time_token") or "")
    canonicalized_request = _drop_ignored_fields(
        request,
        list(normalized_contract.get("ignore_request_fields") or []),
    )
    tool_call_record = _build_tool_call_record(
        canonicalized_request=canonicalized_request,
        response=response,
        request_hash=request_hash,
        response_schema_version=response_schema_version,
        normalized_contract=normalized_contract,
        time_token=time_token,
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
        "tool_call_record": tool_call_record.to_dict(),
    }
    step = record_execution_step(
        "external_system_call",
        payload=payload,
        required=required,
    )
    # Route tool call through CoreState event bus (all execution through events).
    push_core_state_event(
        "tool_called",
        _tool_called_event_payload(
            normalized_operation=normalized_operation,
            normalized_system=normalized_system,
            status=status,
            time_token=time_token,
            request_hash=request_hash,
            response_hash=response_hash,
            deterministic_id=deterministic_id,
        ),
    )
    return step


def build_external_system_call_graph(steps: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    node_ids: list[str] = []
    for step in list(steps or []):
        if not _is_external_system_call_step(step):
            continue
        node_id, node_payload = _external_call_node(step)
        node_ids.append(node_id)
        nodes.append(node_payload)

    edges = _external_call_sequence_edges(node_ids)
    graph_payload = {"nodes": nodes, "edges": edges}
    graph = ExternalSystemCallGraph(
        nodes=nodes,
        edges=edges,
        graph_hash=_stable_sha256(graph_payload),
    )
    return graph.to_dict()


def _is_external_system_call_step(step: dict[str, Any]) -> bool:
    return str(step.get("operation") or "").strip().lower() == "external_system_call"


def _resolve_tool_call_hashes(
    tool_call_record: dict[str, Any], payload: dict[str, Any]
) -> tuple[str, str, str]:
    request_hash = str(
        tool_call_record.get("canonicalized_input_hash") or payload.get("request_hash") or ""
    )
    response_hash = str(
        tool_call_record.get("raw_output_hash") or payload.get("response_hash") or ""
    )
    time_token = str(
        tool_call_record.get("stable_time_token") or payload.get("time_token") or ""
    )
    return request_hash, response_hash, time_token


def _external_call_node(step: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    seq = int(step.get("seq") or 0)
    payload = dict(step.get("payload") or {})
    tool_call_record = dict(payload.get("tool_call_record") or {})
    operation = str(payload.get("operation") or "external_system_call").strip().lower()
    system = str(payload.get("system") or "unknown").strip().lower()
    request_hash, response_hash, time_token = _resolve_tool_call_hashes(tool_call_record, payload)
    execution_result = dict(payload.get("metadata", {}).get("execution_result") or {})
    status = str(execution_result.get("status") or payload.get("status") or "ok").strip().lower() or "ok"
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
    return node_id, node.to_dict()


def _external_call_sequence_edges(node_ids: list[str]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for idx in range(1, len(node_ids)):
        edges.append(
            {"from": node_ids[idx - 1], "to": node_ids[idx], "type": "sequence"},
        )
    return edges


class RuntimeTraceViolation(RuntimeError):
    """Raised when a trace-required operation executes without an active trace."""


class UnboundCoreStateMutationError(RuntimeError):
    """Raised when a write is attempted without a turn-bound CoreState in strict mode."""


class ExecutionTraceRecorder:
    """Turn-scoped recorder for authoritative execution operations."""

    def __init__(
        self,
        *,
        trace_token: str = "",
        prompt: str,
        metadata: dict[str, Any] | None = None,
        **legacy_kwargs: Any,
    ):
        legacy_trace = legacy_kwargs.pop("trace_id", "")
        if legacy_kwargs:
            unknown = ", ".join(sorted(str(name) for name in legacy_kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")
        self.trace_id = str(trace_token or legacy_trace or "")
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

# ---------------------------------------------------------------------------
# CoreState event bus — turn-scoped mutable reference to the active CoreState.
# All writes during a turn flow through push_core_state_event(), which applies
# the reducer and updates this contextvar.  At turn-end, control_plane reads
# the final value and persists it as the authoritative session state.
# ---------------------------------------------------------------------------
_ACTIVE_CORE_STATE: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "dadbot_active_core_state",
    default=None,
)


def get_active_core_state() -> Any:
    """Return the active CoreState for the current turn, or None if not bound."""
    return _ACTIVE_CORE_STATE.get()


def core_state_strict_mutation_mode_enabled() -> bool:
    """Return whether unbound mutation attempts must hard-fail.

    Default behavior:
    - CI: enabled by default (unless explicitly disabled)
    - Local/dev: disabled by default (unless explicitly enabled)

    Override with DADBOT_STRICT_CORESTATE_MUTATIONS.
    """
    raw = str(os.environ.get("DADBOT_STRICT_CORESTATE_MUTATIONS") or "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    # CI defaults to strict; local/dev remains permissive until rollout completes.
    return str(os.environ.get("CI") or "").strip().lower() in {"1", "true", "yes", "on"}


def require_bound_core_state_for_mutation(
    *,
    source: str,
    changed_keys: list[str] | None = None,
) -> Any:
    """Require a turn-bound CoreState for mutation paths when strict mode is on."""
    active = get_active_core_state()
    if active is not None:
        return active
    if not core_state_strict_mutation_mode_enabled():
        return None
    violation = {
        "source": str(source or ""),
        "changed_keys": list(changed_keys or []),
        "has_core_state": False,
        "strict_mode": True,
    }
    raise UnboundCoreStateMutationError(
        "All mutations must be CoreState-bound. "
        "Memory cannot change without a canonical event and reducer transition. "
        "Reducer-only mutation policy violation: no CoreState bound "
        f"(violation={violation!r})",
    )


def push_core_state_event(
    event_type: str,
    payload: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Apply an InputEvent to the active CoreState via the pure reducer.

    Returns the new CoreState, or None if no CoreState is bound for this turn.
    This is the single write path: event → reducer → CoreState.
    """
    event_payload = dict(payload or {})
    event_metadata = dict(metadata or {})
    invariant_metadata: dict[str, Any] = {}
    trace_id = str(event_metadata.get("trace_id") or "").strip()
    if trace_id:
        invariant_metadata["trace_id"] = trace_id
    enforce_mutation_entry_invariants(
        mutation_kind="core_state_event",
        source="execution_context.push_core_state_event",
        changed_keys=sorted(str(key) for key in event_payload.keys()),
        metadata=invariant_metadata,
    )

    current = _ACTIVE_CORE_STATE.get()
    if current is None:
        return None
    # Lazy import avoids module-level circular dependency risk.
    from dadbot.core.core_state import InputEvent, transition  # noqa: PLC0415
    next_state = transition(
        current,
        InputEvent(
            event_type=str(event_type or ""),
            payload=event_payload,
            metadata=event_metadata,
        ),
    )
    _ACTIVE_CORE_STATE.set(next_state)
    return next_state


@contextlib.contextmanager
def bind_core_state(initial_state: Any):
    """Bind a CoreState to the current async context for the duration of a turn.

    Must be entered before the turn executor runs so all downstream mutations
    (memory writes, tool calls, etc.) can update the shared CoreState via
    push_core_state_event().
    """
    token = _ACTIVE_CORE_STATE.set(initial_state)
    try:
        yield
    finally:
        _ACTIVE_CORE_STATE.reset(token)


def open_core_state_scope(initial_state: Any) -> Any:
    """Set the active CoreState and return the contextvar token.

    Pair with close_core_state_scope(token) in a try/finally block when a
    context manager is inconvenient (e.g. wrapping an existing try/except/finally).
    """
    return _ACTIVE_CORE_STATE.set(initial_state)


def close_core_state_scope(token: Any) -> None:
    """Reset the active CoreState to the state before open_core_state_scope."""
    _ACTIVE_CORE_STATE.reset(token)


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
        trace_token=synthetic_trace_id,
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


def _strip_temporal_payload_keys(payload: dict[str, Any]) -> dict[str, Any]:
    temporal_only_keys = {
        "completed_at",
        "timestamp",
        "wall_time",
        "created_at",
        "updated_at",
        "ts",
        "time",
    }
    normalized = dict(payload or {})
    for key in temporal_only_keys:
        normalized.pop(key, None)
    return normalized


def _normalized_external_call_record(
    *,
    tool_call_record: dict[str, Any],
    normalized_token: str,
) -> dict[str, Any]:
    normalized_contract = _normalize_determinism_contract(
        dict(tool_call_record.get("determinism_contract") or {}),
    )
    canonicalized_input_payload = _drop_ignored_fields(
        dict(tool_call_record.get("canonicalized_input_payload") or {}),
        list(normalized_contract.get("ignore_request_fields") or []),
    )
    raw_output_payload = _drop_ignored_fields(
        dict(tool_call_record.get("raw_output_payload") or {}),
        list(normalized_contract.get("ignore_response_fields") or []),
    )
    return ToolCallRecord(
        canonicalized_input_payload=dict(canonicalized_input_payload or {}),
        canonicalized_input_hash=_stable_sha256(canonicalized_input_payload),
        raw_output_payload=dict(raw_output_payload or {}),
        raw_output_hash=_stable_sha256(raw_output_payload),
        response_schema_version=str(
            tool_call_record.get("response_schema_version") or "1.0",
        ),
        determinism_contract=normalized_contract,
        stable_time_token=normalized_token,
    ).to_dict()


def _normalize_external_system_call_payload(
    *,
    seq: int,
    operation: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    normalized_payload = dict(payload or {})
    system = str(normalized_payload.get("system") or "").strip().lower()
    tool_call_record = dict(normalized_payload.get("tool_call_record") or {})
    request_hash = str(
        tool_call_record.get("canonicalized_input_hash")
        or normalized_payload.get("request_hash")
        or "",
    )
    normalized_token = _normalize_time_token(
        seq=seq,
        operation=operation,
        system=system,
        request_hash=request_hash,
    )
    normalized_payload["time_token"] = normalized_token
    if tool_call_record:
        normalized_payload["tool_call_record"] = _normalized_external_call_record(
            tool_call_record=tool_call_record,
            normalized_token=normalized_token,
        )
    return normalized_payload


def _normalized_steps(raw_steps: list[Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for index, item in enumerate(list(raw_steps or [])):
        if not isinstance(item, dict):
            continue
        raw_seq = item.get("seq")
        seq = int(raw_seq if isinstance(raw_seq, int) else index)
        operation = str(item.get("operation") or "").strip().lower()
        payload = _strip_temporal_payload_keys(dict(item.get("payload") or {}))

        if operation == "external_system_call":
            payload = _normalize_external_system_call_payload(
                seq=seq,
                operation=operation,
                payload=payload,
            )

        steps.append(
            {
                "seq": seq,
                "operation": operation,
                "payload": payload,
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


def _is_tool_invocation_operation(operation: str) -> bool:
    return operation in {"external_system_call", "tool_call", "tool_dispatch"}


def _tool_name_from_payload(payload: dict[str, Any]) -> str:
    return str(payload.get("system") or payload.get("tool") or payload.get("name") or "")


def _tool_status_from_payload(payload: dict[str, Any]) -> str:
    execution_result = dict(payload.get("metadata", {}).get("execution_result") or {})
    return str(execution_result.get("status") or payload.get("status") or "")


def _tool_projection_with_record(
    *,
    operation: str,
    tool_name: str,
    status: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    contract = _normalize_determinism_contract(dict(record.get("determinism_contract") or {}))
    canonicalized_input_payload = _drop_ignored_fields(
        dict(record.get("canonicalized_input_payload") or {}),
        list(contract.get("ignore_request_fields") or []),
    )
    normalized_output_payload = _drop_ignored_fields(
        dict(record.get("raw_output_payload") or {}),
        list(contract.get("ignore_response_fields") or []),
    )
    return {
        "operation": operation,
        "tool": tool_name,
        "status": status,
        "request_hash": _stable_sha256(canonicalized_input_payload),
        "response_hash": _stable_sha256(normalized_output_payload),
        "response_schema_version": str(record.get("response_schema_version") or "1.0"),
        "stable_time_token": str(record.get("stable_time_token") or ""),
        "determinism_contract_hash": _stable_sha256(contract),
    }


def _tool_projection_live(*, operation: str, tool_name: str, status: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation": operation,
        "tool": tool_name,
        "status": status,
        "request_hash": str(payload.get("request_hash") or ""),
        "response_hash": str(payload.get("response_hash") or ""),
        "response_schema_version": "",
        "stable_time_token": str(payload.get("time_token") or ""),
        "determinism_contract_hash": "",
    }


def build_tool_invocation_projection(
    execution_trace_context: dict[str, Any],
    *,
    live_tool_mode: bool = False,
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for step in list(execution_trace_context.get("steps") or []):
        operation = str(step.get("operation") or "")
        payload = dict(step.get("payload") or {})
        if not _is_tool_invocation_operation(operation):
            continue

        tool_name = _tool_name_from_payload(payload)
        status = _tool_status_from_payload(payload)

        if not live_tool_mode and isinstance(payload.get("tool_call_record"), dict):
            record = dict(payload.get("tool_call_record") or {})
            tools.append(
                _tool_projection_with_record(
                    operation=operation,
                    tool_name=tool_name,
                    status=status,
                    record=record,
                ),
            )
            continue

        tools.append(
            _tool_projection_live(
                operation=operation,
                tool_name=tool_name,
                status=status,
                payload=payload,
            ),
        )
    return tools


def _require_execution_context_contract(context: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    state = getattr(context, "state", None)
    metadata = getattr(context, "metadata", None)
    if not isinstance(state, dict):
        raise TypeError(
            "Execution trace contract violation: context.state must be a dict",
        )
    if not isinstance(metadata, dict):
        raise TypeError(
            "Execution trace contract violation: context.metadata must be a dict",
        )
    return dict(state), dict(metadata)


def _normalize_execution_response(result: Any) -> str:
    return str((result[0] if isinstance(result, tuple) else result) or "")


def _coerce_record_list(values: list[Any]) -> list[dict[str, Any]]:
    return [item if isinstance(item, dict) else {"value": item} for item in list(values or [])]


def _build_memory_snapshot_used(
    *,
    context: Any,
    state: dict[str, Any],
    determinism: dict[str, Any],
) -> dict[str, Any]:
    state_memory_snapshot, memory_snapshot_contract_warning = _coerce_memory_snapshot(state.get("memory_snapshot"))
    if memory_snapshot_contract_warning:
        logger.warning(
            "Execution trace context built with malformed memory_snapshot (trace_id=%s, warning=%s)",
            str(getattr(context, "trace_id", "") or ""),
            memory_snapshot_contract_warning,
        )
    return {
        "memory_fingerprint": str(determinism.get("memory_fingerprint") or ""),
        "memory_structured": dict(state_memory_snapshot.get("memory_structured") or {}),
        "memory_full_history_id": str(state_memory_snapshot.get("memory_full_history_id") or ""),
        "contract_ok": not bool(memory_snapshot_contract_warning),
        "contract_warning": str(memory_snapshot_contract_warning or ""),
    }


def _build_model_call_parameters(determinism: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": str(determinism.get("llm_provider") or ""),
        "model": str(determinism.get("llm_model") or ""),
        "seed_policy": str(determinism.get("seed_policy") or ""),
        "temperature_policy": str(determinism.get("temperature_policy") or ""),
    }


def _collect_recorded_steps(recorder: ExecutionTraceRecorder | None) -> tuple[list[dict[str, Any]], list[str]]:
    recorded_steps = recorder.steps if recorder is not None else []
    operations = [str(step.get("operation") or "") for step in recorded_steps]
    return recorded_steps, operations


def build_execution_trace_context(
    *,
    context,
    result,
    recorder: ExecutionTraceRecorder | None = None,
) -> dict[str, Any]:
    state, metadata = _require_execution_context_contract(context)
    determinism = dict(metadata.get("determinism") or {})

    normalized_response = _normalize_execution_response(result)

    retrieval_set = list(
        state.get("memory_retrieval_set") or state.get("retrieval_set") or state.get("memory_retrieval_results") or [],
    )
    retrieval_records = _coerce_record_list(retrieval_set)

    tool_outputs = list(state.get("tool_results") or [])
    tool_output_records = _coerce_record_list(tool_outputs)

    memory_snapshot_used = _build_memory_snapshot_used(
        context=context,
        state=state,
        determinism=determinism,
    )

    model_call_parameters = _build_model_call_parameters(determinism)

    model_output = state.get("model_output") or state.get("inference_output") or normalized_response

    trace_recorder = recorder or active_execution_trace()
    recorded_steps, operations = _collect_recorded_steps(trace_recorder)
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


def _snapshot_outputs_per_step(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "seq": int(step.get("seq") or 0),
            "operation": str(step.get("operation") or ""),
            "input": _step_input_payload(step),
            "output": _step_output_payload(step),
            "payload_hash": _stable_sha256(dict(step.get("payload") or {})),
        }
        for step in steps
    ]


def _snapshot_model_inputs(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _step_input_payload(step)
        for step in steps
        if str(step.get("operation") or "") == "model_call"
    ]


def _snapshot_memory_delta_summary(
    *,
    state: dict[str, Any],
    metadata: dict[str, Any],
    memory_write_intents: list[Any],
) -> dict[str, Any]:
    return dict(
        state.get("memory_delta_summary")
        or metadata.get("memory_delta_summary")
        or {
            "version": "1.0",
            "intent_count": len(memory_write_intents),
            "intents": list(memory_write_intents),
            "memory_retrieval_set_after_commit": list(state.get("memory_retrieval_set") or []),
        },
    )


def _snapshot_inputs(context: Any, control_plane: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": str(getattr(context, "user_input", "") or ""),
        "trace_id": str(getattr(context, "trace_id", "") or ""),
        "session_id": str(control_plane.get("session_id") or ""),
    }


def _snapshot_memory_snapshot(
    *,
    state: dict[str, Any],
    memory_snapshot_used: dict[str, Any],
) -> dict[str, Any]:
    return {
        **memory_snapshot_used,
        "retrieval_set": list(state.get("memory_retrieval_set") or []),
    }


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

    outputs_per_step = _snapshot_outputs_per_step(steps)
    model_inputs = _snapshot_model_inputs(steps)
    memory_write_intents = list(
        state.get("memory_write_intents")
        or metadata.get("memory_write_intents")
        or [],
    )
    memory_delta_summary = _snapshot_memory_delta_summary(
        state=state,
        metadata=metadata,
        memory_write_intents=memory_write_intents,
    )

    snapshot = {
        "inputs": _snapshot_inputs(context, control_plane),
        "memory_snapshot": _snapshot_memory_snapshot(
            state=state,
            memory_snapshot_used=memory_snapshot_used,
        ),
        "model_inputs": {
            "parameters": dict(model_call_parameters),
            "calls": model_inputs,
        },
        "outputs_per_step": outputs_per_step,
        "final_output": normalized_response,
        "memory_write_intents": memory_write_intents,
        "memory_delta_summary": memory_delta_summary,
    }
    snapshot["snapshot_hash"] = _stable_sha256(snapshot)
    return snapshot


def _build_dag_nodes(steps: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    nodes: list[dict[str, Any]] = []
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
    return nodes, node_ids


def _build_iteration_edges(
    steps: list[dict[str, Any]],
    node_ids: list[str],
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
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
            edges.append({"from": source, "to": node_id, "type": "iteration"})
    return edges


def _build_execution_dag(
    steps: list[dict[str, Any]],
    *,
    external_system_calls: dict[str, Any] | None = None,
) -> dict[str, Any]:
    nodes, node_ids = _build_dag_nodes(steps)

    edges: list[dict[str, Any]] = [
        {"from": node_ids[idx - 1], "to": node_ids[idx], "type": "sequence"}
        for idx in range(1, len(node_ids))
    ]
    edges.extend(_build_iteration_edges(steps, node_ids))

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
