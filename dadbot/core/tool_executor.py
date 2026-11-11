"""Kernel-owned single execution spine for all agentic tool calls.

Service tool execution goes through ``execute_tool`` or ``execute_tool_async``.
The private sandbox implementation is unreachable outside the allowed core
execution-spine modules and enforced by CI RULE16_TOOL_SANDBOX_ISOLATION.
External async calls share recording/replay checks but never use the synchronous
sandbox's shared cache or silently turn uncertain effects into retryable failures.

Why a function, not a class?
-----------------------------
The ToolSandbox is intentionally a per-call object — one sandbox per tool
invocation, scoped to a single agentic turn step.  Wrapping it in a singleton
or a long-lived object would break the isolation guarantee.  This module owns
the instantiation decision; callers see only the ``execute_tool`` interface.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import platform
import sys
import time
from collections.abc import Awaitable, Callable
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from dadbot.contracts import SovereignEvent, SovereignEventType, ToolExecutionPayload, ToolResultPayload
from dadbot.core._tool_sandbox import ToolExecutionRecord, _ToolSandbox
from dadbot.core.runtime_errors import ReplayInvariantViolation
from dadbot.core.tool_recording import ToolIOLedger, ToolIORecord
from dadbot.pii_scrubber import scrub_text
from dadbot_system.runtime_signals import start_span

logger = logging.getLogger(__name__)


def _stable_payload_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


_SENSITIVE_PARAMETER_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "password",
        "secret",
        "token",
    },
)
_MAX_AUDIT_VALUE_LENGTH = 256
_MAX_AUDIT_PARAMETERS_LENGTH = 2048


def _safe_audit_value(value: Any, *, key: str = "") -> Any:
    normalized_key = key.lower().replace("-", "_")
    if normalized_key in _SENSITIVE_PARAMETER_NAMES or any(
        marker in normalized_key
        for marker in (
            "token",
            "secret",
            "password",
            "credential",
            "authorization",
            "cookie",
            "base64",
            "api_key",
            "apikey",
            "private_key",
        )
    ):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {
            str(child_key): _safe_audit_value(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe_audit_value(item) for item in value[:20]]
    if isinstance(value, str):
        scrubbed = scrub_text(value[:_MAX_AUDIT_VALUE_LENGTH])[0]
        if len(value) > _MAX_AUDIT_VALUE_LENGTH:
            scrubbed += "[TRUNCATED]"
        return scrubbed
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return f"<{type(value).__name__}>"


def _audit_scope(turn_context: Any) -> dict[str, str]:
    metadata = getattr(turn_context, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    control_plane = metadata.get("control_plane")
    control_plane = control_plane if isinstance(control_plane, dict) else {}
    identity = {
        "tenant_id": str(control_plane.get("tenant_id") or metadata.get("tenant_id") or ""),
        "session_id": str(control_plane.get("session_id") or metadata.get("session_id") or ""),
        "trace_id": str(getattr(turn_context, "trace_id", "") or ""),
    }
    return {key: value for key, value in identity.items() if value}


def _emit_tool_audit(
    *,
    turn_context: Any,
    tool_name: str,
    parameters: dict[str, Any],
    status: str,
    input_hash: str,
    output_hash: str = "",
    error_type: str = "",
    authority: dict[str, str] | None = None,
) -> None:  # noqa: PLR0913
    from dadbot_system.delegation import live_authority_audit_context

    resolved_authority = dict(authority or live_authority_audit_context())
    audit = {
        "event": "dadbot.tool.execution",
        "timestamp": datetime.now(UTC).isoformat(),
        **_audit_scope(turn_context),
        **resolved_authority,
        "tool_name": str(tool_name or ""),
        "input_parameters": _safe_audit_value(parameters),
        "input_hash": input_hash,
        "output_hash": output_hash,
        "status": str(status or "unknown"),
        "authorization_decision": {
            "authorization_denied": "denied",
            "authority_unavailable": "unavailable",
            "replayed": "not_required_replay",
        }.get(
            str(status or "").strip().lower(),
            "trusted_local" if resolved_authority.get("authority_mode") == "trusted_local" else "verified",
        ),
    }
    if error_type:
        audit["error_type"] = error_type
    try:
        encoded = json.dumps(audit, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError):
        audit["input_parameters"] = "[UNSERIALIZABLE]"
        encoded = json.dumps(audit, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if len(encoded) > _MAX_AUDIT_PARAMETERS_LENGTH + 2048:
        audit["input_parameters"] = "[TRUNCATED]"
        encoded = json.dumps(audit, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    logger.info("%s", encoded)


def _verify_tool_authority(
    turn_context: Any, tool_name: str, parameters: dict[str, Any], input_hash: str, *, span: Any | None = None,
) -> None:
    from dadbot_system.delegation import (
        AuthorityUnavailableError,
        AuthorizationError,
        live_authority_audit_context,
        verify_live_authority,
    )

    try:
        verify_live_authority()
    except (AuthorizationError, AuthorityUnavailableError) as exc:
        if span is not None:
            span.set_attribute("dadbot.tool.is_error", True)
        _emit_tool_audit(
            turn_context=turn_context,
            tool_name=tool_name,
            parameters=parameters,
            status="authority_unavailable" if isinstance(exc, AuthorityUnavailableError) else "authorization_denied",
            input_hash=input_hash,
            error_type=type(exc).__name__,
            authority=live_authority_audit_context(),
        )
        raise


def _execute_live_tool(
    *,
    tool_name: str,
    parameters: dict[str, Any],
    input_hash: str,
    executor: Callable[[], Any],
    compensating_action: Callable[[], None] | None,
    turn_context: Any,
) -> tuple[ToolExecutionRecord, float]:
    with start_span(
        "dadbot.tool.execute",
        **{
            "dadbot.tool.name": str(tool_name or ""),
            "dadbot.server.name": "dadbot",
        },
    ) as span:
        _verify_tool_authority(turn_context, tool_name, parameters, input_hash, span=span)

        started = time.perf_counter()
        record = _ToolSandbox().execute(
            tool_name=tool_name,
            parameters=parameters,
            executor=executor,
            compensating_action=compensating_action,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        if span is not None:
            span.set_attribute("dadbot.tool.is_error", record.status == "failed")
    return record, latency_ms


def _emit_tool_execution_event(
    *,
    turn_context: Any,
    tool_name: str,
    parameters: dict[str, Any] | None,
    record: ToolExecutionRecord,
    latency_ms: float,
    authority: dict[str, str] | None = None,
) -> None:
    state = getattr(turn_context, "state", None)
    metadata = getattr(turn_context, "metadata", None)
    if not isinstance(state, dict) or not isinstance(metadata, dict):
        return

    previous_checksum = str(metadata.get("sovereign_event_checksum") or "")
    normalized_params = dict(parameters or {})
    input_hash = _stable_payload_hash({"tool_name": str(tool_name or ""), "parameters": normalized_params})
    output_hash = _stable_payload_hash(record.result)
    resolved_authority = dict(authority or {})
    status = str(record.status or "").strip().lower()
    authorization_decision = (
        "not_required_replay"
        if status == "replayed"
        else "trusted_local"
        if resolved_authority.get("authority_mode") == "trusted_local"
        else "verified"
    )
    tool_result_payload = ToolResultPayload(
        status=str(record.status or ""),
        output_hash=output_hash,
        error=str(record.error or ""),
        latency_ms=round(float(latency_ms), 3),
        metadata={
            "idempotency_key": str(record.idempotency_key or ""),
            "cached": bool(str(record.status or "").strip().lower() == "cached"),
            "authority_mode": str(resolved_authority.get("authority_mode") or "unknown"),
            "authorization_decision": authorization_decision,
        },
    )
    event = SovereignEvent(
        turn_id=str(getattr(turn_context, "trace_id", "") or ""),
        event_type=SovereignEventType.TOOL_EXECUTION.value,
        payload=ToolExecutionPayload(
            tool_name=str(tool_name or ""),
            status=str(record.status or "pending"),
            input_hash=input_hash,
            output_hash=output_hash,
            tool_result=tool_result_payload,
            metadata={"error": str(record.error or "")},
        ),
        previous_checksum=previous_checksum,
    )
    stream = list(state.get("sovereign_events") or [])
    stream.append(event.to_ledger_event())
    state["sovereign_events"] = stream
    metadata["sovereign_event_checksum"] = event.checksum
    metadata["sovereign_event_count"] = len(stream)


def _executor_tool_version(executor: Callable[[], Any], metadata: dict[str, Any]) -> str:
    explicit = str(metadata.get("tool_version") or "").strip()
    if explicit:
        return explicit
    code = getattr(executor, "__code__", None)
    code_payload: dict[str, Any] = {}
    if code is not None:
        code_payload = {
            "co_code": getattr(code, "co_code", b"").hex(),
            "co_consts": list(getattr(code, "co_consts", ()) or ()),
            "co_names": list(getattr(code, "co_names", ()) or ()),
            "co_argcount": int(getattr(code, "co_argcount", 0) or 0),
            "co_kwonlyargcount": int(getattr(code, "co_kwonlyargcount", 0) or 0),
        }
    return _stable_payload_hash(
        {
            "module": str(getattr(executor, "__module__", "") or ""),
            "qualname": str(getattr(executor, "__qualname__", getattr(executor, "__name__", "")) or ""),
            "code": code_payload,
        },
    )


def _environment_fingerprint(metadata: dict[str, Any]) -> str:
    explicit = str(metadata.get("environment_fingerprint") or "").strip()
    if explicit:
        return explicit
    determinism = dict(metadata.get("determinism") or {})
    det_env = str(determinism.get("env_hash") or "").strip()
    if det_env:
        return det_env
    return _stable_payload_hash(
        {
            "python_version": sys.version,
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
        },
    )


def _ensure_mutable_metadata(turn_context: Any) -> dict[str, Any]:
    metadata = getattr(turn_context, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    normalized = dict(metadata or {})
    turn_context.metadata = normalized
    return normalized


def _recording_scope(turn_context: Any) -> dict[str, str]:
    scope = _audit_scope(turn_context)
    request_id = str(turn_context.metadata.get("request_id") or "")
    if request_id:
        scope["request_id"] = request_id
    return scope


def _prepare_tool_replay(  # noqa: PLR0913
    *,
    tool_name: str,
    parameters: dict[str, Any],
    executor: Callable[[], Any],
    compensating_action: Callable[[], None] | None,
    turn_context: Any | None,
    require_scoped: bool = False,
) -> tuple[ToolExecutionRecord, float] | None:
    if turn_context is None:
        return None
    metadata = _ensure_mutable_metadata(turn_context)
    tool_version = _executor_tool_version(executor, metadata)
    environment_fingerprint = _environment_fingerprint(metadata)
    metadata.setdefault("tool_version", tool_version)
    metadata.setdefault("environment_fingerprint", environment_fingerprint)
    if not metadata.get("replay_mode", False):
        return None
    input_hash = _stable_payload_hash({"tool_name": str(tool_name or ""), "parameters": parameters})
    coordinates = {
        "tool_name": str(tool_name or ""), "input_hash": input_hash,
        "tool_version": tool_version, "environment_fingerprint": environment_fingerprint,
    }
    restored_ledger = metadata.get("_tool_io_ledger")
    if not isinstance(restored_ledger, ToolIOLedger):
        raise ReplayInvariantViolation(
            f"Replay mode requires a restored tool ledger for tool={tool_name!r}", context=coordinates,
        )
    restored = restored_ledger.lookup(
        tool_name, input_hash, tool_version=tool_version, environment_fingerprint=environment_fingerprint,
    )
    if restored is None:
        raise ReplayInvariantViolation(f"Replay mode requires a ledger hit for tool={tool_name!r}", context=coordinates)
    if not restored.is_success() or (require_scoped and restored.status not in {"succeeded", "replayed"}):
        raise ReplayInvariantViolation("Replay requires a successful tool outcome", context=coordinates)
    if (
        _stable_payload_hash({"tool_name": restored.tool_name, "parameters": restored.input_payload}) != input_hash
        or _stable_payload_hash(restored.output_payload) != restored.output_hash
    ):
        raise ReplayInvariantViolation("Recorded tool IO integrity check failed", context=coordinates)
    scope = restored.metadata.get("scope")
    if (require_scoped or scope is not None) and scope != _recording_scope(turn_context):
        raise ReplayInvariantViolation("Recorded tool IO scope does not match the request", context=coordinates)
    record = ToolExecutionRecord(
        tool_name=str(tool_name or ""),
        idempotency_key=restored.lookup_key(tool_version=tool_version, environment_fingerprint=environment_fingerprint),
        status="replayed", result=deepcopy(restored.output_payload), error="", compensating_action=compensating_action,
    )
    return record, restored.latency_ms


def _record_tool_completion(
    *,
    turn_context: Any | None,
    tool_name: str,
    parameters: dict[str, Any],
    record: ToolExecutionRecord,
    latency_ms: float,
) -> None:
    from dadbot_system.delegation import live_authority_audit_context

    is_replayed = record.status == "replayed"
    authority = {"authority_mode": "replay"} if is_replayed else live_authority_audit_context()
    input_hash = _stable_payload_hash({"tool_name": str(tool_name or ""), "parameters": parameters})
    if turn_context is not None:
        metadata = _ensure_mutable_metadata(turn_context)

        io_log = getattr(turn_context, "_tool_io_ledger", None)
        if not isinstance(io_log, ToolIOLedger):
            io_log = ToolIOLedger()
            turn_context._tool_io_ledger = io_log

        output_payload = {}
        if record.status in {"succeeded", "cached", "replayed", "uncertain"} and record.result is not None:
            output_payload = (
                record.result if isinstance(record.result, dict) else {"result": record.result}
            )

        output_hash = _stable_payload_hash(output_payload)
        io_record = ToolIORecord(
            sequence=len(io_log.records) + 1,
            tool_name=str(tool_name or ""),
            input_hash=input_hash,
            input_payload=deepcopy(parameters),
            output_payload=deepcopy(output_payload),
            output_hash=output_hash,
            status=record.status,
            latency_ms=round(latency_ms, 3),
            error=str(record.error or ""),
            metadata={
                "tool_version": str(metadata["tool_version"]),
                "environment_fingerprint": str(metadata["environment_fingerprint"]),
                "replay_mode": is_replayed,
                "scope": _recording_scope(turn_context),
                **({"replay_key": record.idempotency_key} if is_replayed else {}),
            },
        )
        io_log.append(io_record)

        if not is_replayed:
            _emit_tool_execution_event(
                turn_context=turn_context, tool_name=tool_name, parameters=parameters,
                record=record, latency_ms=latency_ms, authority=authority,
            )

    _emit_tool_audit(
        turn_context=turn_context,
        tool_name=tool_name,
        parameters=parameters,
        status=record.status,
        input_hash=input_hash,
        output_hash=_stable_payload_hash(record.result),
        error_type=(
            record.error if record.status == "uncertain" else "ToolExecutionError" if record.status == "failed" else ""
        ),
        authority=authority,
    )


def execute_tool(
    *,
    tool_name: str,
    parameters: dict[str, Any] | None = None,
    executor: Callable[[], Any],
    compensating_action: Callable[[], None] | None = None,
    turn_context: Any | None = None,
) -> ToolExecutionRecord:
    """Execute a synchronous tool through the scoped sandbox and shared IO/replay checks."""
    normalized_params = dict(parameters or {})
    replay = _prepare_tool_replay(
        tool_name=tool_name, parameters=normalized_params, executor=executor,
        compensating_action=compensating_action, turn_context=turn_context,
    )
    record, latency_ms = replay if replay is not None else _execute_live_tool(
        tool_name=tool_name, parameters=normalized_params,
        input_hash=_stable_payload_hash({"tool_name": str(tool_name or ""), "parameters": normalized_params}),
        executor=executor, compensating_action=compensating_action, turn_context=turn_context,
    )
    _record_tool_completion(
        turn_context=turn_context, tool_name=tool_name, parameters=normalized_params,
        record=record, latency_ms=latency_ms,
    )
    return record


async def execute_tool_async(
    *,
    tool_name: str,
    parameters: dict[str, Any],
    executor: Callable[[], Awaitable[dict[str, Any]]],
    turn_context: Any | None = None,
) -> ToolExecutionRecord:
    """Await external effects in the caller's task, without shared caching or automatic retries."""
    normalized_params = deepcopy(parameters)
    replay = _prepare_tool_replay(
        tool_name=tool_name, parameters=normalized_params, executor=executor,
        compensating_action=None, turn_context=turn_context, require_scoped=True,
    )
    if replay is not None:
        record, latency_ms = replay
    else:
        input_hash = _stable_payload_hash({"tool_name": str(tool_name or ""), "parameters": normalized_params})
        with start_span(
            "dadbot.tool.execute", **{"dadbot.tool.name": tool_name, "dadbot.server.name": "dadbot"},
        ) as span:
            _verify_tool_authority(turn_context, tool_name, normalized_params, input_hash, span=span)
            started = time.perf_counter()
            try:
                result = await executor()
                if not isinstance(result, dict):
                    raise TypeError("Async tool execution must return a result mapping")
                if "is_error" in result and not isinstance(result["is_error"], bool):
                    raise TypeError("Async tool error flag must be a boolean")
                record = ToolExecutionRecord(
                    tool_name=tool_name, idempotency_key=input_hash,
                    status="uncertain" if result.get("is_error") else "succeeded", result=result, error="",
                )
            except (Exception, asyncio.CancelledError) as exc:
                record = ToolExecutionRecord(
                    tool_name=tool_name, idempotency_key=input_hash, status="uncertain",
                    result=None, error=type(exc).__name__,
                )
                _record_tool_completion(
                    turn_context=turn_context, tool_name=tool_name, parameters=normalized_params,
                    record=record, latency_ms=(time.perf_counter() - started) * 1000.0,
                )
                if span is not None:
                    span.set_attribute("dadbot.tool.is_error", True)
                raise
            latency_ms = (time.perf_counter() - started) * 1000.0
            if span is not None:
                span.set_attribute("dadbot.tool.is_error", record.status == "uncertain")
    _record_tool_completion(
        turn_context=turn_context, tool_name=tool_name, parameters=normalized_params,
        record=record, latency_ms=latency_ms,
    )
    return record
