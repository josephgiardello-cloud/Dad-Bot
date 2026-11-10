"""Kernel-owned single execution spine for all agentic tool calls.

All tool execution in the service layer MUST go through ``execute_tool``.
The private sandbox implementation is unreachable outside the allowed core
execution-spine modules and enforced by CI RULE16_TOOL_SANDBOX_ISOLATION.

Why a function, not a class?
-----------------------------
The ToolSandbox is intentionally a per-call object — one sandbox per tool
invocation, scoped to a single agentic turn step.  Wrapping it in a singleton
or a long-lived object would break the isolation guarantee.  This module owns
the instantiation decision; callers see only the ``execute_tool`` interface.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from dadbot.contracts import SovereignEvent, SovereignEventType, ToolExecutionPayload, ToolResultPayload
from dadbot.core._tool_sandbox import ToolExecutionRecord, _ToolSandbox
from dadbot.core.runtime_errors import ReplayInvariantViolation
from dadbot.core.tool_recording import ToolIOLedger, ToolIORecord
from dadbot.pii_scrubber import scrub_text
from dadbot.telemetry.otel_bridge import start_span

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


def _execute_live_tool(
    *,
    tool_name: str,
    parameters: dict[str, Any],
    input_hash: str,
    executor: Callable[[], Any],
    compensating_action: Callable[[], None] | None,
    turn_context: Any,
) -> tuple[ToolExecutionRecord, float]:
    from dadbot_system.delegation import (
        AuthorityUnavailableError,
        AuthorizationError,
        live_authority_audit_context,
        verify_live_authority,
    )

    with start_span(
        "dadbot.tool.execute",
        **{
            "dadbot.tool.name": str(tool_name or ""),
            "dadbot.server.name": "dadbot",
        },
    ) as span:
        try:
            verify_live_authority()
        except (AuthorizationError, AuthorityUnavailableError) as exc:
            authority = live_authority_audit_context()
            _emit_tool_audit(
                turn_context=turn_context,
                tool_name=tool_name,
                parameters=parameters,
                status=(
                    "authority_unavailable"
                    if isinstance(exc, AuthorityUnavailableError)
                    else "authorization_denied"
                ),
                input_hash=input_hash,
                error_type=type(exc).__name__,
                authority=authority,
            )
            if span is not None:
                span.set_attribute("dadbot.tool.is_error", True)
            raise

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
    try:
        setattr(turn_context, "metadata", normalized)
    except Exception:
        pass
    return normalized


def execute_tool(
    *,
    tool_name: str,
    parameters: dict[str, Any] | None = None,
    executor: Callable[[], Any],
    compensating_action: Callable[[], None] | None = None,
    turn_context: Any | None = None,
) -> ToolExecutionRecord:
    """Single kernel-owned execution spine for all agentic tool calls.

    Creates a scoped private tool sandbox, executes the tool with idempotency and
    failure isolation, and returns the execution record. Records all tool IO
    (input + output) to turn_context._tool_io_ledger for deterministic replay.

    Parameters
    ----------
    tool_name:
        Canonical name of the tool (e.g. ``"set_reminder"``).
    parameters:
        Tool call parameters dict.  Used for idempotency key derivation.
    executor:
        Zero-argument callable that performs the actual tool work.
    compensating_action:
        Optional zero-argument callable registered for LIFO rollback.
        Callers that need the execution result inside the compensating action
        should close over a result holder populated after this function returns.
    turn_context:
        Turn context (TurnContext). If provided, tool IO is recorded to
        turn_context._tool_io_ledger for checkpoint persistence.
    """
    # Layer 1: Record tool IO
    normalized_params = dict(parameters or {})
    input_hash = _stable_payload_hash(
        {"tool_name": str(tool_name or ""), "parameters": normalized_params}
    )

    record: ToolExecutionRecord | None = None
    latency_ms = 0.0
    is_replayed = False
    audit_authority: dict[str, str] | None = None

    if turn_context is not None:
        metadata = _ensure_mutable_metadata(turn_context)
        replay_mode = bool(metadata.get("replay_mode", False))
        tool_version = _executor_tool_version(executor, metadata)
        environment_fingerprint = _environment_fingerprint(metadata)
        metadata.setdefault("tool_version", tool_version)
        metadata.setdefault("environment_fingerprint", environment_fingerprint)

        if replay_mode:
            restored_ledger = metadata.get("_tool_io_ledger")
            if not isinstance(restored_ledger, ToolIOLedger):
                raise ReplayInvariantViolation(
                    f"Replay mode requires a restored tool ledger for tool={tool_name!r}",
                    context={
                        "tool_name": str(tool_name or ""),
                        "input_hash": input_hash,
                        "tool_version": tool_version,
                        "environment_fingerprint": environment_fingerprint,
                    },
                )

            restored_record = restored_ledger.lookup(
                tool_name,
                input_hash,
                tool_version=tool_version,
                environment_fingerprint=environment_fingerprint,
            )
            if restored_record is None:
                raise ReplayInvariantViolation(
                    f"Replay mode requires a ledger hit for tool={tool_name!r}",
                    context={
                        "tool_name": str(tool_name or ""),
                        "input_hash": input_hash,
                        "tool_version": tool_version,
                        "environment_fingerprint": environment_fingerprint,
                    },
                )

            record = ToolExecutionRecord(
                tool_name=str(tool_name or ""),
                idempotency_key=restored_record.lookup_key(
                    tool_version=tool_version,
                    environment_fingerprint=environment_fingerprint,
                ),
                status="replayed",
                result=restored_record.output_payload or {},
                error="",
                compensating_action=compensating_action,
            )

            io_log = getattr(turn_context, "_tool_io_ledger", None)
            if not isinstance(io_log, ToolIOLedger):
                io_log = ToolIOLedger()
                setattr(turn_context, "_tool_io_ledger", io_log)
            io_record = ToolIORecord(
                sequence=len(io_log.records) + 1,
                tool_name=str(tool_name or ""),
                input_hash=input_hash,
                input_payload=normalized_params,
                output_payload=restored_record.output_payload or {},
                output_hash=restored_record.output_hash,
                status="replayed",
                latency_ms=restored_record.latency_ms,
                error="",
                metadata={
                    "tool_version": tool_version,
                    "environment_fingerprint": environment_fingerprint,
                    "replay_mode": True,
                    "replay_key": restored_record.lookup_key(
                        tool_version=tool_version,
                        environment_fingerprint=environment_fingerprint,
                    ),
                },
            )
            io_log.append(io_record)
            latency_ms = restored_record.latency_ms
            is_replayed = True
            audit_authority = {"authority_mode": "replay"}

    if record is None:
        record, latency_ms = _execute_live_tool(
            tool_name=tool_name,
            parameters=normalized_params,
            input_hash=input_hash,
            executor=executor,
            compensating_action=compensating_action,
            turn_context=turn_context,
        )

    if audit_authority is None:
        from dadbot_system.delegation import live_authority_audit_context

        audit_authority = live_authority_audit_context()

    if turn_context is not None and not is_replayed:
        metadata = _ensure_mutable_metadata(turn_context)
        tool_version = str(metadata.get("tool_version") or _executor_tool_version(executor, metadata))
        environment_fingerprint = str(
            metadata.get("environment_fingerprint") or _environment_fingerprint(metadata),
        )
        metadata.setdefault("tool_version", tool_version)
        metadata.setdefault("environment_fingerprint", environment_fingerprint)

        io_log = getattr(turn_context, "_tool_io_ledger", None)
        if not isinstance(io_log, ToolIOLedger):
            io_log = ToolIOLedger()
            setattr(turn_context, "_tool_io_ledger", io_log)

        output_payload = {}
        if record.status == "succeeded":
            output_payload = (
                record.result if isinstance(record.result, dict) else {"result": record.result}
            )

        output_hash = _stable_payload_hash(output_payload)
        io_record = ToolIORecord(
            sequence=len(io_log.records) + 1,
            tool_name=str(tool_name or ""),
            input_hash=input_hash,
            input_payload=normalized_params,
            output_payload=output_payload,
            output_hash=output_hash,
            status=record.status,
            latency_ms=round(latency_ms, 3),
            error=str(record.error or ""),
            metadata={
                "tool_version": tool_version,
                "environment_fingerprint": environment_fingerprint,
                "replay_mode": False,
            },
        )
        io_log.append(io_record)

        _emit_tool_execution_event(
            turn_context=turn_context,
            tool_name=tool_name,
            parameters=parameters,
            record=record,
            latency_ms=latency_ms,
            authority=audit_authority,
        )

    _emit_tool_audit(
        turn_context=turn_context,
        tool_name=tool_name,
        parameters=normalized_params,
        status=record.status,
        input_hash=input_hash,
        output_hash=_stable_payload_hash(record.result),
        error_type="ToolExecutionError" if record.status == "failed" else "",
        authority=audit_authority,
    )

    return record
