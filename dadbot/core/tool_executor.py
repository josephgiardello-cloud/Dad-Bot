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
import platform
import sys
import time
from collections.abc import Callable
from typing import Any

from dadbot.contracts import SovereignEvent, SovereignEventType, ToolExecutionPayload, ToolResultPayload
from dadbot.core._tool_sandbox import ToolExecutionRecord, _ToolSandbox
from dadbot.core.runtime_errors import ReplayInvariantViolation
from dadbot.core.tool_recording import ToolIOLedger, ToolIORecord


def _stable_payload_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


def _emit_tool_execution_event(
    *,
    turn_context: Any,
    tool_name: str,
    parameters: dict[str, Any] | None,
    record: ToolExecutionRecord,
    latency_ms: float,
) -> None:
    state = getattr(turn_context, "state", None)
    metadata = getattr(turn_context, "metadata", None)
    if not isinstance(state, dict) or not isinstance(metadata, dict):
        return

    previous_checksum = str(metadata.get("sovereign_event_checksum") or "")
    normalized_params = dict(parameters or {})
    input_hash = _stable_payload_hash({"tool_name": str(tool_name or ""), "parameters": normalized_params})
    output_hash = _stable_payload_hash(record.result)
    tool_result_payload = ToolResultPayload(
        status=str(record.status or ""),
        output_hash=output_hash,
        error=str(record.error or ""),
        latency_ms=round(float(latency_ms), 3),
        metadata={
            "idempotency_key": str(record.idempotency_key or ""),
            "cached": bool(str(record.status or "").strip().lower() == "cached"),
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

    if record is None:
        sandbox = _ToolSandbox()
        started = time.perf_counter()
        record = sandbox.execute(
            tool_name=tool_name,
            parameters=parameters,
            executor=executor,
            compensating_action=compensating_action,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

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
        )

    return record
