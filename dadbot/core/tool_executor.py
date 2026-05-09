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
import time
from collections.abc import Callable
from typing import Any

from dadbot.contracts import SovereignEvent, SovereignEventType, ToolExecutionPayload, ToolResultPayload
from dadbot.core._tool_sandbox import ToolExecutionRecord, _ToolSandbox


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
    failure isolation, and returns the execution record.  Never raises.

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
    """
    sandbox = _ToolSandbox()
    started = time.perf_counter()
    record = sandbox.execute(
        tool_name=tool_name,
        parameters=parameters,
        executor=executor,
        compensating_action=compensating_action,
    )
    if turn_context is not None:
        latency_ms = (time.perf_counter() - started) * 1000.0
        _emit_tool_execution_event(
            turn_context=turn_context,
            tool_name=tool_name,
            parameters=parameters,
            record=record,
            latency_ms=latency_ms,
        )
    return record
