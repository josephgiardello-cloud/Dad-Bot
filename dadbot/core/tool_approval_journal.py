from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.graph_context import TurnContext
from dadbot.core.ledger_writer import LedgerWriter
from dadbot.core.runtime_errors import ReplayInvariantViolation
from dadbot.core.state_lineage import canonical_state_hash
from dadbot.core.tool_executor import execute_tool_async
from dadbot.core.tool_recording import ToolIOLedger
from dadbot_system.delegation import require_tenant_authority_if_active, verify_live_authority
from dadbot_system.execution_identity import require_scoped_tenant_id

logger = logging.getLogger(__name__)

_EVENT = "TOOL_APPROVAL_UPDATED"
_IO_EVENT = "TOOL_IO_RECORDED"
_TERMINAL = frozenset({"rejected", "expired", "succeeded", "uncertain"})


@dataclass(frozen=True)
class ApprovedToolCall:
    session_id: str
    trace_id: str
    request_id: str
    tool_name: str
    input_hash: str


class ToolApprovalJournal:
    """Commit approval/dispatch boundaries and replay only durably completed IO."""

    def __init__(self, ledger: ExecutionLedger, *, tenant_id: str) -> None:
        self.ledger = ledger
        self.tenant_id = require_scoped_tenant_id(tenant_id)
        self.writer = LedgerWriter(ledger)

    def _latest(self) -> dict[tuple[str, str], dict[str, Any]]:
        latest = {}
        for event in self.ledger.read():
            if event.get("type") == _EVENT and event.get("tenant_id") == self.tenant_id:
                payload = event.get("payload")
                if not isinstance(payload, dict) or not all(
                    isinstance(payload.get(key), str) and payload[key]
                    for key in ("request_id", "trace_id", "tool_name", "input_hash", "status")
                ):
                    raise RuntimeError("Recorded approval has invalid identity or status")
                latest[(event["session_id"], payload["request_id"])] = event
        return latest

    def _recorded(self, call: ApprovedToolCall) -> dict[str, Any] | None:
        require_tenant_authority_if_active(self.tenant_id)
        event = self._latest().get((call.session_id, call.request_id))
        if event is not None and any(
            event["payload"][key] != getattr(call, key) for key in ("tool_name", "input_hash", "trace_id")
        ):
            raise ValueError("Approval identity does not match the recorded proposal")
        return event

    def _transition(self, call: ApprovedToolCall, status: str, *, expected: set[str | None], **details: Any) -> None:
        with self.ledger._lock:
            prior = self._recorded(call)
            previous_status = prior["payload"]["status"] if prior is not None else None
            if previous_status not in expected:
                raise RuntimeError(f"Approval transition {previous_status!r} -> {status!r} is not permitted")
            self.writer.write_event(
                _EVENT,
                tenant_id=self.tenant_id,
                session_id=call.session_id,
                trace_id=call.trace_id,
                kernel_step_id="tools.approval",
                committed=True,
                payload={
                    "request_id": call.request_id,
                    "trace_id": call.trace_id,
                    "tool_name": call.tool_name,
                    "input_hash": call.input_hash,
                    "status": status,
                    **details,
                },
            )

    def propose(self, call: ApprovedToolCall) -> None:
        if not all((call.session_id, call.trace_id, call.request_id, call.tool_name, call.input_hash)):
            raise ValueError("Approval requires complete tenant, session, request and tool identity")
        self._transition(call, "pending", expected={None})

    def decide(self, call: ApprovedToolCall, *, approved: bool) -> None:
        self._transition(call, "approved" if approved else "rejected", expected={"pending"})

    def expire(self, call: ApprovedToolCall) -> None:
        self._transition(call, "expired", expected={"pending", "approved"})

    def status(self, call: ApprovedToolCall) -> str:
        event = self._recorded(call)
        if event is None:
            raise RuntimeError("Approval is not recorded")
        return str(event["payload"]["status"])

    def expire_recovered(self) -> None:
        with self.ledger._lock:
            for event in self._latest().values():
                payload = event["payload"]
                status = payload["status"]
                if status in _TERMINAL:
                    continue
                call = ApprovedToolCall(
                    session_id=event["session_id"],
                    trace_id=payload["trace_id"],
                    request_id=payload["request_id"],
                    tool_name=payload["tool_name"],
                    input_hash=payload["input_hash"],
                )
                if status == "started":
                    self._transition(call, "uncertain", expected={"started"}, reason="runtime_restart")
                elif status in {"pending", "approved"}:
                    self.expire(call)
                else:
                    raise RuntimeError(f"Unknown recorded approval status: {status!r}")

    async def execute(
        self,
        call: ApprovedToolCall,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        executor: Callable[[], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        arguments = deepcopy(arguments)
        input_hash = canonical_state_hash({"tool_name": tool_name, "arguments": arguments})
        if tool_name != call.tool_name or input_hash != call.input_hash:
            raise ValueError("Approved tool name or arguments changed")
        verify_live_authority()
        self._transition(call, "started", expected={"approved"})
        context = self._tool_context(call)
        try:
            record = await execute_tool_async(
                tool_name=tool_name, parameters=arguments, executor=executor, turn_context=context,
            )
        except (Exception, asyncio.CancelledError) as exc:
            self._transition(call, "uncertain", expected={"started"}, error_type=type(exc).__name__)
            logger.warning("Approved tool outcome is uncertain: %s", type(exc).__name__)
            raise
        result = record.result
        if not isinstance(result, dict):
            raise TypeError("Approved tool execution must return a result mapping")
        if record.status == "uncertain":
            self._transition(call, "uncertain", expected={"started"}, output_hash=canonical_state_hash(result))
            return result
        io = getattr(context, "_tool_io_ledger", None)
        if not isinstance(io, ToolIOLedger) or len(io.records) != 1:
            raise RuntimeError("Approved tool execution did not produce exactly one kernel IO record")
        snapshot = io.to_dict()
        json.dumps(snapshot, allow_nan=False)
        with self.ledger._lock:
            event = self.writer.write_event(
                _IO_EVENT, tenant_id=self.tenant_id, session_id=call.session_id, trace_id=call.trace_id,
                kernel_step_id="tools.approval", committed=True,
                payload={
                    "request_id": call.request_id, "tool_io": snapshot, "tool_io_hash": canonical_state_hash(snapshot),
                },
            )
            self._transition(
                call, "succeeded", expected={"started"},
                output_hash=canonical_state_hash(result), io_event_id=event["event_id"],
            )
        return result

    def _tool_context(self, call: ApprovedToolCall) -> TurnContext:
        return TurnContext(
            user_input="", trace_id=call.trace_id,
            metadata={
                "tenant_id": self.tenant_id, "session_id": call.session_id, "request_id": call.request_id,
                "tool_version": "mcp.approved.v1",
            },
        )

    async def replay(self, call: ApprovedToolCall, *, arguments: dict[str, Any]) -> dict[str, Any]:
        """Read a completed receipt through kernel replay; never connect or dispatch."""
        arguments = deepcopy(arguments)
        if canonical_state_hash({"tool_name": call.tool_name, "arguments": arguments}) != call.input_hash:
            raise ValueError("Approved tool name or arguments changed")
        completion = self._recorded(call)
        if (
            completion is None or completion.get("committed") is not True
            or completion["payload"]["status"] != "succeeded"
        ):
            raise RuntimeError("Replay requires a durably successful approval")
        event_id = completion["payload"].get("io_event_id")
        event = next((event for event in self.ledger.read() if event.get("event_id") == event_id), None)
        if (
            event is None or event.get("committed") is not True
            or event.get("type") != _IO_EVENT or event.get("tenant_id") != self.tenant_id
            or event.get("session_id") != call.session_id or event.get("trace_id") != call.trace_id
            or event.get("payload", {}).get("request_id") != call.request_id
        ):
            raise ReplayInvariantViolation("Completed tool IO receipt is missing or has a different scope")
        payload = event["payload"]
        snapshot = payload.get("tool_io")
        if not isinstance(snapshot, dict) or canonical_state_hash(snapshot) != payload.get("tool_io_hash"):
            raise ReplayInvariantViolation("Completed tool IO receipt integrity check failed")
        records = snapshot.get("records")
        if (
            not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict)
            or not all(isinstance(records[0].get(key), dict) for key in ("input_payload", "output_payload", "metadata"))
        ):
            raise ReplayInvariantViolation("Completed tool IO receipt has an invalid record shape")
        io = ToolIOLedger.from_dict(snapshot)
        if len(io.records) != 1 or canonical_state_hash(io.to_dict()) != payload["tool_io_hash"]:
            raise ReplayInvariantViolation("Completed tool IO receipt has an invalid record shape")
        if canonical_state_hash(io.records[0].output_payload) != completion["payload"].get("output_hash"):
            raise ReplayInvariantViolation("Completed tool output does not match its approval receipt")
        context = self._tool_context(call)
        context.metadata.update({"replay_mode": True, "_tool_io_ledger": io})

        async def forbidden_dispatch() -> dict[str, Any]:
            raise ReplayInvariantViolation("Completed tool replay must not dispatch an external effect")

        record = await execute_tool_async(
            tool_name=call.tool_name, parameters=arguments, executor=forbidden_dispatch, turn_context=context,
        )
        result = record.result
        return result
