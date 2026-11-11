from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_writer import LedgerWriter
from dadbot.core.state_lineage import canonical_state_hash
from dadbot_system.delegation import require_tenant_authority_if_active, verify_live_authority
from dadbot_system.execution_identity import require_scoped_tenant_id

logger = logging.getLogger(__name__)

_EVENT = "TOOL_APPROVAL_UPDATED"
_TERMINAL = frozenset({"rejected", "expired", "succeeded", "uncertain"})


@dataclass(frozen=True)
class ApprovedToolCall:
    session_id: str
    trace_id: str
    request_id: str
    tool_name: str
    input_hash: str


class ToolApprovalJournal:
    """Commit approval/dispatch boundaries without replaying external effects."""

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
        input_hash = canonical_state_hash({"tool_name": tool_name, "arguments": arguments})
        if tool_name != call.tool_name or input_hash != call.input_hash:
            raise ValueError("Approved tool name or arguments changed")
        verify_live_authority()
        self._transition(call, "started", expected={"approved"})
        try:
            result = await executor()
        except (Exception, asyncio.CancelledError) as exc:
            self._transition(call, "uncertain", expected={"started"}, error_type=type(exc).__name__)
            logger.warning("Approved tool outcome is uncertain: %s", type(exc).__name__)
            raise
        if not isinstance(result, dict):
            self._transition(call, "uncertain", expected={"started"}, error_type="TypeError")
            raise TypeError("Approved tool execution must return a result mapping")
        status = "uncertain" if result.get("is_error") else "succeeded"
        self._transition(
            call,
            status,
            expected={"started"},
            output_hash=canonical_state_hash(result),
        )
        return result
