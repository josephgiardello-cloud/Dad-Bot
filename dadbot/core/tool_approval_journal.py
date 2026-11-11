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
from dadbot.core.tenant_session import current_tenant_session
from dadbot.core.tool_executor import execute_tool_async
from dadbot.core.tool_recording import ToolIOLedger
from dadbot_system.delegation import require_tenant_authority_if_active, verify_live_authority
from dadbot_system.execution_identity import require_scoped_tenant_id

logger = logging.getLogger(__name__)

_EVENT = "TOOL_APPROVAL_UPDATED"
_IO_EVENT = "TOOL_IO_RECORDED"
_REVIEW_EVENT = "TOOL_ACTION_REVIEWED"
_TERMINAL = frozenset({"rejected", "expired", "succeeded", "uncertain"})


@dataclass(frozen=True)
class ApprovedToolCall:
    session_id: str
    trace_id: str
    request_id: str
    tool_name: str
    input_hash: str


@dataclass(frozen=True)
class ToolActionState:
    call: ApprovedToolCall
    outcome_event_id: str
    status: str
    review_outcome: str = ""

    @property
    def retry_policy(self) -> str:
        if self.status == "uncertain":
            return {
                "": "manual_review_required",
                "performed": "do_not_repeat",
                "not_performed": "new_proposal_required",
            }[self.review_outcome]
        return {
            "pending": "approval_required", "approved": "dispatch_pending",
            "started": "await_completion_no_retry",
            "rejected": "new_proposal_required", "expired": "new_proposal_required",
            "succeeded": "do_not_repeat",
        }[self.status]


class ToolApprovalJournal:
    """Commit approval/dispatch boundaries and replay only durably completed IO."""

    def __init__(self, ledger: ExecutionLedger, *, tenant_id: str) -> None:
        self.ledger = ledger
        self.tenant_id = require_scoped_tenant_id(tenant_id)
        self.writer = LedgerWriter(ledger)

    def _latest(self, events: list[dict[str, Any]] | None = None) -> dict[tuple[str, str], dict[str, Any]]:
        latest = {}
        for event in events if events is not None else self.ledger.read():
            if event.get("type") == _EVENT and event.get("tenant_id") == self.tenant_id:
                payload = event.get("payload")
                if event.get("committed") is not True or not isinstance(payload, dict) or not all(
                    isinstance(payload.get(key), str) and payload[key]
                    for key in ("request_id", "trace_id", "tool_name", "input_hash", "status")
                ):
                    raise RuntimeError("Recorded approval has invalid identity or status")
                latest[(event["session_id"], payload["request_id"])] = event
        return latest

    @staticmethod
    def _call_for_event(event: dict[str, Any]) -> ApprovedToolCall:
        payload = event["payload"]
        return ApprovedToolCall(
            session_id=event["session_id"], trace_id=payload["trace_id"],
            request_id=payload["request_id"], tool_name=payload["tool_name"], input_hash=payload["input_hash"],
        )

    def _require_session_scope(self, session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("Action review requires a session identity")
        active = current_tenant_session()
        if active is not None and (active.tenant_id != self.tenant_id or active.session_id != session_id):
            raise ValueError("Action review scope does not match the active session")
        require_tenant_authority_if_active(self.tenant_id)

    def _reviews(self, events: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
        reviews: dict[tuple[str, str], dict[str, Any]] = {}
        for event in events:
            if event.get("type") != _REVIEW_EVENT or event.get("tenant_id") != self.tenant_id:
                continue
            payload = event.get("payload")
            if (
                event.get("committed") is not True or not isinstance(payload, dict)
                or payload.get("review_source") != "user_confirmation"
                or payload.get("outcome") not in ("performed", "not_performed")
                or not all(isinstance(payload.get(key), str) and payload[key] for key in (
                    "request_id", "trace_id", "tool_name", "input_hash", "outcome_event_id",
                ))
                or not isinstance(payload.get("review_scope"), dict)
                or not all(isinstance(value, str) and value for value in payload["review_scope"].values())
                or payload["review_scope"] != {
                    "tenant_id": self.tenant_id, "session_id": event["session_id"],
                    "trace_id": event["trace_id"], "request_id": payload["review_scope"].get("request_id"),
                }
                or not payload["review_scope"].get("request_id")
            ):
                raise RuntimeError("Recorded action review has invalid identity or outcome")
            key = (event["session_id"], payload["request_id"])
            previous = reviews.get(key)
            if previous is not None and previous["payload"] != payload:
                raise RuntimeError("Recorded action has conflicting reviews")
            reviews[key] = event
        return reviews

    def actions(self, session_id: str) -> list[ToolActionState]:
        """Project durable action outcomes and user confirmations without exposing IO."""
        self._require_session_scope(session_id)
        with self.ledger._lock:
            events = self.ledger.read()
            reviews = self._reviews(events)
            actions = []
            for key, event in self._latest(events).items():
                if key[0] != session_id:
                    continue
                call = self._call_for_event(event)
                status = event["payload"]["status"]
                if status not in _TERMINAL | {"pending", "approved", "started"}:
                    raise RuntimeError("Recorded action has an unknown outcome")
                review = reviews.pop(key, None)
                if review is not None and (
                    status != "uncertain" or review["payload"]["outcome_event_id"] != event["event_id"]
                    or any(review["payload"][field] != getattr(call, field) for field in (
                        "trace_id", "tool_name", "input_hash",
                    ))
                ):
                    raise RuntimeError("Recorded action review does not match its uncertain outcome")
                actions.append(ToolActionState(
                    call=call, outcome_event_id=event["event_id"], status=status,
                    review_outcome=review["payload"]["outcome"] if review is not None else "",
                ))
            if any(key[0] == session_id for key in reviews):
                raise RuntimeError("Recorded action review has no matching approval")
            return actions

    def review_action(
        self, call: ApprovedToolCall, *, outcome_event_id: str, outcome: str, turn_context: TurnContext,
    ) -> dict[str, Any]:
        self._require_session_scope(call.session_id)
        if not isinstance(outcome, str) or outcome not in ("performed", "not_performed"):
            raise ValueError("Action review outcome must be performed or not_performed")
        scope = self._execution_scope(turn_context, call)
        verify_live_authority()
        with self.ledger._lock:
            recorded = self._recorded(call)
            if recorded is None or recorded["payload"]["status"] != "uncertain":
                raise RuntimeError("Only uncertain action outcomes can be reviewed")
            if recorded["event_id"] != outcome_event_id:
                raise ValueError("Action review outcome event is stale or does not match")
            previous = self._reviews(self.ledger.read()).get((call.session_id, call.request_id))
            if previous is not None:
                if previous["payload"]["outcome"] != outcome:
                    raise RuntimeError("This action already has a different review")
                self.actions(call.session_id)
                return previous
            return self.writer.write_event(
                _REVIEW_EVENT, tenant_id=self.tenant_id, session_id=call.session_id,
                trace_id=turn_context.trace_id, kernel_step_id="tools.action_review", committed=True,
                payload={
                    "request_id": call.request_id, "trace_id": call.trace_id,
                    "tool_name": call.tool_name, "input_hash": call.input_hash,
                    "outcome_event_id": outcome_event_id, "outcome": outcome,
                    "review_source": "user_confirmation", "review_scope": scope,
                },
            )

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
                call = self._call_for_event(event)
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
        turn_context: TurnContext | None = None,
    ) -> dict[str, Any]:
        arguments = deepcopy(arguments)
        input_hash = canonical_state_hash({"tool_name": tool_name, "arguments": arguments})
        if tool_name != call.tool_name or input_hash != call.input_hash:
            raise ValueError("Approved tool name or arguments changed")
        context = turn_context if turn_context is not None else self._tool_context(call)
        scope = self._execution_scope(context, call)
        previous_io = getattr(context, "_tool_io_ledger", None)
        previous_count = len(previous_io.records) if isinstance(previous_io, ToolIOLedger) else 0
        previous_version = context.metadata.get("tool_version")
        verify_live_authority()
        self._transition(call, "started", expected={"approved"})
        context.metadata["tool_version"] = "mcp.approved.v1"
        try:
            record = await execute_tool_async(
                tool_name=tool_name, parameters=arguments, executor=executor, turn_context=context,
            )
        except (Exception, asyncio.CancelledError) as exc:
            self._transition(call, "uncertain", expected={"started"}, error_type=type(exc).__name__)
            logger.warning("Approved tool outcome is uncertain: %s", type(exc).__name__)
            raise
        finally:
            if previous_version is None:
                context.metadata.pop("tool_version", None)
            else:
                context.metadata["tool_version"] = previous_version
        result = record.result
        if not isinstance(result, dict):
            raise TypeError("Approved tool execution must return a result mapping")
        if record.status == "uncertain":
            self._transition(call, "uncertain", expected={"started"}, output_hash=canonical_state_hash(result))
            return result
        io = getattr(context, "_tool_io_ledger", None)
        if not isinstance(io, ToolIOLedger) or len(io.records) != previous_count + 1:
            raise RuntimeError("Approved tool execution did not produce exactly one kernel IO record")
        receipt = ToolIOLedger()
        receipt.append(io.records[-1])
        snapshot = receipt.to_dict()
        json.dumps(snapshot, allow_nan=False)
        with self.ledger._lock:
            event = self.writer.write_event(
                _IO_EVENT, tenant_id=self.tenant_id, session_id=call.session_id, trace_id=call.trace_id,
                kernel_step_id="tools.approval", committed=True,
                payload={
                    "request_id": call.request_id, "execution_scope": scope,
                    "tool_io": snapshot, "tool_io_hash": canonical_state_hash(snapshot),
                },
            )
            self._transition(
                call, "succeeded", expected={"started"},
                output_hash=canonical_state_hash(result), io_event_id=event["event_id"],
                execution_scope=scope,
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

    def _execution_scope(self, context: TurnContext, call: ApprovedToolCall) -> dict[str, str]:
        metadata = context.metadata
        control_plane = metadata.get("control_plane") or {}
        scope = {
            "tenant_id": str(metadata.get("tenant_id") or control_plane.get("tenant_id") or ""),
            "session_id": str(metadata.get("session_id") or control_plane.get("session_id") or ""),
            "trace_id": str(context.trace_id or ""),
            "request_id": str(metadata.get("request_id") or ""),
        }
        if (
            scope["tenant_id"] != self.tenant_id or scope["session_id"] != call.session_id
            or not all(scope.values())
            or any(control_plane.get(key) not in (None, scope[key]) for key in ("tenant_id", "session_id"))
        ):
            raise ValueError("Tool execution context does not match the approval tenant/thread")
        return scope

    async def replay(
        self, call: ApprovedToolCall, *, arguments: dict[str, Any], turn_context: TurnContext | None = None,
    ) -> dict[str, Any]:
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
        context = turn_context if turn_context is not None else self._tool_context(call)
        self._validate_receipt_scope(context, call, completion["payload"], payload)
        snapshot = payload.get("tool_io")
        if not isinstance(snapshot, dict) or canonical_state_hash(snapshot) != payload.get("tool_io_hash"):
            raise ReplayInvariantViolation("Completed tool IO receipt integrity check failed")
        records = snapshot.get("records")
        if (
            not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict)
            or not all(isinstance(records[0].get(key), dict) for key in ("input_payload", "output_payload", "metadata"))
        ):
            raise ReplayInvariantViolation("Completed tool IO receipt has an invalid record shape")
        try:
            io = ToolIOLedger.from_checkpoint(snapshot)
        except ValueError as exc:
            raise ReplayInvariantViolation("Completed tool IO receipt has an invalid record shape") from exc
        if len(io.records) != 1 or canonical_state_hash(io.to_dict()) != payload["tool_io_hash"]:
            raise ReplayInvariantViolation("Completed tool IO receipt has an invalid record shape")
        if canonical_state_hash(io.records[0].output_payload) != completion["payload"].get("output_hash"):
            raise ReplayInvariantViolation("Completed tool output does not match its approval receipt")
        context.metadata.update({"replay_mode": True, "_tool_io_ledger": io})
        context.metadata["tool_version"] = "mcp.approved.v1"

        async def forbidden_dispatch() -> dict[str, Any]:
            raise ReplayInvariantViolation("Completed tool replay must not dispatch an external effect")

        record = await execute_tool_async(
            tool_name=call.tool_name, parameters=arguments, executor=forbidden_dispatch, turn_context=context,
        )
        result = record.result
        return result

    def _validate_receipt_scope(
        self, context: TurnContext, call: ApprovedToolCall,
        completion: dict[str, Any], receipt: dict[str, Any],
    ) -> None:
        original_scope = self._execution_scope(self._tool_context(call), call)
        recorded_scope = completion.get("execution_scope", original_scope)
        if (
            receipt.get("execution_scope", original_scope) != recorded_scope
            or self._execution_scope(context, call) != recorded_scope
        ):
            raise ReplayInvariantViolation("Completed tool IO execution scope does not match the replay context")
