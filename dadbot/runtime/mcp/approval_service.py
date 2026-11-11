from __future__ import annotations

import inspect
import logging
import os
from copy import deepcopy
from typing import Any
from uuid import uuid4

from dadbot.core.execution_contract import SovereignContext, TurnDelivery, TurnResponse, live_turn_request
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.graph_context import TurnContext
from dadbot.core.runtime_errors import ReplayInvariantViolation
from dadbot.core.state_lineage import canonical_state_hash
from dadbot.core.tool_approval_journal import ApprovedToolCall, ToolActionState, ToolApprovalJournal
from dadbot.runtime.mcp.tool_client import (
    MCPApprovalRequest,
    MCPToolClient,
    configured_ha_mcp_tools,
    configured_ha_mcp_url,
    resume_mcp_with_fallback,
    run_mcp_with_fallback,
)
from dadbot.services._tool_pipeline_mixin import needs_tool_planning

logger = logging.getLogger(__name__)


class MCPApprovalService:
    """Prepare Home Assistant tool candidates and retain per-thread approvals."""

    def __init__(self, bot: Any) -> None:
        self.bot = bot
        self._pending: dict[str, tuple[MCPApprovalRequest, list[dict[str, Any]], str, ApprovedToolCall]] = {}
        self._journal: ToolApprovalJournal | None = None
        if isinstance(getattr(bot, "_execution_ledger", None), ExecutionLedger):
            self._approval_journal()

    def _approval_journal(self) -> ToolApprovalJournal:
        if self._journal is None:
            ledger = getattr(self.bot, "_execution_ledger", None)
            if not isinstance(ledger, ExecutionLedger):
                raise RuntimeError("MCP approvals require the canonical execution ledger")
            journal = ToolApprovalJournal(ledger, tenant_id=self.bot.config.tenant_id)
            journal.expire_recovered()
            self._journal = journal
        return self._journal

    def _queue_approval(
        self,
        request: MCPApprovalRequest,
        attachments: list[dict[str, Any]],
        mcp_url: str,
        *,
        thread_id: str,
        trace_id: str,
    ) -> None:
        call = ApprovedToolCall(
            session_id=thread_id,
            trace_id=trace_id,
            request_id=request.request_id,
            tool_name=request.tool_name,
            input_hash=canonical_state_hash({"tool_name": request.tool_name, "arguments": request.arguments}),
        )
        journal = self._approval_journal()
        previous = self._pending.get(thread_id)
        if previous is not None:
            journal.expire(previous[3])
        journal.propose(call)
        self._pending[thread_id] = (request, attachments, mcp_url, call)

    @property
    def is_enabled(self) -> bool:
        return bool(configured_ha_mcp_url())

    def get_pending_mcp_approval(self, thread_id: str) -> MCPApprovalRequest | None:
        pending = self._pending.get(str(thread_id or "default"))
        return pending[0] if pending else None

    def get_mcp_action_history(self, thread_id: str) -> list[ToolActionState]:
        if thread_id != str(self.bot.active_thread_id):
            raise ValueError("MCP action history must target the active thread")
        return self._approval_journal().actions(thread_id)

    def _review_target(self, thread_id: str, request_id: str, outcome_event_id: str) -> ToolActionState:
        action = next((
            action for action in self.get_mcp_action_history(thread_id) if action.call.request_id == request_id
        ), None)
        if action is None or action.outcome_event_id != outcome_event_id:
            raise ValueError("MCP action review no longer matches the recorded outcome")
        if action.status != "uncertain":
            raise ValueError("MCP action review requires an uncertain outcome")
        return action

    def _build_messages(
        self,
        context: Any,
        rich_context: dict[str, Any],
        attachments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        raw_messages = rich_context.get("messages")
        messages: list[dict[str, Any]] = []
        if isinstance(raw_messages, list):
            for item in raw_messages:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "user").strip().lower() or "user"
                content = str(item.get("content") or item.get("text") or "").strip()
                if role in {"system", "user", "assistant"} and content:
                    messages.append({"role": role, "content": content})

        user_input = str(getattr(context, "user_input", "") or "").strip()
        if not messages and user_input:
            messages.append({"role": "user", "content": user_input})
        if not messages:
            raise RuntimeError("MCP approval service requires a non-empty turn message")

        if not any(message["role"] == "system" for message in messages):
            prompt_assembly = getattr(self.bot, "prompt_assembly", None)
            build_system_prompt = getattr(prompt_assembly, "build_request_system_prompt", None)
            if not callable(build_system_prompt):
                raise RuntimeError("MCP approval service requires a wired system-prompt builder")
            state = getattr(context, "state", None)
            mood = str(state.get("mood") or "neutral") if isinstance(state, dict) else "neutral"
            latest_user_input = next(
                (
                    str(message["content"])
                    for message in reversed(messages)
                    if message["role"] == "user"
                ),
                user_input,
            )
            system_prompt = str(
                build_system_prompt(latest_user_input, mood, attachments) or "",
            ).strip()
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

        image_payloads = list(
            dict.fromkeys(
                str(item.get("image_b64") or item.get("data_b64") or "").strip()
                for item in attachments
                if str(item.get("type") or "").strip().lower() == "image"
                and str(item.get("image_b64") or item.get("data_b64") or "").strip()
            ),
        )[:6]
        if image_payloads:
            for message in reversed(messages):
                if message["role"] == "user":
                    message["images"] = image_payloads
                    break
        return messages

    @staticmethod
    def _collect_attachments(context: Any, rich_context: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[Any] = []
        for source in (rich_context, getattr(context, "state", None)):
            if not isinstance(source, dict):
                continue
            for key in ("attachments", "norm_attachments"):
                value = source.get(key)
                if isinstance(value, (list, tuple)):
                    candidates.extend(value)
        context_attachments = getattr(context, "attachments", None)
        if isinstance(context_attachments, (list, tuple)):
            candidates.extend(context_attachments)
        return [
            dict(item)
            for item in candidates
            if isinstance(item, dict)
            and str(item.get("type") or "").strip().lower() in {"image", "audio", "document"}
        ][:6]

    def _model_call(
        self, messages: list[dict[str, Any]], *, purpose: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        model_port = getattr(self.bot, "model_port", None)
        generate = getattr(model_port, "generate", None)
        if not callable(generate):
            raise RuntimeError("MCP approval service requires a wired model port")
        model_name = str(getattr(self.bot, "ACTIVE_MODEL", "") or "").strip()
        if not model_name:
            raise RuntimeError("MCP approval service requires a configured active model")
        generation_kwargs = {}
        if purpose == "mcp_tool_plan" and self.bot.model_runtime.normalized_llm_provider() == "ollama":
            generation_kwargs["think"] = False
        response = generate(
            messages,
            model=model_name,
            response_format=response_format if purpose == "mcp_tool_plan" else None,
            purpose=purpose,
            **generation_kwargs,
        )
        reply = str(response or "").strip()
        if not reply:
            raise RuntimeError("MCP model returned an empty response")
        return reply

    async def generate_candidate(
        self,
        context: Any,
        rich_context: dict[str, Any] | None = None,
    ) -> tuple[str, bool] | None:
        mcp_url = configured_ha_mcp_url()
        if not mcp_url:
            return None
        rich_context = dict(rich_context or {})
        attachments = self._collect_attachments(context, rich_context)
        allowed_tools = configured_ha_mcp_tools()
        history = rich_context.get("messages") or self.bot.prompt_history()
        if not needs_tool_planning(
            getattr(context, "user_input", ""),
            [{"function": {"name": name}} for name in allowed_tools],
            attachments,
            recent_history=history,
        ):
            logger.info("MCP tool planning skipped: no tool intent")
            return None
        messages = self._build_messages(context, rich_context, attachments)
        client = MCPToolClient(
            mcp_url,
            bearer_token=str(os.environ.get("DADBOT_HA_MCP_TOKEN") or ""),
        )
        result = await run_mcp_with_fallback(
            messages,
            client,
            allowed_tools,
            model_call=lambda candidate_messages, schema: self._model_call(
                candidate_messages,
                purpose="mcp_tool_plan",
                response_format=schema,
            ),
            fallback_call=lambda candidate_messages: self._model_call(
                candidate_messages,
                purpose="mcp_fallback",
            ),
        )
        if isinstance(result, MCPApprovalRequest):
            metadata = getattr(context, "metadata", {}) or {}
            thread_id = str(
                metadata.get("control_plane", {}).get("session_id")
                or metadata.get("session_id")
                or getattr(self.bot, "_streamlit_active_thread_id", None)
                or getattr(self.bot, "active_thread_id", "default")
                or "default"
            )
            self._queue_approval(
                result, attachments, mcp_url,
                thread_id=thread_id, trace_id=str(getattr(context, "trace_id", "") or result.request_id),
            )
            reply = "A Home Assistant action is ready for your review. It will not run until you approve it."
        else:
            reply = str(result or "").strip()
        if not reply:
            raise RuntimeError("MCP approval service returned an empty candidate")
        return reply, False

    @staticmethod
    def decision_text(*, approved: bool) -> str:
        return "Approve this Home Assistant action." if approved else "Reject this Home Assistant action."

    async def resolve_mcp_approval(
        self, thread_id: str, *, approved: bool, request_id: str | None = None,
    ) -> str:
        if type(approved) is not bool:
            raise ValueError("MCP approval decision must be an explicit boolean")
        key = str(thread_id or "default")
        if key != str(self.bot.active_thread_id):
            raise ValueError("MCP approval decision must target the active thread")
        pending = self._pending.get(key)
        if pending is None:
            if request_id is not None:
                raise ValueError("MCP approval button no longer matches a pending proposal")
            return "This Home Assistant approval is no longer pending. No additional action was sent."
        call = pending[3]
        if request_id is not None and request_id != call.request_id:
            raise ValueError("MCP approval button no longer matches the pending proposal")
        decision_id = "mcp-decision:" + canonical_state_hash({
            "tenant_id": self.bot.config.tenant_id, "session_id": key,
            "approval_id": call.request_id, "approved": approved,
        })
        return await self._submit_continuation(
            call, self.decision_text(approved=approved),
            resolution={"request_id": call.request_id, "approved": approved}, request_id=decision_id,
        )

    @staticmethod
    def review_text(*, outcome: str, request_id: str) -> str:
        if outcome not in ("performed", "not_performed"):
            raise ValueError("MCP action review requires performed or not_performed")
        observed = "the action happened" if outcome == "performed" else "no action happened"
        return f"I checked Home Assistant and verified that {observed} for approval {request_id}."

    async def review_mcp_action(
        self, thread_id: str, *, request_id: str, outcome_event_id: str, outcome: str,
    ) -> str:
        text = self.review_text(outcome=outcome, request_id=request_id)
        action = self._review_target(thread_id, request_id, outcome_event_id)
        return await self._submit_continuation(
            action.call, text,
            resolution={"request_id": request_id, "outcome_event_id": outcome_event_id, "review_outcome": outcome},
            request_id="mcp-review:" + uuid4().hex,
        )

    async def _submit_continuation(
        self, call: ApprovedToolCall, text: str, *, resolution: dict[str, Any], request_id: str,
    ) -> str:
        execute_turn = getattr(self.bot, "execute_turn", None)
        if not callable(execute_turn):
            raise RuntimeError("MCP continuation requires canonical execute_turn")
        request = live_turn_request(
            text, delivery=TurnDelivery.ASYNC,
            context=SovereignContext(
                tenant_id=self.bot.config.tenant_id, session_id=call.session_id, request_id=request_id,
            ),
            metadata={
                "request_id": request_id, "mcp_approval_resolution": resolution,
                "mcp_approval_parent_trace_id": call.trace_id,
            },
        )
        response = execute_turn(request)
        response = await response if inspect.isawaitable(response) else response
        if not isinstance(response, TurnResponse) or not str(response.reply or "").strip():
            raise RuntimeError("Canonical MCP continuation returned an invalid response")
        return str(response.reply)

    @staticmethod
    def _validate_resolution(resolution: Any) -> dict[str, Any]:
        if not isinstance(resolution, dict) or not isinstance(resolution.get("request_id"), str):
            raise ValueError("Graph MCP resolution requires an exact request identity")
        decision = set(resolution) == {"request_id", "approved"} and type(resolution.get("approved")) is bool
        review = (
            set(resolution) == {"request_id", "review_outcome", "outcome_event_id"}
            and resolution.get("review_outcome") in ("performed", "not_performed")
            and isinstance(resolution.get("outcome_event_id"), str) and bool(resolution["outcome_event_id"])
        )
        if not resolution["request_id"] or not (decision or review):
            raise ValueError("Graph MCP resolution requires a boolean decision or an exact action review")
        return resolution

    async def resolve_in_graph(self, context: TurnContext) -> str:
        resolution = self._validate_resolution(context.metadata.get("mcp_approval_resolution"))
        key = str(context.metadata.get("session_id") or "")
        if (
            context.metadata.get("tenant_id") != self.bot.config.tenant_id
            or key != str(self.bot.active_thread_id)
            or context.metadata.get("source_mode") is True
        ):
            raise ValueError("Graph MCP approval resolution has a different tenant/thread or source mode")
        if context.metadata.get("replay_mode") is True:
            reply = context.state.get("mcp_approval_result")
            if (
                context.state.get("mcp_approval_resolution") != resolution
                or not isinstance(reply, str) or not reply.strip()
            ):
                raise ReplayInvariantViolation("MCP approval replay requires the matching checkpoint result")
            return reply
        context.state["decision_outcome"] = "no_tool_needed"
        context.state["tool_execution"] = {
            "executor": "kernel_mcp_v1", "selected_tool": None, "status": "not_dispatched",
        }
        if "review_outcome" in resolution:
            reply = self._review_in_graph(context, resolution)
        else:
            reply = await self._resolve_pending(context, resolution)
        context.state["mcp_approval_resolution"] = dict(resolution)
        context.state["mcp_approval_result"] = reply
        return reply

    def _review_in_graph(self, context: TurnContext, resolution: dict[str, Any]) -> str:
        action = self._review_target(
            context.metadata["session_id"], resolution["request_id"], resolution["outcome_event_id"],
        )
        if context.metadata.get("mcp_approval_parent_trace_id") != action.call.trace_id:
            raise ValueError("Graph MCP action review has a different parent approval")
        self._approval_journal().review_action(
            action.call, outcome_event_id=resolution["outcome_event_id"],
            outcome=resolution["review_outcome"], turn_context=context,
        )
        if resolution["review_outcome"] == "performed":
            return (
                "Recorded your confirmation that the Home Assistant action happened. "
                "This is your observation, not a verified tool-success receipt. "
                "The original uncertain action remains blocked from replay or resend."
            )
        return (
            "Recorded your confirmation that no Home Assistant action happened. "
            "This is your observation, not a provider-verified result. "
            "The original uncertain action will not be replayed or resent. "
            "To try again, request a new action and review its fresh approval."
        )

    async def _resolve_pending(self, context: TurnContext, resolution: dict[str, Any]) -> str:
        key = str(context.metadata["session_id"])
        approved = resolution["approved"]
        pending = self._pending.get(key)
        if pending is None:
            return "This Home Assistant approval is no longer pending. No additional action was sent."

        request, _attachments, original_url, call = pending
        if (
            call.request_id != resolution["request_id"]
            or context.metadata.get("mcp_approval_parent_trace_id") != call.trace_id
        ):
            raise ValueError("Graph MCP approval no longer matches the original proposal")
        journal = self._approval_journal()
        journal.decide(call, approved=approved)
        del self._pending[key]
        current_url = configured_ha_mcp_url()
        if approved and current_url != original_url:
            request.consumed = True
            journal.expire(call)
            return "Home Assistant configuration changed, so the proposed action expired. No action was sent."

        client = MCPToolClient(
            current_url or original_url,
            bearer_token=str(os.environ.get("DADBOT_HA_MCP_TOKEN") or ""),
        )

        async def execute_approved(session: Any, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            frozen_arguments = deepcopy(arguments)
            context.state["decision_outcome"] = "executed_tool"
            context.state["tool_execution"] = {
                "executor": "kernel_mcp_v1", "selected_tool": name, "status": "approved",
            }
            try:
                return await journal.execute(
                    call, tool_name=name, arguments=frozen_arguments,
                    executor=lambda: client.call_tool(session, name, deepcopy(frozen_arguments)),
                    turn_context=context,
                )
            finally:
                context.state["tool_execution"]["status"] = journal.status(call)

        result = await resume_mcp_with_fallback(
            request,
            client,
            configured_ha_mcp_tools(),
            approved=approved,
            model_call=lambda candidate_messages, schema: self._model_call(
                candidate_messages,
                purpose="mcp_tool_plan",
                response_format=schema,
            ),
            tool_call=execute_approved,
        )
        if journal.status(call) == "approved":
            journal.expire(call)
        if isinstance(result, MCPApprovalRequest):
            self._queue_approval(
                result, _attachments, original_url, thread_id=key, trace_id=context.trace_id,
            )
            return "Another Home Assistant action needs your approval before it can run."
        return str(result or "")
