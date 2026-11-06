from __future__ import annotations

import logging
import os
from typing import Any

from dadbot.services._tool_pipeline_mixin import needs_tool_planning
from dadbot.runtime.mcp.tool_client import (
    MCPApprovalRequest,
    MCPToolClient,
    configured_ha_mcp_tools,
    configured_ha_mcp_url,
    mcp_plan_schema,
    resume_mcp_with_fallback,
    run_mcp_with_fallback,
)

logger = logging.getLogger(__name__)


class MCPApprovalService:
    """Prepare Home Assistant tool candidates and retain per-thread approvals."""

    def __init__(self, bot: Any) -> None:
        self.bot = bot
        self._pending: dict[str, tuple[MCPApprovalRequest, list[dict[str, Any]], str]] = {}

    @property
    def is_enabled(self) -> bool:
        return bool(configured_ha_mcp_url())

    def get_pending_mcp_approval(self, thread_id: str) -> MCPApprovalRequest | None:
        pending = self._pending.get(str(thread_id or "default"))
        return pending[0] if pending else None

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

    def _model_call(self, messages: list[dict[str, Any]], *, purpose: str) -> str:
        model_port = getattr(self.bot, "model_port", None)
        generate = getattr(model_port, "generate", None)
        if not callable(generate):
            raise RuntimeError("MCP approval service requires a wired model port")
        model_name = str(getattr(self.bot, "ACTIVE_MODEL", "") or "").strip()
        if not model_name:
            raise RuntimeError("MCP approval service requires a configured active model")
        response = generate(
            messages,
            model=model_name,
            response_format=mcp_plan_schema() if purpose == "mcp_tool_plan" else None,
            purpose=purpose,
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
            model_call=lambda candidate_messages: self._model_call(
                candidate_messages,
                purpose="mcp_tool_plan",
            ),
            fallback_call=lambda candidate_messages: self._model_call(
                candidate_messages,
                purpose="mcp_fallback",
            ),
        )
        if isinstance(result, MCPApprovalRequest):
            thread_id = str(
                getattr(self.bot, "_streamlit_active_thread_id", None)
                or getattr(self.bot, "active_thread_id", "default")
                or "default"
            )
            self._pending[thread_id] = (result, attachments, mcp_url)
            reply = "A Home Assistant action is ready for your review. It will not run until you approve it."
        else:
            reply = str(result or "").strip()
        if not reply:
            raise RuntimeError("MCP approval service returned an empty candidate")
        return reply, False

    async def resolve_mcp_approval(self, thread_id: str, *, approved: bool) -> str:
        key = str(thread_id or "default")
        pending = self._pending.pop(key, None)
        if pending is None:
            return "This Home Assistant approval is no longer pending. No action was sent."

        request, _attachments, original_url = pending
        current_url = configured_ha_mcp_url()
        if approved and current_url != original_url:
            request.consumed = True
            return "Home Assistant configuration changed, so the proposed action expired. No action was sent."

        client = MCPToolClient(
            current_url or original_url,
            bearer_token=str(os.environ.get("DADBOT_HA_MCP_TOKEN") or ""),
        )
        result = await resume_mcp_with_fallback(
            request,
            client,
            configured_ha_mcp_tools(),
            approved=approved,
            model_call=lambda candidate_messages: self._model_call(
                candidate_messages,
                purpose="mcp_tool_plan",
            ),
        )
        if isinstance(result, MCPApprovalRequest):
            self._pending[key] = (result, _attachments, original_url)
            return "Another Home Assistant action needs your approval before it can run."
        return str(result or "")
