from __future__ import annotations

import os
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from dadbot.contracts import AttachmentList, FinalizedTurnResult


def thin_turn_handler_enabled() -> bool:
    # Canonical default is enabled; alternate paths are no longer runtime-selectable.
    value = str(os.environ.get("DADBOT_USE_THIN_TURN_HANDLER", "1") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class TurnContext:
    user_input: str
    session_id: str
    attachments: AttachmentList | None = None
    confluence_key: str | None = None
    metadata: Mapping[str, Any] | None = None
    timeout_seconds: float | None = None


class TurnHandler:
    """Thin, deterministic turn spine adapter.

    This wrapper does not decide policy or mutate global state; it only normalizes
    submit payload shape and delegates into the authoritative execution entrypoint.
    """

    def __init__(
        self,
        *,
        submit_turn: Callable[..., Awaitable[FinalizedTurnResult]],
        policy_store: Any | None = None,
        prompt_builder: Callable[[], str] | None = None,
        memory_ledger: Any | None = None,
        relationship_snapshotter: Callable[[], Mapping[str, Any] | None] | None = None,
    ) -> None:
        self._submit_turn = submit_turn
        self._policy_store = policy_store
        self._prompt_builder = prompt_builder
        self._memory_ledger = memory_ledger
        self._relationship_snapshotter = relationship_snapshotter

    async def _inject_policy_metadata(self, metadata: dict[str, Any]) -> None:
        store = self._policy_store
        if store is None:
            return
        getter = getattr(store, "get_current_policy", None)
        if not callable(getter):
            return
        try:
            policy = await getter()
        except (RuntimeError, ValueError, TypeError):
            return

        if metadata.get("dad_policy_version"):
            return

        version = str(getattr(policy, "version", "") or "").strip()
        if version:
            metadata["dad_policy_version"] = version
            metadata.setdefault("policy_version", version)

        template = {
            "persona_style": dict(getattr(policy, "persona_style", {}) or {}),
            "relationship_rules": dict(getattr(policy, "relationship_rules", {}) or {}),
            "safety_boundaries": dict(getattr(policy, "safety_boundaries", {}) or {}),
            "memory_preferences": dict(getattr(policy, "memory_preferences", {}) or {}),
        }
        metadata.setdefault("dad_policy_template", template)

    async def _inject_prompt_context(self, metadata: dict[str, Any]) -> None:
        builder = self._prompt_builder
        if builder is None or metadata.get("prompt_context"):
            return
        try:
            prompt_context = builder()
            if inspect.isawaitable(prompt_context):
                prompt_context = await prompt_context
        except (RuntimeError, ValueError, TypeError):
            return

        normalized_context = str(prompt_context or "").strip()
        if not normalized_context:
            return

        metadata.setdefault("prompt_context", normalized_context)
        rich_context = dict(metadata.get("rich_context") or {})
        rich_context.setdefault("prompt_context", normalized_context)
        metadata["rich_context"] = rich_context

    async def _append_memory_ledger_event(
        self,
        *,
        ctx: TurnContext,
        metadata: Mapping[str, Any],
        result: FinalizedTurnResult,
    ) -> None:
        ledger = self._memory_ledger
        append_event = getattr(ledger, "append_memory_event", None)
        if not callable(append_event):
            return

        reply, should_end = result
        relationship_projection = dict(metadata.get("relationship_projection") or {})
        payload = {
            "event_type": "turn.finalized",
            "session_id": str(ctx.session_id or "default"),
            "confluence_key": str(ctx.confluence_key or ""),
            "user_input": str(ctx.user_input or ""),
            "reply_preview": str(reply or "")[:240],
            "should_end": bool(should_end),
            "policy_version": str(metadata.get("policy_version") or metadata.get("dad_policy_version") or ""),
            "prompt_context": str(metadata.get("prompt_context") or "")[:240],
            "relationship_trust_level": int(relationship_projection.get("trust_level", 0) or 0),
            "relationship_openness_level": int(relationship_projection.get("openness_level", 0) or 0),
            "relationship_emotional_momentum": str(
                relationship_projection.get("emotional_momentum") or "",
            ),
        }
        try:
            write_result = append_event(payload)
            if inspect.isawaitable(write_result):
                await write_result
        except (RuntimeError, ValueError, TypeError):
            return

    async def _inject_relationship_projection(self, metadata: dict[str, Any]) -> None:
        if metadata.get("relationship_projection"):
            return
        snapshotter = self._relationship_snapshotter
        if snapshotter is None:
            return
        try:
            projection = snapshotter()
            if inspect.isawaitable(projection):
                projection = await projection
        except (RuntimeError, ValueError, TypeError):
            return
        projection_dict = dict(projection or {})
        if not projection_dict:
            return
        metadata["relationship_projection"] = projection_dict

    async def process_turn(self, ctx: TurnContext) -> FinalizedTurnResult:
        outbound_metadata: dict[str, Any] = {
            "confluence_mode": "enforce",
            "confluence_key": str(ctx.confluence_key or "").strip(),
        }
        if ctx.metadata:
            outbound_metadata.update(dict(ctx.metadata))
        await self._inject_policy_metadata(outbound_metadata)
        await self._inject_prompt_context(outbound_metadata)

        result = await self._submit_turn(
            str(ctx.user_input or ""),
            attachments=ctx.attachments,
            session_id=str(ctx.session_id or "default"),
            confluence_key=str(ctx.confluence_key or ""),
            metadata=outbound_metadata,
            timeout_seconds=ctx.timeout_seconds,
        )
        await self._inject_relationship_projection(outbound_metadata)
        await self._append_memory_ledger_event(
            ctx=ctx,
            metadata=outbound_metadata,
            result=result,
        )
        return result
