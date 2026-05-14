from __future__ import annotations

import logging
from dadbot.contracts import (
    AttachmentList,
    ChunkCallback,
    DadBotContext,
    SupportsTurnProcessingRuntime,
)

logger = logging.getLogger(__name__)


class ReplyGenerationManager:
    """Legacy compatibility shim for reply-generation APIs.

    Phase 1 authority collapse: direct reply-generation entrypoints are
    intentionally disabled so imports cannot bypass control-plane authority.
    """

    def __init__(self, bot: DadBotContext | SupportsTurnProcessingRuntime):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot

    @staticmethod
    def _authority_disabled_error(path: str) -> RuntimeError:
        return RuntimeError(
            f"ReplyGenerationManager.{path} authority is disabled; use control_plane.execute_from_graph_context",
        )

    def _record_direct_call_attempt(
        self,
        *,
        path: str,
        stripped_input: str,
        turn_text: str,
        current_mood: str,
        normalized_attachments: AttachmentList,
        stream: bool,
    ) -> None:
        reason = (
            "Direct reply_generation invocation attempted; "
            "authority is restricted to control_plane.execute_from_graph_context"
        )
        logger.warning(reason)
        metadata = {
            "path": path,
            "stream": bool(stream),
            "attachments": len(list(normalized_attachments or [])),
            "authority_disabled": True,
        }
        recorder = getattr(self.bot, "record_shadow_decision", None)
        event = (
            recorder(
                source="reply_generation",
                type="override_attempt",
                content_preview="",
                reason=reason,
                would_replace=False,
                priority=0.0,
                metadata=metadata,
            )
            if callable(recorder)
            else None
        )
        setattr(
            self.bot,
            "_last_reply_generation_shadow",
            {
                "path": path,
                "validation_input": str(stripped_input or turn_text or ""),
                "current_mood": str(current_mood or "neutral"),
                "applied": False,
                "authority_disabled": True,
                "bus_event": event,
            },
        )

    def generate_validated_reply(
        self,
        stripped_input: str,
        turn_text: str,
        current_mood: str,
        normalized_attachments: AttachmentList,
        stream: bool = False,
        chunk_callback: ChunkCallback | None = None,
    ) -> str:
        """Disabled authority surface preserved only for compatibility telemetry."""
        _ = chunk_callback
        self._record_direct_call_attempt(
            path="generate_validated_reply",
            stripped_input=stripped_input,
            turn_text=turn_text,
            current_mood=current_mood,
            normalized_attachments=normalized_attachments,
            stream=stream,
        )
        raise self._authority_disabled_error("generate_validated_reply")

    async def generate_validated_reply_async(
        self,
        stripped_input: str,
        turn_text: str,
        current_mood: str,
        normalized_attachments: AttachmentList,
        stream: bool = False,
        chunk_callback: ChunkCallback | None = None,
    ) -> str:
        """Disabled async authority surface preserved only for compatibility telemetry."""
        _ = chunk_callback
        self._record_direct_call_attempt(
            path="generate_validated_reply_async",
            stripped_input=stripped_input,
            turn_text=turn_text,
            current_mood=current_mood,
            normalized_attachments=normalized_attachments,
            stream=stream,
        )
        raise self._authority_disabled_error("generate_validated_reply_async")


__all__ = ["ReplyGenerationManager"]
