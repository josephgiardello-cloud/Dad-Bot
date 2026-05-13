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
    """Owns Ollama reply generation, supervision, validation, and finalization before turn persistence.
    
    PHASE 1 NOTE: reply_generation is an OBSERVER PATH (telemetry-only in hot execution).
    In main control-plane execution, ResponseEngine is the sole selection authority.
    This manager is preserved for backward compatibility and non-hot-path uses (agent_service, testing).
    """

    def __init__(self, bot: DadBotContext | SupportsTurnProcessingRuntime):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot

    def generate_validated_reply(
        self,
        stripped_input: str,
        turn_text: str,
        current_mood: str,
        normalized_attachments: AttachmentList,
        stream: bool = False,
        chunk_callback: ChunkCallback | None = None,
    ) -> str:
        """Generate and finalize a reply (OBSERVER PATH — not authoritative in hot execution).
        
        This method is preserved for backward compatibility and non-hot-path execution contexts.
        In the main control-plane hot path, ResponseEngine is the sole selection authority.
        """
        validation_input = stripped_input or turn_text
        
        # PHASE 1: Track observer-path invocation
        logger.debug(
            "reply_generation.generate_validated_reply invoked (OBSERVER PATH); "
            "main hot path uses ResponseEngine"
        )
        
        if stream:
            raw_reply = self.bot.call_ollama_chat_stream(
                messages=self.bot.build_chat_request_messages(
                    turn_text,
                    current_mood,
                    normalized_attachments,
                ),
                purpose="chat response",
                chunk_callback=chunk_callback,
            )
            reviewed_reply = (
                raw_reply if self.bot.LIGHT_MODE else self.bot.critique_reply(stripped_input, raw_reply, current_mood)
            )
        else:
            response = self.bot.call_ollama_chat(
                messages=self.bot.build_chat_request_messages(
                    turn_text,
                    current_mood,
                    normalized_attachments,
                ),
                purpose="chat response",
            )
            reply_text = self.bot.extract_ollama_message_content(response)
            reviewed_reply = (
                reply_text if self.bot.LIGHT_MODE else self.bot.critique_reply(stripped_input, reply_text, current_mood)
            )

        validated_reply = self.bot.validate_reply(validation_input, reviewed_reply)
        try:
            shadow_final = self.bot.reply_finalization.finalize(
                validated_reply,
                current_mood,
                validation_input,
            )
            recorder = getattr(self.bot, "record_shadow_decision", None)
            event = (
                recorder(
                    source="reply_generation",
                    type="transform",
                    content_preview=str(shadow_final or ""),
                    reason="Observer-path finalization transform computed for audit only.",
                    would_replace=True,
                    priority=0.50,
                    metadata={
                        "path": "sync",
                        "validated_reply_preview": str(validated_reply or "")[:220],
                    },
                )
                if callable(recorder)
                else None
            )
            setattr(
                self.bot,
                "_last_reply_generation_shadow",
                {
                    "path": "sync",
                    "validation_input": str(validation_input or ""),
                    "validated_reply": str(validated_reply or ""),
                    "shadow_finalized_reply": str(shadow_final or ""),
                    "applied": False,
                    "bus_event": event,
                },
            )
        except Exception:
            logger.debug("reply_generation shadow finalization failed", exc_info=True)
        # Phase 1 authority collapse: observer paths may propose, but never decide.
        return str(validated_reply or "")

    async def generate_validated_reply_async(
        self,
        stripped_input: str,
        turn_text: str,
        current_mood: str,
        normalized_attachments: AttachmentList,
        stream: bool = False,
        chunk_callback: ChunkCallback | None = None,
    ) -> str:
        """Generate and finalize a reply (async variant — OBSERVER PATH)."""
        validation_input = stripped_input or turn_text
        
        # PHASE 1: Track observer-path invocation
        logger.debug(
            "reply_generation.generate_validated_reply_async invoked (OBSERVER PATH); "
            "main hot path uses ResponseEngine"
        )
        
        if stream:
            raw_reply = await self.bot.call_ollama_chat_stream_async(
                messages=self.bot.build_chat_request_messages(
                    turn_text,
                    current_mood,
                    normalized_attachments,
                ),
                purpose="chat response",
                chunk_callback=chunk_callback,
            )
        else:
            response = await self.bot.call_ollama_chat_async(
                messages=self.bot.build_chat_request_messages(
                    turn_text,
                    current_mood,
                    normalized_attachments,
                ),
                purpose="chat response",
            )
            raw_reply = self.bot.extract_ollama_message_content(response)
        reviewed_reply = (
            raw_reply
            if self.bot.LIGHT_MODE
            else await self.bot.critique_reply_async(
                stripped_input,
                raw_reply,
                current_mood,
            )
        )
        validated_reply = self.bot.validate_reply(validation_input, reviewed_reply)
        try:
            shadow_final = await self.bot.reply_finalization.finalize_async(
                validated_reply,
                current_mood,
                validation_input,
            )
            recorder = getattr(self.bot, "record_shadow_decision", None)
            event = (
                recorder(
                    source="reply_generation",
                    type="transform",
                    content_preview=str(shadow_final or ""),
                    reason="Observer-path async finalization transform computed for audit only.",
                    would_replace=True,
                    priority=0.50,
                    metadata={
                        "path": "async",
                        "validated_reply_preview": str(validated_reply or "")[:220],
                    },
                )
                if callable(recorder)
                else None
            )
            setattr(
                self.bot,
                "_last_reply_generation_shadow",
                {
                    "path": "async",
                    "validation_input": str(validation_input or ""),
                    "validated_reply": str(validated_reply or ""),
                    "shadow_finalized_reply": str(shadow_final or ""),
                    "applied": False,
                    "bus_event": event,
                },
            )
        except Exception:
            logger.debug("reply_generation async shadow finalization failed", exc_info=True)
        # Phase 1 authority collapse: observer paths may propose, but never decide.
        return str(validated_reply or "")


__all__ = ["ReplyGenerationManager"]
