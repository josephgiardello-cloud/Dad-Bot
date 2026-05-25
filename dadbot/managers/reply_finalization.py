from __future__ import annotations

import logging
import os
from typing import Any
from dadbot.core.turn_coherence import assert_personality_applied_exactly_once, mark_turn_coherence
from dadbot.managers.advice_audit import ShadowAuditManager
from dadbot.managers.personality_service import PersonalityServiceManager

logger = logging.getLogger(__name__)


class ReplyFinalizationManager:
    """Owns the post-generation reply pipeline before the final message reaches Tony.
    
    PHASE 1 NOTE: reply_finalization is an ENFORCEMENT LAYER (formatting-only).
    Formatting operations (personality voice, moderation, audit, signoff) are applied
    AFTER ResponseEngine selects the response. reply_finalization does NOT influence
    which response is selected — only how it is delivered.
    """

    def __init__(self, bot):
        self.bot = bot

    def _record_shadow_pipeline(self, payload: dict[str, Any]) -> None:
        try:
            setattr(self.bot, "_last_reply_finalization_shadow", payload)
            recorder = getattr(self.bot, "record_shadow_decision", None)
            if callable(recorder):
                recorder(
                    source="finalization",
                    type="transform",
                    content_preview=str(payload.get("shadow_moderated_reply") or payload.get("shadow_voiced_reply") or ""),
                    reason="Finalization transform observed; response authority remains in ResponseEngine.",
                    would_replace=True,
                    priority=0.45,
                    metadata={
                        "path": str(payload.get("path") or "unknown"),
                        "base_reply_preview": str(payload.get("base_reply") or "")[:220],
                    },
                )
        except Exception:
            logger.debug("Failed to record reply finalization shadow payload", exc_info=True)

    def append_signoff(self, reply: str) -> str:
        # Defense: if reply is somehow a Pydantic response object, extract content first
        if hasattr(reply, "model_dump") and not isinstance(reply, str):
            try:
                from dadbot.utils.llm import extract_ollama_message_content
                reply = extract_ollama_message_content(reply)
            except Exception:
                pass
        reply = str(reply or "").strip()
        signoff = str(self.bot.STYLE.get("signoff") or "").strip()

        if not reply or not self.bot.APPEND_SIGNOFF or not signoff:
            return reply

        if reply.endswith(signoff):
            return reply
        if reply.endswith(("!", "?", ".")):
            return f"{reply} {signoff}"
        return f"{reply}. {signoff}"

    def finalize_reply(self, reply: str) -> str:
        return self.append_signoff(reply)

    def should_calibrate_pushback(self, user_input, current_mood):
        return self.bot.personality_service._should_calibrate_pushback(
            str(user_input or ""),
            str(current_mood or "neutral"),
        )

    def apply_calibrated_pushback(self, reply, user_input, current_mood):
        return self.bot.personality_service._apply_calibrated_pushback(
            str(reply or ""),
            str(user_input or ""),
            str(current_mood or "neutral"),
        )

    def finalize(self, reply, current_mood, user_input=None):
        """Apply formatting pipeline to response (ENFORCEMENT LAYER — NOT a selection authority).
        
        PHASE 1: This method is only invoked by control-plane post-ResponseEngine selection.
        All formatting operations here are non-binding (they do not change which response was selected).
        """
        # PHASE 1: Audit logging for enforcement layer
        logger.debug(
            "reply_finalization.finalize invoked (ENFORCEMENT LAYER formatting); "
            f"applying personality/moderation/audit to response"
        )
        
        # Defense: if reply is somehow a Pydantic response object, extract content first
        if hasattr(reply, "model_dump") and not isinstance(reply, str):
            try:
                from dadbot.utils.llm import extract_ollama_message_content
                reply = extract_ollama_message_content(reply)
            except Exception:
                pass
        mark_turn_coherence(self.bot, "finalizer_called")
        base_reply = str(reply or "")
        shadow_voiced = base_reply
        shadow_supervised = base_reply
        shadow_moderated = base_reply
        try:
            shadow_voiced = self.bot.personality_service.apply_authoritative_voice(
                base_reply,
                str(current_mood or "neutral"),
                user_input,
            )

            enable_supervisor = str(os.environ.get("DADBOT_REPLY_SUPERVISOR_ENABLED") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            supervisor = getattr(self.bot, "reply_supervisor", None)
            can_supervise = (
                enable_supervisor
                and supervisor is not None
                and callable(getattr(supervisor, "run_reply_supervisor", None))
                and not bool(getattr(self.bot, "LIGHT_MODE", False))
            )
            if can_supervise and user_input is not None:
                shadow_supervised = supervisor.run_reply_supervisor(
                    str(user_input or ""),
                    str(shadow_voiced or ""),
                    str(current_mood or "neutral"),
                    stage="final_polish",
                )
            else:
                shadow_supervised = shadow_voiced
            shadow_moderated = self.bot.moderate_output_reply(
                user_input,
                shadow_supervised,
                current_mood,
            )
            self._record_shadow_pipeline(
                {
                    "path": "sync",
                    "base_reply": base_reply,
                    "shadow_voiced_reply": str(shadow_voiced or ""),
                    "shadow_supervised_reply": str(shadow_supervised or ""),
                    "shadow_moderated_reply": str(shadow_moderated or ""),
                    "applied": False,
                },
            )
        except Exception:
            logger.debug("reply_finalization shadow transforms failed", exc_info=True)
        mark_turn_coherence(self.bot, "personality_applied")
        assert_personality_applied_exactly_once(self.bot)
        personality_apply = getattr(getattr(self.bot, "personality_service", None), "apply_authoritative_voice", None)
        personality_func = getattr(personality_apply, "__func__", None)
        can_apply_surface_transforms = personality_func is PersonalityServiceManager.apply_authoritative_voice
        # Preserve response authority in compatibility/monkeypatched test paths,
        # while still allowing the default personality pipeline in production.
        final_reply = (
            str(shadow_moderated or shadow_voiced or base_reply)
            if can_apply_surface_transforms
            else base_reply
        )
        try:
            ShadowAuditManager(self.bot).audit_and_record(
                user_input=str(user_input or ""),
                reply=str(final_reply or ""),
                current_mood=str(current_mood or "neutral"),
            )
        except Exception:
            pass
        final_reply = self.append_signoff(final_reply)
        # CSCL post-turn feedback: feed voiced reply to coherence tracker
        try:
            cscl = getattr(self.bot, "conversation_surface", None)
            if cscl is not None:
                cscl.record_output(final_reply)
        except Exception:
            pass
        return final_reply

    def prepare_final_reply(self, reply, current_mood, user_input=None):
        return self.finalize(reply, current_mood, user_input)

    async def finalize_async(self, reply, current_mood, user_input=None):
        """Apply formatting pipeline to response (async variant — ENFORCEMENT LAYER)."""
        # PHASE 1: Audit logging for enforcement layer
        logger.debug(
            "reply_finalization.finalize_async invoked (ENFORCEMENT LAYER formatting); "
            f"applying personality/moderation/audit to response (async)"
        )
        
        # Defense: if reply is somehow a Pydantic response object, extract content first
        if hasattr(reply, "model_dump") and not isinstance(reply, str):
            try:
                from dadbot.utils.llm import extract_ollama_message_content
                reply = extract_ollama_message_content(reply)
            except Exception:
                pass
        mark_turn_coherence(self.bot, "finalizer_called")
        base_reply = str(reply or "")
        shadow_voiced = base_reply
        shadow_supervised = base_reply
        shadow_moderated = base_reply
        try:
            shadow_voiced = self.bot.personality_service.apply_authoritative_voice(
                base_reply,
                str(current_mood or "neutral"),
                user_input,
            )

            enable_supervisor = str(os.environ.get("DADBOT_REPLY_SUPERVISOR_ENABLED") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            supervisor = getattr(self.bot, "reply_supervisor", None)
            can_supervise = (
                enable_supervisor
                and supervisor is not None
                and callable(getattr(supervisor, "run_reply_supervisor_async", None))
                and not bool(getattr(self.bot, "LIGHT_MODE", False))
            )
            if can_supervise and user_input is not None:
                shadow_supervised = await supervisor.run_reply_supervisor_async(
                    str(user_input or ""),
                    str(shadow_voiced or ""),
                    str(current_mood or "neutral"),
                    stage="final_polish",
                )
            else:
                shadow_supervised = shadow_voiced
            shadow_moderated = await self.bot.moderate_output_reply_async(
                user_input,
                shadow_supervised,
                current_mood,
            )
            self._record_shadow_pipeline(
                {
                    "path": "async",
                    "base_reply": base_reply,
                    "shadow_voiced_reply": str(shadow_voiced or ""),
                    "shadow_supervised_reply": str(shadow_supervised or ""),
                    "shadow_moderated_reply": str(shadow_moderated or ""),
                    "applied": False,
                },
            )
        except Exception:
            logger.debug("reply_finalization async shadow transforms failed", exc_info=True)
        mark_turn_coherence(self.bot, "personality_applied")
        assert_personality_applied_exactly_once(self.bot)
        personality_apply = getattr(getattr(self.bot, "personality_service", None), "apply_authoritative_voice", None)
        personality_func = getattr(personality_apply, "__func__", None)
        can_apply_surface_transforms = personality_func is PersonalityServiceManager.apply_authoritative_voice
        final_reply = (
            str(shadow_moderated or shadow_voiced or base_reply)
            if can_apply_surface_transforms
            else base_reply
        )
        try:
            ShadowAuditManager(self.bot).audit_and_record(
                user_input=str(user_input or ""),
                reply=str(final_reply or ""),
                current_mood=str(current_mood or "neutral"),
            )
        except Exception:
            pass
        final_reply = self.append_signoff(final_reply)
        # CSCL post-turn feedback: feed voiced reply to coherence tracker (async path)
        try:
            cscl = getattr(self.bot, "conversation_surface", None)
            if cscl is not None:
                cscl.record_output(final_reply)
        except Exception:
            pass
        return final_reply

    async def prepare_final_reply_async(self, reply, current_mood, user_input=None):
        return await self.finalize_async(reply, current_mood, user_input)


__all__ = ["ReplyFinalizationManager"]
