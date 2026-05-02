from __future__ import annotations

from dadbot.managers.advice_audit import ShadowAuditManager
from dadbot.core.turn_coherence import assert_personality_applied_exactly_once, mark_turn_coherence


class ReplyFinalizationManager:
    """Owns the post-generation reply pipeline before the final message reaches Tony."""

    def __init__(self, bot):
        self.bot = bot

    def append_signoff(self, reply: str) -> str:
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
        mark_turn_coherence(self.bot, "finalizer_called")
        voiced_reply = self.bot.personality_service.apply_authoritative_voice(
            str(reply or ""),
            str(current_mood or "neutral"),
            user_input,
        )
        mark_turn_coherence(self.bot, "personality_applied")
        assert_personality_applied_exactly_once(self.bot)
        final_reply = self.bot.moderate_output_reply(
            user_input,
            voiced_reply,
            current_mood,
        )
        try:
            ShadowAuditManager(self.bot).audit_and_record(
                user_input=str(user_input or ""),
                reply=str(final_reply or ""),
                current_mood=str(current_mood or "neutral"),
            )
        except Exception:
            pass
        return self.append_signoff(final_reply)

    def prepare_final_reply(self, reply, current_mood, user_input=None):
        return self.finalize(reply, current_mood, user_input)

    async def finalize_async(self, reply, current_mood, user_input=None):
        mark_turn_coherence(self.bot, "finalizer_called")
        voiced_reply = self.bot.personality_service.apply_authoritative_voice(
            str(reply or ""),
            str(current_mood or "neutral"),
            user_input,
        )
        mark_turn_coherence(self.bot, "personality_applied")
        assert_personality_applied_exactly_once(self.bot)
        final_reply = await self.bot.moderate_output_reply_async(
            user_input,
            voiced_reply,
            current_mood,
        )
        try:
            ShadowAuditManager(self.bot).audit_and_record(
                user_input=str(user_input or ""),
                reply=str(final_reply or ""),
                current_mood=str(current_mood or "neutral"),
            )
        except Exception:
            pass
        return self.append_signoff(final_reply)

    async def prepare_final_reply_async(self, reply, current_mood, user_input=None):
        return await self.finalize_async(reply, current_mood, user_input)


__all__ = ["ReplyFinalizationManager"]
