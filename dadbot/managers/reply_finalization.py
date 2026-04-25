from __future__ import annotations

import re

from dadbot.managers.advice_audit import ShadowAuditManager


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
        settings = self.bot.relationship_calibration_settings()
        lowered = str(user_input or "").lower()
        if not lowered:
            return False
        if not settings["enabled"]:
            return False
        if self.bot.detect_crisis_signal(lowered):
            return False
        if self.bot.normalize_mood(current_mood) in set(settings["protected_moods"]):
            return False
        return any(re.search(pattern, lowered) for pattern in settings["trigger_patterns"])

    def apply_calibrated_pushback(self, reply, user_input, current_mood):
        settings = self.bot.relationship_calibration_settings()
        if not self.should_calibrate_pushback(user_input, current_mood):
            return reply
        normalized_reply = str(reply or "").strip()
        if not normalized_reply:
            return normalized_reply
        if "truth in love" in normalized_reply.lower() or "i care too much" in normalized_reply.lower():
            return normalized_reply
        opening_line = str(settings["opening_line"] or "").strip()
        return f"{opening_line} {normalized_reply}".strip()

    def finalize(self, reply, current_mood, user_input=None):
        blended_reply = self.apply_calibrated_pushback(reply, user_input, current_mood)
        blended_reply = self.bot.tone_context.blend_daily_checkin_reply(blended_reply, current_mood)
        blended_reply = self.bot.maybe_add_family_echo(blended_reply, user_input, current_mood)
        blended_reply = self.bot.moderate_output_reply(user_input, blended_reply, current_mood)
        try:
            ShadowAuditManager(self.bot).audit_and_record(
                user_input=str(user_input or ""),
                reply=str(blended_reply or ""),
                current_mood=str(current_mood or "neutral"),
            )
        except Exception:
            pass
        return self.append_signoff(blended_reply)

    def prepare_final_reply(self, reply, current_mood, user_input=None):
        return self.finalize(reply, current_mood, user_input)

    async def finalize_async(self, reply, current_mood, user_input=None):
        blended_reply = self.apply_calibrated_pushback(reply, user_input, current_mood)
        blended_reply = self.bot.tone_context.blend_daily_checkin_reply(blended_reply, current_mood)
        blended_reply = self.bot.maybe_add_family_echo(blended_reply, user_input, current_mood)
        blended_reply = await self.bot.moderate_output_reply_async(user_input, blended_reply, current_mood)
        try:
            ShadowAuditManager(self.bot).audit_and_record(
                user_input=str(user_input or ""),
                reply=str(blended_reply or ""),
                current_mood=str(current_mood or "neutral"),
            )
        except Exception:
            pass
        return self.append_signoff(blended_reply)

    async def prepare_final_reply_async(self, reply, current_mood, user_input=None):
        return await self.finalize_async(reply, current_mood, user_input)


__all__ = ["ReplyFinalizationManager"]
