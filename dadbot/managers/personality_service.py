from __future__ import annotations

import re


class PersonalityServiceManager:
    """Single authority for response voice, tone, and emotional stance."""

    def __init__(self, bot):
        self.bot = bot

    def _should_calibrate_pushback(self, user_input: str, current_mood: str) -> bool:
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

    def _apply_calibrated_pushback(
        self,
        reply: str,
        user_input: str,
        current_mood: str,
    ) -> str:
        settings = self.bot.relationship_calibration_settings()
        if not self._should_calibrate_pushback(user_input, current_mood):
            return str(reply or "")
        normalized_reply = str(reply or "").strip()
        if not normalized_reply:
            return normalized_reply
        if "truth in love" in normalized_reply.lower() or "i care too much" in normalized_reply.lower():
            return normalized_reply
        opening_line = str(settings["opening_line"] or "").strip()
        return f"{opening_line} {normalized_reply}".strip()

    def apply_authoritative_voice(
        self,
        reply: str,
        current_mood: str,
        user_input: str | None = None,
    ) -> str:
        """Apply personality decisions exactly once before final formatting/safety."""
        voiced_reply = self._apply_calibrated_pushback(
            str(reply or ""),
            str(user_input or ""),
            str(current_mood or "neutral"),
        )
        voiced_reply = self.bot.tone_context.blend_daily_checkin_reply(
            voiced_reply,
            current_mood,
        )
        voiced_reply = self.bot.maybe_add_family_echo(
            voiced_reply,
            user_input,
            current_mood,
        )
        return str(voiced_reply or "")

    def build_personality_context(self, current_mood: str) -> str | None:
        """Return the mood/personality section for the system prompt.

        Single authority — all callers must go through here rather than
        calling ``tone_context.build_mood_context`` directly.
        """
        return self.bot.tone_context.build_mood_context(current_mood)


__all__ = ["PersonalityServiceManager"]
