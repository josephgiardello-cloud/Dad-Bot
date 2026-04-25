from __future__ import annotations

from dadbot.config import MOOD_TONE_GUIDANCE
from dadbot.contracts import DadBotContext, SupportsToneRuntime


class ToneContextBuilder:
    """Owns mood, escalation, and gentle daily-checkin tone shaping."""

    def __init__(self, bot: DadBotContext | SupportsToneRuntime):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot

    def build_mood_context(self, mood: str) -> str:
        mood = self.bot.normalize_mood(mood)
        guidance = MOOD_TONE_GUIDANCE.get(mood, MOOD_TONE_GUIDANCE["neutral"])
        return f"""
Current detected mood of Tony: {mood.upper()}
Response guideline: {guidance}
Match your tone to this mood while staying warm, supportive, and like a loving dad.
Keep replies natural - do not mention the mood explicitly unless it helps.
""".strip()

    @staticmethod
    def emotional_support_streak(session_moods: list[str]) -> int:
        streak = 0

        for mood in reversed(session_moods):
            if mood in {"sad", "stressed"}:
                streak += 1
            else:
                break

        return streak

    def support_escalation_settings(self) -> tuple[set[str], int, list[str]]:
        tracked_moods = {
            self.bot.normalize_mood(mood)
            for mood in self.bot.SUPPORT_ESCALATION.get("tracked_moods", ["sad", "stressed"])
        }
        tracked_moods.discard("neutral")
        threshold = self.bot.SUPPORT_ESCALATION.get("streak_threshold", 2)
        supportive_lines = self.bot.SUPPORT_ESCALATION.get(
            "supportive_lines",
            ["I'm right here with you, buddy.", "We can take this one step at a time together."],
        )
        if not supportive_lines:
            supportive_lines = ["I'm right here with you, buddy."]

        try:
            threshold = max(1, int(threshold))
        except (TypeError, ValueError):
            threshold = 2

        return tracked_moods or {"sad", "stressed"}, threshold, supportive_lines

    def emotional_support_streak_for_tracked_moods(self, session_moods: list[str], tracked_moods: set[str]) -> int:
        streak = 0

        for mood in reversed(session_moods):
            if self.bot.normalize_mood(mood) in tracked_moods:
                streak += 1
            else:
                break

        return streak

    def build_escalation_context(self, current_mood: str, session_moods: list[str]) -> str | None:
        current_mood = self.bot.normalize_mood(current_mood)
        tracked_moods, threshold, supportive_lines = self.support_escalation_settings()
        streak = self.emotional_support_streak_for_tracked_moods(session_moods, tracked_moods)

        if current_mood not in tracked_moods or streak < threshold:
            return None

        support_examples = " or ".join(f"'{line}'" for line in supportive_lines)

        return f"""
Tony has sounded emotionally heavy for {streak} turns in a row.
Be especially gentle and present.
Invite him to keep talking instead of rushing into solutions.
Offer one specific supportive line such as {support_examples}
Keep the reply grounded, warm, and not overly dramatic.
""".strip()

    def build_daily_checkin_context(self, current_mood: str) -> str | None:
        if not getattr(self.bot, "_pending_daily_checkin_context", False):
            return None

        if self.bot.normalize_mood(current_mood) != "neutral":
            return None

        return (
            "This is Tony's first message of a new day. "
            "Since his tone is neutral so far, let the reply carry a gentle daily check-in feel: warm, casual, and lightly curious about how his day is going, without ignoring what he just said."
        )

    @staticmethod
    def daily_checkin_greeting():
        return "Hey buddy, how's your day going so far? You can give me the short version or the real version."

    def blend_daily_checkin_reply(self, reply: str, current_mood: str) -> str:
        if not getattr(self.bot, "_pending_daily_checkin_context", False):
            return reply

        if self.bot.normalize_mood(current_mood) != "neutral":
            return reply

        normalized_reply = str(reply or "").strip()
        if not normalized_reply:
            return normalized_reply

        if "how's your day" in normalized_reply.lower() or "how is your day" in normalized_reply.lower():
            return normalized_reply

        return f"{normalized_reply} How's your day shaping up so far?"


__all__ = ["ToneContextBuilder"]
