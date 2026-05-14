from __future__ import annotations

from datetime import datetime
from typing import Any

from dadbot.contracts import DadBotContext, SupportsDadBotAccess


class SessionSummaryManager:
    """Owns rolling session-summary prompt construction and refresh logic."""

    def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot

    def build_session_summary_prompt(
        self,
        previous_summary: str | None,
        messages: list[dict[str, Any]],
    ) -> str | None:
        transcript = self.bot.transcript_from_messages(messages)
        if not transcript:
            return None

        prior_summary = previous_summary or "None yet."
        return f"""
You are maintaining a compact rolling summary of a conversation between Tony and his dad.

Previous rolling summary:
{prior_summary}

New conversation turns to fold in:
{transcript}

Write a concise updated summary in 5 short bullet lines or fewer.
Focus on:
- Tony's ongoing concerns or wins
- Important emotional shifts
- Any promises, follow-ups, or unresolved threads
- Details the dad should remember during the rest of this chat

Do not invent anything. Keep it compact and concrete.
""".strip()

    def refresh_session_summary(self, force: bool = False):
        history = self.bot.conversation_history()
        eligible_count = max(0, len(history) - self.bot.RECENT_HISTORY_WINDOW)

        if eligible_count <= 0:
            return self.bot.session_summary
        if not force and len(history) < self.bot.SUMMARY_TRIGGER_MESSAGES:
            return self.bot.session_summary
        if eligible_count <= self.bot.session_summary_covered_messages:
            return self.bot.session_summary

        chunk = history[self.bot.session_summary_covered_messages : eligible_count]
        prompt = self.build_session_summary_prompt(self.bot.session_summary, chunk)
        if prompt is None:
            return self.bot.session_summary

        try:
            response = self.bot.call_ollama_chat(
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1},
                purpose="session summarization",
            )
            summary = self.bot.extract_ollama_message_content(response).strip()
        except (RuntimeError, KeyError, TypeError) as exc:
            self.bot.record_runtime_issue(
                "session summarization",
                "keeping the previous rolling summary",
                exc,
            )
            return self.bot.session_summary

        if summary:
            self.bot.session_summary = summary
            self.bot.session_summary_covered_messages = eligible_count
            self.bot.session_summary_updated_at = datetime.now().isoformat(
                timespec="seconds",
            )

        return self.bot.session_summary


__all__ = ["SessionSummaryManager"]
