"""Mood normalisation and detection-key utilities.

Canonical home for pure mood-string logic that previously lived as @staticmethod
methods on DadBot.  Import directly rather than going through the facade::

    from dadbot.utils.mood import normalize_mood, normalize_mood_detection_key
"""

from __future__ import annotations

import re
from datetime import date

from dadbot.constants import MOOD_ALIASES, MOOD_CATEGORIES


def normalize_mood(mood: str) -> str:
    """Return a canonical mood category label for any free-text mood string."""
    if not mood:
        return "neutral"

    lowered = str(mood).strip().lower()
    if lowered in MOOD_CATEGORIES:
        return lowered
    if lowered in MOOD_ALIASES:
        return MOOD_ALIASES[lowered]

    for alias, mapped_mood in sorted(
        MOOD_ALIASES.items(),
        key=lambda item: -len(item[0]),
    ):
        if alias in lowered:
            return mapped_mood

    return "neutral"


def normalize_mood_detection_key(user_input: str, recent_history=None) -> str:
    """Build a stable cache key for mood detection.

    Includes the current date as a salt so that identical phrases on different
    days do not serve a stale cached mood from a prior session.
    """

    def _norm(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9\s!?]+", " ", str(value or "").lower())
        return re.sub(r"\s+", " ", normalized).strip()

    normalized_input = _norm(user_input)
    if not normalized_input:
        return ""

    recent_lines: list[str] = []
    for message in list(recent_history or [])[-2:]:
        if not isinstance(message, dict):
            continue
        line = _norm(message.get("content", ""))
        if line:
            recent_lines.append(line[:80])

    date_salt = date.today().isoformat()
    return " || ".join([date_salt, *recent_lines, normalized_input])[:240]


def build_style_examples() -> str:
    """Return a short block of illustrative Dad-style conversation examples."""
    return """
Style examples:
- Tony: I had a pretty good day. Dad: That's my boy. I love hearing that, Tony. What was the best part of it? Love you, buddy.
- Tony: I'm overwhelmed. Dad: Hey, buddy, take a breath with me. We can take this one step at a time together. What's feeling heaviest right now? Love you, buddy.
- Tony: I messed up. Dad: You're still my boy, and one rough moment doesn't change that. Let's look at what happened and figure out the next right step. Love you, buddy.
""".strip()


def command_help_text() -> str:
    """Return the short command-reference string shown to the user."""
    return (
        "Quick commands: /status, /dad, /proactive, /quiet on, /quiet off, /quiet status, "
        "/voice on, /voice off, /voice status, /evolve, /reject trait, /help. "
        "Natural commands: 'add to calendar ...', 'list calendar events', and 'draft email to ...'."
    )


__all__ = [
    "build_style_examples",
    "command_help_text",
    "normalize_mood",
    "normalize_mood_detection_key",
]
