from __future__ import annotations

from typing import Any


def save_reviewed_transcript(bot: Any, transcript: str) -> dict[str, Any] | None:
    text = str(transcript or "").strip()
    if not text:
        return None
    return bot.add_memory(text, category="oral_history")
