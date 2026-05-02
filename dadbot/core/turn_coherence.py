from __future__ import annotations

from typing import Any


def reset_turn_coherence(bot: Any) -> None:
    bot._turn_coherence_counts = {
        "personality_applied": 0,
        "memory_included": 0,
        "finalizer_called": 0,
        "tool_decision_origin": 0,
    }


def mark_turn_coherence(bot: Any, key: str) -> None:
    counts = getattr(bot, "_turn_coherence_counts", None)
    if not isinstance(counts, dict):
        reset_turn_coherence(bot)
        counts = getattr(bot, "_turn_coherence_counts", {})
    counts[str(key)] = int(counts.get(str(key), 0) or 0) + 1


def assert_personality_applied_exactly_once(bot: Any) -> None:
    """Hard assertion: personality must be applied exactly once per turn.

    Called immediately after ``mark_turn_coherence(bot, "personality_applied")``
    in ``ReplyFinalizationManager.finalize()`` and ``finalize_async()``.

    Raises ``RuntimeError`` if:
    - Count == 0: personality was never applied (impossible at this call site, but
      catches future refactors that move the mark elsewhere).
    - Count > 1: finalize() was called more than once in the same turn, which
      would double-apply tone and signoff.
    """
    counts = getattr(bot, "_turn_coherence_counts", None)
    if not isinstance(counts, dict):
        return  # coherence tracking not initialised — non-fatal in light mode
    n = int(counts.get("personality_applied", 0) or 0)
    if n != 1:
        raise RuntimeError(
            f"Personality coherence violation: expected exactly 1 application of "
            f"personality_applied per turn, got {n}. "
            "This indicates finalize() was called 0 or >1 times in a single turn.",
        )


def turn_coherence_snapshot(bot: Any) -> dict[str, int]:
    counts = getattr(bot, "_turn_coherence_counts", None)
    if not isinstance(counts, dict):
        return {
            "personality_applied": 0,
            "memory_included": 0,
            "finalizer_called": 0,
            "tool_decision_origin": 0,
        }
    return {
        "personality_applied": int(counts.get("personality_applied", 0) or 0),
        "memory_included": int(counts.get("memory_included", 0) or 0),
        "finalizer_called": int(counts.get("finalizer_called", 0) or 0),
        "tool_decision_origin": int(counts.get("tool_decision_origin", 0) or 0),
    }
