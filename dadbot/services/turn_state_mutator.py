"""turn_state_mutator.py — Isolates all direct context/memory state writes for TurnService.

Owns:
  - Turn pipeline state (bot._last_turn_pipeline) via TurnPipelineTracker
  - Legacy/fallback direct attribute writes (bot._last_recorded_mood, session_moods, etc.)

Does NOT own:
  - mutation_queue.queue() operations — those are already Wave-5 compliant
  - Any LLM I/O — that is owned by LLMCallAdapter
  - Business logic — state writes only
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from dadbot.models import TurnPipelineSnapshot, TurnPipelineStep


class TurnStateMutator:
    """Owns all direct state writes that TurnService performs on bot/context.

    Mutation order is preserved exactly — callers must invoke methods in the
    same sequence as the original code to maintain Wave-5 determinism.
    """

    def __init__(self, bot: Any) -> None:
        self._bot = bot

    # ------------------------------------------------------------------
    # Pipeline timestamp helper (read, not write)
    # ------------------------------------------------------------------

    def pipeline_timestamp(self) -> str:
        temporal = getattr(self._bot, "_current_turn_time_base", None)
        wall_time = str(getattr(temporal, "wall_time", "") or "").strip()
        if wall_time:
            return wall_time
        return datetime.now().isoformat(timespec="seconds")

    # ------------------------------------------------------------------
    # Pipeline state writes
    # ------------------------------------------------------------------

    def store_turn_pipeline(self, payload: dict[str, object]) -> dict[str, object]:
        validated = TurnPipelineSnapshot.model_validate(payload)
        self._bot._last_turn_pipeline = validated.model_dump(mode="python")
        return self._bot._last_turn_pipeline

    def start_turn_pipeline(self, mode: str, user_input: str) -> dict[str, object]:
        return self.store_turn_pipeline(
            {
                "mode": str(mode or "sync").strip() or "sync",
                "user_input": str(user_input or "").strip(),
                "started_at": self.pipeline_timestamp(),
                "steps": [],
            }
        )

    def update_turn_pipeline(self, **fields) -> dict[str, object] | None:
        current = dict(getattr(self._bot, "_last_turn_pipeline", {}) or {})
        if not current:
            return None
        current.update(fields)
        return self.store_turn_pipeline(current)

    def append_turn_pipeline_step(
        self,
        name: str,
        status: str = "completed",
        detail: str = "",
        **metadata,
    ) -> dict[str, object] | None:
        current = dict(getattr(self._bot, "_last_turn_pipeline", {}) or {})
        if not current:
            return None
        steps = list(current.get("steps", []))
        steps.append(
            TurnPipelineStep.model_validate(
                {
                    "name": str(name or "step").strip() or "step",
                    "status": str(status or "completed").strip().lower() or "completed",
                    "detail": str(detail or "").strip(),
                    "timestamp": self.pipeline_timestamp(),
                    "metadata": dict(metadata or {}),
                }
            ).model_dump(mode="python")
        )
        current["steps"] = steps
        return self.store_turn_pipeline(current)

    def complete_turn_pipeline(
        self,
        *,
        final_path: str = "",
        reply_source: str = "",
        should_end: bool = False,
        error: str = "",
    ) -> dict[str, object] | None:
        current = dict(getattr(self._bot, "_last_turn_pipeline", {}) or {})
        if not current:
            return None
        current["completed_at"] = self.pipeline_timestamp()
        if final_path:
            current["final_path"] = str(final_path).strip()
        if reply_source:
            current["reply_source"] = str(reply_source).strip()
        current["should_end"] = bool(should_end)
        current["error"] = str(error or "").strip()
        return self.store_turn_pipeline(current)

    def turn_pipeline_snapshot(self) -> dict[str, object] | None:
        payload = getattr(self._bot, "_last_turn_pipeline", None)
        if not isinstance(payload, dict):
            return None
        return TurnPipelineSnapshot.model_validate(payload).model_dump(mode="python")

    # ------------------------------------------------------------------
    # Legacy/fallback state writes (non-mutation_queue path)
    # Used when turn_context is unavailable or MutationGuard fires
    # ------------------------------------------------------------------

    def write_mood_fallback(self, current_mood: str, should_offer_daily_checkin: bool) -> None:
        """Write mood state via direct attribute path (legacy compat / fallback only)."""
        import logging
        queued_mood = str(current_mood or "neutral")
        self._bot._last_recorded_mood = queued_mood
        self._bot._last_should_offer_daily_checkin = bool(should_offer_daily_checkin)
        self._bot._pending_daily_checkin_context = bool(should_offer_daily_checkin)
        session_lock = getattr(self._bot, "_session_lock", None)
        if session_lock is not None:
            with session_lock:
                session_moods = getattr(self._bot, "session_moods", None)
                if isinstance(session_moods, list):
                    session_moods.append(queued_mood)
        else:
            session_moods = getattr(self._bot, "session_moods", None)
            if isinstance(session_moods, list):
                session_moods.append(queued_mood)
