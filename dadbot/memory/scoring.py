from __future__ import annotations


class MemoryScorer:
    """Pure scoring math for memory entries.

    No LLM calls, no mutations, no side effects.  All scoring weights for
    importance, recency, reinforcement, and contradiction resolution live here
    so ``MemoryCoordinator`` can delegate cleanly.
    """

    def __init__(self, bot) -> None:
        self.bot = bot

    # --- Entry scoring ---

    def score_memory_entry(self, memory: dict) -> dict:
        mood = self.bot.normalize_mood(memory.get("mood") or "neutral")
        emotional_intensity = self._mood_intensity(mood)
        relationship_impact = self._relationship_impact()
        recency = self._recency_score(memory.get("updated_at") or memory.get("created_at"))
        impact_score = max(0.0, float(memory.get("impact_score", 1.0) or 1.0))
        impact_scaled = max(0.0, min(1.0, impact_score / 3.0))
        importance_score = max(
            0.0,
            min(
                1.0,
                0.35 * emotional_intensity + 0.25 * relationship_impact + 0.2 * recency + 0.2 * impact_scaled,
            ),
        )
        return {
            "emotional_intensity": round(emotional_intensity, 3),
            "relationship_impact": round(relationship_impact, 3),
            "importance_score": round(importance_score, 3),
        }

    def consolidated_importance_score(self, entry: dict) -> float:
        mood = self.bot.normalize_mood(entry.get("mood") or "neutral")
        emotional_intensity = self._mood_intensity(mood)
        relationship_impact = self._relationship_impact()
        recency = self._recency_score(entry.get("updated_at") or entry.get("created_at"))
        frequency = self._frequency_score(entry.get("source_count", 1))
        importance = (
            0.4 * emotional_intensity
            + 0.3 * relationship_impact
            + 0.2 * recency
            + 0.1 * frequency
        )
        return round(max(0.0, min(1.0, importance)), 3)

    # --- Contradiction scoring ---

    def contradiction_weight(self, left: dict, right: dict) -> float:
        left_conf = max(0.05, min(1.0, float(left.get("confidence", 0.5) or 0.5)))
        right_conf = max(0.05, min(1.0, float(right.get("confidence", 0.5) or 0.5)))
        left_decay = self._recency_score(left.get("updated_at"))
        right_decay = self._recency_score(right.get("updated_at"))
        return round(((left_conf + right_conf) / 2.0) * ((left_decay + right_decay) / 2.0), 3)

    def consolidated_resolution_rank(self, entry: dict) -> float:
        confidence = max(0.05, min(1.0, float(entry.get("confidence", 0.5) or 0.5)))
        recency = self._recency_score(entry.get("updated_at"))
        importance = max(0.0, min(1.0, float(entry.get("importance_score", 0.0) or 0.0)))
        return round(0.45 * confidence + 0.35 * recency + 0.2 * importance, 4)

    # --- Private helpers ---

    @staticmethod
    def _mood_intensity(mood: str) -> float:
        intensity_map = {
            "positive": 0.55,
            "neutral": 0.25,
            "stressed": 0.95,
            "sad": 0.9,
            "frustrated": 0.8,
            "tired": 0.65,
        }
        return float(intensity_map.get(str(mood or "neutral").strip().lower(), 0.25))

    def _relationship_impact(self) -> float:
        state = self.bot.relationship_state() if callable(getattr(self.bot, "relationship_state", None)) else {}
        try:
            trust = max(0.0, min(100.0, float(state.get("trust_level", 50) or 50)))
            openness = max(0.0, min(100.0, float(state.get("openness_level", 50) or 50)))
        except (TypeError, ValueError, AttributeError):
            return 0.5
        return round((trust + openness) / 200.0, 3)

    def _recency_score(self, timestamp) -> float:
        elapsed_days = self.bot.days_since_iso_date(timestamp)
        if elapsed_days is None:
            return 0.5
        return round(max(0.05, min(1.0, 1.0 - (min(elapsed_days, 365) / 365.0))), 3)

    @staticmethod
    def _frequency_score(source_count) -> float:
        try:
            count = max(1, int(source_count or 1))
        except (TypeError, ValueError):
            count = 1
        return round(max(0.1, min(1.0, count / 5.0)), 3)


__all__ = ["MemoryScorer"]
