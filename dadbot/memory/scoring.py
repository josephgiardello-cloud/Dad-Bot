from __future__ import annotations

import hashlib
import json
from datetime import date


def _stable_signature(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class MemoryScorer:
    """Pure scoring math for memory entries.

    No LLM calls, no mutations, no side effects.  All scoring weights for
    importance, recency, reinforcement, and contradiction resolution live here
    so ``MemoryCoordinator`` can delegate cleanly.
    """

    def __init__(self, bot) -> None:
        self.bot = bot
        self._score_cache: dict[tuple[str, str, str], dict[str, float]] = {}
        self._consolidated_score_cache: dict[tuple[str, str, str], float] = {}

    # --- Entry scoring ---

    def score_memory_entry(self, memory: dict) -> dict:
        cache_key = self._score_cache_key(memory)
        cached = self._score_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        mood = self.bot.normalize_mood(memory.get("mood") or "neutral")
        emotional_intensity = self._mood_intensity(mood)
        relationship_impact = self._relationship_impact()
        recency = self._recency_score(
            memory.get("updated_at") or memory.get("created_at"),
        )
        impact_score = max(0.0, float(memory.get("impact_score", 1.0) or 1.0))
        impact_scaled = max(0.0, min(1.0, impact_score / 3.0))
        importance_score = max(
            0.0,
            min(
                1.0,
                0.35 * emotional_intensity + 0.25 * relationship_impact + 0.2 * recency + 0.2 * impact_scaled,
            ),
        )
        scored = {
            "emotional_intensity": round(emotional_intensity, 3),
            "relationship_impact": round(relationship_impact, 3),
            "importance_score": round(importance_score, 3),
        }
        self._score_cache[cache_key] = dict(scored)
        return scored

    def consolidated_importance_score(self, entry: dict) -> float:
        cache_key = self._consolidated_score_key(entry)
        cached = self._consolidated_score_cache.get(cache_key)
        if cached is not None:
            return cached

        mood = self.bot.normalize_mood(entry.get("mood") or "neutral")
        emotional_intensity = self._mood_intensity(mood)
        relationship_impact = self._relationship_impact()
        recency = self._recency_score(
            entry.get("updated_at") or entry.get("created_at"),
        )
        frequency = self._frequency_score(entry.get("source_count", 1))
        importance = 0.4 * emotional_intensity + 0.3 * relationship_impact + 0.2 * recency + 0.1 * frequency
        scored = round(max(0.0, min(1.0, importance)), 3)
        self._consolidated_score_cache[cache_key] = scored
        return scored

    # --- Contradiction scoring ---

    def contradiction_weight(self, left: dict, right: dict) -> float:
        left_conf = max(0.05, min(1.0, float(left.get("confidence", 0.5) or 0.5)))
        right_conf = max(0.05, min(1.0, float(right.get("confidence", 0.5) or 0.5)))
        left_decay = self._recency_score(left.get("updated_at"))
        right_decay = self._recency_score(right.get("updated_at"))
        return round(
            ((left_conf + right_conf) / 2.0) * ((left_decay + right_decay) / 2.0),
            3,
        )

    def consolidated_resolution_rank(self, entry: dict) -> float:
        confidence = max(0.05, min(1.0, float(entry.get("confidence", 0.5) or 0.5)))
        recency = self._recency_score(entry.get("updated_at"))
        importance = max(
            0.0,
            min(1.0, float(entry.get("importance_score", 0.0) or 0.0)),
        )
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
            openness = max(
                0.0,
                min(100.0, float(state.get("openness_level", 50) or 50)),
            )
        except (TypeError, ValueError, AttributeError):
            return 0.5
        return round((trust + openness) / 200.0, 3)

    def _relationship_signature(self) -> str:
        state = self.bot.relationship_state() if callable(getattr(self.bot, "relationship_state", None)) else {}
        return _stable_signature(
            {
                "trust_level": state.get("trust_level", 50),
                "openness_level": state.get("openness_level", 50),
                "emotional_momentum": state.get("emotional_momentum", "steady"),
                "recurring_topics": state.get("recurring_topics", {}),
                "last_updated": state.get("last_updated"),
            },
        )

    def _score_cache_key(self, memory: dict) -> tuple[str, str, str]:
        return (
            str(memory.get("id") or ""),
            date.today().isoformat(),
            _stable_signature(
                {
                    "mood": self.bot.normalize_mood(memory.get("mood") or "neutral"),
                    "updated_at": memory.get("updated_at") or memory.get("created_at"),
                    "impact_score": memory.get("impact_score", 1.0),
                    "relationship": self._relationship_signature(),
                },
            ),
        )

    def _consolidated_score_key(self, entry: dict) -> tuple[str, str, str]:
        return (
            str(entry.get("id") or entry.get("summary") or ""),
            date.today().isoformat(),
            _stable_signature(
                {
                    "mood": self.bot.normalize_mood(entry.get("mood") or "neutral"),
                    "updated_at": entry.get("updated_at") or entry.get("created_at"),
                    "source_count": entry.get("source_count", 1),
                    "relationship": self._relationship_signature(),
                },
            ),
        )

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
