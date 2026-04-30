from __future__ import annotations

from typing import Any

from dadbot.ux_overlay.models import InteractionState, ResponseProfile


class UXControlAPI:
    """User-facing UX controls that tune only presentation behavior."""

    def __init__(
        self,
        *,
        interaction_state: InteractionState,
        response_profile: ResponseProfile,
        memory_store: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.interaction_state = interaction_state
        self.response_profile = response_profile
        self.memory_store = memory_store or {}

    def set_tone(self, tone: str) -> InteractionState:
        if tone not in {"calm", "friendly", "playful", "serious"}:
            raise ValueError(f"Unsupported tone: {tone}")
        self.interaction_state.emotional_tone = tone
        return self.interaction_state

    def adjust_emotion(self, level: float) -> InteractionState:
        clamped = max(0.0, min(1.0, float(level)))
        self.interaction_state.engagement_level = clamped
        return self.interaction_state

    def set_verbosity(self, verbosity: float) -> ResponseProfile:
        self.response_profile.verbosity = max(0.0, min(1.0, float(verbosity)))
        return self.response_profile

    def set_warmth(self, warmth: float) -> ResponseProfile:
        self.response_profile.warmth = max(0.0, min(1.0, float(warmth)))
        return self.response_profile

    def edit_memory(
        self,
        memory_id: str,
        *,
        summary: str | None = None,
        emotional_weight: float | None = None,
    ) -> dict[str, Any]:
        if memory_id not in self.memory_store:
            raise KeyError(memory_id)
        entry = self.memory_store[memory_id]
        if summary is not None:
            entry["summary"] = str(summary)
        if emotional_weight is not None:
            entry["emotional_weight"] = max(0.0, min(1.0, float(emotional_weight)))
        return entry

    def relationship_reset(self) -> InteractionState:
        self.interaction_state.user_affinity = 0.5
        self.interaction_state.engagement_level = 0.5
        self.interaction_state.continuity_score = 0.5
        self.interaction_state.emotional_tone = "friendly"
        return self.interaction_state
