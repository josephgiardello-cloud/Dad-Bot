from __future__ import annotations

from dataclasses import replace

from dadbot.ux_overlay.models import InteractionState


class InteractionStateEngine:
    """Tracks social continuity and produces updated interaction state.

    This engine does not touch reasoning, graph execution, or truth binding.
    """

    def __init__(self, initial: InteractionState | None = None) -> None:
        self._state = initial or InteractionState()

    @property
    def state(self) -> InteractionState:
        return self._state

    def apply_turn_feedback(
        self,
        *,
        positive_signal: float,
        user_sentiment: str,
        conversation_break: bool,
    ) -> InteractionState:
        affinity_delta = max(-1.0, min(1.0, float(positive_signal))) * 0.1
        new_affinity = max(0.0, min(1.0, self._state.user_affinity + affinity_delta))

        new_continuity = self._state.continuity_score
        if conversation_break:
            new_continuity = max(0.0, new_continuity - 0.15)
        else:
            new_continuity = min(1.0, new_continuity + 0.05)

        if user_sentiment in {"stressed", "sad", "anxious"}:
            tone = "calm"
            engagement = max(0.3, self._state.engagement_level - 0.1)
        elif user_sentiment in {"excited", "happy"}:
            tone = "playful"
            engagement = min(1.0, self._state.engagement_level + 0.1)
        else:
            tone = "friendly"
            engagement = self._state.engagement_level

        self._state = replace(
            self._state,
            user_affinity=new_affinity,
            emotional_tone=tone,
            engagement_level=engagement,
            continuity_score=new_continuity,
        )
        return self._state
