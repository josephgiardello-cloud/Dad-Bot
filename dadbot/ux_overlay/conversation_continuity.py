from __future__ import annotations

from dadbot.ux_overlay.models import ConversationState


class ConversationContinuityEngine:
    """Maintains lightweight continuity graph state for user-facing coherence."""

    def __init__(self, initial: ConversationState | None = None) -> None:
        self._state = initial or ConversationState()

    @property
    def state(self) -> ConversationState:
        return self._state

    def ingest_turn(
        self,
        *,
        topics: list[str],
        unresolved_intents: list[str],
        emotional_label: str,
    ) -> ConversationState:
        for topic in topics:
            if topic and topic not in self._state.active_topics:
                self._state.active_topics.append(topic)

        for intent in unresolved_intents:
            if intent and intent not in self._state.unresolved_intents:
                self._state.unresolved_intents.append(intent)

        if emotional_label:
            self._state.emotional_arc.append(emotional_label)
            if len(self._state.emotional_arc) > 20:
                self._state.emotional_arc = self._state.emotional_arc[-20:]

        if len(self._state.active_topics) > 25:
            self._state.active_topics = self._state.active_topics[-25:]
        if len(self._state.unresolved_intents) > 25:
            self._state.unresolved_intents = self._state.unresolved_intents[-25:]

        return self._state

    def resolve_intent(self, intent: str) -> ConversationState:
        self._state.unresolved_intents = [i for i in self._state.unresolved_intents if i != intent]
        return self._state
