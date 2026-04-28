from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from dadbot.core.execution_boundary import enforce_execution_role

from dadbot.ux_overlay.conversation_continuity import ConversationContinuityEngine
from dadbot.ux_overlay.control_api import UXControlAPI
from dadbot.ux_overlay.interaction_state import InteractionStateEngine
from dadbot.ux_overlay.memory_curation import MemoryCurator
from dadbot.ux_overlay.models import (
    ConversationState,
    CuratedMemory,
    InteractionState,
    ModalAdapter,
    ResponseProfile,
)
from dadbot.ux_overlay.response_shaper import ResponseShapingEngine, ShapedResponse


EXECUTION_ROLE = "experimental"


@dataclass
class SessionUxState:
    enabled: bool = True
    interaction: InteractionState = field(default_factory=InteractionState)
    response_profile: ResponseProfile = field(default_factory=ResponseProfile)
    continuity: ConversationState = field(default_factory=ConversationState)
    modal: ModalAdapter = field(default_factory=ModalAdapter)
    curated_memories: list[CuratedMemory] = field(default_factory=list)
    memory_store: dict[str, dict[str, Any]] = field(default_factory=dict)


class UxOverlayRuntimeAdapter:
    """Single entrypoint for UX overlays, isolated from deterministic core.

    This adapter is intentionally side-band. It does not inspect or mutate the
    execution graph, receipts, truth-binding state, or kernel internals.
    It is not part of the canonical execution kernel.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionUxState] = {}
        self._shaper = ResponseShapingEngine()
        self._memory_curator = MemoryCurator()

    def ensure_session(self, session_id: str) -> SessionUxState:
        sid = str(session_id or "default").strip() or "default"
        if sid not in self._sessions:
            self._sessions[sid] = SessionUxState()
        return self._sessions[sid]

    def set_session_enabled(self, session_id: str, enabled: bool) -> SessionUxState:
        state = self.ensure_session(session_id)
        state.enabled = bool(enabled)
        return state

    def get_session_enabled(self, session_id: str) -> bool:
        return self.ensure_session(session_id).enabled

    def control_api(self, session_id: str) -> UXControlAPI:
        state = self.ensure_session(session_id)
        return UXControlAPI(
            interaction_state=state.interaction,
            response_profile=state.response_profile,
            memory_store=state.memory_store,
        )

    def process_turn(
        self,
        *,
        session_id: str,
        base_response: str,
        user_text: str,
        user_sentiment: str = "neutral",
        positive_signal: float = 0.0,
        conversation_break: bool = False,
        topics: list[str] | None = None,
        unresolved_intents: list[str] | None = None,
        emotional_label: str = "neutral",
        raw_memory_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Apply UX overlay to one turn and return render + diagnostics."""
        enforce_execution_role(module="dadbot.ux_overlay.runtime_entrypoint", role=EXECUTION_ROLE)
        state = self.ensure_session(session_id)

        if not state.enabled:
            return {
                "session_id": session_id,
                "ux_enabled": False,
                "rendered_response": base_response,
                "original_response": base_response,
                "interaction_state": state.interaction,
                "response_profile": state.response_profile,
                "conversation_state": state.continuity,
                "curated_memories": state.curated_memories,
                "modal": state.modal,
            }

        # Layer 1: interaction state updates.
        interaction_engine = InteractionStateEngine(state.interaction)
        state.interaction = interaction_engine.apply_turn_feedback(
            positive_signal=positive_signal,
            user_sentiment=user_sentiment,
            conversation_break=conversation_break,
        )

        # Layer 2: memory curation.
        if raw_memory_events:
            ingested = self._memory_curator.ingestion_filter(raw_memory_events)
            if ingested:
                state.curated_memories = self._memory_curator.compress(ingested)
                # Mirror curated memories into editable control-surface store.
                state.memory_store = {
                    f"mem_{idx}": {
                        "summary": memory.summary,
                        "emotional_weight": memory.emotional_weight,
                        "last_reinforced": memory.last_reinforced.isoformat(),
                    }
                    for idx, memory in enumerate(state.curated_memories)
                }

        # Layer 4: continuity state updates.
        continuity_engine = ConversationContinuityEngine(state.continuity)
        state.continuity = continuity_engine.ingest_turn(
            topics=topics or self._extract_topics(user_text),
            unresolved_intents=unresolved_intents or [],
            emotional_label=emotional_label,
        )

        # Layer 3: response shaping.
        shaped: ShapedResponse = self._shaper.shape(
            content=base_response,
            profile=state.response_profile,
            interaction=state.interaction,
        )

        return {
            "session_id": session_id,
            "ux_enabled": True,
            "rendered_response": shaped.rendered,
            "original_response": shaped.original,
            "shape_metadata": shaped.metadata,
            "interaction_state": state.interaction,
            "response_profile": state.response_profile,
            "conversation_state": state.continuity,
            "curated_memories": state.curated_memories,
            "modal": state.modal,
        }

    @staticmethod
    def _extract_topics(user_text: str, limit: int = 3) -> list[str]:
        tokens = [t.strip(".,!?;:").lower() for t in str(user_text or "").split()]
        tokens = [t for t in tokens if len(t) >= 5]
        if not tokens:
            return []
        seen: list[str] = []
        for token in tokens:
            if token not in seen:
                seen.append(token)
            if len(seen) >= limit:
                break
        return seen

    def snapshot(self, session_id: str) -> dict[str, Any]:
        state = self.ensure_session(session_id)
        return {
            "enabled": state.enabled,
            "interaction": state.interaction,
            "response_profile": state.response_profile,
            "continuity": state.continuity,
            "modal": state.modal,
            "curated_count": len(state.curated_memories),
            "memory_store_count": len(state.memory_store),
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
