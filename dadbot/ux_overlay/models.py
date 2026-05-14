from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class InteractionState:
    """Social interaction state used only for response behavior tuning."""

    user_affinity: float = 0.5
    emotional_tone: str = "friendly"
    engagement_level: float = 0.5
    continuity_score: float = 0.5


@dataclass
class CuratedMemory:
    """Compressed memory unit oriented around narrative continuity."""

    summary: str
    emotional_weight: float
    last_reinforced: datetime = field(default_factory=utc_now)


@dataclass
class ResponseProfile:
    """Presentation-only response behavior controls."""

    verbosity: float = 0.5
    warmth: float = 0.5
    assertiveness: float = 0.5
    curiosity_level: float = 0.5


@dataclass
class ConversationState:
    """Lightweight continuity state for evolving conversation context."""

    active_topics: list[str] = field(default_factory=list)
    unresolved_intents: list[str] = field(default_factory=list)
    emotional_arc: list[str] = field(default_factory=list)


@dataclass
class ModalAdapter:
    """Multimodal feature flags for future UX expansion."""

    text_enabled: bool = True
    voice_enabled: bool = False
    avatar_enabled: bool = False
    streaming_enabled: bool = False
