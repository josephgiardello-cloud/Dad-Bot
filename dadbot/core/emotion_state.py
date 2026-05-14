"""Stateful emotion field used by response selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EmotionState:
    """Companion emotion coordinates.

    valence: -1.0 (negative) -> +1.0 (positive)
    arousal: 0.0 (calm) -> 1.0 (high energy)
    attachment: 0.0 (distant) -> 1.0 (bonded)
    confidence: 0.0 (uncertain) -> 1.0 (assertive)
    """

    valence: float
    arousal: float
    attachment: float
    confidence: float

    @staticmethod
    def _clip_unit(value: float) -> float:
        return max(min(float(value), 1.0), 0.0)

    @staticmethod
    def _clip_valence(value: float) -> float:
        return max(min(float(value), 1.0), -1.0)

    @classmethod
    def neutral(cls) -> "EmotionState":
        return cls(valence=0.0, arousal=0.2, attachment=0.3, confidence=0.6)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "EmotionState":
        return cls(
            valence=cls._clip_valence(float(mapping.get("valence", 0.0))),
            arousal=cls._clip_unit(float(mapping.get("arousal", 0.2))),
            attachment=cls._clip_unit(float(mapping.get("attachment", 0.3))),
            confidence=cls._clip_unit(float(mapping.get("confidence", 0.6))),
        )
