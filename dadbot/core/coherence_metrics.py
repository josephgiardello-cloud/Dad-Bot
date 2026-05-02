"""Measure personality and output coherence across multi-turn sessions.

Purpose: Track whether Dad's tone, voice, and emotional consistency remain
stable under memory load and long sessions. Detects drift early.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ToneVector:
    """Lightweight semantic fingerprint of a reply's tone.

    Captures: voice (paternal/casual/joking), emotional_valence (positive/neutral/concern),
    and response_length tier (brief/moderate/elaborate).
    """

    def __init__(self, reply_text: str):
        self.text = str(reply_text or "").strip()
        self.voice_score = self._compute_voice_score()
        self.valence_score = self._compute_valence_score()
        self.length_tier = self._compute_length_tier()

    def _compute_voice_score(self) -> float:
        """Voice score in [0.0, 1.0]. 0=formal, 1=casual/joking.

        Heuristic: count voice markers (exclamation, puns, emojis, informal pronouns)
        """
        markers = {
            "!": 0.05,
            "LOL": 0.1,
            "haha": 0.08,
            "dad ": 0.15,  # typical Dad voice
            "you know": 0.05,
            "man": 0.03,
            "actually": -0.02,  # formal marker
            "nonetheless": -0.1,
            "therefore": -0.05,
        }
        score = 0.5  # neutral baseline
        text_lower = self.text.lower()

        for marker, delta in markers.items():
            count = text_lower.count(marker.lower())
            score += delta * min(3, count)  # cap at 3 occurrences

        return max(0.0, min(1.0, score))

    def _compute_valence_score(self) -> float:
        """Emotional valence in [-1.0, 1.0]. -1=negative, 0=neutral, 1=positive."""
        positive_words = {
            "great": 0.3,
            "awesome": 0.3,
            "love": 0.25,
            "fun": 0.2,
            "happy": 0.3,
            "good": 0.15,
            "nice": 0.1,
            "proud": 0.25,
        }
        negative_words = {
            "sorry": -0.2,
            "bad": -0.15,
            "hate": -0.3,
            "wrong": -0.1,
            "upset": -0.25,
            "fail": -0.2,
            "problem": -0.1,
            "difficult": -0.1,
        }
        text_lower = self.text.lower()
        score = 0.0

        for word, delta in positive_words.items():
            count = text_lower.count(word)
            score += delta * min(3, count)

        for word, delta in negative_words.items():
            count = text_lower.count(word)
            score += delta * min(3, count)

        return max(-1.0, min(1.0, score))

    def _compute_length_tier(self) -> int:
        """Response length tier: 0=brief (<50 words), 1=moderate (50-200), 2=elaborate (>200)."""
        word_count = len(self.text.split())
        if word_count < 50:
            return 0
        if word_count < 200:
            return 1
        return 2

    def similarity_to(self, other: ToneVector) -> float:
        """Cosine-like similarity in [0.0, 1.0]. 1.0 = identical tone."""
        if not isinstance(other, ToneVector):
            return 0.0

        voice_diff = abs(self.voice_score - other.voice_score)
        valence_diff = abs(self.valence_score - other.valence_score) / 2.0  # normalize -1:1 to 0:1
        length_diff = abs(self.length_tier - other.length_tier) / 2.0  # normalize 0-2 to 0-1

        # Weighted average: voice (40%), valence (40%), length (20%)
        avg_diff = 0.4 * voice_diff + 0.4 * valence_diff + 0.2 * length_diff
        return max(0.0, 1.0 - avg_diff)


class OutputCoherenceTracker:
    """Track multi-turn coherence and detect personality drift."""

    def __init__(self, *, window_size: int = 5):
        self.window_size = max(2, int(window_size))
        self.reply_history: list[tuple[str, ToneVector]] = []

    def record_reply(self, reply_text: str) -> None:
        """Record a reply and update coherence state."""
        tone = ToneVector(reply_text)
        self.reply_history.append((str(reply_text), tone))
        if len(self.reply_history) > self.window_size * 3:
            self.reply_history.pop(0)

    def compute_window_coherence(
        self,
        *,
        window_size: int | None = None,
    ) -> float:
        """Compute tone coherence over recent window. [0.0, 1.0], 1.0=perfect consistency."""
        ws = window_size or self.window_size
        if len(self.reply_history) < 2:
            return 1.0  # insufficient data

        recent = self.reply_history[-ws:] if len(self.reply_history) >= ws else self.reply_history

        if len(recent) < 2:
            return 1.0

        similarities: list[float] = []
        for i in range(len(recent) - 1):
            _, tone1 = recent[i]
            _, tone2 = recent[i + 1]
            similarities.append(tone1.similarity_to(tone2))

        return sum(similarities) / len(similarities) if similarities else 1.0

    def detect_personality_drift(
        self,
        *,
        threshold: float = 0.75,
    ) -> dict[str, Any]:
        """Detect if personality has drifted below threshold.

        Returns:
        {
            'drifted': bool,
            'coherence': float,
            'threshold': float,
            'window_replies': int,
        }
        """
        coherence = self.compute_window_coherence()
        drifted = coherence < threshold

        if drifted:
            logger.warning(
                "Personality drift detected: coherence=%.3f < threshold=%.3f",
                coherence,
                threshold,
            )

        return {
            "drifted": drifted,
            "coherence": round(coherence, 3),
            "threshold": threshold,
            "window_replies": len(self.reply_history[-self.window_size :]),
        }

    def summarize_tone_profile(self) -> dict[str, Any]:
        """Return a summary of the current tone profile."""
        if not self.reply_history:
            return {"status": "empty"}

        recent_window = self.reply_history[-self.window_size :]
        tones = [tone for _, tone in recent_window]

        avg_voice = sum(t.voice_score for t in tones) / len(tones) if tones else 0.5
        avg_valence = sum(t.valence_score for t in tones) / len(tones) if tones else 0.0

        length_tiers = [t.length_tier for t in tones]
        most_common_length = max(
            set(length_tiers),
            key=length_tiers.count,
            default=1,
        )

        return {
            "avg_voice_score": round(avg_voice, 3),  # 0=formal, 1=casual
            "avg_valence_score": round(avg_valence, 3),  # -1=negative, 1=positive
            "dominant_length_tier": most_common_length,  # 0=brief, 1=moderate, 2=elaborate
            "window_size": len(recent_window),
            "voice_profile": (
                "casual/joking" if avg_voice > 0.65 else "balanced" if avg_voice > 0.35 else "formal"
            ),
            "emotional_profile": (
                "positive" if avg_valence > 0.3 else "neutral" if avg_valence > -0.3 else "concerned"
            ),
        }
