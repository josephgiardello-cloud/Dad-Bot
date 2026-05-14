"""UX Behavioral Tuning Layer — Making Dad Feel Alive.

Gap 3 of the remaining architecture: the system is correct but lacks the
layer that makes it feel "alive" instead of just "correct."

The distinction is:
  Correct: gives the right answer
  Alive:   gives the right answer in the right way, at the right moment,
           with the right emotional register

This module provides:

  PersonalityProfile       — Dad's personality as a numeric vector (5 dimensions)
  ConversationMood         — Current emotional state of the conversation
  TurnEmotionalContext     — Per-turn signals fed into the smoother
  ResponseShapingDirective — Concrete per-turn output: tone, humor, hedging, etc.
  PersonalitySmoother      — Stateful engine that prevents jarring personality swings
  build_dad_personality()  — Dad's canonical personality baseline
  build_default_smoother() — Convenience factory

Information flow
----------------
  System state (health, faults)   ┐
  User signals (sentiment, depth) ├→ TurnEmotionalContext
  Topic stress                    ┘
          ↓
  PersonalitySmoother.observe_turn()
          ↓
  ResponseShapingDirective
          ↓
  Model prompt construction (via .to_prompt_hints())

The PersonalitySmoother uses exponential moving average to prevent jarring
personality swings.  With smoothing_factor=0.3, it takes ~3 turns of
sustained stress to fully shift the personality to its stress-adapted state.

Usage
-----
    smoother = build_default_smoother()

    # Each turn:
    directive = smoother.observe_turn(TurnEmotionalContext(
        user_sentiment="frustrated",
        system_health=SystemHealthStatus.DEGRADED,
        interaction_depth=3,
    ))
    # directive.tone          == "empathetic"
    # directive.humor_level   == 0.12  (dialed back for frustrated user)
    # directive.to_prompt_hints() == ["Tone: empathetic", "Keep humor minimal.", ...]
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from dadbot.core.system_state_model import SystemHealthStatus


# ---------------------------------------------------------------------------
# Personality vector
# ---------------------------------------------------------------------------


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


@dataclass
class PersonalityProfile:
    """Dad's personality as a vector of five continuous dimensions.

    All attributes must be in [0, 1].

    Attributes
    ----------
    warmth:
        How caring/supportive vs detached/transactional.
    humor:
        How much humor/levity to inject.
    directness:
        How direct vs hedged/roundabout.
    formality:
        0 = fully casual; 1 = fully formal.
    patience:
        How much to elaborate vs stay brief and efficient.
    """

    warmth: float = 0.8
    humor: float = 0.6
    directness: float = 0.7
    formality: float = 0.2
    patience: float = 0.7

    def __post_init__(self) -> None:
        for attr in ("warmth", "humor", "directness", "formality", "patience"):
            v = getattr(self, attr)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"PersonalityProfile.{attr}={v!r} must be in [0, 1]")

    def blend(self, other: "PersonalityProfile", weight: float) -> "PersonalityProfile":
        """Return a weighted blend of self toward other.

        weight=0.0 → self unchanged; weight=1.0 → fully other.
        """
        w = _clamp01(weight)
        return PersonalityProfile(
            warmth=_clamp01(self.warmth + (other.warmth - self.warmth) * w),
            humor=_clamp01(self.humor + (other.humor - self.humor) * w),
            directness=_clamp01(self.directness + (other.directness - self.directness) * w),
            formality=_clamp01(self.formality + (other.formality - self.formality) * w),
            patience=_clamp01(self.patience + (other.patience - self.patience) * w),
        )


# ---------------------------------------------------------------------------
# Mood
# ---------------------------------------------------------------------------


class ConversationMood(Enum):
    """Emotional state of the conversation at a given turn."""

    CALM = "calm"
    """Normal, relaxed interaction."""

    ENGAGED = "engaged"
    """Active, curious, productive."""

    TENSE = "tense"
    """Topic or system state is stressful."""

    FRUSTRATED = "frustrated"
    """User is frustrated; requires care and brevity."""

    PLAYFUL = "playful"
    """Light-hearted; humor is appropriate."""


# ---------------------------------------------------------------------------
# Turn context
# ---------------------------------------------------------------------------

_SENTIMENT_STRESS: dict[str, float] = {
    "positive": 0.0,
    "neutral": 0.2,
    "confused": 0.35,
    "frustrated": 0.7,
    "angry": 1.0,
}

_HEALTH_STRESS: dict[SystemHealthStatus, float] = {
    SystemHealthStatus.HEALTHY: 0.0,
    SystemHealthStatus.DEGRADED: 0.4,
    SystemHealthStatus.CRITICAL: 0.8,
    SystemHealthStatus.UNKNOWN: 0.15,
}


@dataclass
class TurnEmotionalContext:
    """Per-turn emotional signals consumed by PersonalitySmoother.

    Attributes
    ----------
    user_sentiment:
        Inferred user sentiment. One of: "positive", "neutral", "confused",
        "frustrated", "angry".
    topic_stress_level:
        [0, 1]. How difficult/stressful the current topic is.
    system_health:
        Current overall system health (from SystemStateSnapshot).
    interaction_depth:
        Number of turns exchanged so far (0 = first turn).
    last_tool_failed:
        Whether the most recent tool invocation failed.
    """

    user_sentiment: str = "neutral"
    topic_stress_level: float = 0.0
    system_health: SystemHealthStatus = SystemHealthStatus.HEALTHY
    interaction_depth: int = 0
    last_tool_failed: bool = False

    @property
    def combined_stress(self) -> float:
        """Composite stress level in [0, 1] from all signals."""
        s_sentiment = _SENTIMENT_STRESS.get(self.user_sentiment, 0.2)
        s_health = _HEALTH_STRESS.get(self.system_health, 0.15)
        s_tool = 0.3 if self.last_tool_failed else 0.0
        raw = (
            s_sentiment * 0.4
            + self.topic_stress_level * 0.3
            + s_health * 0.2
            + s_tool * 0.1
        )
        return _clamp01(raw)


# ---------------------------------------------------------------------------
# Response shaping directive
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResponseShapingDirective:
    """Concrete per-turn output of PersonalitySmoother.observe_turn().

    Every field is a shaping signal that can be injected into prompt
    construction or used to post-process model output.

    Attributes
    ----------
    tone:
        Primary tone label: "warm" | "empathetic" | "playful" | "cautious" |
        "matter-of-fact".
    verbosity:
        Response length guidance: "brief" | "normal" | "detailed".
    humor_level:
        [0, 1]. How much humor/levity to include.
    hedge_level:
        [0, 1]. How much to qualify/hedge statements.
    acknowledge_system_state:
        Whether to mention tool failures or system degradation in the response.
    warmth_factor:
        [0, 1]. Warmth multiplier for the response.
    current_mood:
        The conversation mood that produced this directive.
    """

    tone: str
    verbosity: str
    humor_level: float
    hedge_level: float
    acknowledge_system_state: bool
    warmth_factor: float
    current_mood: ConversationMood

    def to_prompt_hints(self) -> list[str]:
        """Convert to plain-English hints suitable for prompt injection."""
        hints: list[str] = [f"Tone: {self.tone}"]
        if self.verbosity == "brief":
            hints.append("Keep the response concise.")
        elif self.verbosity == "detailed":
            hints.append("Feel free to elaborate.")
        if self.humor_level < 0.2:
            hints.append("Keep humor minimal.")
        elif self.humor_level > 0.6:
            hints.append("It's appropriate to be lighthearted.")
        if self.hedge_level > 0.5:
            hints.append("Acknowledge uncertainty where appropriate.")
        if self.acknowledge_system_state:
            hints.append("Briefly acknowledge any tool issues if relevant.")
        return hints


# ---------------------------------------------------------------------------
# Internal helpers for the smoother
# ---------------------------------------------------------------------------


def _derive_stress_profile(
    baseline: PersonalityProfile,
    stress: float,
) -> PersonalityProfile:
    """Compute what the personality should be at the given stress level."""
    return PersonalityProfile(
        warmth=_clamp01(baseline.warmth + 0.15),                   # More supportive
        humor=_clamp01(baseline.humor - stress * 0.65),            # Dial back humor
        directness=_clamp01(baseline.directness - stress * 0.3),   # Softer
        formality=_clamp01(baseline.formality + stress * 0.15),    # Slightly more formal
        patience=_clamp01(baseline.patience + stress * 0.2),       # More patient
    )


def _classify_mood(
    context: TurnEmotionalContext,
    profile: PersonalityProfile,
) -> ConversationMood:
    stress = context.combined_stress
    sentiment = context.user_sentiment
    if sentiment in {"frustrated", "angry"}:
        return ConversationMood.FRUSTRATED
    if stress > 0.55:
        return ConversationMood.TENSE
    if sentiment == "positive" and profile.humor > 0.5 and context.interaction_depth >= 2:
        return ConversationMood.PLAYFUL
    if context.interaction_depth > 2 and stress < 0.3:
        return ConversationMood.ENGAGED
    return ConversationMood.CALM


def _derive_tone(mood: ConversationMood, profile: PersonalityProfile) -> str:
    if mood == ConversationMood.FRUSTRATED:
        return "empathetic"
    if mood == ConversationMood.TENSE:
        return "cautious" if profile.directness < 0.5 else "matter-of-fact"
    if mood == ConversationMood.PLAYFUL:
        return "playful"
    if profile.warmth >= 0.65:
        return "warm"
    return "matter-of-fact"


def _derive_verbosity(
    mood: ConversationMood,
    profile: PersonalityProfile,
    interaction_depth: int,
) -> str:
    if mood == ConversationMood.FRUSTRATED:
        return "brief"   # Don't overwhelm frustrated users
    if profile.patience >= 0.8 and interaction_depth > 3:
        return "detailed"
    if profile.patience < 0.4:
        return "brief"
    return "normal"


# ---------------------------------------------------------------------------
# Smoother
# ---------------------------------------------------------------------------


class PersonalitySmoother:
    """Stateful personality engine with exponential smoothing.

    Prevents jarring personality swings by blending the target personality
    toward the current state gradually over multiple turns.

    Parameters
    ----------
    baseline:
        Dad's baseline personality.  The smoother returns toward this when
        stress subsides.
    smoothing_factor:
        How fast the current personality tracks the target.
        0.0 = fully reactive (instant change per turn).
        1.0 = never changes.
        0.3 = roughly 3-turn lag (recommended default).
    acknowledge_health_threshold:
        Combined stress level above which the directive should acknowledge
        system health issues in the response.
    """

    def __init__(
        self,
        baseline: PersonalityProfile,
        *,
        smoothing_factor: float = 0.3,
        acknowledge_health_threshold: float = 0.5,
    ) -> None:
        if not (0.0 <= smoothing_factor <= 1.0):
            raise ValueError("smoothing_factor must be in [0, 1]")
        self._baseline = baseline
        self._current = PersonalityProfile(
            warmth=baseline.warmth,
            humor=baseline.humor,
            directness=baseline.directness,
            formality=baseline.formality,
            patience=baseline.patience,
        )
        self._smoothing = smoothing_factor
        self._ack_threshold = acknowledge_health_threshold
        self._current_mood: ConversationMood = ConversationMood.CALM
        self._turn_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_mood(self) -> ConversationMood:
        return self._current_mood

    @property
    def current_personality(self) -> PersonalityProfile:
        return self._current

    @property
    def baseline(self) -> PersonalityProfile:
        return self._baseline

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def observe_turn(self, context: TurnEmotionalContext) -> ResponseShapingDirective:
        """Process one turn and return a ResponseShapingDirective.

        The smoother updates its internal personality state then returns a
        directive that reflects both the new context and the smoothed history.
        """
        self._turn_count += 1
        stress = context.combined_stress

        # Compute target personality for current stress
        target = _derive_stress_profile(self._baseline, stress)

        # Exponential smooth: current = current*(1-s) + target*s
        s = self._smoothing
        self._current = PersonalityProfile(
            warmth=_clamp01(self._current.warmth * (1 - s) + target.warmth * s),
            humor=_clamp01(self._current.humor * (1 - s) + target.humor * s),
            directness=_clamp01(self._current.directness * (1 - s) + target.directness * s),
            formality=_clamp01(self._current.formality * (1 - s) + target.formality * s),
            patience=_clamp01(self._current.patience * (1 - s) + target.patience * s),
        )

        self._current_mood = _classify_mood(context, self._current)

        tone = _derive_tone(self._current_mood, self._current)
        verbosity = _derive_verbosity(
            self._current_mood, self._current, context.interaction_depth
        )
        acknowledge = (
            stress >= self._ack_threshold
            or context.last_tool_failed
            or context.system_health == SystemHealthStatus.CRITICAL
        )
        hedge_level = _clamp01(
            stress * 0.7 + (1.0 - self._current.directness) * 0.3
        )

        return ResponseShapingDirective(
            tone=tone,
            verbosity=verbosity,
            humor_level=round(self._current.humor, 3),
            hedge_level=round(hedge_level, 3),
            acknowledge_system_state=acknowledge,
            warmth_factor=round(self._current.warmth, 3),
            current_mood=self._current_mood,
        )

    def reset(self) -> None:
        """Reset to baseline — call at the start of a new conversation."""
        self._current = PersonalityProfile(
            warmth=self._baseline.warmth,
            humor=self._baseline.humor,
            directness=self._baseline.directness,
            formality=self._baseline.formality,
            patience=self._baseline.patience,
        )
        self._current_mood = ConversationMood.CALM
        self._turn_count = 0


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def build_dad_personality() -> PersonalityProfile:
    """Dad's canonical personality baseline.

    Dad is:
    - Warm (0.85):       cares deeply about the user
    - Humorous (0.65):   naturally funny, reads the room
    - Direct (0.70):     doesn't beat around the bush
    - Casual (0.15):     low formality — family-style interaction
    - Patient (0.75):    will elaborate when it matters
    """
    return PersonalityProfile(
        warmth=0.85,
        humor=0.65,
        directness=0.70,
        formality=0.15,
        patience=0.75,
    )


def build_default_smoother(*, smoothing_factor: float = 0.3) -> PersonalitySmoother:
    """Build a PersonalitySmoother with Dad's canonical personality."""
    return PersonalitySmoother(
        baseline=build_dad_personality(),
        smoothing_factor=smoothing_factor,
    )


__all__ = [
    "PersonalityProfile",
    "ConversationMood",
    "TurnEmotionalContext",
    "ResponseShapingDirective",
    "PersonalitySmoother",
    "build_dad_personality",
    "build_default_smoother",
]
