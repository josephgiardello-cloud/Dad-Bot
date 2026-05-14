"""Conversation Surface Control Layer (CSCL).

This is the coordinator that enforces:
  - Tone stability         — no jarring personality swings across turns
  - Emotional continuity   — responses reflect the arc of the conversation
  - Personality coherence  — Dad stays Dad regardless of topic
  - Pacing                 — response energy matches user energy
  - Narrative memory shaping — depth builds over engaged turns

What it wires together (all pre-existing, all previously unwired):
  PersonalitySmoother   (ux_behavioral_tuning) — smoothed personality EMA
  OutputCoherenceTracker (coherence_metrics)   — drift detection across replies
  PacingSignal           (local)               — user message energy matching
  NarrativeArc           (local)               — cross-turn engagement depth

Information flow per turn:
  compute_turn_directive(user_input, mood, ...) → prompt section string
       ↓  (injected into PromptAssemblyManager.contextual_request_sections)
  LLM generates response
       ↓
  record_output(reply_text)
       ↑  (called from ReplyFinalizationManager.finalize, post signoff)
  OutputCoherenceTracker accumulates drift signals
"""

from __future__ import annotations

import logging
from typing import Any

from dadbot.core.coherence_metrics import OutputCoherenceTracker
from dadbot.core.ux_behavioral_tuning import (
    ConversationMood,
    TurnEmotionalContext,
    build_default_smoother,
)

try:
    from dadbot.core.system_state_model import SystemHealthStatus
except ImportError:
    SystemHealthStatus = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

_COHERENCE_DRIFT_THRESHOLD = 0.35  # below this → inject voice reminder

# User sentiment string → stress float (mirrors ux_behavioral_tuning internals)
_SENTIMENT_TO_STRESS: dict[str, float] = {
    "positive": 0.0,
    "neutral": 0.2,
    "confused": 0.35,
    "frustrated": 0.7,
    "angry": 1.0,
}


class _PacingSignal:
    """Track user message length EMA and compute a verbosity modifier.

    If the user writes short bursts → "brief".
    If the user writes longer, exploratory messages → "expand".
    Otherwise → trust the PersonalitySmoother's verbosity setting ("match").
    """

    _EMA_ALPHA = 0.4  # smoothing for message-length EMA

    def __init__(self) -> None:
        self._ema_words: float = 20.0  # neutral starting estimate

    def observe(self, user_input: str) -> None:
        wc = len(str(user_input or "").split())
        self._ema_words = self._ema_words * (1 - self._EMA_ALPHA) + wc * self._EMA_ALPHA

    def modifier(self) -> str:
        """Return "brief", "match", or "expand" based on recent user message length."""
        if self._ema_words < 8:
            return "brief"
        if self._ema_words > 45:
            return "expand"
        return "match"

    def reset(self) -> None:
        self._ema_words = 20.0


class _NarrativeArc:
    """Count consecutive turns at or above a given engagement level.

    A streak of ENGAGED / PLAYFUL turns signals that the conversation has
    built real depth — responses should acknowledge and reinforce that.
    """

    _DEPTH_MOODS = {ConversationMood.ENGAGED, ConversationMood.PLAYFUL}

    def __init__(self) -> None:
        self._depth_streak: int = 0
        self._total_turns: int = 0

    def observe(self, mood: ConversationMood) -> None:
        self._total_turns += 1
        if mood in self._DEPTH_MOODS:
            self._depth_streak += 1
        else:
            self._depth_streak = 0

    @property
    def depth_streak(self) -> int:
        return self._depth_streak

    @property
    def total_turns(self) -> int:
        return self._total_turns

    def reset(self) -> None:
        self._depth_streak = 0
        self._total_turns = 0


class ConversationSurfaceLayer:
    """Session-scoped coordinator for conversation surface control.

    One instance per conversation session.  Reset on session reset.

    Usage (turn pipeline):
        # Pre-turn (in PromptAssemblyManager):
        section = bot.conversation_surface.compute_turn_directive(
            user_input, current_mood,
            interaction_depth=len(bot.history or []) // 2,
        )
        # section is a plain-English prompt block

        # Post-turn (in ReplyFinalizationManager, after append_signoff):
        bot.conversation_surface.record_output(final_reply)
    """

    def __init__(self, bot: Any) -> None:
        self._bot = bot
        self._smoother = build_default_smoother(smoothing_factor=0.3)
        self._coherence = OutputCoherenceTracker(window_size=8)
        self._pacing = _PacingSignal()
        self._arc = _NarrativeArc()
        self._last_directive_text: str = ""

    # ------------------------------------------------------------------
    # Pre-turn
    # ------------------------------------------------------------------

    def compute_turn_directive(
        self,
        user_input: str,
        current_mood: str,
        *,
        interaction_depth: int | None = None,
        system_health: Any = None,
        last_tool_failed: bool | None = None,
    ) -> str:
        """Compute a CSCL surface directive for this turn.

        Returns a string suitable for direct inclusion as a prompt section.
        Also updates internal pacing state from this turn's user input.
        Bot-aware fields (interaction_depth, system_health, last_tool_failed)
        are resolved from self._bot when not explicitly supplied.
        """
        self._pacing.observe(user_input)

        if interaction_depth is None:
            history = getattr(self._bot, "history", None)
            interaction_depth = len(history or []) // 2

        if system_health is None:
            system_health = getattr(self._bot, "system_health_status", None)

        if last_tool_failed is None:
            last_tool_failed = bool(getattr(self._bot, "_last_tool_failed", False))

        sentiment = self._mood_to_sentiment(current_mood)
        health_status = self._resolve_health(system_health)

        ctx = TurnEmotionalContext(
            user_sentiment=sentiment,
            topic_stress_level=self._topic_stress(user_input),
            system_health=health_status,
            interaction_depth=interaction_depth,
            last_tool_failed=last_tool_failed,
        )

        directive = self._smoother.observe_turn(ctx)
        self._arc.observe(directive.current_mood)

        # Merge pacing modifier into verbosity
        pacing_mod = self._pacing.modifier()
        effective_verbosity = directive.verbosity if pacing_mod == "match" else pacing_mod

        # Coherence drift check
        drift_warning = ""
        drift = self._coherence.compute_window_coherence()
        if drift is not None and drift < _COHERENCE_DRIFT_THRESHOLD:
            drift_warning = "\nTone drift detected across recent replies — maintain your consistent warm dad voice."

        hints = directive.to_prompt_hints()
        # Inject pacing if it overrides smoother verbosity
        if pacing_mod != "match" and pacing_mod != directive.verbosity:
            hints = [h for h in hints if not h.startswith("Keep") and "concise" not in h and "elaborate" not in h]
            if pacing_mod == "brief":
                hints.append("User is writing briefly — match their pace, keep response concise.")
            else:
                hints.append("User is writing in depth — feel free to expand your response.")

        # Narrative arc
        arc_note = ""
        if self._arc.depth_streak >= 6:
            arc_note = f"\nDeep session arc ({self._arc.depth_streak} turns) — this is a meaningful exchange, treat it with full presence."
        elif self._arc.depth_streak >= 3:
            arc_note = f"\nNarrative depth: {self._arc.depth_streak} turns of engaged conversation — let your response reflect that connection."

        section = (
            "[CONVERSATION SURFACE DIRECTIVE]\n"
            + "\n".join(hints)
            + arc_note
            + drift_warning
        )

        self._last_directive_text = section
        return section

    # ------------------------------------------------------------------
    # Post-turn
    # ------------------------------------------------------------------

    def record_output(self, reply_text: str) -> None:
        """Feed the final reply into the coherence tracker."""
        if reply_text:
            self._coherence.record_reply(str(reply_text))

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def reset_session(self) -> None:
        """Reset all session state. Call at the start of a new conversation."""
        self._smoother.reset()
        self._coherence = OutputCoherenceTracker(window_size=8)
        self._pacing.reset()
        self._arc.reset()
        self._last_directive_text = ""

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def current_mood(self) -> ConversationMood:
        return self._smoother.current_mood

    @property
    def arc_depth(self) -> int:
        return self._arc.depth_streak

    @property
    def last_directive_text(self) -> str:
        return self._last_directive_text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mood_to_sentiment(self, mood: str) -> str:
        """Map a mood label to a sentiment string for TurnEmotionalContext."""
        mood = str(mood or "neutral").lower().strip()
        mood_map = {
            "happy": "positive",
            "excited": "positive",
            "positive": "positive",
            "playful": "positive",
            "engaged": "positive",
            "neutral": "neutral",
            "calm": "neutral",
            "curious": "neutral",
            "confused": "confused",
            "sad": "frustrated",
            "stressed": "frustrated",
            "frustrated": "frustrated",
            "angry": "angry",
            "anxious": "frustrated",
        }
        return mood_map.get(mood, "neutral")

    def _resolve_health(self, system_health: Any) -> Any:
        """Resolve health status to SystemHealthStatus enum."""
        if SystemHealthStatus is None:
            return None
        if system_health is None:
            return SystemHealthStatus.HEALTHY
        if isinstance(system_health, SystemHealthStatus):
            return system_health
        # Accept string labels
        label = str(system_health).upper()
        try:
            return SystemHealthStatus[label]
        except KeyError:
            return SystemHealthStatus.HEALTHY

    def _topic_stress(self, user_input: str) -> float:
        """Heuristic topic stress from user input text."""
        text = str(user_input or "").lower()
        stress_markers = [
            "help", "broken", "error", "crash", "fail", "wrong", "issue",
            "problem", "can't", "cannot", "stuck", "confused", "urgent",
            "need", "fix", "broken", "hate",
        ]
        hit = sum(1 for m in stress_markers if m in text)
        return min(1.0, hit * 0.1)


__all__ = ["ConversationSurfaceLayer"]
