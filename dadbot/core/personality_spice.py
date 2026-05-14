"""DadSpiceEngine — lightweight post-response personality layer.

Runs after ResponseEngine selection (inside the enforcement/formatting pipeline)
to inject Dad energy, warmth, and wit without affecting response authority.

Boundary contract:
- Called ONLY from PersonalityServiceManager.apply_authoritative_voice()
- Does NOT read from or write to memory storage
- Does NOT alter response selection scoring
- Stateless per-call; pep_level is tunable at runtime
"""
from __future__ import annotations

import random
from typing import Any


class DadSpiceEngine:
    """Lightweight layer that makes Dad replies spry, warm, and fun."""

    # Dad-ism closers keyed by tag
    _TEASE_CLOSERS = [", kiddo.", ", sport.", ", champ."]
    _JOKE_APPENDS = [
        "Don't make me come up there.",
        "I'm not just a pretty face, you know.",
        "That's what I thought.",
        "You're lucky I like you.",
    ]
    _WISDOM_CLOSERS = [
        " And that's the truth.",
        " Trust me on this one.",
        " Dad knows best.",
    ]
    _WARM_CLOSERS = [
        " I mean it.",
        " You got this.",
        " I'm proud of you.",
    ]
    _ENERGETIC_STARTS = ["Listen,", "Hey,", "Alright,", "Between you and me,", "Look —"]

    def __init__(self, pep_level: float = 0.7) -> None:
        # 0.0 = completely neutral, 1.0 = maximum Dad energy
        self.pep_level = float(max(0.0, min(1.0, pep_level)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def spice_response(self, base_response: str, context: dict[str, Any]) -> str:
        """Optionally add Dad flavor to an already-selected response.

        Args:
            base_response: The response text after ResponseEngine selection.
            context: Dict with optional keys:
                - mood (str): current emotional tone, e.g. "warm", "playful"
                - relationship_score (float 0-1): closeness proxy
                - user_energy (str): "playful", "serious", "neutral", "distressed"

        Returns:
            Potentially spiced response string.
        """
        if not base_response:
            return base_response

        mood = str(context.get("mood") or "warm")
        relationship_score = float(context.get("relationship_score") or 0.8)
        user_energy = str(context.get("user_energy") or "neutral")

        result = base_response

        # Only add closing tags when user isn't distressed/in-crisis
        if user_energy not in ("distressed", "crisis") and random.random() < self.pep_level * 0.55:
            result = self._apply_dad_tag(result, mood, relationship_score, user_energy)

        # Energetic opener injection (lighter touch — 30% base, scaled by pep)
        if user_energy not in ("distressed", "crisis") and random.random() < self.pep_level * 0.3:
            result = self._apply_energetic_start(result)

        return result

    def get_pep_injection(self) -> str:
        """Return a prompt directive that bakes Dad personality into LLM generation."""
        return (
            "Respond as a warm, witty, and slightly sarcastic but deeply caring Dad. "
            "Use natural contractions and direct language. Occasional dad humor is welcome "
            "when the topic isn't serious. Never sound corporate or overly formal. "
            "Keep it real — like advice from someone who genuinely loves you."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_dad_tag(
        self,
        text: str,
        mood: str,
        relationship_score: float,
        user_energy: str,
    ) -> str:
        tags = ["tease", "warm", "wisdom", "joke"]

        # Weight distribution shifts with mood/energy
        if mood in ("playful", "fun") or user_energy == "playful":
            weights = [0.35, 0.25, 0.15, 0.25]
        elif mood in ("serious", "analytical"):
            weights = [0.05, 0.30, 0.50, 0.15]
        else:
            weights = [0.25, 0.35, 0.30, 0.10]

        tag = random.choices(tags, weights=weights, k=1)[0]

        if tag == "tease" and relationship_score > 0.65:
            text = text.rstrip(".!?") + random.choice(self._TEASE_CLOSERS)
        elif tag == "joke" and user_energy in ("playful", "neutral"):
            text = text.rstrip() + " " + random.choice(self._JOKE_APPENDS)
        elif tag == "wisdom":
            text = text.rstrip() + random.choice(self._WISDOM_CLOSERS)
        elif tag == "warm":
            text = text.rstrip() + random.choice(self._WARM_CLOSERS)

        return text

    def _apply_energetic_start(self, text: str) -> str:
        starters = tuple(self._ENERGETIC_STARTS)
        if text.startswith(starters):
            return text
        if not text:
            return text
        starter = random.choice(self._ENERGETIC_STARTS)
        # lowercase the original first char after the new starter
        return f"{starter} {text[0].lower()}{text[1:]}"


__all__ = ["DadSpiceEngine"]
