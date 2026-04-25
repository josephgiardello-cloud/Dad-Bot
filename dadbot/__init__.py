"""Transitional DadBot package surface for the ongoing monolith split."""

from .config import MOOD_ALIASES, MOOD_CATEGORIES, MOOD_TONE_GUIDANCE, PERSONA_PRESETS, DadRuntimeConfig
from .prompts import DadPrompts

__all__ = [
    "DadPrompts",
    "DadRuntimeConfig",
    "MOOD_ALIASES",
    "MOOD_CATEGORIES",
    "MOOD_TONE_GUIDANCE",
    "PERSONA_PRESETS",
]
