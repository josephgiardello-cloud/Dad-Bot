"""Transitional DadBot package surface for the ongoing monolith split."""

from .assistant_runtime import AssistantRuntime
from .config import (
    MOOD_ALIASES,
    MOOD_CATEGORIES,
    MOOD_TONE_GUIDANCE,
    PERSONA_PRESETS,
    DadRuntimeConfig,
)
from .prompts import DadPrompts

__all__ = [
    "AssistantRuntime",
    "MOOD_ALIASES",
    "MOOD_CATEGORIES",
    "MOOD_TONE_GUIDANCE",
    "PERSONA_PRESETS",
    "DadPrompts",
    "DadRuntimeConfig",
]
