from __future__ import annotations

from dataclasses import dataclass

from dadbot.ux_overlay.models import InteractionState, ResponseProfile


@dataclass
class ShapedResponse:
    """Presentation wrapper; preserves original response content for traceability."""

    original: str
    rendered: str
    metadata: dict[str, float | str]


class ResponseShapingEngine:
    """Applies UX tone/verbosity overlays without changing reasoning structure."""

    def shape(
        self,
        *,
        content: str,
        profile: ResponseProfile,
        interaction: InteractionState,
    ) -> ShapedResponse:
        rendered = content.strip()

        if profile.warmth >= 0.7 or interaction.emotional_tone in {
            "calm",
            "friendly",
            "playful",
        }:
            rendered = f"I hear you. {rendered}"

        if profile.curiosity_level >= 0.7:
            rendered = f"{rendered} What feels most important to focus on next?"

        if profile.verbosity < 0.35 and len(rendered) > 220:
            rendered = rendered[:217] + "..."

        if profile.assertiveness >= 0.7:
            rendered = rendered + " Let's commit to one concrete next step."

        return ShapedResponse(
            original=content,
            rendered=rendered,
            metadata={
                "warmth": profile.warmth,
                "verbosity": profile.verbosity,
                "assertiveness": profile.assertiveness,
                "curiosity_level": profile.curiosity_level,
                "tone": interaction.emotional_tone,
                "engagement_level": interaction.engagement_level,
            },
        )
