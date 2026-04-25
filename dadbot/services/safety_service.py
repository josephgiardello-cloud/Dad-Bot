from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Relationship trust thresholds (mirrors RelationshipManager.level_label logic)
_TONY_STRONG = 75
_TONY_GROWING = 60
_TONY_STEADY = 45  # below this â†’ "guarded"


def _tony_level(score: int) -> str:
    if score >= _TONY_STRONG:
        return "strong"
    if score >= _TONY_GROWING:
        return "growing"
    if score >= _TONY_STEADY:
        return "steady"
    return "guarded"


class SafetyService:
    """Service wrapper for post-inference response validation.

    In addition to the standard ``validate_reply`` policy check, this node
    reads the TONY relationship score stored by ``AgentService`` in
    ``turn_context.state["tony_score"]`` and applies level-appropriate
    guardrails so the score is a *functional constraint*, not just a number
    in a JSON file:

    - **guarded** (< 45): run ``prepare_final_reply`` to normalise tone and
      strip any accidental over-familiarity.
    - **steady / growing / strong**: standard validation only.

    The resolved level is written to ``turn_context.metadata`` for telemetry.
    """

    def __init__(self, bot: Any):
        self.bot = bot

    def enforce_policies(self, turn_context: Any, candidate: Any) -> Any:
        if not isinstance(candidate, tuple) or not candidate:
            return candidate

        reply = str(candidate[0] or "")
        user_input = str(getattr(turn_context, "user_input", "") or "")

        # Standard reply validation
        if hasattr(self.bot, "validate_reply"):
            try:
                reply = self.bot.validate_reply(user_input, reply)
            except Exception as exc:
                logger.warning("SafetyService: validate_reply raised: %s", exc)

        # TONY score: resolve trust level and tag metadata
        tony_score = int(turn_context.state.get("tony_score") or 50)
        level = _tony_level(tony_score)
        turn_context.state["tony_level"] = level
        turn_context.metadata["tony_score"] = tony_score
        turn_context.metadata["tony_level"] = level

        # Guarded relationship: run reply finalization to normalize tone
        if level == "guarded":
            mood = turn_context.state.get("mood", "neutral")
            try:
                reply = self.bot.reply_finalization.finalize(reply, mood, user_input)
            except Exception as exc:
                logger.debug("SafetyService: reply finalization (guarded) failed: %s", exc)

        return (reply, *candidate[1:])
