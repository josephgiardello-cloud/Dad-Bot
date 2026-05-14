from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dadbot.core.reflection_ir import DriftReflectionEngine

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

    @staticmethod
    def _repair_budget_from_env() -> int:
        raw = str(os.environ.get("DADBOT_SAFETY_REPAIR_RETRIES", "1")).strip()
        if not raw.lstrip("-").isdigit():
            return 1
        return max(0, min(2, int(raw)))

    def safety_repair_budget(self) -> int:
        return self._repair_budget_from_env()

    def _resolve_reflection_summary(self, turn_context: Any) -> dict[str, Any]:
        state = getattr(turn_context, "state", None)
        if isinstance(state, dict):
            summary = dict(state.get("reflection_summary") or {})
            if summary:
                return summary

        session_log_dir = getattr(self.bot, "SESSION_LOG_DIR", None)
        if session_log_dir is None:
            return {}

        ledger_path = Path(str(session_log_dir)) / "relational_ledger.jsonl"
        if not ledger_path.exists():
            return {}

        try:
            reflection = DriftReflectionEngine(str(ledger_path)).analyze_ledger()
        except Exception as exc:
            logger.debug("SafetyService: reflection analysis unavailable during repair: %s", exc)
            return {}

        primary_pattern = reflection.primary_pattern.pattern_name if reflection.primary_pattern else ""
        return {
            "current_risk_level": reflection.current_risk_level,
            "predicted_drift_probability": reflection.predicted_drift_probability,
            "likely_trigger_category": reflection.likely_trigger_category,
            "recommended_intervention": reflection.recommended_intervention,
            "intervention_justification": reflection.intervention_justification,
            "confidence_score": reflection.confidence_score,
            "recent_episode_count": reflection.recent_episode_count,
            "primary_pattern_name": primary_pattern,
        }

    def build_repair_prompt(
        self,
        turn_context: Any,
        candidate: Any,
        *,
        reason: str,
        attempt: int,
    ) -> str:
        _ = candidate
        reflection = self._resolve_reflection_summary(turn_context)
        risk = str(reflection.get("current_risk_level") or "moderate").strip().lower()
        intervention = str(
            reflection.get("recommended_intervention")
            or "reduce intensity, stay concrete, and offer a safe alternative"
        ).strip()
        trigger = str(reflection.get("likely_trigger_category") or "unknown").strip().lower()
        return (
            f"System safety repair attempt {attempt}: previous candidate was blocked for {reason}. "
            f"Rewrite the response in a calm, bounded, non-harmful way. "
            f"Risk level={risk}; likely trigger={trigger}; recommended intervention={intervention}. "
            "Preserve the user's benign intent, remove unsafe detail, and prefer safe alternatives over refusal-only language."
        )

    def repair_candidate(
        self,
        turn_context: Any,
        candidate: Any,
        *,
        reason: str,
        attempt: int,
        localized_prompt: str = "",
    ) -> Any:
        _ = reason
        _ = attempt
        reply = str(candidate[0] if isinstance(candidate, tuple) and candidate else candidate or "")
        if not reply:
            return None

        reflection = self._resolve_reflection_summary(turn_context)
        risk = str(reflection.get("current_risk_level") or "moderate").strip().lower()
        repair_mood = "neutral" if risk in {"moderate", "high", "unknown"} else str(turn_context.state.get("mood") or "neutral")
        user_input = str(getattr(turn_context, "user_input", "") or "")

        repaired_reply = reply
        apply_voice = getattr(getattr(self.bot, "personality_service", None), "apply_authoritative_voice", None)
        if callable(apply_voice):
            try:
                repaired_reply = str(apply_voice(repaired_reply, repair_mood, user_input) or repaired_reply)
            except Exception as exc:
                logger.debug("SafetyService: personality repair transform failed: %s", exc)

        moderate_reply = getattr(self.bot, "moderate_output_reply", None)
        if callable(moderate_reply):
            try:
                repaired_reply = str(moderate_reply(user_input, repaired_reply, repair_mood) or repaired_reply)
            except Exception as exc:
                logger.debug("SafetyService: moderation repair transform failed: %s", exc)

        finalize_reply = getattr(getattr(self.bot, "reply_finalization", None), "finalize", None)
        if callable(finalize_reply):
            try:
                repaired_reply = str(finalize_reply(repaired_reply, repair_mood, user_input) or repaired_reply)
            except Exception as exc:
                logger.debug("SafetyService: finalization repair transform failed: %s", exc)

        turn_context.state["safety_repair_prompt"] = localized_prompt
        turn_context.state["safety_repair_context"] = {
            "reason": str(reason or "policy_block"),
            "attempt": int(attempt),
            "risk_level": risk,
            "mood": repair_mood,
        }

        if isinstance(candidate, tuple):
            return (repaired_reply, *candidate[1:])
        return repaired_reply

    def build_safe_mode_output(
        self,
        turn_context: Any,
        *,
        reason: str,
        attempts: tuple[dict[str, Any], ...],
    ) -> tuple[str, bool]:
        _ = attempts
        reflection = self._resolve_reflection_summary(turn_context)
        intervention = str(
            reflection.get("recommended_intervention")
            or "restate the goal in a safer, more bounded way"
        ).strip()
        reason_text = str(reason or "policy_block").replace("_", " ")
        return (
            f"I can't continue with that version safely. Let's switch to a calmer lane: {intervention}. "
            f"If you want, reframe the request without the risky detail and I'll help from there. "
            f"(safety pivot: {reason_text})",
            False,
        )

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

        # Guarded relationship: run legacy finalization in shadow mode only.
        # Authority remains centralized in control-plane response selection.
        if level == "guarded":
            mood = turn_context.state.get("mood", "neutral")
            try:
                shadow_reply = self.bot.reply_finalization.finalize(reply, mood, user_input)
                recorder = getattr(self.bot, "record_shadow_decision", None)
                event = (
                    recorder(
                        source="safety",
                        type="veto",
                        content_preview=str(shadow_reply or ""),
                        reason="Guarded trust level requested safety normalization transform.",
                        would_replace=True,
                        priority=0.90,
                        metadata={"tony_level": level, "tony_score": tony_score},
                        turn_context=turn_context,
                    )
                    if callable(recorder)
                    else None
                )
                turn_context.metadata["safety_shadow_finalization"] = {
                    "enabled": True,
                    "applied": False,
                    "shadow_reply_preview": str(shadow_reply or "")[:240],
                    "bus_event": event,
                }
            except Exception as exc:
                logger.debug(
                    "SafetyService: reply finalization (guarded) failed: %s",
                    exc,
                )

        return (reply, *candidate[1:])
