"""Goal re-synthesis logic for sustained high-friction sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GoalAdjustmentProposal:
    """Concrete proposal for adapting goals under sustained friction."""

    action: str
    rationale: str
    revised_goal: str
    suggested_constraints: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class GoalResynthesisResult:
    """Outcome of goal re-synthesis evaluation."""

    should_re_synthesize: bool = False
    urgency: str = "none"
    message: str = ""
    proposal: GoalAdjustmentProposal | None = None


class GoalRecalibrationEngine:
    """Builds specific goal-adjustment proposals from friction evidence."""

    def synthesize(
        self,
        *,
        goals: list[dict[str, Any]],
        friction_analysis: dict[str, Any],
        reflection_summary: dict[str, Any] | None = None,
    ) -> GoalResynthesisResult:
        composite_score = float(friction_analysis.get("composite_score") or 0.0)
        should_trigger = bool(friction_analysis.get("should_trigger_re_synthesis", False))
        primary_factor = str(friction_analysis.get("primary_friction_factor") or "")
        recommendation = str(friction_analysis.get("recommended_intervention") or "")
        confidence = float(friction_analysis.get("confidence") or 0.0)

        if not should_trigger:
            return GoalResynthesisResult(
                should_re_synthesize=False,
                urgency="monitor",
                message="Friction is elevated but not sustained enough for goal re-synthesis yet.",
            )

        base_goal = self._select_goal(goals)
        urgency = self._classify_urgency(composite_score)
        revised_goal, constraints, next_steps = self._adapt_goal(base_goal, primary_factor)

        trigger_category = ""
        if isinstance(reflection_summary, dict):
            trigger_category = str(reflection_summary.get("likely_trigger_category") or "")

        rationale_parts = [
            f"Composite friction score {composite_score:.2f} indicates sustained execution drag.",
            recommendation or "Intervention required to prevent repeated divergence.",
        ]
        if trigger_category:
            rationale_parts.append(f"Recent drift trigger category: {trigger_category}.")
        rationale = " ".join(part for part in rationale_parts if part)

        proposal = GoalAdjustmentProposal(
            action=primary_factor or "scope_reduction",
            rationale=rationale,
            revised_goal=revised_goal,
            suggested_constraints=constraints,
            next_steps=next_steps,
            confidence=max(0.0, min(1.0, confidence)),
        )

        return GoalResynthesisResult(
            should_re_synthesize=True,
            urgency=urgency,
            message="Sustained friction detected. Goal re-synthesis is recommended before continuing.",
            proposal=proposal,
        )

    def _select_goal(self, goals: list[dict[str, Any]]) -> str:
        for item in goals:
            if not isinstance(item, dict):
                continue
            candidate = str(item.get("description") or item.get("goal") or "").strip()
            if candidate:
                return candidate
        return "Complete the current objective with a narrower, validated scope"

    def _classify_urgency(self, composite_score: float) -> str:
        if composite_score >= 0.85:
            return "immediate"
        if composite_score >= 0.70:
            return "high"
        return "moderate"

    def _adapt_goal(
        self,
        base_goal: str,
        primary_factor: str,
    ) -> tuple[str, list[str], list[str]]:
        normalized = str(primary_factor or "").strip().lower()

        if normalized == "context_exhaustion":
            return (
                f"Finish the smallest shippable slice of: {base_goal}",
                [
                    "Cap current session to 20 focused minutes",
                    "Defer non-blocking tasks to a follow-up session",
                ],
                [
                    "Define one acceptance check for this slice",
                    "Execute only that check before expanding scope",
                ],
            )

        if normalized == "topic_drift":
            return (
                f"Re-state and complete the core objective only: {base_goal}",
                [
                    "Reject tasks that do not map directly to the objective",
                    "Use a one-sentence relevance check before each action",
                ],
                [
                    "Write the single target outcome",
                    "List up to 3 actions that directly satisfy that outcome",
                ],
            )

        if normalized == "recovery_failure":
            return (
                f"Reduce complexity and recover momentum on: {base_goal}",
                [
                    "Use one deterministic validation path",
                    "Avoid branching into alternative approaches this cycle",
                ],
                [
                    "Select one known-good approach",
                    "Ship a minimal proof before optimization",
                ],
            )

        if normalized == "pattern_recurrence":
            return (
                f"Re-structure execution environment for: {base_goal}",
                [
                    "Remove repeated trigger contexts",
                    "Separate planning and execution into distinct blocks",
                ],
                [
                    "Name the recurring trigger",
                    "Create one direct mitigation before resuming",
                ],
            )

        return (
            f"Split and complete a reduced milestone of: {base_goal}",
            [
                "Limit scope to one measurable deliverable",
                "Stop after the deliverable is validated",
            ],
            [
                "Define a pass/fail checkpoint",
                "Execute checkpoint and reassess friction",
            ],
        )

    @staticmethod
    def to_context_payload(result: GoalResynthesisResult) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "should_re_synthesize": bool(result.should_re_synthesize),
            "urgency": str(result.urgency or "none"),
            "message": str(result.message or ""),
        }
        if result.proposal is None:
            return payload

        payload["proposal"] = {
            "action": result.proposal.action,
            "rationale": result.proposal.rationale,
            "revised_goal": result.proposal.revised_goal,
            "suggested_constraints": list(result.proposal.suggested_constraints),
            "next_steps": list(result.proposal.next_steps),
            "confidence": float(result.proposal.confidence),
        }
        return payload
