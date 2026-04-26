"""CritiqueEngine: heuristic evaluator for the plan→execute→critique→revise→converge loop.

The CritiqueEngine scores a candidate reply against the TurnPlan produced by
PlannerNode.  InferenceNode runs it after each ``run_agent`` call to decide
whether the reply is good enough or needs a revision pass.

Design principles:
- No LLM calls — pure heuristics to keep latency near-zero.
- Soft, configurable threshold: ``PASS_THRESHOLD`` (default 0.65).
- At most ``_MAX_LOOP_ITERATIONS`` rounds (default 2) to prevent spirals.
- Revision hint is injected into context.state so the agent can read it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Candidate must score at or above this to be accepted without revision.
PASS_THRESHOLD: float = 0.65

#: Maximum number of inference → critique cycles per turn.
MAX_LOOP_ITERATIONS: int = 2

# Phrases that indicate a fallback / error response that should always be revised.
_FALLBACK_PHRASES = frozenset({
    "something went sideways",
    "try again in a moment",
    "unable to generate",
    "internal error",
    "i couldn't complete",
    "[sub-task failed",
})

# Empathetic response signals — required for EMOTIONAL_SHARE strategy.
_EMPATHY_TOKENS = frozenset({
    "understand", "hear you", "sounds like", "that must", "it's okay",
    "it makes sense", "i can see", "feel", "sorry", "difficult", "tough",
    "must be", "know how", "here for you", "gotcha", "support",
})

# Directive/answer signals — expected for DIRECT_ANSWER strategy.
_ANSWER_HEDGE_MIN_LEN = 15


# ---------------------------------------------------------------------------
# CritiqueResult value object
# ---------------------------------------------------------------------------

@dataclass
class CritiqueResult:
    """The outcome of one critique cycle."""

    score: float          # 0.0–1.0 quality score
    passed: bool          # True when score ≥ PASS_THRESHOLD and no hard failures
    issues: list[str]     # symbolic issue tags (empty on pass)
    revision_hint: str    # human-readable guidance for the revision pass
    iteration: int        # which loop iteration this critique covers


# ---------------------------------------------------------------------------
# CritiqueEngine
# ---------------------------------------------------------------------------

class CritiqueEngine:
    """Score a candidate reply against a TurnPlan and decide whether to revise.

    Usage::

        engine = CritiqueEngine()
        result = engine.critique(reply, user_input, turn_plan, iteration=0)
        if not result.passed:
            context.state["_critique_revision_context"] = result.revision_hint
            # re-run inference...
    """

    def __init__(
        self,
        pass_threshold: float = PASS_THRESHOLD,
        max_iterations: int = MAX_LOOP_ITERATIONS,
    ) -> None:
        self.pass_threshold = pass_threshold
        self.max_iterations = max_iterations

    def critique(
        self,
        candidate: str,
        user_input: str,
        turn_plan: dict[str, Any],
        iteration: int,
    ) -> CritiqueResult:
        """Evaluate ``candidate`` and return a CritiqueResult."""
        reply = str(candidate or "").strip()
        issues: list[str] = []
        score = 1.0

        # --- Hard failure checks (penalise heavily) ---

        if len(reply) < 5:
            issues.append("reply_empty")
            score -= 0.7

        if any(phrase in reply.lower() for phrase in _FALLBACK_PHRASES):
            issues.append("fallback_detected")
            score -= 0.5

        # --- Strategy-specific checks ---

        strategy = str(turn_plan.get("strategy") or "direct_answer")
        intent_type = str(turn_plan.get("intent_type") or "statement")
        complexity = str(turn_plan.get("complexity") or "simple")

        if strategy == "empathy_first":
            reply_tokens = set(re.split(r"\W+", reply.lower()))
            if not (reply_tokens & _EMPATHY_TOKENS):
                issues.append("missing_empathy")
                score -= 0.25

        if intent_type == "question":
            # Check: reply picks up at least one content word from the question.
            q_tokens = set(
                t for t in re.split(r"\W+", user_input.lower())
                if len(t) > 3 and t not in {"what", "when", "where", "which", "this", "that"}
            )
            r_tokens = set(re.split(r"\W+", reply.lower()))
            if q_tokens and not (q_tokens & r_tokens):
                issues.append("reply_misses_question")
                score -= 0.2

        # --- Brevity penalty for complex turns ---

        if complexity in ("moderate", "complex") and len(reply) < _ANSWER_HEDGE_MIN_LEN:
            issues.append("reply_too_brief")
            score -= 0.15

        # --- Final score ---

        score = max(0.0, round(score, 3))
        hard_failure = any(
            i in ("reply_empty", "fallback_detected") for i in issues
        )
        passed = (score >= self.pass_threshold) and not hard_failure

        # Build a single revision hint string.
        hints: list[str] = []
        if "reply_empty" in issues or "fallback_detected" in issues:
            hints.append("Provide a complete, on-topic reply")
        if "missing_empathy" in issues:
            hints.append("Acknowledge the user's feelings before responding")
        if "reply_misses_question" in issues:
            hints.append(f"Directly address the question: '{user_input[:80]}'")
        if "reply_too_brief" in issues:
            hints.append("Give a more thorough answer")
        revision_hint = "; ".join(hints)

        return CritiqueResult(
            score=score,
            passed=passed,
            issues=issues,
            revision_hint=revision_hint,
            iteration=iteration,
        )

    def needs_revision(self, result: CritiqueResult) -> bool:
        """True when the candidate should be re-generated on the next iteration."""
        return not result.passed and result.iteration < self.max_iterations - 1
