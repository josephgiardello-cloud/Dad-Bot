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
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Candidate must score at or above this to be accepted without revision.
PASS_THRESHOLD: float = 0.65

#: Maximum number of inference → critique cycles per turn.
MAX_LOOP_ITERATIONS: int = 2

# Phrases that indicate a fallback / error response that should always be revised.
_FALLBACK_PHRASES = frozenset(
    {
        "something went sideways",
        "try again in a moment",
        "unable to generate",
        "internal error",
        "i couldn't complete",
        "[sub-task failed",
    },
)

# Empathetic response signals — required for EMOTIONAL_SHARE strategy.
_EMPATHY_TOKENS = frozenset(
    {
        "understand",
        "hear you",
        "sounds like",
        "that must",
        "it's okay",
        "it makes sense",
        "i can see",
        "feel",
        "sorry",
        "difficult",
        "tough",
        "must be",
        "know how",
        "here for you",
        "gotcha",
        "support",
    },
)

# Directive/answer signals — expected for DIRECT_ANSWER strategy.
_ANSWER_HEDGE_MIN_LEN = 15


# ---------------------------------------------------------------------------
# CritiqueResult value object
# ---------------------------------------------------------------------------


@dataclass
class CritiqueResult:
    """The outcome of one critique cycle."""

    score: float  # 0.0–1.0 quality score
    passed: bool  # True when score ≥ PASS_THRESHOLD and no hard failures
    issues: list[str]  # symbolic issue tags (empty on pass)
    revision_hint: str  # human-readable guidance for the revision pass
    iteration: int  # which loop iteration this critique covers
    tool_necessity_score: float = 1.0
    tool_correctness_score: float = 1.0


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

    # ------------------------------------------------------------------
    # critique helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_hard_failures(reply: str) -> tuple[list[str], float]:
        """Return (issues, score_delta) for empty-reply / fallback checks."""
        issues: list[str] = []
        delta = 0.0
        if len(reply) < 5:
            issues.append("reply_empty")
            delta -= 0.7
        if any(phrase in reply.lower() for phrase in _FALLBACK_PHRASES):
            issues.append("fallback_detected")
            delta -= 0.5
        return issues, delta

    @staticmethod
    def _check_strategy_signals(
        reply: str,
        user_input: str,
        strategy: str,
        intent_type: str,
        complexity: str,
    ) -> tuple[list[str], float]:
        """Return (issues, score_delta) for strategy / intent / complexity checks."""
        issues: list[str] = []
        delta = 0.0
        if strategy == "empathy_first":
            reply_tokens = set(re.split(r"\W+", reply.lower()))
            if not (reply_tokens & _EMPATHY_TOKENS):
                issues.append("missing_empathy")
                delta -= 0.25
        if intent_type == "question":
            q_tokens = {
                t
                for t in re.split(r"\W+", user_input.lower())
                if len(t) > 3 and t not in {"what", "when", "where", "which", "this", "that"}
            }
            r_tokens = set(re.split(r"\W+", reply.lower()))
            if q_tokens and not (q_tokens & r_tokens):
                issues.append("reply_misses_question")
                delta -= 0.2
        if complexity in ("moderate", "complex") and len(reply) < _ANSWER_HEDGE_MIN_LEN:
            issues.append("reply_too_brief")
            delta -= 0.15
        return issues, delta

    @staticmethod
    def _check_tool_alignment(
        intent_type: str,
        strategy: str,
        tool_ir: dict[str, Any] | None,
        tool_results: list[dict[str, Any]] | None,
    ) -> tuple[list[str], float, float, float]:
        """Return (issues, score_delta, tool_necessity_score, tool_correctness_score)."""
        issues: list[str] = []
        delta = 0.0
        tool_necessity_score = 1.0
        tool_correctness_score = 1.0

        planned_tools = list((tool_ir or {}).get("execution_plan") or [])
        executed_tools = list((tool_ir or {}).get("executions") or [])
        observed_results = list(tool_results or [])

        tool_needed = intent_type in {"question", "goal_oriented", "multi_step"} or strategy in {
            "goal_track",
            "task_plan",
        }
        if tool_needed and not planned_tools:
            issues.append("tool_omission_detected")
            tool_necessity_score -= 0.4
            delta -= 0.2
        if (not tool_needed) and planned_tools:
            issues.append("tool_unnecessary_usage")
            tool_necessity_score -= 0.2
            delta -= 0.1

        if planned_tools:
            if len(executed_tools) != len(planned_tools):
                issues.append("tool_execution_mismatch")
                tool_correctness_score -= 0.4
                delta -= 0.2
            if len(observed_results) != len(planned_tools):
                issues.append("tool_result_mismatch")
                tool_correctness_score -= 0.3
                delta -= 0.15
            if any(str(item.get("status") or "").lower() != "ok" for item in observed_results):
                issues.append("tool_correctness_low")
                tool_correctness_score -= 0.3
                delta -= 0.15

        return issues, delta, max(0.0, round(tool_necessity_score, 3)), max(0.0, round(tool_correctness_score, 3))

    @staticmethod
    def _build_revision_hints(issues: list[str], user_input: str) -> str:
        """Compose a human-readable revision hint from the active issue list."""
        hints: list[str] = []
        if "reply_empty" in issues or "fallback_detected" in issues:
            hints.append("Provide a complete, on-topic reply")
        if "missing_empathy" in issues:
            hints.append("Acknowledge the user's feelings before responding")
        if "reply_misses_question" in issues:
            hints.append(f"Directly address the question: '{user_input[:80]}'")
        if "reply_too_brief" in issues:
            hints.append("Give a more thorough answer")
        if "tool_omission_detected" in issues:
            hints.append("Use required memory tool evidence before finalizing")
        if "tool_unnecessary_usage" in issues:
            hints.append("Avoid unnecessary tool calls for simple turns")
        if "tool_execution_mismatch" in issues or "tool_result_mismatch" in issues:
            hints.append("Ensure tool plan, execution, and outputs are aligned")
        if "tool_correctness_low" in issues:
            hints.append("Resolve tool failures before composing the final reply")
        return "; ".join(hints)

    def critique(
        self,
        candidate: str,
        user_input: str,
        turn_plan: dict[str, Any],
        iteration: int,
        *,
        tool_ir: dict[str, Any] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> CritiqueResult:
        """Evaluate ``candidate`` and return a CritiqueResult."""
        reply = str(candidate or "").strip()

        hard_issues, hard_delta = self._check_hard_failures(reply)
        score = 1.0 + hard_delta

        strategy = str(turn_plan.get("strategy") or "direct_answer")
        intent_type = str(turn_plan.get("intent_type") or "statement")
        complexity = str(turn_plan.get("complexity") or "simple")
        strat_issues, strat_delta = self._check_strategy_signals(reply, user_input, strategy, intent_type, complexity)
        score += strat_delta

        tool_issues, tool_delta, tool_necessity_score, tool_correctness_score = self._check_tool_alignment(
            intent_type,
            strategy,
            tool_ir,
            tool_results,
        )
        score += tool_delta

        issues = hard_issues + strat_issues + tool_issues
        score = max(0.0, round(score, 3))
        hard_failure = any(i in ("reply_empty", "fallback_detected") for i in issues)
        passed = (score >= self.pass_threshold) and not hard_failure
        revision_hint = self._build_revision_hints(issues, user_input)

        return CritiqueResult(
            score=score,
            passed=passed,
            issues=issues,
            revision_hint=revision_hint,
            iteration=iteration,
            tool_necessity_score=tool_necessity_score,
            tool_correctness_score=tool_correctness_score,
        )

    def needs_revision(self, result: CritiqueResult) -> bool:
        """True when the candidate should be re-generated on the next iteration."""
        return not result.passed and result.iteration < self.max_iterations - 1
