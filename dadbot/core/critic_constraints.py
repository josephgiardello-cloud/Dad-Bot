"""L3-P3 — Critic as Formal Constraint System.

Replaces the heuristic CritiqueEngine scorer with a declarative constraint
satisfaction model:

- CritiqueViolationType: formal violation taxonomy
- CritiqueConstraint: a single declarative rule (id, type, predicate, weight)
- ConstraintViolation: a recorded violation with severity
- ConstraintCritiqueResult: pass/fail + satisfaction ratio + violations
- ConstraintCritiqueEngine: evaluates constraints, computes satisfaction ratio

Design principle:
    score = sum(satisfied_weights) / sum(total_weights)

A reply passes iff score >= pass_threshold and no HARD violations are present.
A "HARD" violation is one with recoverable=False.

Backward compatibility:
    ConstraintCritiqueEngine.default() returns an engine pre-loaded with
    constraints that mirror the original CritiqueEngine heuristics, so
    existing tests continue to pass.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Violation taxonomy
# ---------------------------------------------------------------------------


class CritiqueViolationType(Enum):
    """Formal violation types.  Each maps to exactly one constraint category."""
    EMPTY_REPLY = "empty_reply"
    FALLBACK_DETECTED = "fallback_detected"
    MISSING_EMPATHY = "missing_empathy"
    MISSING_QUESTION_COVERAGE = "missing_question_coverage"
    BREVITY_VIOLATION = "brevity_violation"
    TOOL_OMISSION = "tool_omission"
    TOOL_REDUNDANCY = "tool_redundancy"
    TOOL_EXECUTION_MISMATCH = "tool_execution_mismatch"
    TOOL_RESULT_MISMATCH = "tool_result_mismatch"
    TOOL_CORRECTNESS_FAILURE = "tool_correctness_failure"


# ---------------------------------------------------------------------------
# Constraint value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CritiqueConstraint:
    """A single declarative critique rule.

    ``predicate(reply, user_input, turn_plan, tool_ir, tool_results)``
    returns True when the constraint is *violated*.
    ``weight`` is the fractional budget lost on violation.
    ``recoverable`` distinguishes soft violations (recoverable=True) from
    hard failures (recoverable=False) that always block a pass.
    """
    id: str
    violation_type: CritiqueViolationType
    weight: float
    recoverable: bool
    predicate: Callable[[str, str, dict[str, Any], dict[str, Any], list[dict[str, Any]]], bool]

    def evaluate(
        self,
        reply: str,
        user_input: str,
        turn_plan: dict[str, Any],
        tool_ir: dict[str, Any],
        tool_results: list[dict[str, Any]],
    ) -> bool:
        """Return True iff this constraint is violated."""
        try:
            return bool(self.predicate(reply, user_input, turn_plan, tool_ir, tool_results))
        except Exception:
            return False


@dataclass(frozen=True)
class ConstraintViolation:
    """A recorded violation from a single constraint evaluation."""
    constraint_id: str
    violation_type: CritiqueViolationType
    weight: float
    recoverable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "violation_type": self.violation_type.value,
            "weight": self.weight,
            "recoverable": self.recoverable,
        }


# ---------------------------------------------------------------------------
# Constraint critique result
# ---------------------------------------------------------------------------


@dataclass
class ConstraintCritiqueResult:
    """Result of a constraint-based critique evaluation.

    ``satisfaction_ratio``:
        (total_weight - violated_weight) / total_weight
        e.g. 0.75 means 75% of constraint weight was satisfied.

    ``passed``:
        True iff satisfaction_ratio >= pass_threshold AND no hard violations.
    """
    violations: list[ConstraintViolation]
    satisfaction_ratio: float   # 0.0–1.0
    passed: bool
    revision_hint: str
    iteration: int
    hard_failure: bool          # True iff any non-recoverable constraint violated

    def issue_tags(self) -> list[str]:
        return [v.violation_type.value for v in self.violations]

    def to_dict(self) -> dict[str, Any]:
        return {
            "violations": [v.to_dict() for v in self.violations],
            "satisfaction_ratio": round(self.satisfaction_ratio, 4),
            "passed": self.passed,
            "revision_hint": self.revision_hint,
            "iteration": self.iteration,
            "hard_failure": self.hard_failure,
            "issue_tags": self.issue_tags(),
        }


# ---------------------------------------------------------------------------
# Constraint library helpers
# ---------------------------------------------------------------------------

_FALLBACK_PHRASES = frozenset({
    "something went sideways", "try again in a moment",
    "unable to generate", "internal error",
    "i couldn't complete", "[sub-task failed",
})

_EMPATHY_TOKENS = frozenset({
    "understand", "hear you", "sounds like", "that must", "it's okay",
    "it makes sense", "i can see", "feel", "sorry", "difficult", "tough",
    "must be", "know how", "here for you", "gotcha", "support",
})


def _is_empty_reply(reply: str, _u: str, _p: dict, _t: dict, _r: list) -> bool:
    return len(reply.strip()) < 5


def _is_fallback(reply: str, _u: str, _p: dict, _t: dict, _r: list) -> bool:
    return any(phrase in reply.lower() for phrase in _FALLBACK_PHRASES)


def _missing_empathy(reply: str, _u: str, plan: dict, _t: dict, _r: list) -> bool:
    if str(plan.get("strategy") or "") != "empathy_first":
        return False
    tokens = set(re.split(r"\W+", reply.lower()))
    return not (tokens & _EMPATHY_TOKENS)


def _missing_question_coverage(reply: str, user: str, plan: dict, _t: dict, _r: list) -> bool:
    if str(plan.get("intent_type") or "") != "question":
        return False
    q_tokens = {t for t in re.split(r"\W+", user.lower()) if len(t) > 3 and t not in {"what","when","where","which","this","that"}}
    r_tokens = set(re.split(r"\W+", reply.lower()))
    return bool(q_tokens and not (q_tokens & r_tokens))


def _brevity_violation(reply: str, _u: str, plan: dict, _t: dict, _r: list) -> bool:
    return str(plan.get("complexity") or "") in ("moderate", "complex") and len(reply.strip()) < 15


def _tool_omission(reply: str, user: str, plan: dict, tool_ir: dict, _r: list) -> bool:
    intent_type = str(plan.get("intent_type") or "")
    strategy = str(plan.get("strategy") or "")
    tool_needed = intent_type in {"question", "goal_oriented", "multi_step"} or strategy in {"goal_track", "task_plan"}
    planned = list(tool_ir.get("execution_plan") or [])
    return bool(tool_needed and not planned)


def _tool_redundancy(reply: str, user: str, plan: dict, tool_ir: dict, _r: list) -> bool:
    intent_type = str(plan.get("intent_type") or "")
    strategy = str(plan.get("strategy") or "")
    tool_needed = intent_type in {"question", "goal_oriented", "multi_step"} or strategy in {"goal_track", "task_plan"}
    planned = list(tool_ir.get("execution_plan") or [])
    return bool((not tool_needed) and planned)


def _tool_execution_mismatch(_r: str, _u: str, _p: dict, tool_ir: dict, results: list) -> bool:
    planned = list(tool_ir.get("execution_plan") or [])
    executed = list(tool_ir.get("executions") or [])
    return bool(planned and len(executed) != len(planned))


def _tool_result_mismatch(_r: str, _u: str, _p: dict, tool_ir: dict, results: list) -> bool:
    planned = list(tool_ir.get("execution_plan") or [])
    return bool(planned and len(results) != len(planned))


def _tool_correctness_failure(_r: str, _u: str, _p: dict, _t: dict, results: list) -> bool:
    return any(str(item.get("status") or "").lower() != "ok" for item in results)


# ---------------------------------------------------------------------------
# Default constraint set
# ---------------------------------------------------------------------------

DEFAULT_CONSTRAINTS: list[CritiqueConstraint] = [
    CritiqueConstraint("empty_reply", CritiqueViolationType.EMPTY_REPLY, weight=0.70, recoverable=False, predicate=_is_empty_reply),
    CritiqueConstraint("fallback_detected", CritiqueViolationType.FALLBACK_DETECTED, weight=0.50, recoverable=False, predicate=_is_fallback),
    CritiqueConstraint("missing_empathy", CritiqueViolationType.MISSING_EMPATHY, weight=0.25, recoverable=True, predicate=_missing_empathy),
    CritiqueConstraint("missing_question_coverage", CritiqueViolationType.MISSING_QUESTION_COVERAGE, weight=0.20, recoverable=True, predicate=_missing_question_coverage),
    CritiqueConstraint("brevity_violation", CritiqueViolationType.BREVITY_VIOLATION, weight=0.15, recoverable=True, predicate=_brevity_violation),
    CritiqueConstraint("tool_omission", CritiqueViolationType.TOOL_OMISSION, weight=0.20, recoverable=True, predicate=_tool_omission),
    CritiqueConstraint("tool_redundancy", CritiqueViolationType.TOOL_REDUNDANCY, weight=0.10, recoverable=True, predicate=_tool_redundancy),
    CritiqueConstraint("tool_execution_mismatch", CritiqueViolationType.TOOL_EXECUTION_MISMATCH, weight=0.20, recoverable=True, predicate=_tool_execution_mismatch),
    CritiqueConstraint("tool_result_mismatch", CritiqueViolationType.TOOL_RESULT_MISMATCH, weight=0.15, recoverable=True, predicate=_tool_result_mismatch),
    CritiqueConstraint("tool_correctness_failure", CritiqueViolationType.TOOL_CORRECTNESS_FAILURE, weight=0.15, recoverable=True, predicate=_tool_correctness_failure),
]


# ---------------------------------------------------------------------------
# ConstraintCritiqueEngine
# ---------------------------------------------------------------------------


class ConstraintCritiqueEngine:
    """Declarative constraint satisfaction critique engine.

    Replaces the heuristic scorer with:
    - Explicit constraint registry
    - Formal violation types
    - Satisfaction ratio: sum(satisfied) / sum(total)
    - Hard-failure gate: non-recoverable violations always block pass
    """

    def __init__(
        self,
        constraints: list[CritiqueConstraint] | None = None,
        pass_threshold: float = 0.65,
        max_iterations: int = 2,
    ) -> None:
        self.constraints = list(constraints or DEFAULT_CONSTRAINTS)
        self.pass_threshold = float(pass_threshold)
        self.max_iterations = int(max_iterations)

    @classmethod
    def default(cls) -> "ConstraintCritiqueEngine":
        return cls(constraints=DEFAULT_CONSTRAINTS, pass_threshold=0.65, max_iterations=2)

    def evaluate(
        self,
        candidate: str,
        user_input: str,
        turn_plan: dict[str, Any],
        iteration: int,
        *,
        tool_ir: dict[str, Any] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> ConstraintCritiqueResult:
        """Evaluate all constraints and return a ConstraintCritiqueResult."""
        reply = str(candidate or "").strip()
        plan = dict(turn_plan or {})
        ir = dict(tool_ir or {})
        results = list(tool_results or [])

        total_weight = sum(c.weight for c in self.constraints)
        violated_weight = 0.0
        violations: list[ConstraintViolation] = []
        hard_failure = False

        for constraint in self.constraints:
            if constraint.evaluate(reply, user_input, plan, ir, results):
                violations.append(ConstraintViolation(
                    constraint_id=constraint.id,
                    violation_type=constraint.violation_type,
                    weight=constraint.weight,
                    recoverable=constraint.recoverable,
                ))
                violated_weight += constraint.weight
                if not constraint.recoverable:
                    hard_failure = True

        satisfaction_ratio = (
            (total_weight - violated_weight) / total_weight
            if total_weight > 0 else 1.0
        )
        satisfaction_ratio = max(0.0, min(1.0, round(satisfaction_ratio, 4)))
        passed = (satisfaction_ratio >= self.pass_threshold) and not hard_failure

        # Build revision hint from violations.
        hints: list[str] = []
        for v in violations:
            if v.violation_type == CritiqueViolationType.EMPTY_REPLY:
                hints.append("Provide a complete, on-topic reply")
            elif v.violation_type == CritiqueViolationType.FALLBACK_DETECTED:
                hints.append("Avoid fallback/error phrasing; provide a real answer")
            elif v.violation_type == CritiqueViolationType.MISSING_EMPATHY:
                hints.append("Acknowledge the user's feelings before responding")
            elif v.violation_type == CritiqueViolationType.MISSING_QUESTION_COVERAGE:
                hints.append(f"Directly address the question: '{user_input[:80]}'")
            elif v.violation_type == CritiqueViolationType.BREVITY_VIOLATION:
                hints.append("Give a more thorough answer")
            elif v.violation_type == CritiqueViolationType.TOOL_OMISSION:
                hints.append("Use required memory tool evidence before finalizing")
            elif v.violation_type == CritiqueViolationType.TOOL_REDUNDANCY:
                hints.append("Avoid unnecessary tool calls for simple turns")
            elif v.violation_type in (CritiqueViolationType.TOOL_EXECUTION_MISMATCH, CritiqueViolationType.TOOL_RESULT_MISMATCH):
                hints.append("Ensure tool plan, execution, and outputs are aligned")
            elif v.violation_type == CritiqueViolationType.TOOL_CORRECTNESS_FAILURE:
                hints.append("Resolve tool failures before composing the final reply")

        return ConstraintCritiqueResult(
            violations=violations,
            satisfaction_ratio=satisfaction_ratio,
            passed=passed,
            revision_hint="; ".join(hints),
            iteration=iteration,
            hard_failure=hard_failure,
        )

    def needs_revision(self, result: ConstraintCritiqueResult) -> bool:
        """True when the candidate should be re-generated on the next iteration."""
        return not result.passed and result.iteration < self.max_iterations - 1

    def constraint_by_id(self, constraint_id: str) -> CritiqueConstraint | None:
        for c in self.constraints:
            if c.id == constraint_id:
                return c
        return None


__all__ = [
    "ConstraintCritiqueEngine",
    "ConstraintCritiqueResult",
    "ConstraintViolation",
    "CritiqueConstraint",
    "CritiqueViolationType",
    "DEFAULT_CONSTRAINTS",
]
