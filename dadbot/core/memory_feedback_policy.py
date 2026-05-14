"""Memory → Execution Behavioral Feedback Loop.

Gap 4 of the causal loop: execution writes to memory, but memory was not
yet feeding back into:

  - tool selection (which tool to call for a given task)
  - failure handling decisions (should the policy engine be more aggressive?)
  - scheduling prioritization (how urgent is this execution?)
  - response shaping (how should the model surface this to the user?)

This module closes the loop:

    memory ↔ execution ↔ policy decisions

Core types
----------
ToolMemoryProfile        — Aggregated statistics built from CausalMemoryEntry history
ToolSelectionScore       — Ranked candidate produced by ToolSelectionAdvisor
ToolSelectionAdvisor     — Scores/ranks tool candidates using their memory profiles
MemoryPolicyAdjustment   — Adjustments to a PolicyDecision based on memory context
MemoryAwarePolicyContext — Wraps FailurePolicyEngine to inject memory-based overrides

Usage
-----
    # Build profiles from historical memory
    profiles = ToolMemoryProfile.build_profiles(causal_entries)

    # Rank candidate tools for a task
    advisor = ToolSelectionAdvisor(profiles)
    ranked = advisor.rank(["tool_a", "tool_b", "tool_c"])
    best_tool = ranked[0].tool_name

    # Use memory to modify policy decisions
    ctx = MemoryAwarePolicyContext(policy_engine, profiles)
    adjusted = ctx.decide_with_memory(result=result, contract=contract, attempt=attempt)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from dadbot.core.tool_memory_causal_contract import CausalMemoryEntry
from dadbot.core.failure_policy_engine import (
    FailurePolicyEngine,
    PolicyAction,
    PolicyDecision,
)


# ---------------------------------------------------------------------------
# Tool memory profile
# ---------------------------------------------------------------------------


@dataclass
class ToolMemoryProfile:
    """Aggregated statistics for a single tool derived from its CausalMemoryEntry history.

    Attributes
    ----------
    tool_name:
        The tool this profile describes.
    total_executions:
        Total number of recorded executions.
    success_count:
        Number of executions with status == "ok".
    failure_count:
        Number of executions with non-ok status.
    avg_latency_ms:
        Average execution latency across all recorded entries.
    most_common_failure_class:
        The failure_class that appeared most often (or None if no failures).
    most_common_policy_action:
        The policy_action that appeared most often.
    escalation_count:
        Number of times policy_action was "escalate".
    abort_count:
        Number of times policy_action was "abort".
    max_attempts_seen:
        Highest attempt number seen across all entries (signals persistent retries).
    """

    tool_name: str
    total_executions: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_latency_ms: float = 0.0
    most_common_failure_class: str | None = None
    most_common_policy_action: str | None = None
    escalation_count: int = 0
    abort_count: int = 0
    max_attempts_seen: int = 0

    @property
    def failure_rate(self) -> float:
        """Fraction of executions that failed.  Returns 0.0 if no history."""
        if self.total_executions == 0:
            return 0.0
        return self.failure_count / self.total_executions

    @property
    def success_rate(self) -> float:
        return 1.0 - self.failure_rate

    @property
    def is_healthy(self) -> bool:
        """True when failure rate < 0.3 and no escalations/aborts dominate."""
        return (
            self.failure_rate < 0.3
            and self.escalation_count < max(1, self.total_executions // 4)
            and self.abort_count < max(1, self.total_executions // 5)
        )

    @property
    def reliability_score(self) -> float:
        """Composite score in [0, 1]: higher = more reliable.

        Combines success rate with a penalty for escalations and aborts.
        """
        if self.total_executions == 0:
            return 0.5  # No history → neutral

        base = self.success_rate
        escalation_penalty = min(0.3, self.escalation_count / max(1, self.total_executions) * 0.5)
        abort_penalty = min(0.4, self.abort_count / max(1, self.total_executions) * 0.6)
        retry_penalty = min(0.2, max(0, self.max_attempts_seen - 1) * 0.05)

        score = base - escalation_penalty - abort_penalty - retry_penalty
        return max(0.0, min(1.0, score))

    @classmethod
    def build_profiles(
        cls,
        entries: Sequence[CausalMemoryEntry],
    ) -> dict[str, "ToolMemoryProfile"]:
        """Build a profile per tool from a flat list of CausalMemoryEntry objects.

        Returns
        -------
        dict[str, ToolMemoryProfile]
            Keyed by tool_name.
        """
        profiles: dict[str, ToolMemoryProfile] = {}
        # Intermediate accumulators
        latencies: dict[str, list[float]] = {}
        failure_classes: dict[str, list[str]] = {}
        policy_actions: dict[str, list[str]] = {}

        for entry in entries:
            name = entry.tool_name
            if name not in profiles:
                profiles[name] = cls(tool_name=name)
                latencies[name] = []
                failure_classes[name] = []
                policy_actions[name] = []

            p = profiles[name]
            p.total_executions += 1

            if entry.status.lower() == "ok":
                p.success_count += 1
            else:
                p.failure_count += 1
                if entry.failure_class:
                    failure_classes[name].append(entry.failure_class)

            latencies[name].append(entry.latency_ms)

            if entry.policy_action:
                pa = entry.policy_action.lower()
                policy_actions[name].append(pa)
                if pa == "escalate":
                    p.escalation_count += 1
                elif pa == "abort":
                    p.abort_count += 1

            p.max_attempts_seen = max(p.max_attempts_seen, entry.attempt)

        # Finalise aggregates
        for name, p in profiles.items():
            lats = latencies[name]
            p.avg_latency_ms = sum(lats) / len(lats) if lats else 0.0

            fc_list = failure_classes[name]
            p.most_common_failure_class = _mode(fc_list)

            pa_list = policy_actions[name]
            p.most_common_policy_action = _mode(pa_list)

        return profiles


def _mode(values: list[str]) -> str | None:
    """Return the most frequent value, or None if the list is empty."""
    if not values:
        return None
    return max(set(values), key=values.count)


# ---------------------------------------------------------------------------
# Tool selection scoring
# ---------------------------------------------------------------------------


@dataclass
class ToolSelectionScore:
    """A candidate tool with its memory-informed selection score.

    Attributes
    ----------
    tool_name:
        The tool being evaluated.
    score:
        Selection score in [0, 1].  Higher = prefer this tool.
    profile:
        The ToolMemoryProfile used to compute the score (None if no history).
    reason:
        Human-readable explanation of the score.
    """

    tool_name: str
    score: float
    profile: ToolMemoryProfile | None = None
    reason: str = ""

    def __lt__(self, other: "ToolSelectionScore") -> bool:
        return self.score < other.score


class ToolSelectionAdvisor:
    """Ranks tool candidates using their memory profiles.

    Tools with no history receive a configurable neutral score (default 0.5).
    Tools with a high reliability_score are ranked first.

    Parameters
    ----------
    profiles:
        A dict of ToolMemoryProfile keyed by tool_name (as returned by
        ToolMemoryProfile.build_profiles).
    no_history_score:
        Score assigned to tools without any recorded history.
    """

    def __init__(
        self,
        profiles: dict[str, ToolMemoryProfile],
        *,
        no_history_score: float = 0.5,
    ) -> None:
        self._profiles = dict(profiles)
        self._no_history_score = max(0.0, min(1.0, no_history_score))

    def score_tool(self, tool_name: str) -> ToolSelectionScore:
        """Score a single tool candidate."""
        profile = self._profiles.get(tool_name)
        if profile is None or profile.total_executions == 0:
            return ToolSelectionScore(
                tool_name=tool_name,
                score=self._no_history_score,
                profile=profile,
                reason="no memory history — using neutral score",
            )
        score = profile.reliability_score
        reason_parts: list[str] = [
            f"reliability={score:.2f}",
            f"success_rate={profile.success_rate:.2f}",
            f"executions={profile.total_executions}",
        ]
        if not profile.is_healthy:
            reason_parts.append("UNHEALTHY")
        if profile.most_common_failure_class:
            reason_parts.append(f"common_failure={profile.most_common_failure_class}")
        return ToolSelectionScore(
            tool_name=tool_name,
            score=score,
            profile=profile,
            reason=", ".join(reason_parts),
        )

    def rank(self, tool_names: Sequence[str]) -> list[ToolSelectionScore]:
        """Rank a list of tool candidates by memory-informed score, descending."""
        scored = [self.score_tool(name) for name in tool_names]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored

    def best(self, tool_names: Sequence[str]) -> ToolSelectionScore | None:
        """Return the single highest-scoring tool, or None if the list is empty."""
        ranked = self.rank(tool_names)
        return ranked[0] if ranked else None


# ---------------------------------------------------------------------------
# Memory-aware policy adjustment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryPolicyAdjustment:
    """Represents how memory context modified a base policy decision.

    Attributes
    ----------
    original_action:
        The action returned by the base FailurePolicyEngine.
    adjusted_action:
        The final action after applying memory-based modulations.
    adjustment_reason:
        Why the adjustment was made (or why no adjustment was needed).
    memory_signal:
        The key statistic from the profile that drove the adjustment.
    adjusted:
        True if the action was changed from the original.
    """

    original_action: PolicyAction
    adjusted_action: PolicyAction
    adjustment_reason: str
    memory_signal: str = ""

    @property
    def adjusted(self) -> bool:
        return self.original_action != self.adjusted_action


class MemoryAwarePolicyContext:
    """Wraps FailurePolicyEngine with memory-based policy overrides.

    When the memory profile for a failing tool shows persistent escalation or
    abort patterns, the policy is strengthened (e.g., skip RETRY → go directly
    to ESCALATE).  When the tool has a strong history of eventual success, the
    policy is relaxed (ESCALATE → RETRY with note).

    Parameters
    ----------
    engine:
        The base FailurePolicyEngine.
    profiles:
        Memory profiles for tools (as returned by ToolMemoryProfile.build_profiles).
    escalation_threshold:
        Fraction of escalation-class executions above which the engine will
        preemptively escalate instead of retrying.  Default 0.5.
    abort_threshold:
        Fraction of abort-class executions above which the engine will
        preemptively abort instead of retrying.  Default 0.6.
    """

    def __init__(
        self,
        engine: FailurePolicyEngine,
        profiles: dict[str, ToolMemoryProfile],
        *,
        escalation_threshold: float = 0.5,
        abort_threshold: float = 0.6,
    ) -> None:
        self._engine = engine
        self._profiles = dict(profiles)
        self._escalation_threshold = escalation_threshold
        self._abort_threshold = abort_threshold

    def decide_with_memory(
        self,
        *,
        result: Any,           # ToolExecutionResult
        contract: Any,         # ToolExecutionContract
        attempt: int,
    ) -> tuple[PolicyDecision, MemoryPolicyAdjustment]:
        """Run base policy decision then apply memory-based adjustments.

        Returns
        -------
        (PolicyDecision, MemoryPolicyAdjustment)
            The (possibly adjusted) decision and an explanation of any change.
        """
        base_decision = self._engine.decide(
            result=result,
            contract=contract,
            current_attempt=attempt,
        )
        tool_name = str(getattr(result, "tool_name", "") or "")
        profile = self._profiles.get(tool_name)

        if profile is None or profile.total_executions < 3:
            # Not enough history to override
            return base_decision, MemoryPolicyAdjustment(
                original_action=base_decision.action,
                adjusted_action=base_decision.action,
                adjustment_reason="insufficient memory history — base decision unchanged",
                memory_signal="total_executions < 3",
            )

        original_action = base_decision.action
        adjusted_action = original_action
        adjustment_reason = "memory confirms base decision"
        memory_signal = f"reliability={profile.reliability_score:.2f}"

        # Abort override: if the tool fails and aborts persistently, stop retrying
        if original_action == PolicyAction.RETRY:
            abort_rate = profile.abort_count / profile.total_executions
            if abort_rate >= self._abort_threshold:
                adjusted_action = PolicyAction.ABORT
                adjustment_reason = (
                    f"memory: abort_rate={abort_rate:.2f} >= threshold={self._abort_threshold:.2f}; "
                    "skipping retry — tool historically unrecoverable"
                )
                memory_signal = f"abort_rate={abort_rate:.2f}"

            elif profile.escalation_count / profile.total_executions >= self._escalation_threshold:
                esc_rate = profile.escalation_count / profile.total_executions
                adjusted_action = PolicyAction.ESCALATE
                adjustment_reason = (
                    f"memory: escalation_rate={esc_rate:.2f} >= threshold={self._escalation_threshold:.2f}; "
                    "promoting to escalate — tool has persistent failure pattern"
                )
                memory_signal = f"escalation_rate={esc_rate:.2f}"

        # Relax override: if base says ESCALATE but tool is historically healthy, try one more retry
        elif original_action == PolicyAction.ESCALATE and profile.is_healthy:
            adjusted_action = PolicyAction.RETRY
            adjustment_reason = (
                f"memory: tool is historically healthy "
                f"(reliability={profile.reliability_score:.2f}); "
                "relaxing ESCALATE → RETRY"
            )
            memory_signal = f"reliability={profile.reliability_score:.2f}"

        adjustment = MemoryPolicyAdjustment(
            original_action=original_action,
            adjusted_action=adjusted_action,
            adjustment_reason=adjustment_reason,
            memory_signal=memory_signal,
        )

        if adjusted_action == original_action:
            return base_decision, adjustment

        # Rebuild decision with adjusted action
        adjusted_decision = PolicyDecision(
            action=adjusted_action,
            reason=f"{base_decision.reason} [memory-adjusted: {adjustment_reason}]",
            retry_delay_seconds=base_decision.retry_delay_seconds,
            escalation_category=base_decision.escalation_category,
            confidence=max(0.5, base_decision.confidence - 0.1),  # Slight confidence reduction on override
            metadata={
                **(base_decision.metadata or {}),
                "memory_adjusted": True,
                "original_action": original_action.value,
                "adjustment_reason": adjustment_reason,
            },
        )
        return adjusted_decision, adjustment


# ---------------------------------------------------------------------------
# Scheduling priority signal
# ---------------------------------------------------------------------------


def compute_scheduling_priority(
    profile: ToolMemoryProfile | None,
    *,
    base_priority: int = 5,
) -> int:
    """Compute an integer scheduling priority for a tool based on its memory profile.

    Lower number = higher priority (follows Unix convention).

    Rules:
    - Healthy tools get base_priority.
    - Unhealthy tools (high failure/escalation) get lower priority (higher number).
    - Tools with abort history are deprioritized most.
    - Tools with no history get base_priority.

    Returns
    -------
    int
        Priority value in [1, 10].  1 = highest, 10 = lowest.
    """
    if profile is None or profile.total_executions == 0:
        return base_priority

    score = profile.reliability_score  # [0, 1]

    # Map [0, 1] reliability → [10, 1] priority (inverted, clamped)
    # reliability=1.0 → priority=1 (highest)
    # reliability=0.0 → priority=10 (lowest)
    raw = 10 - int(score * 9)
    return max(1, min(10, raw))


# ---------------------------------------------------------------------------
# Response shaping signal
# ---------------------------------------------------------------------------


def suggest_response_tone(profile: ToolMemoryProfile | None) -> str:
    """Suggest how the model should surface tool status to the user based on memory.

    Returns one of: "confident", "cautious", "warn", "escalate"
    """
    if profile is None or profile.total_executions == 0:
        return "confident"

    if profile.reliability_score >= 0.8:
        return "confident"
    if profile.reliability_score >= 0.5 and profile.is_healthy:
        return "cautious"
    if profile.abort_count > 0 or profile.escalation_count > profile.total_executions // 3:
        return "warn"
    return "escalate"


__all__ = [
    "ToolMemoryProfile",
    "ToolSelectionScore",
    "ToolSelectionAdvisor",
    "MemoryPolicyAdjustment",
    "MemoryAwarePolicyContext",
    "compute_scheduling_priority",
    "suggest_response_tone",
]
