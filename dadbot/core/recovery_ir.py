"""Phase D: Recovery Strategy Selection.

Transforms policy decisions into recovery strategies:
- What went wrong (PolicyDecisionIR)?
- How should we recover (RecoveryDecision)?
- What action to take (RecoveryStrategy)?

Architecture:
  PolicyDecisionIR (from Phase C)
       ↓
  RecoverySelector (strategy selection logic)
       ↓
  RecoveryDecision {strategy, checkpoint_id, bounded_attempts, ...}
       ↓
  Recovery execution (Phase E context)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dadbot.core.policy_ir import PolicyDecisionIR
from dadbot.core.runtime_types import PolicyEffectType, ToolExecutionStatus, ToolResult


class RecoveryStrategy(str, Enum):
    """Strategy for recovering from policy effects or tool failures."""

    RETRY_SAME = "retry_same"  # Retry the same tool invocation
    REPLAY_CHECKPOINT = "replay_checkpoint"  # Restart from saved checkpoint
    DEGRADE_GRACEFULLY = "degrade_gracefully"  # Use fallback/degraded output
    REQUIRE_APPROVAL = "require_approval"  # Wait for user approval
    HALT_SAFE = "halt_safe"  # Stop execution, preserve state for manual recovery


@dataclass
class RecoveryDecision:
    """Recovery strategy selection based on policy and execution state.
    
    Bridges PolicyDecisionIR (what policies say) and execution recovery
    (what to do about it).
    """

    # The recovery strategy to apply
    strategy: RecoveryStrategy

    # Why this strategy was selected (audit trail)
    reason: str = ""

    # Maximum attempts for RETRY_SAME strategy
    bounded_attempts: int = 1

    # Checkpoint ID to replay from (for REPLAY_CHECKPOINT)
    checkpoint_id: str = ""


    # Fallback output for DEGRADE_GRACEFULLY
    degraded_output: Any = None

    # User approval required (for REQUIRE_APPROVAL)
    requires_user_approval: bool = False

    # Original tool result (for context)
    tool_result: ToolResult | None = None

    # Matched policy rules (audit trail)
    matched_rules: tuple[str, ...] = field(default_factory=tuple)

    def is_retriable(self) -> bool:
        """Strategy allows retry attempts."""
        return self.strategy == RecoveryStrategy.RETRY_SAME and self.bounded_attempts > 0

    def requires_external_input(self) -> bool:
        """Strategy requires user/external input."""
        return self.strategy in [
            RecoveryStrategy.REQUIRE_APPROVAL,
        ]
        return self.strategy == RecoveryStrategy.REQUIRE_APPROVAL

    def is_terminal(self) -> bool:
        """Strategy is terminal (no more recovery possible)."""
        return self.strategy == RecoveryStrategy.HALT_SAFE


@dataclass
class RecoveryContext:
    """Context for recovery decision: tool result + policy decision + history."""

    tool_result: ToolResult
    policy_decision: PolicyDecisionIR
    # Number of times this tool has already been retried
    retry_count: int = 0
    # Previous recovery decisions (for escalation logic)
    previous_strategies: list[RecoveryStrategy] = field(default_factory=list)
    # User context (permissions, approval status, etc.)
    user_context: dict[str, Any] = field(default_factory=dict)


class RecoverySelector:
    """Selects recovery strategy based on policy decision and execution context."""

    # Maximum retries for transient failures
    MAX_RETRIES = 3

    # Tool statuses considered transient (retriable)
    TRANSIENT_STATUSES = {ToolExecutionStatus.TIMEOUT, ToolExecutionStatus.DEGRADED}

    # Tool statuses considered fatal (not retriable)
    FATAL_STATUSES = {ToolExecutionStatus.DENIED, ToolExecutionStatus.ERROR}

    def select(self, context: RecoveryContext) -> RecoveryDecision:
        """Select recovery strategy given tool result and policy decision.

        Decision logic:
        1. If tool succeeded and no policy effects: no recovery needed
        2. If tool timed out and retriable: RETRY_SAME
        3. If policy denied: REQUIRE_APPROVAL or ESCALATE_PERMISSION
        4. If large retry count: DEGRADE_GRACEFULLY or HALT_SAFE
        5. If fatal error: HALT_SAFE
        6. Otherwise: HALT_SAFE (safe default)
        """
        # Case 1: Success, no policy effects
        if (
            context.tool_result.status == ToolExecutionStatus.OK
            and not context.policy_decision.output_was_modified
        ):
            return RecoveryDecision(
                strategy=RecoveryStrategy.DEGRADE_GRACEFULLY,
                reason="Tool succeeded, no recovery needed",
                degraded_output=context.policy_decision.final_output,
                tool_result=context.tool_result,
                matched_rules=context.policy_decision.matched_rules,
            )

        # Case 2: Transient failure (timeout, degraded) and retries available
        if (
            context.tool_result.status in self.TRANSIENT_STATUSES
            and context.retry_count < self.MAX_RETRIES
        ):
            return RecoveryDecision(
                strategy=RecoveryStrategy.RETRY_SAME,
                reason=f"Transient failure ({context.tool_result.status.value}), retrying",
                bounded_attempts=self.MAX_RETRIES - context.retry_count,
                tool_result=context.tool_result,
                matched_rules=context.policy_decision.matched_rules,
            )

        # Case 3: Policy denied tool, check if escalation requested
        # Case 4: Retry limit exceeded, degrade gracefully
        if context.retry_count >= self.MAX_RETRIES:
            return RecoveryDecision(
                strategy=RecoveryStrategy.DEGRADE_GRACEFULLY,
                reason=f"Retry limit ({self.MAX_RETRIES}) exceeded",
                degraded_output={"error": "Tool unavailable after retries", "degraded": True},
                tool_result=context.tool_result,
                matched_rules=context.policy_decision.matched_rules,
            )

        # Case 5: Approval required by policy
        if any(
            e.effect_type == PolicyEffectType.REQUIRE_APPROVAL
            for e in context.policy_decision.emitted_effects
        ):
            return RecoveryDecision(
                strategy=RecoveryStrategy.REQUIRE_APPROVAL,
                reason="Policy requires user approval",
                requires_user_approval=True,
                tool_result=context.tool_result,
                matched_rules=context.policy_decision.matched_rules,
            )

        # Case 6: Fatal error, halt safely
        if context.tool_result.status in self.FATAL_STATUSES:
            return RecoveryDecision(
                strategy=RecoveryStrategy.HALT_SAFE,
                reason=f"Fatal error: {context.tool_result.error or context.tool_result.status.value}",
                tool_result=context.tool_result,
                matched_rules=context.policy_decision.matched_rules,
            )

        # Default: halt safely (conservative)
        return RecoveryDecision(
            strategy=RecoveryStrategy.HALT_SAFE,
            reason="No matching recovery strategy",
            tool_result=context.tool_result,
            matched_rules=context.policy_decision.matched_rules,
        )


class RecoveryChain:
    """Executes a sequence of recovery actions with bounded attempts."""

    def __init__(self, selector: RecoverySelector | None = None) -> None:
        """Initialize recovery chain.

        Args:
            selector: RecoverySelector instance (uses default if None)
        """
        self.selector = selector or RecoverySelector()
        self.decisions: list[RecoveryDecision] = []
        self.attempt_count: int = 0

    def attempt(self, context: RecoveryContext) -> RecoveryDecision:
        """Execute one recovery decision.

        Args:
            context: RecoveryContext with tool result and policy decision

        Returns:
            RecoveryDecision with selected strategy
        """
        decision = self.selector.select(context)
        self.decisions.append(decision)

        # Update attempt count for next iteration
        if decision.strategy == RecoveryStrategy.RETRY_SAME:
            self.attempt_count += 1

        return decision

    def can_retry(self, decision: RecoveryDecision) -> bool:
        """Check if recovery chain can execute another retry.

        Args:
            decision: Most recent recovery decision

        Returns:
            True if strategy allows further recovery
        """
        if decision.strategy == RecoveryStrategy.RETRY_SAME:
            return decision.is_retriable()
        return False

    def recovery_summary(self) -> dict[str, Any]:
        """Summarize recovery chain execution.

        Returns:
            Summary dict with decisions, attempt count, final strategy
        """
        if not self.decisions:
            return {"status": "no_decisions", "attempts": 0}

        final_decision = self.decisions[-1]
        return {
            "status": "completed",
            "attempts": self.attempt_count,
            "total_decisions": len(self.decisions),
            "final_strategy": final_decision.strategy.value,
            "requires_external_input": final_decision.requires_external_input(),
            "is_terminal": final_decision.is_terminal(),
        }


__all__ = [
    "RecoveryStrategy",
    "RecoveryDecision",
    "RecoveryContext",
    "RecoverySelector",
    "RecoveryChain",
]
