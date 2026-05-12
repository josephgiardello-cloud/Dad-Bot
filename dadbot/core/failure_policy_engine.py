"""Failure policy engine: deterministic mapping from failure classification to system action.

This module is the "decision brain" that turns "failure detected" into "what happens next".
It consumes FailureClass + context and returns a deterministic action (retry, escalate, abort, reconcile, degrade).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from dadbot.core.tool_execution_contract import (
    FailureClass,
    FailureSeverity,
    ToolExecutionContract,
)
from dadbot.core.external_tool_runtime import ToolExecutionResult


class PolicyAction(str, Enum):
    """Deterministic actions the system can take when a failure is detected."""

    RETRY = "retry"
    """Retry the same operation (with backoff if transient, with policy adjustment if needed)."""

    ESCALATE = "escalate"
    """Send to human or higher-level decision maker (approval gate, investigation queue, etc.)."""

    ABORT = "abort"
    """Terminate operation; failure is unrecoverable and no fallback applies."""

    FALLBACK = "fallback"
    """Try next tool in fallback chain (if available)."""

    RECONCILE = "reconcile"
    """Investigate state inconsistency; unclear what happened, need manual inspection or drift detection."""

    DEGRADE = "degrade"
    """Continue with reduced confidence; operation succeeded but quality is degraded (partial result, latency spike, etc.)."""


@dataclass(frozen=True)
class PolicyDecision:
    """Formal decision output from failure policy engine."""

    action: PolicyAction
    """What should happen next."""

    reason: str
    """Human-readable justification."""

    retry_delay_seconds: float | None = None
    """If action=RETRY, suggested delay before retry (for backoff)."""

    escalation_category: str | None = None
    """If action=ESCALATE, category for routing (e.g., 'rate_limit_backoff', 'permission_denied', 'investigation')."""

    confidence: float = 1.0
    """Confidence in this decision (0-1; lower = more uncertain)."""

    metadata: dict[str, Any] | None = None
    """Additional context for debugging or monitoring."""


class FailurePolicyRules:
    """Deterministic decision rules for failure handling.

    Maps failure class + severity + context → action.
    """

    # Transient failure retry budgets
    TRANSIENT_RETRY_LIMIT: int = 5  # Max retries for transient failures
    TRANSIENT_BASE_DELAY_MS: float = 100.0  # Initial backoff
    TRANSIENT_MAX_DELAY_SECONDS: float = 30.0  # Max backoff

    # Permanent failure escalation categories
    RATE_LIMIT_ESCALATION = "rate_limit_backoff"
    PERMISSION_DENIED_ESCALATION = "permission_denied_escalate"
    TOOL_ERROR_ESCALATION = "tool_error_debug"
    RESOURCE_UNAVAILABLE_ESCALATION = "tool_unavailable_fallback"
    ISOLATION_VIOLATION_ESCALATION = "isolation_violated_abort"

    @staticmethod
    def classify_failure_to_action(
        failure_class: FailureClass,
        severity: FailureSeverity,
        escalation_key: str | None = None,
        retryable: bool = False,
    ) -> PolicyAction:
        """Classify failure to primary action (before attempt/context filtering).

        This is the base classification; context (attempt count, etc.) modulates it.
        """
        if severity == FailureSeverity.TRANSIENT:
            return PolicyAction.RETRY  # Transient failures are candidates for retry

        if severity == FailureSeverity.PERMANENT:
            if escalation_key == FailurePolicyRules.RATE_LIMIT_ESCALATION:
                return PolicyAction.RETRY  # Rate limit is actually transient; retry with longer backoff
            if escalation_key == FailurePolicyRules.RESOURCE_UNAVAILABLE_ESCALATION:
                return PolicyAction.FALLBACK  # Tool not available; try fallback
            if escalation_key in {
                FailurePolicyRules.PERMISSION_DENIED_ESCALATION,
                FailurePolicyRules.TOOL_ERROR_ESCALATION,
            }:
                return PolicyAction.ESCALATE  # Needs human decision
            return PolicyAction.ABORT  # Other permanent failures: stop

        if severity == FailureSeverity.PARTIAL:
            return PolicyAction.DEGRADE  # Partial success; continue with reduced confidence

        if severity == FailureSeverity.UNKNOWN:
            return PolicyAction.RECONCILE  # Unknown failures require investigation

        return PolicyAction.ABORT  # Default to abort for safety

    @staticmethod
    def compute_retry_delay(
        attempt: int,
        base_delay_ms: float = TRANSIENT_BASE_DELAY_MS,
        max_delay_seconds: float = TRANSIENT_MAX_DELAY_SECONDS,
        jitter_ratio: float = 0.2,
    ) -> float:
        """Compute exponential backoff with jitter.

        Args:
            attempt: Attempt number (1-indexed)
            base_delay_ms: Initial delay in milliseconds
            max_delay_seconds: Maximum delay in seconds
            jitter_ratio: Proportion of jitter to add (0-1)

        Returns:
            Delay in seconds.
        """
        # Exponential: 100ms, 200ms, 400ms, 800ms, ...
        exponential_ms = base_delay_ms * (2.0 ** max(0, attempt - 1))
        capped_ms = min(exponential_ms, max_delay_seconds * 1000.0)

        # Add jitter: ±(jitter_ratio * capped_ms)
        jitter_ms = capped_ms * jitter_ratio * 0.5  # Simple: add up to jitter_ratio of capped_ms
        result_ms = capped_ms + jitter_ms
        return result_ms / 1000.0


class FailurePolicyEngine:
    """Decision engine: maps failure classification to deterministic actions."""

    def __init__(self, rules: FailurePolicyRules | None = None) -> None:
        self._rules = rules or FailurePolicyRules()

    def decide(
        self,
        *,
        result: ToolExecutionResult,
        contract: ToolExecutionContract,
        current_attempt: int,
    ) -> PolicyDecision:
        """Deterministic policy decision for a failed tool execution.

        Args:
            result: Tool execution result with failure_class, severity, escalation_key in metadata
            contract: Tool contract (provides max_total_attempts)
            current_attempt: Current attempt number (1-indexed)

        Returns:
            PolicyDecision with action and rationale.
        """
        metadata = dict(result.metadata or {})
        failure_class_str = str(metadata.get("failure_class") or "unknown")
        severity_str = str(metadata.get("failure_severity") or "unknown")
        escalation_key = str(metadata.get("escalation_key") or "").strip() or None
        retryable = bool(metadata.get("failure_retryable", False))

        # Parse failure class
        try:
            failure_class = FailureClass(failure_class_str)
        except (ValueError, KeyError):
            failure_class = FailureClass.UNKNOWN

        # Parse severity
        try:
            severity = FailureSeverity(severity_str)
        except (ValueError, KeyError):
            severity = FailureSeverity.UNKNOWN

        # Base action classification
        primary_action = self._rules.classify_failure_to_action(
            failure_class,
            severity,
            escalation_key=escalation_key,
            retryable=retryable,
        )

        # Modulate based on attempt count
        if primary_action == PolicyAction.RETRY:
            if current_attempt >= contract.max_total_attempts:
                # Retry budget exhausted
                return PolicyDecision(
                    action=PolicyAction.ESCALATE,
                    reason=f"Retry budget exhausted after {current_attempt} attempts (max={contract.max_total_attempts})",
                    escalation_category="retry_budget_exceeded",
                    confidence=0.95,
                    metadata={
                        "failure_class": failure_class.value,
                        "severity": severity.value,
                        "attempts_exhausted": current_attempt,
                    },
                )

            # Compute backoff
            delay_seconds = self._rules.compute_retry_delay(current_attempt)
            return PolicyDecision(
                action=PolicyAction.RETRY,
                reason=f"Transient failure (attempt {current_attempt}/{contract.max_total_attempts}); retrying with backoff",
                retry_delay_seconds=delay_seconds,
                confidence=0.9,
                metadata={
                    "failure_class": failure_class.value,
                    "severity": severity.value,
                    "attempt": current_attempt,
                    "max_attempts": contract.max_total_attempts,
                },
            )

        if primary_action == PolicyAction.ESCALATE:
            return PolicyDecision(
                action=PolicyAction.ESCALATE,
                reason=f"Permanent failure requires escalation: {failure_class.value}",
                escalation_category=escalation_key or "unknown_permanent",
                confidence=0.95,
                metadata={
                    "failure_class": failure_class.value,
                    "severity": severity.value,
                    "escalation_key": escalation_key,
                    "attempt": current_attempt,
                },
            )

        if primary_action == PolicyAction.FALLBACK:
            return PolicyDecision(
                action=PolicyAction.FALLBACK,
                reason=f"Primary tool unavailable ({failure_class.value}); try fallback",
                confidence=0.85,
                metadata={
                    "failure_class": failure_class.value,
                    "severity": severity.value,
                },
            )

        if primary_action == PolicyAction.DEGRADE:
            return PolicyDecision(
                action=PolicyAction.DEGRADE,
                reason=f"Partial/degraded result accepted; continue with reduced confidence",
                confidence=0.7,
                metadata={
                    "failure_class": failure_class.value,
                    "severity": severity.value,
                    "result_usable": True,
                },
            )

        if primary_action == PolicyAction.RECONCILE:
            return PolicyDecision(
                action=PolicyAction.RECONCILE,
                reason=f"Unknown failure classification; state reconciliation recommended",
                escalation_category="unknown_reconciliation",
                confidence=0.5,
                metadata={
                    "failure_class": failure_class.value,
                    "severity": severity.value,
                    "investigation_required": True,
                },
            )

        # Default: abort
        return PolicyDecision(
            action=PolicyAction.ABORT,
            reason=f"Failure unrecoverable; aborting operation",
            confidence=0.95,
            metadata={
                "failure_class": failure_class.value,
                "severity": severity.value,
                "attempt": current_attempt,
            },
        )


class FailurePolicyContext:
    """Context for policy decisions that may involve system-wide state."""

    def __init__(
        self,
        engine: FailurePolicyEngine | None = None,
        rate_limit_backoff_factor: float = 2.0,
        approval_gate_available: bool = True,
    ) -> None:
        self._engine = engine or FailurePolicyEngine()
        self._rate_limit_backoff_factor = rate_limit_backoff_factor
        self._approval_gate_available = approval_gate_available

    def decide_with_system_context(
        self,
        *,
        result: ToolExecutionResult,
        contract: ToolExecutionContract,
        current_attempt: int,
        has_fallback_tools: bool = False,
    ) -> PolicyDecision:
        """Decide policy action with system context (e.g., fallback availability).

        Modulates the policy based on what options are available:
        - If no fallback tools, FALLBACK→ABORT
        - If no approval gate, ESCALATE→ABORT
        """
        decision = self._engine.decide(
            result=result,
            contract=contract,
            current_attempt=current_attempt,
        )

        # Modulate based on system availability
        if decision.action == PolicyAction.FALLBACK and not has_fallback_tools:
            return PolicyDecision(
                action=PolicyAction.ABORT,
                reason="No fallback tools available; aborting instead of fallback",
                confidence=max(0.9, decision.confidence - 0.1),
                metadata=dict(decision.metadata or {}, **{"fallback_unavailable": True}),
            )

        if (
            decision.action == PolicyAction.ESCALATE
            and decision.escalation_category == FailurePolicyRules.PERMISSION_DENIED_ESCALATION
            and not self._approval_gate_available
        ):
            return PolicyDecision(
                action=PolicyAction.ABORT,
                reason="Approval gate unavailable; aborting instead of escalating",
                confidence=max(0.8, decision.confidence - 0.2),
                metadata=dict(decision.metadata or {}, **{"escalation_unavailable": True}),
            )

        return decision


__all__ = [
    "FailurePolicyContext",
    "FailurePolicyEngine",
    "FailurePolicyRules",
    "PolicyAction",
    "PolicyDecision",
]
