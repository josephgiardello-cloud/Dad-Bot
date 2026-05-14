"""Tests for failure policy engine: decision brain validation."""

import pytest

from dadbot.core.failure_policy_engine import (
    FailurePolicyContext,
    FailurePolicyEngine,
    FailurePolicyRules,
    PolicyAction,
    PolicyDecision,
)
from dadbot.core.tool_execution_contract import (
    FailureClass,
    FailureSeverity,
    ReplaySemantics,
    ToolExecutionContract,
    ToolInputSchema,
    ToolOutputSchema,
)
from dadbot.core.external_tool_runtime import ToolExecutionResult, ToolExecutionStatus


class TestFailurePolicyRules:
    """Test deterministic failure → action classification."""

    def test_transient_failure_returns_retry(self):
        """Transient failures map to retry."""
        action = FailurePolicyRules.classify_failure_to_action(
            failure_class=FailureClass.NETWORK_TIMEOUT,
            severity=FailureSeverity.TRANSIENT,
        )
        assert action == PolicyAction.RETRY

    def test_permanent_failure_returns_abort_or_escalate(self):
        """Permanent failures map to abort or escalate depending on escalation_key."""
        # Generic permanent → abort
        action = FailurePolicyRules.classify_failure_to_action(
            failure_class=FailureClass.CLIENT_ERROR,
            severity=FailureSeverity.PERMANENT,
            escalation_key="unknown",
        )
        assert action == PolicyAction.ABORT

        # Permission denied → escalate
        action = FailurePolicyRules.classify_failure_to_action(
            failure_class=FailureClass.PERMISSION_DENIED,
            severity=FailureSeverity.PERMANENT,
            escalation_key=FailurePolicyRules.PERMISSION_DENIED_ESCALATION,
        )
        assert action == PolicyAction.ESCALATE

    def test_partial_failure_returns_degrade(self):
        """Partial failures map to degrade."""
        action = FailurePolicyRules.classify_failure_to_action(
            failure_class=FailureClass.PARTIAL_OUTPUT,
            severity=FailureSeverity.PARTIAL,
        )
        assert action == PolicyAction.DEGRADE

    def test_unknown_failure_returns_reconcile(self):
        """Unknown failures map to reconcile."""
        action = FailurePolicyRules.classify_failure_to_action(
            failure_class=FailureClass.UNKNOWN,
            severity=FailureSeverity.UNKNOWN,
        )
        assert action == PolicyAction.RECONCILE

    def test_rate_limit_escalation_maps_to_retry(self):
        """Rate limit (technically permanent) maps to retry (with longer backoff)."""
        action = FailurePolicyRules.classify_failure_to_action(
            failure_class=FailureClass.NETWORK_RATE_LIMIT,
            severity=FailureSeverity.TRANSIENT,  # Actually transient despite looking permanent
            escalation_key=FailurePolicyRules.RATE_LIMIT_ESCALATION,
        )
        assert action == PolicyAction.RETRY

    def test_resource_unavailable_maps_to_fallback(self):
        """Tool unavailable maps to fallback."""
        action = FailurePolicyRules.classify_failure_to_action(
            failure_class=FailureClass.RESOURCE_UNAVAILABLE,
            severity=FailureSeverity.PERMANENT,
            escalation_key=FailurePolicyRules.RESOURCE_UNAVAILABLE_ESCALATION,
        )
        assert action == PolicyAction.FALLBACK


class TestRetryBackoff:
    """Test exponential backoff computation."""

    def test_retry_delay_increases_exponentially(self):
        """Retry delays increase exponentially with attempt number."""
        delay_1 = FailurePolicyRules.compute_retry_delay(attempt=1, jitter_ratio=0.0)
        delay_2 = FailurePolicyRules.compute_retry_delay(attempt=2, jitter_ratio=0.0)
        delay_3 = FailurePolicyRules.compute_retry_delay(attempt=3, jitter_ratio=0.0)

        assert delay_1 < delay_2 < delay_3
        # Exponential with base 2: 100ms, 200ms, 400ms
        assert 0.09 < delay_1 < 0.11  # ~100ms
        assert 0.19 < delay_2 < 0.21  # ~200ms
        assert 0.39 < delay_3 < 0.41  # ~400ms

    def test_retry_delay_capped_at_max(self):
        """Retry delays do not exceed max_delay_seconds."""
        delay_high = FailurePolicyRules.compute_retry_delay(
            attempt=10,
            max_delay_seconds=5.0,
            jitter_ratio=0.0,
        )
        assert delay_high <= 5.0

    def test_retry_delay_with_jitter(self):
        """Jitter adds randomness to delays."""
        # Multiple calls with jitter should produce slightly different results
        delays = [
            FailurePolicyRules.compute_retry_delay(attempt=2, jitter_ratio=0.2)
            for _ in range(3)
        ]
        # All should be in similar range but not identical
        assert len(set(delays)) == len(delays) or len(set(delays)) > 1  # At least some variation


class TestFailurePolicyEngine:
    """Test failure policy engine decisions."""

    def test_transient_retry_within_budget(self):
        """Transient failure within retry budget → retry action."""
        contract = ToolExecutionContract(
            tool_name="test_tool",
            version="1.0.0",
            input_schema=ToolInputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            output_schema=ToolOutputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            replay_semantics=ReplaySemantics(
                idempotency_key_factors=frozenset(),
                policy_context_factors=frozenset(),
                cache_ttl_seconds=0,
                determinism_guarantee="strict",
            ),
            max_total_attempts=5,
        )

        result = ToolExecutionResult(
            tool_name="test_tool",
            status=ToolExecutionStatus.TIMEOUT,
            output=None,
            error="timeout",
            attempts=1,
            latency_ms=5000.0,
            confidence=0.0,
            metadata={
                "failure_class": FailureClass.NETWORK_TIMEOUT.value,
                "failure_severity": FailureSeverity.TRANSIENT.value,
                "failure_retryable": True,
                "escalation_key": "network_timeout_retryable",
            },
        )

        engine = FailurePolicyEngine()
        decision = engine.decide(result=result, contract=contract, current_attempt=1)

        assert decision.action == PolicyAction.RETRY
        assert decision.retry_delay_seconds is not None
        assert decision.retry_delay_seconds > 0

    def test_transient_retry_budget_exhausted(self):
        """Transient failure with exhausted retry budget → escalate."""
        contract = ToolExecutionContract(
            tool_name="test_tool",
            version="1.0.0",
            input_schema=ToolInputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            output_schema=ToolOutputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            replay_semantics=ReplaySemantics(
                idempotency_key_factors=frozenset(),
                policy_context_factors=frozenset(),
                cache_ttl_seconds=0,
                determinism_guarantee="strict",
            ),
            max_total_attempts=3,
        )

        result = ToolExecutionResult(
            tool_name="test_tool",
            status=ToolExecutionStatus.TIMEOUT,
            output=None,
            error="timeout",
            attempts=3,
            latency_ms=5000.0,
            confidence=0.0,
            metadata={
                "failure_class": FailureClass.NETWORK_TIMEOUT.value,
                "failure_severity": FailureSeverity.TRANSIENT.value,
                "failure_retryable": True,
                "escalation_key": "network_timeout_retryable",
            },
        )

        engine = FailurePolicyEngine()
        decision = engine.decide(result=result, contract=contract, current_attempt=3)

        assert decision.action == PolicyAction.ESCALATE
        assert "budget exhausted" in decision.reason

    def test_permanent_permission_denied_escalates(self):
        """Permission denied → escalate."""
        contract = ToolExecutionContract(
            tool_name="test_tool",
            version="1.0.0",
            input_schema=ToolInputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            output_schema=ToolOutputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            replay_semantics=ReplaySemantics(
                idempotency_key_factors=frozenset(),
                policy_context_factors=frozenset(),
                cache_ttl_seconds=0,
                determinism_guarantee="strict",
            ),
            max_total_attempts=3,
        )

        result = ToolExecutionResult(
            tool_name="test_tool",
            status=ToolExecutionStatus.ERROR,
            output=None,
            error="permission_denied",
            attempts=1,
            latency_ms=100.0,
            confidence=0.0,
            metadata={
                "failure_class": FailureClass.PERMISSION_DENIED.value,
                "failure_severity": FailureSeverity.PERMANENT.value,
                "failure_retryable": False,
                "escalation_key": FailurePolicyRules.PERMISSION_DENIED_ESCALATION,
            },
        )

        engine = FailurePolicyEngine()
        decision = engine.decide(result=result, contract=contract, current_attempt=1)

        assert decision.action == PolicyAction.ESCALATE
        assert decision.escalation_category == FailurePolicyRules.PERMISSION_DENIED_ESCALATION

    def test_partial_output_degrades(self):
        """Partial output → degrade."""
        contract = ToolExecutionContract(
            tool_name="test_tool",
            version="1.0.0",
            input_schema=ToolInputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            output_schema=ToolOutputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            replay_semantics=ReplaySemantics(
                idempotency_key_factors=frozenset(),
                policy_context_factors=frozenset(),
                cache_ttl_seconds=0,
                determinism_guarantee="strict",
            ),
            max_total_attempts=3,
        )

        result = ToolExecutionResult(
            tool_name="test_tool",
            status=ToolExecutionStatus.PARTIAL,
            output={"partial_data": "incomplete"},
            error="",
            attempts=1,
            latency_ms=200.0,
            confidence=0.5,
            metadata={
                "failure_class": FailureClass.PARTIAL_OUTPUT.value,
                "failure_severity": FailureSeverity.PARTIAL.value,
                "failure_retryable": False,
                "escalation_key": "partial_output_accepted",
            },
        )

        engine = FailurePolicyEngine()
        decision = engine.decide(result=result, contract=contract, current_attempt=1)

        assert decision.action == PolicyAction.DEGRADE
        assert decision.confidence < 1.0

    def test_unknown_failure_reconciles(self):
        """Unknown failure → reconcile."""
        contract = ToolExecutionContract(
            tool_name="test_tool",
            version="1.0.0",
            input_schema=ToolInputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            output_schema=ToolOutputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            replay_semantics=ReplaySemantics(
                idempotency_key_factors=frozenset(),
                policy_context_factors=frozenset(),
                cache_ttl_seconds=0,
                determinism_guarantee="strict",
            ),
            max_total_attempts=3,
        )

        result = ToolExecutionResult(
            tool_name="test_tool",
            status=ToolExecutionStatus.ERROR,
            output=None,
            error="something weird happened",
            attempts=1,
            latency_ms=100.0,
            confidence=0.0,
            metadata={
                "failure_class": FailureClass.UNKNOWN.value,
                "failure_severity": FailureSeverity.UNKNOWN.value,
                "failure_retryable": True,
                "escalation_key": "unknown",
            },
        )

        engine = FailurePolicyEngine()
        decision = engine.decide(result=result, contract=contract, current_attempt=1)

        assert decision.action == PolicyAction.RECONCILE


class TestFailurePolicyContext:
    """Test policy decisions with system context."""

    def test_fallback_unavailable_downgrades_to_abort(self):
        """When no fallback tools, FALLBACK→ABORT."""
        contract = ToolExecutionContract(
            tool_name="test_tool",
            version="1.0.0",
            input_schema=ToolInputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            output_schema=ToolOutputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            replay_semantics=ReplaySemantics(
                idempotency_key_factors=frozenset(),
                policy_context_factors=frozenset(),
                cache_ttl_seconds=0,
                determinism_guarantee="strict",
            ),
            max_total_attempts=3,
        )

        result = ToolExecutionResult(
            tool_name="test_tool",
            status=ToolExecutionStatus.ERROR,
            output=None,
            error="tool_not_registered",
            attempts=1,
            latency_ms=100.0,
            confidence=0.0,
            metadata={
                "failure_class": FailureClass.RESOURCE_UNAVAILABLE.value,
                "failure_severity": FailureSeverity.PERMANENT.value,
                "failure_retryable": False,
                "escalation_key": FailurePolicyRules.RESOURCE_UNAVAILABLE_ESCALATION,
            },
        )

        context = FailurePolicyContext()
        decision = context.decide_with_system_context(
            result=result,
            contract=contract,
            current_attempt=1,
            has_fallback_tools=False,  # No fallback available
        )

        assert decision.action == PolicyAction.ABORT
        assert "fallback" in decision.reason.lower() or "fallback_unavailable" in str(decision.metadata or {})

    def test_escalate_unavailable_downgrades_to_abort(self):
        """When approval gate unavailable, ESCALATE(permission)→ABORT."""
        contract = ToolExecutionContract(
            tool_name="test_tool",
            version="1.0.0",
            input_schema=ToolInputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            output_schema=ToolOutputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            replay_semantics=ReplaySemantics(
                idempotency_key_factors=frozenset(),
                policy_context_factors=frozenset(),
                cache_ttl_seconds=0,
                determinism_guarantee="strict",
            ),
            max_total_attempts=3,
        )

        result = ToolExecutionResult(
            tool_name="test_tool",
            status=ToolExecutionStatus.ERROR,
            output=None,
            error="permission_denied",
            attempts=1,
            latency_ms=100.0,
            confidence=0.0,
            metadata={
                "failure_class": FailureClass.PERMISSION_DENIED.value,
                "failure_severity": FailureSeverity.PERMANENT.value,
                "failure_retryable": False,
                "escalation_key": FailurePolicyRules.PERMISSION_DENIED_ESCALATION,
            },
        )

        context = FailurePolicyContext(approval_gate_available=False)
        decision = context.decide_with_system_context(
            result=result,
            contract=contract,
            current_attempt=1,
            has_fallback_tools=False,
        )

        assert decision.action == PolicyAction.ABORT

    def test_normal_escalate_when_gate_available(self):
        """When approval gate available, ESCALATE works normally."""
        contract = ToolExecutionContract(
            tool_name="test_tool",
            version="1.0.0",
            input_schema=ToolInputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            output_schema=ToolOutputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            replay_semantics=ReplaySemantics(
                idempotency_key_factors=frozenset(),
                policy_context_factors=frozenset(),
                cache_ttl_seconds=0,
                determinism_guarantee="strict",
            ),
            max_total_attempts=3,
        )

        result = ToolExecutionResult(
            tool_name="test_tool",
            status=ToolExecutionStatus.ERROR,
            output=None,
            error="permission_denied",
            attempts=1,
            latency_ms=100.0,
            confidence=0.0,
            metadata={
                "failure_class": FailureClass.PERMISSION_DENIED.value,
                "failure_severity": FailureSeverity.PERMANENT.value,
                "failure_retryable": False,
                "escalation_key": FailurePolicyRules.PERMISSION_DENIED_ESCALATION,
            },
        )

        context = FailurePolicyContext(approval_gate_available=True)
        decision = context.decide_with_system_context(
            result=result,
            contract=contract,
            current_attempt=1,
            has_fallback_tools=False,
        )

        assert decision.action == PolicyAction.ESCALATE
        assert decision.escalation_category == FailurePolicyRules.PERMISSION_DENIED_ESCALATION


class TestPolicyDecision:
    """Test policy decision structure."""

    def test_policy_decision_with_retry(self):
        """Decision with retry includes delay."""
        decision = PolicyDecision(
            action=PolicyAction.RETRY,
            reason="transient failure",
            retry_delay_seconds=1.5,
            confidence=0.9,
        )
        assert decision.action == PolicyAction.RETRY
        assert decision.retry_delay_seconds == 1.5
        assert decision.confidence == 0.9

    def test_policy_decision_with_escalation(self):
        """Decision with escalation includes category."""
        decision = PolicyDecision(
            action=PolicyAction.ESCALATE,
            reason="permission required",
            escalation_category="permission_denied",
            confidence=0.95,
        )
        assert decision.action == PolicyAction.ESCALATE
        assert decision.escalation_category == "permission_denied"

    def test_policy_decision_metadata(self):
        """Decision metadata preserved."""
        metadata = {"attempt": 2, "reason": "rate_limited"}
        decision = PolicyDecision(
            action=PolicyAction.RETRY,
            reason="rate limited",
            metadata=metadata,
        )
        assert decision.metadata == metadata
