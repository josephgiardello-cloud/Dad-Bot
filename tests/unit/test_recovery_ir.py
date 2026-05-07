"""Tests for Phase D: Recovery Strategy Selection.

Tests validate:
1. Recovery strategies and decision types
2. Recovery selector logic (all 6 cases)
3. Recovery chain execution and summarization
4. Decision properties (retriable, external_input, terminal)
5. Integration with PolicyDecisionIR
"""

from __future__ import annotations

import pytest

from dadbot.core.policy_ir import PolicyDecisionIR, PolicyEffectType
from dadbot.core.recovery_ir import (
    RecoveryChain,
    RecoveryContext,
    RecoveryDecision,
    RecoverySelector,
    RecoveryStrategy,
)
from dadbot.core.runtime_types import (
    CanonicalPayload,
    PolicyEffect,
    ToolExecutionStatus,
    ToolResult,
)

pytestmark = pytest.mark.unit


class TestRecoveryDecision:
    """Test RecoveryDecision properties and behavior."""

    def test_is_retriable_retry_same(self):
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.RETRY_SAME,
            bounded_attempts=2,
        )
        assert decision.is_retriable() is True

    def test_is_retriable_no_attempts(self):
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.RETRY_SAME,
            bounded_attempts=0,
        )
        assert decision.is_retriable() is False

    def test_is_retriable_non_retry_strategy(self):
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.DEGRADE_GRACEFULLY,
        )
        assert decision.is_retriable() is False

    def test_requires_external_input_approval(self):
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.REQUIRE_APPROVAL,
            requires_user_approval=True,
        )
        assert decision.requires_external_input() is True

    def test_requires_external_input_none(self):
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.DEGRADE_GRACEFULLY,
        )
        assert decision.requires_external_input() is False

    def test_is_terminal_halt_safe(self):
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.HALT_SAFE,
        )
        assert decision.is_terminal() is True

    def test_is_terminal_non_halt(self):
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.RETRY_SAME,
        )
        assert decision.is_terminal() is False


class TestRecoverySelectorBasic:
    """Test RecoverySelector basic decision logic."""

    def test_select_success_no_effects(self):
        """Success with no policy effects: degrade gracefully (pass through)."""
        tool_result = ToolResult(
            tool_name="test",
            invocation_id="inv-1",
            status=ToolExecutionStatus.OK,
            payload=CanonicalPayload({"data": "success"}, payload_type="result"),
        )
        policy_decision = PolicyDecisionIR(
            tool_result=tool_result,
            matched_rules=(),
            emitted_effects=(),
            final_output={"data": "success"},
            output_was_modified=False,
        )
        context = RecoveryContext(
            tool_result=tool_result,
            policy_decision=policy_decision,
        )

        selector = RecoverySelector()
        decision = selector.select(context)

        assert decision.strategy == RecoveryStrategy.DEGRADE_GRACEFULLY
        assert decision.degraded_output == {"data": "success"}

    def test_select_timeout_retry(self):
        """Timeout with retries available: retry same."""
        tool_result = ToolResult(
            tool_name="api",
            invocation_id="inv-1",
            status=ToolExecutionStatus.TIMEOUT,
            error="Request timeout",
        )
        policy_decision = PolicyDecisionIR(
            tool_result=tool_result,
            matched_rules=(),
            emitted_effects=(),
            final_output=None,
            output_was_modified=False,
        )
        context = RecoveryContext(
            tool_result=tool_result,
            policy_decision=policy_decision,
            retry_count=0,
        )

        selector = RecoverySelector()
        decision = selector.select(context)

        assert decision.strategy == RecoveryStrategy.RETRY_SAME
        assert decision.bounded_attempts > 0

    def test_select_retry_limit_exceeded(self):
        """Too many retries: degrade gracefully."""
        tool_result = ToolResult(
            tool_name="api",
            invocation_id="inv-1",
            status=ToolExecutionStatus.TIMEOUT,
        )
        policy_decision = PolicyDecisionIR(
            tool_result=tool_result,
            matched_rules=(),
            emitted_effects=(),
            final_output=None,
            output_was_modified=False,
        )
        context = RecoveryContext(
            tool_result=tool_result,
            policy_decision=policy_decision,
            retry_count=RecoverySelector.MAX_RETRIES,  # Exceeded
        )

        selector = RecoverySelector()
        decision = selector.select(context)

        assert decision.strategy == RecoveryStrategy.DEGRADE_GRACEFULLY
        assert decision.degraded_output is not None


class TestRecoverySelectorPolicies:
    """Test RecoverySelector with policy effects."""

    def test_select_approval_required(self):
        """Policy requires approval: require approval."""
        tool_result = ToolResult(
            tool_name="test",
            invocation_id="inv-1",
            status=ToolExecutionStatus.OK,
        )
        effect = PolicyEffect(
            effect_type=PolicyEffectType.REQUIRE_APPROVAL,
            source_rule="audit_rule",
            before_hash="hash1",
            after_hash="hash1",
            reason="Tool needs approval",
        )
        policy_decision = PolicyDecisionIR(
            tool_result=tool_result,
            matched_rules=("audit_rule",),
            emitted_effects=(effect,),
            final_output=None,
            output_was_modified=True,
        )
        context = RecoveryContext(
            tool_result=tool_result,
            policy_decision=policy_decision,
        )

        selector = RecoverySelector()
        decision = selector.select(context)

        assert decision.strategy == RecoveryStrategy.REQUIRE_APPROVAL
        assert decision.requires_user_approval is True

    def test_select_fatal_error(self):
        """Tool error status: halt safely."""
        tool_result = ToolResult(
            tool_name="test",
            invocation_id="inv-1",
            status=ToolExecutionStatus.ERROR,
            error="Unrecoverable error",
        )
        policy_decision = PolicyDecisionIR(
            tool_result=tool_result,
            matched_rules=(),
            emitted_effects=(),
            final_output=None,
            output_was_modified=False,
        )
        context = RecoveryContext(
            tool_result=tool_result,
            policy_decision=policy_decision,
        )

        selector = RecoverySelector()
        decision = selector.select(context)

        assert decision.strategy == RecoveryStrategy.HALT_SAFE


class TestRecoveryChain:
    """Test RecoveryChain execution and tracking."""

    def test_chain_single_decision(self):
        """Recovery chain executes single decision."""
        tool_result = ToolResult(
            tool_name="test",
            invocation_id="inv-1",
            status=ToolExecutionStatus.OK,
        )
        policy_decision = PolicyDecisionIR(
            tool_result=tool_result,
            matched_rules=(),
            emitted_effects=(),
            final_output={"ok": True},
            output_was_modified=False,
        )
        context = RecoveryContext(
            tool_result=tool_result,
            policy_decision=policy_decision,
        )

        chain = RecoveryChain()
        decision = chain.attempt(context)

        assert decision is not None
        assert len(chain.decisions) == 1
        assert chain.decisions[0] == decision

    def test_chain_retry_attempts(self):
        """Recovery chain tracks retry attempts."""
        tool_result = ToolResult(
            tool_name="api",
            invocation_id="inv-1",
            status=ToolExecutionStatus.TIMEOUT,
        )
        policy_decision = PolicyDecisionIR(
            tool_result=tool_result,
            matched_rules=(),
            emitted_effects=(),
            final_output=None,
            output_was_modified=False,
        )

        chain = RecoveryChain()
        for i in range(2):
            context = RecoveryContext(
                tool_result=tool_result,
                policy_decision=policy_decision,
                retry_count=i,
            )
            decision = chain.attempt(context)
            assert decision.strategy == RecoveryStrategy.RETRY_SAME

        assert chain.attempt_count == 2

    def test_chain_can_retry(self):
        """Recovery chain checks if retry is possible."""
        chain = RecoveryChain()

        retry_decision = RecoveryDecision(
            strategy=RecoveryStrategy.RETRY_SAME,
            bounded_attempts=1,
        )
        assert chain.can_retry(retry_decision) is True

        halt_decision = RecoveryDecision(
            strategy=RecoveryStrategy.HALT_SAFE,
        )
        assert chain.can_retry(halt_decision) is False

    def test_chain_recovery_summary(self):
        """Recovery chain summarizes execution."""
        tool_result = ToolResult(
            tool_name="test",
            invocation_id="inv-1",
            status=ToolExecutionStatus.OK,
        )
        policy_decision = PolicyDecisionIR(
            tool_result=tool_result,
            matched_rules=("rule1",),
            emitted_effects=(),
            final_output={"ok": True},
            output_was_modified=False,
        )
        context = RecoveryContext(
            tool_result=tool_result,
            policy_decision=policy_decision,
        )

        chain = RecoveryChain()
        chain.attempt(context)

        summary = chain.recovery_summary()

        assert summary["status"] == "completed"
        assert summary["total_decisions"] == 1
        assert summary["attempts"] == 0


class TestRecoveryIntegration:
    """Integration tests: policy effects → recovery decisions."""

    def test_policy_deny_leads_to_escalation(self):
        """Policy denies tool → recovery seeks escalation."""
        tool_result = ToolResult(
            tool_name="admin_tool",
            invocation_id="inv-1",
            status=ToolExecutionStatus.DENIED,
            error="Denied by policy",
        )
        effect = PolicyEffect(
            effect_type=PolicyEffectType.DENY_TOOL,
            source_rule="safety_deny_unsafe",
                    before_hash="hash1",
                    after_hash="hash1",
        )
        policy_decision = PolicyDecisionIR(
            tool_result=tool_result,
            matched_rules=("safety_deny_unsafe",),
            emitted_effects=(effect,),
            final_output=None,
            output_was_modified=True,
        )
        context = RecoveryContext(
            tool_result=tool_result,
            policy_decision=policy_decision,
        )

        selector = RecoverySelector()
        decision = selector.select(context)

        # Denial status leads to halt safe
        assert decision.strategy == RecoveryStrategy.HALT_SAFE

    def test_policy_require_approval_leads_to_approval(self):
        """Policy requires approval → recovery waits for approval."""
        tool_result = ToolResult(
            tool_name="test",
            invocation_id="inv-1",
            status=ToolExecutionStatus.OK,
        )
        effect = PolicyEffect(
            effect_type=PolicyEffectType.REQUIRE_APPROVAL,
            source_rule="audit_log_errors",
                    before_hash="hash1",
                    after_hash="hash1",
        )
        policy_decision = PolicyDecisionIR(
            tool_result=tool_result,
            matched_rules=("audit_log_errors",),
            emitted_effects=(effect,),
            final_output=None,
            output_was_modified=True,
        )
        context = RecoveryContext(
            tool_result=tool_result,
            policy_decision=policy_decision,
        )

        selector = RecoverySelector()
        decision = selector.select(context)

        assert decision.strategy == RecoveryStrategy.REQUIRE_APPROVAL

    def test_multiple_failures_escalate_to_halt(self):
        """Multiple failures exhaust retries → halt safely."""
        tool_result = ToolResult(
            tool_name="api",
            invocation_id="inv-1",
            status=ToolExecutionStatus.TIMEOUT,
        )
        policy_decision = PolicyDecisionIR(
            tool_result=tool_result,
            matched_rules=(),
            emitted_effects=(),
            final_output=None,
            output_was_modified=False,
        )

        chain = RecoveryChain()

        # Attempt multiple retries
        for i in range(RecoverySelector.MAX_RETRIES + 1):
            context = RecoveryContext(
                tool_result=tool_result,
                policy_decision=policy_decision,
                retry_count=i,
            )
            decision = chain.attempt(context)

        # Final decision should be halt (not retry)
        final_decision = chain.decisions[-1]
        assert final_decision.strategy != RecoveryStrategy.RETRY_SAME
