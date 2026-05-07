"""Integration test: Policy IR (Phase C) → Recovery IR (Phase D).

Validates end-to-end flow:
1. ToolRegistry produces ToolResult
2. PolicyCompilerIR produces PolicyDecisionIR
3. RecoverySelector produces RecoveryDecision
4. Full audit trail preserved

Proves Phases C and D integrate correctly.
"""

from __future__ import annotations

import pytest

from dadbot.core.policy_ir import (
    PolicyCompilerIR,
    PolicyCondition,
    PolicyRule,
    PolicyRuleCondition,
)
from dadbot.core.recovery_ir import (
    RecoveryChain,
    RecoveryContext,
    RecoverySelector,
    RecoveryStrategy,
)
from dadbot.core.runtime_types import (
    CanonicalPayload,
    ExecutionIdentity,
    PolicyEffectType,
    ToolDeterminismClass,
    ToolExecutionStatus,
    ToolInvocation,
    ToolResult,
    ToolSideEffectClass,
    ToolSpec,
)
from dadbot.core.tool_registry import ToolRegistry

pytestmark = pytest.mark.unit


@pytest.fixture
def registry_with_tools():
    """Create registry with success and failure tools."""
    reg = ToolRegistry()

    # Success tool
    success_spec = ToolSpec(
        name="success_tool",
        version="1.0.0",
        determinism=ToolDeterminismClass.READ_ONLY,
        side_effect_class=ToolSideEffectClass.PURE,
        capabilities=frozenset({"output"}),
    )

    def success_executor(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(
            tool_name="success_tool",
            invocation_id=invocation.invocation_id,
            status=ToolExecutionStatus.OK,
            payload=CanonicalPayload({"result": "success"}, payload_type="result"),
        )

    reg.register(success_spec, success_executor)

    # Timeout tool
    timeout_spec = ToolSpec(
        name="timeout_tool",
        version="1.0.0",
        determinism=ToolDeterminismClass.DETERMINISTIC,
        side_effect_class=ToolSideEffectClass.STATEFUL,
        capabilities=frozenset({"external_call"}),
    )

    def timeout_executor(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(
            tool_name="timeout_tool",
            invocation_id=invocation.invocation_id,
            status=ToolExecutionStatus.TIMEOUT,
            error="Request timed out",
        )

    reg.register(timeout_spec, timeout_executor)

    return reg


@pytest.fixture
def policy_compiler():
    """Create policy compiler with audit rules."""
    rules = [
        PolicyRule(
            rule_id="audit_errors",
            rule_name="Audit Tool Errors",
            conditions=(
                PolicyCondition(
                    PolicyRuleCondition.EXECUTION_STATUS,
                    params={"statuses": [ToolExecutionStatus.ERROR]},
                ),
            ),
            effects=(PolicyEffectType.REQUIRE_APPROVAL,),
            priority=80,
        ),
        PolicyRule(
            rule_id="audit_timeout",
            rule_name="Audit Timeouts",
            conditions=(
                PolicyCondition(
                    PolicyRuleCondition.EXECUTION_STATUS,
                    params={"statuses": [ToolExecutionStatus.TIMEOUT]},
                ),
            ),
            effects=(PolicyEffectType.REQUIRE_APPROVAL,),
            priority=70,
        ),
    ]
    return PolicyCompilerIR(rules)


def test_phase_c_d_success_flow(registry_with_tools, policy_compiler):
    """Full flow: success tool → no policies → no recovery needed."""
    # Phase B: Execute tool through registry
    spec, executor = registry_with_tools.resolve("success_tool")
    caller = ExecutionIdentity(
        caller_trace_id="trace-1",
        caller_role="user",
        caller_context="test",
    )
    invocation = ToolInvocation(
        invocation_id="inv-1",
        tool_spec=spec,
        arguments={},
        caller=caller,
    )
    tool_result = executor(invocation)

    # Phase C: Flow through policy compiler
    policy_decision = policy_compiler.evaluate_with_effects(tool_result)

    # Phase D: Select recovery
    context = RecoveryContext(
        tool_result=tool_result,
        policy_decision=policy_decision,
    )
    selector = RecoverySelector()
    recovery_decision = selector.select(context)

    # Validate flow
    assert tool_result.status == ToolExecutionStatus.OK
    assert len(policy_decision.emitted_effects) == 0
    assert recovery_decision.strategy == RecoveryStrategy.DEGRADE_GRACEFULLY


def test_phase_c_d_timeout_flow(registry_with_tools, policy_compiler):
    """Full flow: timeout tool → policy audit → recovery retry."""
    # Phase B: Execute timeout tool
    spec, executor = registry_with_tools.resolve("timeout_tool")
    caller = ExecutionIdentity(
        caller_trace_id="trace-2",
        caller_role="user",
        caller_context="test",
    )
    invocation = ToolInvocation(
        invocation_id="inv-2",
        tool_spec=spec,
        arguments={},
        caller=caller,
    )
    tool_result = executor(invocation)

    # Phase C: Flow through policy compiler
    policy_decision = policy_compiler.evaluate_with_effects(tool_result)

    # Phase D: Select recovery
    context = RecoveryContext(
        tool_result=tool_result,
        policy_decision=policy_decision,
        retry_count=0,
    )
    selector = RecoverySelector()
    recovery_decision = selector.select(context)

    # Validate flow
    assert tool_result.status == ToolExecutionStatus.TIMEOUT
    # Policy should trigger audit rule
    assert len(policy_decision.emitted_effects) > 0
    # Recovery should retry
    assert recovery_decision.strategy == RecoveryStrategy.RETRY_SAME


def test_phase_c_d_retry_exhaustion_flow(registry_with_tools, policy_compiler):
    """Full flow: timeout → multiple retries → degrade gracefully."""
    spec, executor = registry_with_tools.resolve("timeout_tool")
    caller = ExecutionIdentity(
        caller_trace_id="trace-3",
        caller_role="user",
        caller_context="test",
    )

    # Execute tool
    invocation = ToolInvocation(
        invocation_id="inv-3",
        tool_spec=spec,
        arguments={},
        caller=caller,
    )
    tool_result = executor(invocation)

    # Evaluate policy
    policy_decision = policy_compiler.evaluate_with_effects(tool_result)

    # Recovery chain with multiple attempts
    chain = RecoveryChain()

    # First N attempts: retry
    for i in range(RecoverySelector.MAX_RETRIES):
        context = RecoveryContext(
            tool_result=tool_result,
            policy_decision=policy_decision,
            retry_count=i,
        )
        decision = chain.attempt(context)
        assert decision.strategy == RecoveryStrategy.RETRY_SAME

    # Final attempt: degrade gracefully
    final_context = RecoveryContext(
        tool_result=tool_result,
        policy_decision=policy_decision,
        retry_count=RecoverySelector.MAX_RETRIES,
    )
    final_decision = chain.attempt(final_context)

    assert final_decision.strategy == RecoveryStrategy.DEGRADE_GRACEFULLY
    assert chain.recovery_summary()["total_decisions"] == RecoverySelector.MAX_RETRIES + 1


def test_phase_c_d_audit_trail_preservation(registry_with_tools, policy_compiler):
    """Audit trail preserved through policy → recovery pipeline."""
    # Execute tool
    spec, executor = registry_with_tools.resolve("timeout_tool")
    caller = ExecutionIdentity(
        caller_trace_id="trace-4",
        caller_role="admin",
        caller_context="audit_test",
    )
    invocation = ToolInvocation(
        invocation_id="inv-4",
        tool_spec=spec,
        arguments={},
        caller=caller,
    )
    tool_result = executor(invocation)

    # Phase C: Policies matched and effects emitted
    context = {
        "tool_result": tool_result,
        "tool_spec": spec,
        "caller_identity": caller,
    }
    policy_decision = policy_compiler.evaluate_with_effects(tool_result, context)

    # Phase D: Recovery decision preserves policy context
    context = RecoveryContext(
        tool_result=tool_result,
        policy_decision=policy_decision,
    )
    selector = RecoverySelector()
    recovery_decision = selector.select(context)

    # Audit trail should be complete
    assert len(policy_decision.matched_rules) > 0
    assert recovery_decision.matched_rules == policy_decision.matched_rules
    assert recovery_decision.tool_result is tool_result
    assert recovery_decision.reason != ""
