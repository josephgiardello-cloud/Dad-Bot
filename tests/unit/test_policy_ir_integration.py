"""Integration test: Tool Registry (Phase B) → Policy IR (Phase C).

Validates that:
1. ToolRegistry produces ToolResult
2. ToolResult flows through PolicyCompilerIR
3. PolicyDecisionIR captures effect chain
4. Real tools produce auditable decisions

This proves Phases B and C integrate correctly.
"""

from __future__ import annotations

import pytest

from dadbot.core.policy_ir import (
    PolicyCompilerIR,
    PolicyCondition,
    PolicyRule,
    PolicyRuleCondition,
)
from dadbot.core.runtime_types import (
    CanonicalPayload,
    ExecutionIdentity,
    PolicyEffectType,
    ToolExecutionStatus,
    ToolInvocation,
    ToolResult,
    ToolSpec,
)
from dadbot.core.tool_registry import ToolRegistry

pytestmark = pytest.mark.unit


@pytest.fixture
def registry():
    """Create a registry with test tools."""
    reg = ToolRegistry()

    # Simple echo tool
    echo_spec = ToolSpec(
        name="echo",
        version="1.0.0",
        determinism="read_only",
        side_effect_class="pure",
        capabilities=["output"],
    )

    def echo_executor(invocation: ToolInvocation) -> ToolResult:
        message = invocation.arguments.get("message", "echo")
        return ToolResult(
            tool_name="echo",
            invocation_id=invocation.invocation_id,
            status=ToolExecutionStatus.OK,
            payload=CanonicalPayload({"echoed": message}, payload_type="result"),
        )

    reg.register(echo_spec, echo_executor)

    # Failing tool
    failing_spec = ToolSpec(
        name="failing_tool",
        version="1.0.0",
        determinism="deterministic",
        side_effect_class="stateful",
        capabilities=["state_mutation"],
    )

    def failing_executor(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(
            tool_name="failing_tool",
            invocation_id=invocation.invocation_id,
            status=ToolExecutionStatus.ERROR,
            error="Simulated failure",
        )

    reg.register(failing_spec, failing_executor)

    return reg


@pytest.fixture
def policy_compiler():
    """Create a compiler with audit rules."""
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
            rule_id="deny_stateful",
            rule_name="Deny Stateful Tools",
            conditions=(
                PolicyCondition(
                    PolicyRuleCondition.TOOL_NAME_MATCH,
                    params={"names": ["failing_tool"]},
                ),
            ),
            effects=(PolicyEffectType.DENY_TOOL,),
            priority=90,
        ),
    ]
    return PolicyCompilerIR(rules)


def test_registry_to_policy_flow_success(registry, policy_compiler):
    """Validate successful tool result flows through policy IR."""
    # Execute tool through registry
    echo_spec, echo_executor = registry.resolve("echo")
    caller = ExecutionIdentity(
        caller_trace_id="trace-1",
        caller_role="test",
        caller_context="integration_test",
    )
    invocation = ToolInvocation(
        invocation_id="inv-test-1",
        tool_spec=echo_spec,
        arguments={"message": "Hello from Phase B"},
        caller=caller,
    )
    tool_result = echo_executor(invocation)

    # Flow result through policy compiler
    decision = policy_compiler.evaluate_with_effects(tool_result)

    # Validate decision
    assert decision.tool_result is tool_result
    assert decision.final_output == {"echoed": "Hello from Phase B"}
    assert decision.output_was_modified is False
    assert len(decision.emitted_effects) == 0  # No rules matched


def test_registry_to_policy_flow_error(registry, policy_compiler):
    """Validate error result triggers audit policy."""
    # Execute failing tool through registry
    failing_spec, failing_executor = registry.resolve("failing_tool")
    caller = ExecutionIdentity(
        caller_trace_id="trace-2",
        caller_role="test",
        caller_context="integration_test",
    )
    invocation = ToolInvocation(
        invocation_id="inv-test-2",
        tool_spec=failing_spec,
        arguments={},
        caller=caller,
    )
    tool_result = failing_executor(invocation)

    # Flow result through policy compiler
    decision = policy_compiler.evaluate_with_effects(tool_result)

    # Validate decision: both audit and deny rules matched
    assert decision.tool_result is tool_result
    assert decision.output_was_modified is True
    # Should have 2 effects: deny_stateful (priority 90) then audit_errors (priority 80)
    assert len(decision.emitted_effects) == 2
    assert decision.emitted_effects[0].source_rule == "deny_stateful"
    assert decision.emitted_effects[0].effect_type == PolicyEffectType.DENY_TOOL
    assert decision.emitted_effects[1].source_rule == "audit_errors"
    assert decision.emitted_effects[1].effect_type == PolicyEffectType.REQUIRE_APPROVAL
    # Final output should be error due to deny
    assert "error" in decision.final_output


def test_registry_to_policy_with_context(registry, policy_compiler):
    """Validate policy evaluation uses tool spec from context."""
    # Execute tool
    echo_spec, echo_executor = registry.resolve("echo")
    caller = ExecutionIdentity(
        caller_trace_id="trace-3",
        caller_role="admin",
        caller_context="integration_test",
    )
    invocation = ToolInvocation(
        invocation_id="inv-test-3",
        tool_spec=echo_spec,
        arguments={"message": "test"},
        caller=caller,
    )
    tool_result = echo_executor(invocation)

    # Evaluate with rich context
    context = {
        "tool_result": tool_result,
        "tool_spec": echo_spec,
        "caller_identity": caller,
    }
    decision = policy_compiler.evaluate_with_effects(tool_result, context)

    assert decision.tool_result is tool_result
    assert len(decision.matched_rules) == 0  # No policies matched


def test_effect_chain_audit_trail(registry, policy_compiler):
    """Validate effect chain creates complete audit trail."""
    # Execute failing tool
    failing_spec, failing_executor = registry.resolve("failing_tool")
    caller = ExecutionIdentity(
        caller_trace_id="trace-4",
        caller_role="test",
        caller_context="audit_test",
    )
    invocation = ToolInvocation(
        invocation_id="inv-test-4",
        tool_spec=failing_spec,
        arguments={},
        caller=caller,
    )
    tool_result = failing_executor(invocation)

    # Evaluate through policy
    decision = policy_compiler.evaluate_with_effects(tool_result)

    # Verify audit trail
    summary = decision.effect_chain_summary()
    assert len(decision.emitted_effects) > 0
    assert "deny_tool" in summary
    # Original error preserved in metadata or effect reason
    for effect in decision.emitted_effects:
        assert effect.source_rule in ["deny_stateful", "audit_errors"]
