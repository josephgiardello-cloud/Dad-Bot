"""Tests for Phase C: Policy Intermediate Representation.

Tests validate:
1. Policy rules with conditions
2. Effect evaluation and chaining
3. Effect synthesis (output transformation)
4. End-to-end evaluation from ToolResult to PolicyDecisionIR
5. Built-in safety and audit rules
"""

from __future__ import annotations

import pytest

from dadbot.core.policy_ir import (
    PolicyCompilerIR,
    PolicyCondition,
    PolicyDecisionIR,
    PolicyEvaluator,
    PolicyRule,
    PolicyRuleCondition,
    AUDIT_RULE_LOG_ERRORS,
    AUDIT_RULE_LOG_LARGE_OUTPUT,
    SAFETY_RULE_DENY_UNSAFE_TOOLS,
    default_effect_synthesizer,
)
from dadbot.core.runtime_types import (
    CanonicalPayload,
    PolicyEffectType,
    ToolExecutionStatus,
    ToolResult,
)

pytestmark = pytest.mark.unit


class TestPolicyConditions:
    """Test individual policy condition matching."""

    def test_tool_name_match_condition(self):
        condition = PolicyCondition(
            PolicyRuleCondition.TOOL_NAME_MATCH,
            params={"names": ["weather", "calendar"]},
        )

        tool_result = ToolResult(
            tool_name="weather",
            invocation_id="inv-1",
            status=ToolExecutionStatus.OK,
            payload=CanonicalPayload({"temp": 72}, payload_type="result"),
        )
        context = {"tool_result": tool_result}

        assert condition.matches(context) is True

    def test_tool_name_match_condition_not_in_list(self):
        condition = PolicyCondition(
            PolicyRuleCondition.TOOL_NAME_MATCH,
            params={"names": ["weather", "calendar"]},
        )

        tool_result = ToolResult(
            tool_name="api_call",
            invocation_id="inv-1",
            status=ToolExecutionStatus.OK,
            payload=CanonicalPayload({}, payload_type="result"),
        )
        context = {"tool_result": tool_result}

        assert condition.matches(context) is False

    def test_execution_status_match_condition(self):
        condition = PolicyCondition(
            PolicyRuleCondition.EXECUTION_STATUS,
            params={"statuses": [ToolExecutionStatus.ERROR, ToolExecutionStatus.TIMEOUT]},
        )

        tool_result = ToolResult(
            tool_name="test",
            invocation_id="inv-1",
            status=ToolExecutionStatus.TIMEOUT,
            error="Request timeout",
        )
        context = {"tool_result": tool_result}

        assert condition.matches(context) is True

    def test_error_pattern_match_condition(self):
        condition = PolicyCondition(
            PolicyRuleCondition.ERROR_PATTERN,
            params={"pattern": r"timeout|connection.*refused"},
        )

        tool_result = ToolResult(
            tool_name="api",
            invocation_id="inv-1",
            status=ToolExecutionStatus.ERROR,
            error="Connection refused by server",
        )
        context = {"tool_result": tool_result}

        assert condition.matches(context) is True

    def test_error_pattern_no_match(self):
        condition = PolicyCondition(
            PolicyRuleCondition.ERROR_PATTERN,
            params={"pattern": r"timeout"},
        )

        tool_result = ToolResult(
            tool_name="api",
            invocation_id="inv-1",
            status=ToolExecutionStatus.OK,
            payload=CanonicalPayload({}, payload_type="result"),
        )
        context = {"tool_result": tool_result}

        assert condition.matches(context) is False


class TestPolicyRules:
    """Test policy rule evaluation."""

    def test_rule_with_single_condition_matches(self):
        rule = PolicyRule(
            rule_id="test_1",
            rule_name="Test Rule",
            conditions=(
                PolicyCondition(
                    PolicyRuleCondition.TOOL_NAME_MATCH,
                    params={"names": ["weather"]},
                ),
            ),
            effects=(PolicyEffectType.DENY_TOOL,),
        )

        tool_result = ToolResult(
            tool_name="weather",
            invocation_id="inv-1",
            status=ToolExecutionStatus.OK,
        )
        context = {"tool_result": tool_result}

        assert rule.matches(context) is True

    def test_rule_with_multiple_conditions_all_must_match(self):
        rule = PolicyRule(
            rule_id="test_2",
            rule_name="Multi-Condition Rule",
            conditions=(
                PolicyCondition(
                    PolicyRuleCondition.TOOL_NAME_MATCH,
                    params={"names": ["api_call"]},
                ),
                PolicyCondition(
                    PolicyRuleCondition.EXECUTION_STATUS,
                    params={"statuses": [ToolExecutionStatus.ERROR]},
                ),
            ),
            effects=(PolicyEffectType.REQUIRE_APPROVAL,),
        )

        # Both conditions match
        tool_result = ToolResult(
            tool_name="api_call",
            invocation_id="inv-1",
            status=ToolExecutionStatus.ERROR,
            error="API error",
        )
        context = {"tool_result": tool_result}
        assert rule.matches(context) is True

        # Only first condition matches
        tool_result_ok = ToolResult(
            tool_name="api_call",
            invocation_id="inv-2",
            status=ToolExecutionStatus.OK,
        )
        context_ok = {"tool_result": tool_result_ok}
        assert rule.matches(context_ok) is False

    def test_rule_with_no_conditions_always_matches(self):
        rule = PolicyRule(
            rule_id="always",
            rule_name="Always Match",
            conditions=(),
            effects=(PolicyEffectType.REQUIRE_APPROVAL,),
        )

        tool_result = ToolResult(
            tool_name="anything",
            invocation_id="inv-1",
            status=ToolExecutionStatus.OK,
        )
        context = {"tool_result": tool_result}

        assert rule.matches(context) is True


class TestPolicyEvaluator:
    """Test policy evaluator and effect emission."""

    def test_evaluator_emits_effects_for_matching_rules(self):
        rules = [
            PolicyRule(
                rule_id="rule1",
                rule_name="First Rule",
                conditions=(
                    PolicyCondition(
                        PolicyRuleCondition.TOOL_NAME_MATCH,
                        params={"names": ["test_tool"]},
                    ),
                ),
                effects=(PolicyEffectType.DENY_TOOL,),
                priority=50,
            ),
        ]

        evaluator = PolicyEvaluator(rules)

        tool_result = ToolResult(
            tool_name="test_tool",
            invocation_id="inv-1",
            status=ToolExecutionStatus.OK,
        )

        effects = evaluator.evaluate(tool_result)

        assert len(effects) == 1
        assert effects[0].effect_type == PolicyEffectType.DENY_TOOL
        assert effects[0].source_rule == "rule1"

    def test_evaluator_respects_priority_order(self):
        rules = [
            PolicyRule(
                rule_id="low_priority",
                rule_name="Low Priority Rule",
                conditions=(),
                effects=(PolicyEffectType.STRIP_FACET,),
                priority=10,
            ),
            PolicyRule(
                rule_id="high_priority",
                rule_name="High Priority Rule",
                conditions=(),
                effects=(PolicyEffectType.DENY_TOOL,),
                priority=90,
            ),
        ]

        evaluator = PolicyEvaluator(rules)
        tool_result = ToolResult(
            tool_name="test",
            invocation_id="inv-1",
            status=ToolExecutionStatus.OK,
        )

        effects = evaluator.evaluate(tool_result)

        # High priority rule should be evaluated first
        assert len(effects) == 2
        assert effects[0].source_rule == "high_priority"
        assert effects[1].source_rule == "low_priority"

    def test_evaluator_no_matching_rules(self):
        rules = [
            PolicyRule(
                rule_id="never_match",
                rule_name="Never Matches",
                conditions=(
                    PolicyCondition(
                        PolicyRuleCondition.TOOL_NAME_MATCH,
                        params={"names": ["nonexistent"]},
                    ),
                ),
                effects=(PolicyEffectType.DENY_TOOL,),
            ),
        ]

        evaluator = PolicyEvaluator(rules)
        tool_result = ToolResult(
            tool_name="other_tool",
            invocation_id="inv-1",
            status=ToolExecutionStatus.OK,
        )

        effects = evaluator.evaluate(tool_result)

        assert len(effects) == 0


class TestEffectSynthesis:
    """Test applying policy effects to tool results."""

    def test_deny_tool_effect(self):
        tool_result = ToolResult(
            tool_name="test",
            invocation_id="inv-1",
            status=ToolExecutionStatus.OK,
            payload=CanonicalPayload({"data": "sensitive"}, payload_type="output"),
        )

        rule = PolicyRule(
            rule_id="deny",
            rule_name="Deny Tool",
            conditions=(),
            effects=(PolicyEffectType.DENY_TOOL,),
        )

        evaluator = PolicyEvaluator([rule])
        effects = evaluator.evaluate(tool_result)

        output, was_modified = default_effect_synthesizer(tool_result, effects)

        assert was_modified is True
        assert "error" in output
        assert output["error"] == "Tool invocation denied by policy"

    def test_strip_facet_effect(self):
        tool_result = ToolResult(
            tool_name="test",
            invocation_id="inv-1",
            status=ToolExecutionStatus.OK,
            payload=CanonicalPayload(
                {"answer": 42, "sarcasm": "yeah right"},
                payload_type="output",
            ),
        )

        rule = PolicyRule(
            rule_id="strip",
            rule_name="Strip Sarcasm",
            conditions=(),
            effects=(PolicyEffectType.STRIP_FACET,),
        )

        evaluator = PolicyEvaluator([rule])
        effects = evaluator.evaluate(tool_result)

        output, was_modified = default_effect_synthesizer(tool_result, effects)

        assert was_modified is True
        assert "sarcasm" not in output
        assert output["answer"] == 42

    def test_force_degradation_effect(self):
        tool_result = ToolResult(
            tool_name="test",
            invocation_id="inv-1",
            status=ToolExecutionStatus.OK,
            payload=CanonicalPayload({"rich_output": True}, payload_type="output"),
        )

        rule = PolicyRule(
            rule_id="degrade",
            rule_name="Degrade Output",
            conditions=(),
            effects=(PolicyEffectType.FORCE_DEGRADATION,),
        )

        evaluator = PolicyEvaluator([rule])
        effects = evaluator.evaluate(tool_result)

        output, was_modified = default_effect_synthesizer(tool_result, effects)

        assert was_modified is True
        assert output["degraded"] is True


class TestPolicyCompilerIR:
    """Test end-to-end policy evaluation."""

    def test_compiler_evaluates_tool_result_end_to_end(self):
        rules = [
            PolicyRule(
                rule_id="safety_1",
                rule_name="Deny unsafe tools",
                conditions=(
                    PolicyCondition(
                        PolicyRuleCondition.TOOL_NAME_MATCH,
                        params={"names": ["exec", "eval"]},
                    ),
                ),
                effects=(PolicyEffectType.DENY_TOOL,),
                priority=90,
            ),
        ]

        compiler = PolicyCompilerIR(rules)

        tool_result = ToolResult(
            tool_name="exec",
            invocation_id="inv-1",
            status=ToolExecutionStatus.OK,
            payload=CanonicalPayload({"code": "print('hi')"}, payload_type="output"),
        )

        decision = compiler.evaluate_with_effects(tool_result)

        assert isinstance(decision, PolicyDecisionIR)
        assert decision.tool_result is tool_result
        assert decision.output_was_modified is True
        assert "error" in decision.final_output
        assert len(decision.emitted_effects) == 1

    def test_compiler_multiple_effects_applied_in_order(self):
        rules = [
            PolicyRule(
                rule_id="first",
                rule_name="First Effect",
                conditions=(),
                effects=(PolicyEffectType.STRIP_FACET,),
                priority=60,
            ),
            PolicyRule(
                rule_id="second",
                rule_name="Second Effect",
                conditions=(),
                effects=(PolicyEffectType.FORCE_DEGRADATION,),
                priority=50,
            ),
        ]

        compiler = PolicyCompilerIR(rules)

        tool_result = ToolResult(
            tool_name="test",
            invocation_id="inv-1",
            status=ToolExecutionStatus.OK,
            payload=CanonicalPayload(
                {"answer": 42, "sarcasm": "yeah"},
                payload_type="output",
            ),
        )

        decision = compiler.evaluate_with_effects(tool_result)

        # Both effects should be emitted
        assert len(decision.emitted_effects) == 2
        assert decision.emitted_effects[0].source_rule == "first"
        assert decision.emitted_effects[1].source_rule == "second"
        # Final output should have effects applied
        assert "degraded" in decision.final_output

    def test_compiler_effect_chain_summary(self):
        rules = [
            PolicyRule(
                rule_id="deny",
                rule_name="Deny",
                conditions=(),
                effects=(PolicyEffectType.DENY_TOOL,),
            ),
        ]

        compiler = PolicyCompilerIR(rules)
        tool_result = ToolResult(
            tool_name="test",
            invocation_id="inv-1",
            status=ToolExecutionStatus.OK,
        )

        decision = compiler.evaluate_with_effects(tool_result)

        summary = decision.effect_chain_summary()
        assert "deny_tool" in summary


class TestBuiltInRules:
    """Test built-in safety and audit rules."""

    def test_safety_rule_deny_unsafe_tools(self):
        tool_result = ToolResult(
            tool_name="exec",
            invocation_id="inv-1",
            status=ToolExecutionStatus.OK,
        )

        assert SAFETY_RULE_DENY_UNSAFE_TOOLS.matches({"tool_result": tool_result})

    def test_safety_rule_deny_other_tools(self):
        tool_result = ToolResult(
            tool_name="weather",
            invocation_id="inv-1",
            status=ToolExecutionStatus.OK,
        )

        assert not SAFETY_RULE_DENY_UNSAFE_TOOLS.matches({"tool_result": tool_result})

    def test_audit_rule_log_errors(self):
        tool_result = ToolResult(
            tool_name="test",
            invocation_id="inv-1",
            status=ToolExecutionStatus.ERROR,
            error="Test error",
        )

        assert AUDIT_RULE_LOG_ERRORS.matches({"tool_result": tool_result})

    def test_audit_rule_log_errors_no_match_on_success(self):
        tool_result = ToolResult(
            tool_name="test",
            invocation_id="inv-1",
            status=ToolExecutionStatus.OK,
        )

        assert not AUDIT_RULE_LOG_ERRORS.matches({"tool_result": tool_result})
