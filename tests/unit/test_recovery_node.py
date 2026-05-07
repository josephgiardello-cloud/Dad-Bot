"""Tests for Phase E RecoveryNode execution in pipeline context."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from dadbot.core.graph_context import TurnContext
from dadbot.core.graph_pipeline_nodes import RecoveryNode
from dadbot.core.policy_ir import PolicyDecisionIR
from dadbot.core.runtime_types import (
    PolicyEffect,
    PolicyEffectType,
    ToolExecutionStatus,
    ToolResult,
)

pytestmark = pytest.mark.unit


@dataclass
class _Registry:
    def get(self, _name: str):
        return None


def _policy_decision(
    *,
    status: ToolExecutionStatus,
    final_output: object = None,
    effects: tuple[PolicyEffect, ...] = (),
) -> PolicyDecisionIR:
    tool_result = ToolResult(
        tool_name="tool",
        invocation_id="inv-1",
        status=status,
        payload=None,
        error="error" if status == ToolExecutionStatus.ERROR else "",
    )
    return PolicyDecisionIR(
        tool_result=tool_result,
        matched_rules=("rule",),
        emitted_effects=effects,
        final_output=final_output,
        output_was_modified=False,
    )


@pytest.mark.asyncio
async def test_recovery_node_pass_through_on_success() -> None:
    node = RecoveryNode()
    ctx = TurnContext(user_input="hello")
    ctx.state["safe_result"] = "ok"
    ctx.state["safety_policy_decision"] = {"action": "passthrough"}

    await node.execute(_Registry(), ctx)

    assert ctx.state["recovery_strategy"] == "degrade_gracefully"
    assert ctx.state["safe_result"] == "ok"
    assert ctx.state["recovery_action"]["action_type"] == "return_degraded"


@pytest.mark.asyncio
async def test_recovery_node_retry_sets_retry_requested() -> None:
    node = RecoveryNode()
    ctx = TurnContext(user_input="hello")
    ctx.state["safe_result"] = "stale"
    ctx.state["policy_decision_ir"] = _policy_decision(
        status=ToolExecutionStatus.TIMEOUT,
        final_output="stale",
    )

    await node.execute(_Registry(), ctx)

    assert ctx.state["recovery_strategy"] == "retry_same"
    assert ctx.state["recovery_retry_requested"] is True
    assert ctx.state["recovery_retry_count"] == 1


@pytest.mark.asyncio
async def test_recovery_node_retry_exhausted_degrades_output() -> None:
    node = RecoveryNode()
    ctx = TurnContext(user_input="hello")
    ctx.state["safe_result"] = "initial"
    ctx.state["recovery_retry_count"] = 3
    ctx.state["policy_decision_ir"] = _policy_decision(
        status=ToolExecutionStatus.TIMEOUT,
        final_output="initial",
    )

    await node.execute(_Registry(), ctx)

    assert ctx.state["recovery_strategy"] == "degrade_gracefully"
    assert ctx.state["safe_result"] == {"error": "Tool unavailable after retries", "degraded": True}


@pytest.mark.asyncio
async def test_recovery_node_requires_approval_sets_prompt() -> None:
    node = RecoveryNode()
    ctx = TurnContext(user_input="hello")
    ctx.state["safe_result"] = "pending"
    approval_effect = PolicyEffect(
        effect_type=PolicyEffectType.REQUIRE_APPROVAL,
        source_rule="approval_rule",
        before_hash="h1",
        after_hash="h1",
    )
    ctx.state["policy_decision_ir"] = _policy_decision(
        status=ToolExecutionStatus.OK,
        final_output="pending",
        effects=(approval_effect,),
    )
    ctx.state["policy_decision_ir"] = PolicyDecisionIR(
        tool_result=ctx.state["policy_decision_ir"].tool_result,
        matched_rules=ctx.state["policy_decision_ir"].matched_rules,
        emitted_effects=ctx.state["policy_decision_ir"].emitted_effects,
        final_output=ctx.state["policy_decision_ir"].final_output,
        output_was_modified=True,
    )

    await node.execute(_Registry(), ctx)

    assert ctx.state["recovery_strategy"] == "require_approval"
    assert ctx.state["recovery_requires_approval"] is True
    assert ctx.state["safe_result"][0].startswith("I need your approval")


@pytest.mark.asyncio
async def test_recovery_node_halt_safe_sets_halt_message() -> None:
    node = RecoveryNode()
    ctx = TurnContext(user_input="hello")
    ctx.state["safe_result"] = "x"
    ctx.state["policy_decision_ir"] = _policy_decision(
        status=ToolExecutionStatus.ERROR,
        final_output="x",
    )

    await node.execute(_Registry(), ctx)

    assert ctx.state["recovery_strategy"] == "halt_safe"
    assert ctx.state["recovery_halted"] is True
    assert "paused for safety" in ctx.state["safe_result"][0].lower()
