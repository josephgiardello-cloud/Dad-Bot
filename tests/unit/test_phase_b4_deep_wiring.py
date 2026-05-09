from __future__ import annotations

from dadbot.contracts import SovereignEvent
from dadbot.core.graph_context import TurnContext
from dadbot.core.policy_ir import PolicyCompilerIR, PolicyCondition, PolicyRule, PolicyRuleCondition
from dadbot.core.runtime_types import (
    CanonicalPayload,
    PolicyEffectType,
    ToolExecutionStatus,
    ToolResult,
)
from dadbot.core.tool_executor import execute_tool


def _build_turn_context(trace_id: str) -> TurnContext:
    return TurnContext(user_input="test", trace_id=trace_id)


def test_tool_executor_emits_checksum_linked_tool_events() -> None:
    turn_context = _build_turn_context("turn-b4-tools")

    first = execute_tool(
        tool_name="web_search",
        parameters={"query": "first"},
        executor=lambda: {"ok": 1},
        turn_context=turn_context,
    )
    second = execute_tool(
        tool_name="set_reminder",
        parameters={"title": "buy milk"},
        executor=lambda: {"id": "r-1"},
        turn_context=turn_context,
    )

    assert first.status in {"succeeded", "cached", "failed"}
    assert second.status in {"succeeded", "cached", "failed"}

    events = list(turn_context.state.get("sovereign_events") or [])
    assert len(events) == 2

    first_event = SovereignEvent.model_validate(events[0])
    second_event = SovereignEvent.model_validate(events[1])

    assert first_event.event_type == "TOOL_EXECUTION"
    assert second_event.event_type == "TOOL_EXECUTION"
    assert first_event.verify_checksum("")
    assert second_event.verify_checksum(first_event.checksum)


def test_policy_compiler_ir_emits_policy_veto_with_veto_reason() -> None:
    turn_context = _build_turn_context("turn-b4-policy")
    compiler = PolicyCompilerIR(
        rules=[
            PolicyRule(
                rule_id="deny_dangerous_tool",
                rule_name="Deny dangerous tool",
                conditions=(
                    PolicyCondition(
                        PolicyRuleCondition.TOOL_NAME_MATCH,
                        params={"names": ["danger_tool"]},
                    ),
                ),
                effects=(PolicyEffectType.DENY_TOOL,),
                priority=100,
            ),
        ],
    )

    result = ToolResult(
        tool_name="danger_tool",
        invocation_id="inv-b4-policy",
        status=ToolExecutionStatus.OK,
        payload=CanonicalPayload({"value": 1}, payload_type="json"),
    )

    decision = compiler.evaluate_with_effects(result, turn_context=turn_context)

    assert decision.emitted_effects
    events = list(turn_context.state.get("sovereign_events") or [])
    assert len(events) == 1

    event = SovereignEvent.model_validate(events[0])
    payload = event.payload.model_dump(mode="json")

    assert event.event_type == "POLICY_VETO"
    assert str(payload.get("veto_reason", {}).get("code") or "") == "deny_tool"
    assert str(payload.get("reason") or "")
    assert event.verify_checksum("")
