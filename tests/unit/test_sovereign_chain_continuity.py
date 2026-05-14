from __future__ import annotations

from pathlib import Path

import pytest

from dadbot.contracts import GenericSovereignPayload, PlannerDecisionPayload, SovereignEvent, SovereignEventType
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.graph_context import TurnContext
from dadbot.core.ledger_backend import FileWALLedgerBackend
from dadbot.core.policy_ir import PolicyCompilerIR, PolicyCondition, PolicyRule, PolicyRuleCondition
from dadbot.core.runtime_types import CanonicalPayload, PolicyEffectType, ToolExecutionStatus, ToolResult
from dadbot.core.tool_executor import execute_tool


def _append_sovereign_event(turn_context: TurnContext, event: SovereignEvent) -> None:
    stream = list(turn_context.state.get("sovereign_events") or [])
    stream.append(event.to_ledger_event())
    turn_context.state["sovereign_events"] = stream
    turn_context.metadata["sovereign_event_checksum"] = event.checksum
    turn_context.metadata["sovereign_event_count"] = len(stream)


def _assert_chain_contiguous(event_stream: list[dict]) -> None:
    assert len(event_stream) >= 4
    prior_checksum = ""
    prior_event = None
    turn_ids: set[str] = set()
    for raw in event_stream:
        event = SovereignEvent.model_validate(raw)
        turn_ids.add(str(event.turn_id or ""))
        if prior_event is not None:
            assert event.previous_checksum == prior_checksum
        assert event.verify_checksum(prior_checksum)
        prior_checksum = event.checksum
        prior_event = event
    assert len(turn_ids) == 1


def test_sovereign_chain_continuity_tool_to_policy_veto() -> None:
    turn_context = TurnContext(user_input="weather in boston", trace_id="turn-chain-001")

    start_event = SovereignEvent(
        turn_id=turn_context.trace_id,
        event_type=SovereignEventType.LOGIC_BRANCH.value,
        payload={
            "kind": "LOGIC_BRANCH",
            "branch_name": "turn_start",
            "condition": "ingress",
            "outcome": "accepted",
            "metadata": {},
        },
    )
    _append_sovereign_event(turn_context, start_event)

    planner_event = SovereignEvent(
        turn_id=turn_context.trace_id,
        event_type=SovereignEventType.PLANNER_DECISION.value,
        payload=PlannerDecisionPayload(
            planner_node="planner",
            selected_branch="tool:get_weather",
            rationale="weather requested",
        ),
        previous_checksum=start_event.checksum,
    )
    _append_sovereign_event(turn_context, planner_event)

    record = execute_tool(
        tool_name="get_weather",
        parameters={"location": "Boston"},
        executor=lambda: {"forecast": "clear with restricted-term in diagnostics"},
        turn_context=turn_context,
    )

    policy = PolicyCompilerIR(
        rules=[
            PolicyRule(
                rule_id="safety_restricted_term",
                rule_name="Restricted term safety veto",
                conditions=(
                    PolicyCondition(
                        PolicyRuleCondition.TOOL_NAME_MATCH,
                        params={"names": ["get_weather"]},
                    ),
                ),
                effects=(PolicyEffectType.DENY_TOOL,),
                priority=100,
            ),
        ],
    )
    policy.evaluate_with_effects(
        ToolResult(
            tool_name="get_weather",
            invocation_id="inv-weather-1",
            status=ToolExecutionStatus.OK,
            payload=CanonicalPayload(record.result, payload_type="json"),
        ),
        context={"restricted_term_hit": True, "term": "restricted-term"},
        turn_context=turn_context,
    )

    stream = list(turn_context.state.get("sovereign_events") or [])
    _assert_chain_contiguous(stream)

    tool_event = SovereignEvent.model_validate(stream[2])
    veto_event = SovereignEvent.model_validate(stream[3])
    assert tool_event.event_type == SovereignEventType.TOOL_EXECUTION.value
    assert veto_event.event_type == SovereignEventType.POLICY_VETO.value
    assert veto_event.previous_checksum == tool_event.checksum


def test_sovereign_chain_continuity_rejects_phantom_injection_gap() -> None:
    turn_context = TurnContext(user_input="weather in boston", trace_id="turn-chain-002")

    first = SovereignEvent(
        turn_id=turn_context.trace_id,
        event_type=SovereignEventType.LOGIC_BRANCH.value,
        payload={
            "kind": "LOGIC_BRANCH",
            "branch_name": "turn_start",
            "condition": "ingress",
            "outcome": "accepted",
            "metadata": {},
        },
    )
    _append_sovereign_event(turn_context, first)

    second = SovereignEvent(
        turn_id=turn_context.trace_id,
        event_type=SovereignEventType.PLANNER_DECISION.value,
        payload=PlannerDecisionPayload(
            planner_node="planner",
            selected_branch="tool:get_weather",
            rationale="weather requested",
        ),
        previous_checksum=first.checksum,
    )
    _append_sovereign_event(turn_context, second)

    record = execute_tool(
        tool_name="get_weather",
        parameters={"location": "Boston"},
        executor=lambda: {"forecast": "restricted-term"},
        turn_context=turn_context,
    )

    policy = PolicyCompilerIR(
        rules=[
            PolicyRule(
                rule_id="safety_restricted_term",
                rule_name="Restricted term safety veto",
                conditions=(
                    PolicyCondition(
                        PolicyRuleCondition.TOOL_NAME_MATCH,
                        params={"names": ["get_weather"]},
                    ),
                ),
                effects=(PolicyEffectType.DENY_TOOL,),
                priority=100,
            ),
        ],
    )
    policy.evaluate_with_effects(
        ToolResult(
            tool_name="get_weather",
            invocation_id="inv-weather-2",
            status=ToolExecutionStatus.OK,
            payload=CanonicalPayload(record.result, payload_type="json"),
        ),
        context={"restricted_term_hit": True, "term": "restricted-term"},
        turn_context=turn_context,
    )

    stream = list(turn_context.state.get("sovereign_events") or [])
    assert len(stream) >= 4

    tool_event = SovereignEvent.model_validate(stream[2])
    phantom = SovereignEvent(
        turn_id=turn_context.trace_id,
        event_type=SovereignEventType.GENERIC.value,
        payload=GenericSovereignPayload(data={"rogue": True}),
        previous_checksum=tool_event.checksum,
    ).to_ledger_event()

    tampered_stream = list(stream)
    tampered_stream.insert(3, phantom)

    with pytest.raises(AssertionError):
        _assert_chain_contiguous(tampered_stream)


def test_recovery_gate_refuses_one_byte_tamper_in_jsonl(tmp_path: Path) -> None:
    wal_path = tmp_path / "continuity_gate.jsonl"

    source = ExecutionLedger(backend=FileWALLedgerBackend(wal_path))
    source.write(
        {
            "type": "JOB_SUBMITTED",
            "session_id": "session-continuity",
            "trace_id": "trace-continuity",
            "kernel_step_id": "k-1",
            "timestamp": "2026-05-08T00:00:00Z",
            "payload": {"step": 1},
        },
    )
    source.write(
        {
            "type": "JOB_STARTED",
            "session_id": "session-continuity",
            "trace_id": "trace-continuity",
            "kernel_step_id": "k-2",
            "timestamp": "2026-05-08T00:00:01Z",
            "payload": {"step": 2},
        },
    )

    original = wal_path.read_text(encoding="utf-8")
    tampered = original.replace('"step": 2', '"step": 3', 1)
    assert tampered != original
    wal_path.write_text(tampered, encoding="utf-8")

    reloaded = ExecutionLedger(backend=FileWALLedgerBackend(wal_path))
    with pytest.raises(RuntimeError, match="checksum chain verification failed"):
        reloaded.load_from_backend()
