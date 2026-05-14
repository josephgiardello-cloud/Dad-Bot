from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from dadbot.contracts import PlannerDecisionPayload, SovereignEvent, SovereignEventType, ToolExecutionPayload
from dadbot.managers.runtime_interface import SovereignMonitor

pytestmark = pytest.mark.unit


def _wait_until(predicate, timeout_seconds: float = 3.0) -> bool:
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _seed_valid_stream(context: SimpleNamespace) -> None:
    first = SovereignEvent(
        turn_id="turn-hud-1",
        event_type=SovereignEventType.PLANNER_DECISION.value,
        payload=PlannerDecisionPayload(
            planner_node="planner",
            selected_branch="tool",
            rationale="run tool",
        ),
    )
    second = SovereignEvent(
        turn_id="turn-hud-1",
        event_type=SovereignEventType.TOOL_EXECUTION.value,
        payload=ToolExecutionPayload(
            tool_name="get_weather",
            status="ok",
            input_hash="in1",
            output_hash="out1",
        ),
        previous_checksum=first.checksum,
    )
    context.state["sovereign_events"] = [first.to_ledger_event(), second.to_ledger_event()]
    context.metadata["sovereign_event_checksum"] = second.checksum


def test_sovereign_monitor_tracks_chain_health_and_veto_flash(tmp_path) -> None:
    ledger_path = tmp_path / "relational_ledger.jsonl"
    ledger_path.write_text('{"ok": true}\n', encoding="utf-8")

    context = SimpleNamespace(state={}, metadata={})
    _seed_valid_stream(context)

    bot = SimpleNamespace(_active_turn_context=context, _last_turn_context=context)
    monitor = SovereignMonitor(bot=bot, poll_seconds=0.1, ledger_path=str(ledger_path))

    monitor.start()
    try:
        assert _wait_until(lambda: monitor._last_event_count >= 2)

        veto = SovereignEvent(
            turn_id="turn-hud-1",
            event_type=SovereignEventType.POLICY_VETO.value,
            payload={
                "kind": "POLICY_VETO",
                "policy_rule": "safety_rule",
                "reason": "restricted term",
                "severity": "high",
                "veto_reason": {
                    "code": "deny_tool",
                    "message": "restricted term",
                    "severity": "high",
                    "metadata": {},
                },
                "metadata": {},
            },
            previous_checksum=str(context.metadata.get("sovereign_event_checksum") or ""),
        )
        stream = list(context.state.get("sovereign_events") or [])
        stream.append(veto.to_ledger_event())
        context.state["sovereign_events"] = stream
        context.metadata["sovereign_event_checksum"] = veto.checksum

        assert _wait_until(lambda: monitor._last_flash_until > 0.0)
    finally:
        monitor.stop()
        monitor.join(timeout=2.0)


def test_sovereign_monitor_triggers_refusal_on_malformed_out_of_band_write(tmp_path) -> None:
    ledger_path = tmp_path / "relational_ledger.jsonl"
    ledger_path.write_text('{"ok": true}\n', encoding="utf-8")

    context = SimpleNamespace(state={}, metadata={})
    _seed_valid_stream(context)

    bot = SimpleNamespace(_active_turn_context=context, _last_turn_context=context)
    monitor = SovereignMonitor(bot=bot, poll_seconds=0.1, ledger_path=str(ledger_path))

    monitor.start()
    try:
        assert _wait_until(lambda: monitor._last_event_count >= 2)

        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write('{"malformed": ')  # intentionally incomplete JSON

        assert _wait_until(lambda: str(context.state.get("refusal_state") or "") == "integrity_failure")
        assert bool(context.metadata.get("integrity_failure", False)) is True
    finally:
        monitor.stop()
        monitor.join(timeout=2.0)
