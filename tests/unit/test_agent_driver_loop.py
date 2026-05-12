from __future__ import annotations

import pytest

from dadbot.core.execution_contract import TurnDelivery, TurnResponse
from dadbot.runtime.agent_driver_loop import AgentDriverLoop, DriverLoopPolicy
from dadbot.runtime_core.bus import EventBus

pytestmark = pytest.mark.unit


class _KernelStub:
    def __init__(self, replies: list[tuple[str, bool]]) -> None:
        self.replies = list(replies)
        self.calls: list[str] = []

    def execute_turn(self, request, **_kwargs):
        assert request.delivery == TurnDelivery.SYNC
        self.calls.append(str(request.input.text or ""))
        if not self.replies:
            return TurnResponse(reply="", should_end=False)
        reply, should_end = self.replies.pop(0)
        return TurnResponse(reply=reply, should_end=should_end)


class _FailingKernelStub:
    def __init__(self, fail_count: int) -> None:
        self.fail_count = int(fail_count)

    def execute_turn(self, request, **_kwargs):
        _ = request
        if self.fail_count > 0:
            self.fail_count -= 1
            raise RuntimeError("synthetic failure")
        return TurnResponse(reply="recovered", should_end=False)


def test_driver_loop_stops_on_kernel_should_end():
    kernel = _KernelStub(replies=[("reflecting", False), ("final", True), ("unused", False)])
    loop = AgentDriverLoop(kernel, policy=DriverLoopPolicy(max_turns=5, max_failures=1, max_consecutive_noop=2))

    result = loop.run("start")

    assert result.stop_reason == "kernel_should_end"
    assert result.completed_turns == 2
    assert [record.commit_status for record in result.records] == ["committed", "committed"]
    assert kernel.calls == ["start", "reflecting"]


def test_driver_loop_stops_on_noop_threshold():
    kernel = _KernelStub(replies=[("", False), ("", False), ("never", False)])
    loop = AgentDriverLoop(kernel, policy=DriverLoopPolicy(max_turns=6, max_failures=1, max_consecutive_noop=2))

    result = loop.run("start")

    assert result.stop_reason == "noop_threshold"
    assert result.completed_turns == 2
    assert result.consecutive_noop == 2


def test_driver_loop_respects_failure_budget():
    kernel = _FailingKernelStub(fail_count=3)
    loop = AgentDriverLoop(kernel, policy=DriverLoopPolicy(max_turns=5, max_failures=1, max_consecutive_noop=2))

    result = loop.run("start")

    assert result.stop_reason == "failure_budget"
    assert result.failures == 2
    assert all(record.commit_status == "failed" for record in result.records)


def test_driver_loop_emits_telemetry_events():
    """EventBus receives loop_started, loop_turn_completed, and loop_stopped."""
    bus = EventBus()
    kernel = _KernelStub(replies=[("hi", False), ("bye", True)])
    loop = AgentDriverLoop(kernel, policy=DriverLoopPolicy(max_turns=5), event_bus=bus)

    loop.run("hello", session_id="sess-1")

    events = bus.drain()
    types = [e.type for e in events]
    assert types[0] == "loop_started"
    assert types.count("loop_turn_completed") == 2
    assert types[-1] == "loop_stopped"
    # all events carry the session thread_id
    assert all(e.thread_id == "sess-1" for e in events)
    # loop_stopped payload has stop_reason
    stopped = next(e for e in events if e.type == "loop_stopped")
    assert stopped.payload["stop_reason"] == "kernel_should_end"


def test_driver_loop_retries_after_system_observation_without_noop_penalty():
    kernel = _KernelStub(replies=[("ok", True)])
    loop = AgentDriverLoop(kernel, policy=DriverLoopPolicy(max_turns=3, max_failures=1, max_consecutive_noop=2))

    calls = {"n": 0}

    def reflection_hook(_ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "action_input": "",
                "should_continue": True,
                "system_observation": "System observation: invalid tool attempted, retry.",
            }
        return {"action_input": "retry with valid tool", "should_continue": True}

    result = loop.run("start", reflection_hook=reflection_hook)

    assert result.stop_reason == "kernel_should_end"
    assert result.completed_turns == 1
    assert result.records[0].commit_status == "skipped"
    assert result.records[1].commit_status == "committed"
    assert kernel.calls == ["retry with valid tool"]
