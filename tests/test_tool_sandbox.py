"""Tests for ToolSandbox: idempotency, failure isolation, rollback semantics."""
import pytest
from dadbot.core.tool_sandbox import ToolSandbox


def test_tool_sandbox_executes_and_returns_result():
    sandbox = ToolSandbox()
    record = sandbox.execute(
        tool_name="set_reminder",
        parameters={"title": "Call dentist", "due_text": "tomorrow"},
        executor=lambda: {"id": "r1", "title": "Call dentist"},
    )
    assert record.status == "succeeded"
    assert record.result == {"id": "r1", "title": "Call dentist"}
    assert record.error == ""


def test_tool_sandbox_idempotency_blocks_duplicate_execution():
    sandbox = ToolSandbox()
    calls = []

    def _remind():
        calls.append(1)
        return {"id": "r1", "title": "Call dentist"}

    sandbox.execute(
        tool_name="set_reminder",
        parameters={"title": "Call dentist", "due_text": "tomorrow"},
        executor=_remind,
    )
    record2 = sandbox.execute(
        tool_name="set_reminder",
        parameters={"title": "Call dentist", "due_text": "tomorrow"},
        executor=_remind,
    )
    # Second call must be a cache hit; executor must only run once.
    assert record2.status == "cached"
    assert record2.result == {"id": "r1", "title": "Call dentist"}
    assert len(calls) == 1


def test_tool_sandbox_different_params_are_different_keys():
    sandbox = ToolSandbox()
    calls = []

    def _remind():
        calls.append(1)
        return {"id": f"r{len(calls)}", "title": "Remind"}

    sandbox.execute(
        tool_name="set_reminder",
        parameters={"title": "Call dentist"},
        executor=_remind,
    )
    record2 = sandbox.execute(
        tool_name="set_reminder",
        parameters={"title": "Call bank"},   # different params → different key
        executor=_remind,
    )
    assert record2.status == "succeeded"
    assert len(calls) == 2


def test_tool_sandbox_isolates_executor_failures():
    sandbox = ToolSandbox()

    def _bad_executor():
        raise RuntimeError("Reminder service is down")

    record = sandbox.execute(
        tool_name="set_reminder",
        parameters={"title": "Explode"},
        executor=_bad_executor,
    )
    assert record.status == "failed"
    assert "Reminder service is down" in record.error
    assert record.result is None


def test_tool_sandbox_rollback_runs_compensating_actions_in_lifo_order():
    sandbox = ToolSandbox()
    order = []

    def _remind_a():
        return {"id": "a"}

    def _remind_b():
        return {"id": "b"}

    sandbox.execute(
        tool_name="set_reminder",
        parameters={"title": "A"},
        executor=_remind_a,
        compensating_action=lambda: order.append("rollback_a"),
    )
    sandbox.execute(
        tool_name="set_reminder",
        parameters={"title": "B"},
        executor=_remind_b,
        compensating_action=lambda: order.append("rollback_b"),
    )

    outcomes = sandbox.rollback()
    assert [o["rolled_back"] for o in outcomes] == [True, True]
    # LIFO: B was added second, must be rolled back first.
    assert order == ["rollback_b", "rollback_a"]


def test_tool_sandbox_rollback_tolerates_compensating_action_failure():
    sandbox = ToolSandbox()

    def _compensate():
        raise RuntimeError("delete failed")

    sandbox.execute(
        tool_name="set_reminder",
        parameters={"title": "X"},
        executor=lambda: {"id": "x"},
        compensating_action=_compensate,
    )

    outcomes = sandbox.rollback()
    assert outcomes[0]["rolled_back"] is False
    assert "delete failed" in outcomes[0]["error"]


def test_tool_sandbox_snapshot_reflects_execution_history():
    sandbox = ToolSandbox()
    sandbox.execute(
        tool_name="web_search",
        parameters={"query": "weather"},
        executor=lambda: {"heading": "Weather", "summary": "Sunny"},
    )

    snap = sandbox.snapshot()
    assert snap["executed_count"] == 1
    assert snap["succeeded_count"] == 1
    assert snap["failed_count"] == 0
    assert snap["records"][0]["tool"] == "web_search"
    assert snap["records"][0]["status"] == "succeeded"


def test_tool_sandbox_failed_execution_not_cached():
    sandbox = ToolSandbox()
    calls = []

    def _flaky():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient failure")
        return {"id": "ok"}

    record1 = sandbox.execute(
        tool_name="set_reminder",
        parameters={"title": "Flaky"},
        executor=_flaky,
    )
    assert record1.status == "failed"

    # Second attempt with same params must NOT be a cache hit since first failed.
    record2 = sandbox.execute(
        tool_name="set_reminder",
        parameters={"title": "Flaky"},
        executor=_flaky,
    )
    assert record2.status == "succeeded"
    assert len(calls) == 2
