"""Tests for the tool runtime test adapter: idempotency and rollback semantics."""

from dadbot.core.testing.tool_runtime_test_adapter import ToolRuntimeTestAdapter


def test_tool_sandbox_executes_and_returns_result():
    runtime = ToolRuntimeTestAdapter()
    record = runtime.execute_tool(
        tool_name="set_reminder",
        parameters={"title": "Call dentist", "due_text": "tomorrow"},
        executor=lambda: {"id": "r1", "title": "Call dentist"},
    )
    assert record.status == "succeeded"
    assert record.result == {"id": "r1", "title": "Call dentist"}
    assert record.error == ""


def test_tool_sandbox_idempotency_blocks_duplicate_execution():
    runtime = ToolRuntimeTestAdapter()
    calls = []

    def _remind():
        calls.append(1)
        return {"id": "r1", "title": "Call dentist"}

    runtime.execute(
        tool_name="set_reminder",
        parameters={"title": "Call dentist", "due_text": "tomorrow"},
        executor=_remind,
    )
    record2 = runtime.execute(
        tool_name="set_reminder",
        parameters={"title": "Call dentist", "due_text": "tomorrow"},
        executor=_remind,
    )
    # Second call must be a cache hit; executor must only run once.
    assert record2.status == "cached"
    assert record2.result == {"id": "r1", "title": "Call dentist"}
    assert len(calls) == 1


def test_tool_sandbox_different_params_are_different_keys():
    runtime = ToolRuntimeTestAdapter()
    calls = []

    def _remind():
        calls.append(1)
        return {"id": f"r{len(calls)}", "title": "Remind"}

    runtime.execute(
        tool_name="set_reminder",
        parameters={"title": "Call dentist"},
        executor=_remind,
    )
    record2 = runtime.execute(
        tool_name="set_reminder",
        parameters={"title": "Call bank"},  # different params → different key
        executor=_remind,
    )
    assert record2.status == "succeeded"
    assert len(calls) == 2


def test_tool_sandbox_isolates_executor_failures():
    runtime = ToolRuntimeTestAdapter()

    def _bad_executor():
        raise RuntimeError("Reminder service is down")

    record = runtime.execute(
        tool_name="set_reminder",
        parameters={"title": "Explode"},
        executor=_bad_executor,
    )
    assert record.status == "failed"
    assert "Reminder service is down" in record.error
    assert record.result is None


def test_tool_sandbox_rollback_runs_compensating_actions_in_lifo_order():
    runtime = ToolRuntimeTestAdapter()
    order = []

    def _remind_a():
        return {"id": "a"}

    def _remind_b():
        return {"id": "b"}

    runtime.execute(
        tool_name="set_reminder",
        parameters={"title": "A"},
        executor=_remind_a,
        compensating_action=lambda: order.append("rollback_a"),
    )
    runtime.execute(
        tool_name="set_reminder",
        parameters={"title": "B"},
        executor=_remind_b,
        compensating_action=lambda: order.append("rollback_b"),
    )

    outcomes = runtime.rollback()
    assert [o["rolled_back"] for o in outcomes] == [True, True]
    # LIFO: B was added second, must be rolled back first.
    assert order == ["rollback_b", "rollback_a"]


def test_tool_sandbox_rollback_tolerates_compensating_action_failure():
    runtime = ToolRuntimeTestAdapter()

    def _compensate():
        raise RuntimeError("delete failed")

    runtime.execute(
        tool_name="set_reminder",
        parameters={"title": "X"},
        executor=lambda: {"id": "x"},
        compensating_action=_compensate,
    )

    outcomes = runtime.rollback()
    assert outcomes[0]["rolled_back"] is False
    assert "delete failed" in outcomes[0]["error"]


def test_tool_sandbox_snapshot_reflects_execution_history():
    runtime = ToolRuntimeTestAdapter()
    runtime.execute(
        tool_name="web_search",
        parameters={"query": "weather"},
        executor=lambda: {"heading": "Weather", "summary": "Sunny"},
    )

    snap = runtime.snapshot()
    assert snap["executed_count"] == 1
    assert snap["succeeded_count"] == 1
    assert snap["failed_count"] == 0
    assert snap["records"][0]["tool"] == "web_search"
    assert snap["records"][0]["status"] == "succeeded"


def test_tool_sandbox_failed_execution_not_cached():
    runtime = ToolRuntimeTestAdapter()
    calls = []

    def _flaky():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient failure")
        return {"id": "ok"}

    record1 = runtime.execute(
        tool_name="set_reminder",
        parameters={"title": "Flaky"},
        executor=_flaky,
    )
    assert record1.status == "failed"

    # Second attempt with same params must NOT be a cache hit since first failed.
    record2 = runtime.execute(
        tool_name="set_reminder",
        parameters={"title": "Flaky"},
        executor=_flaky,
    )
    assert record2.status == "succeeded"
    assert len(calls) == 2
