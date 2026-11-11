from __future__ import annotations

import asyncio
from contextvars import ContextVar
from copy import deepcopy
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from dadbot.core.graph_context import TurnContext
from dadbot.core.runtime_errors import ReplayInvariantViolation
from dadbot.core.tool_recording import ToolIOLedger

pytestmark = pytest.mark.unit


def _context(**changes):
    return TurnContext(
        user_input="",
        trace_id="trace-a",
        metadata={
            "tenant_id": "tenant-a", "session_id": "session-a", "request_id": "approval-a",
            "tool_version": "tool-v1", "environment_fingerprint": "env-v1", **changes,
        },
    )


def _execute(context, executor):
    from dadbot.core.tool_executor import execute_tool_async

    return execute_tool_async(
        tool_name="light_control", parameters={"entity_id": "light.kitchen"},
        executor=executor, turn_context=context,
    )


def _io(context) -> ToolIOLedger:
    io = getattr(context, "_tool_io_ledger", None)
    assert isinstance(io, ToolIOLedger)
    return io


def test_async_execution_awaits_in_the_callers_task_and_records_complete_io():
    marker = ContextVar("async_tool_marker", default="before")
    context = _context()

    async def run():
        parent_task = asyncio.current_task()

        async def effect():
            assert asyncio.current_task() is parent_task
            marker.set("after")
            return {"is_error": False, "result": {"state": "off"}}

        record = await _execute(context, effect)
        assert marker.get() == "after"
        return record

    record = asyncio.run(run())
    assert record.status == "succeeded"
    io = _io(context)
    assert io.successful_calls() == 1
    assert io.records[0].output_payload == record.result
    assert io.records[0].metadata["scope"] == {
        "tenant_id": "tenant-a", "session_id": "session-a", "trace_id": "trace-a", "request_id": "approval-a",
    }
    assert context.state["sovereign_events"]


def test_async_replay_uses_the_shared_kernel_ledger_without_effects():
    original = _context()
    result = {"is_error": False, "result": {"state": "off"}}
    asyncio.run(_execute(original, AsyncMock(return_value=result)))
    restored = ToolIOLedger.from_dict(_io(original).to_dict())
    context = _context(replay_mode=True, _tool_io_ledger=restored)
    forbidden = AsyncMock(side_effect=AssertionError("replay dispatched"))
    record = asyncio.run(_execute(context, forbidden))
    assert record.status == "replayed"
    assert record.result == result
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "coordinate", ["tenant_id", "session_id", "request_id", "tool_version", "environment_fingerprint"],
)
def test_async_replay_rejects_changed_scope_or_version(coordinate):
    original = _context()
    asyncio.run(_execute(original, AsyncMock(return_value={"result": "done"})))
    context = _context(
        replay_mode=True, _tool_io_ledger=_io(original), **{coordinate: "other"},
    )
    executor = AsyncMock(return_value={})
    with pytest.raises(ReplayInvariantViolation):
        asyncio.run(_execute(context, executor))
    executor.assert_not_called()


def test_async_replay_rejects_tampered_output():
    original = _context()
    asyncio.run(_execute(original, AsyncMock(return_value={"result": "done"})))
    io = deepcopy(_io(original))
    io.records[0].output_payload["result"] = "changed"
    context = _context(replay_mode=True, _tool_io_ledger=io)
    executor = AsyncMock(return_value={})
    with pytest.raises(ReplayInvariantViolation, match="integrity"):
        asyncio.run(_execute(context, executor))
    executor.assert_not_called()


@pytest.mark.parametrize("error", [TimeoutError("unverified"), asyncio.CancelledError()])
def test_async_failure_is_uncertain_and_not_replayable(error):
    original = _context()
    with pytest.raises(type(error)):
        asyncio.run(_execute(original, AsyncMock(side_effect=error)))
    io = _io(original)
    assert io.records[0].status == "uncertain"
    context = _context(replay_mode=True, _tool_io_ledger=io)
    executor = AsyncMock(return_value={})
    with pytest.raises(ReplayInvariantViolation, match="successful"):
        asyncio.run(_execute(context, executor))
    executor.assert_not_called()


def test_async_tool_error_result_is_not_cached_or_replayed():
    context = _context()
    executor = AsyncMock(return_value={"is_error": True, "result": "unknown"})
    assert asyncio.run(_execute(context, executor)).status == "uncertain"
    assert asyncio.run(_execute(_context(request_id="approval-b"), executor)).status == "uncertain"
    assert executor.await_count == 2
    io = _io(context)
    with pytest.raises(ReplayInvariantViolation, match="successful"):
        asyncio.run(_execute(_context(replay_mode=True, _tool_io_ledger=io), executor))
    assert executor.await_count == 2


def test_async_live_authority_denial_blocks_executor():
    from dadbot_system.delegation import require_live_effect_authority
    from dadbot_system.security import AuthorizationError

    executor = AsyncMock(return_value={})
    with require_live_effect_authority(), pytest.raises(AuthorizationError):
        asyncio.run(_execute(_context(), executor))
    executor.assert_not_called()


def test_cached_sync_result_preserves_output_in_the_shared_io_ledger():
    from dadbot.core.tool_executor import execute_tool

    parameters = {"_idempotency_key": uuid4().hex}
    output = {"result": "done"}
    execute_tool(tool_name="sync-tool", parameters=parameters, executor=lambda: output, turn_context=_context())
    context = _context()
    record = execute_tool(
        tool_name="sync-tool", parameters=parameters,
        executor=lambda: pytest.fail("cached tool dispatched"), turn_context=context,
    )
    assert record.status == "cached"
    assert _io(context).records[0].output_payload == output
    assert _io(context).successful_calls() == 1


@pytest.mark.parametrize("output", [None, {"is_error": "false"}])
def test_async_invalid_result_is_uncertain(output):
    context = _context()
    with pytest.raises(TypeError):
        asyncio.run(_execute(context, AsyncMock(return_value=output)))
    assert _io(context).records[0].status == "uncertain"
