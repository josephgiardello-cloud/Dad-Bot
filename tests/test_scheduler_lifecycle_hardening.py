from __future__ import annotations

import asyncio

import pytest

from dadbot.core.control_plane import (
    ExecutionControlPlane,
    ExecutionJob,
    SessionRegistry,
    TurnTerminalState,
    _SCHEDULER_EXCEPTION_MAPPER,
    _set_terminal_turn_state,
)
from dadbot.core.runtime_errors import InvariantViolation


def _last_ledger_event(plane: ExecutionControlPlane, event_type: str) -> dict:
    events = [e for e in plane.ledger_events() if str(e.get("type") or "") == event_type]
    assert events, f"expected at least one {event_type} event"
    return dict(events[-1])


def _terminal_state_from_event(event: dict) -> str:
    payload = dict(event.get("payload") or {})
    metadata = dict(payload.get("metadata") or {})
    execution_state = dict(metadata.get("execution_state") or {})
    return str(
        execution_state.get("terminal_turn_state")
        or metadata.get("terminal_turn_state")
        or payload.get("terminal_turn_state")
        or "",
    )


@pytest.mark.asyncio
async def test_cancel_mid_turn() -> None:
    async def _executor(_session: dict, _job: ExecutionJob) -> tuple[str, bool]:
        raise asyncio.CancelledError("cancelled by test")

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)

    with pytest.raises(asyncio.CancelledError):
        await plane.submit_turn(session_id="cancel-mid-turn", user_input="hello")

    failed = _last_ledger_event(plane, "JOB_FAILED")
    assert _terminal_state_from_event(failed) == TurnTerminalState.CANCELLED.value
    assert len(plane._scheduler._jobs) == 0


@pytest.mark.asyncio
async def test_timeout_mid_tool_execution() -> None:
    async def _executor(_session: dict, _job: ExecutionJob) -> tuple[str, bool]:
        raise TimeoutError("tool timeout")

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)

    with pytest.raises(TimeoutError):
        await plane.submit_turn(session_id="timeout-mid-tool", user_input="hello")

    failed = _last_ledger_event(plane, "JOB_FAILED")
    assert _terminal_state_from_event(failed) == TurnTerminalState.TIMEOUT.value


@pytest.mark.asyncio
async def test_exception_during_execution_phase() -> None:
    async def _executor(_session: dict, _job: ExecutionJob) -> tuple[str, bool]:
        raise RuntimeError("boom")

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)

    with pytest.raises(RuntimeError):
        await plane.submit_turn(session_id="execution-failure", user_input="hello")

    failed = _last_ledger_event(plane, "JOB_FAILED")
    assert _terminal_state_from_event(failed) == TurnTerminalState.FAILED.value


def test_retry_recovery_correctness() -> None:
    assert _SCHEDULER_EXCEPTION_MAPPER.from_success(recovered=True) == TurnTerminalState.RECOVERED
    assert _SCHEDULER_EXCEPTION_MAPPER.from_success(recovered=False) == TurnTerminalState.SUCCESS


def test_double_finalization_prevention() -> None:
    job = ExecutionJob(session_id="s", user_input="u", metadata={"execution_state": {}}, trace_id="t", job_id="j")

    _set_terminal_turn_state(
        job,
        terminal_state=TurnTerminalState.SUCCESS,
        reason="initial-finalize",
    )

    with pytest.raises(InvariantViolation):
        _set_terminal_turn_state(
            job,
            terminal_state=TurnTerminalState.FAILED,
            reason="second-finalize",
        )


@pytest.mark.asyncio
@pytest.mark.soak
async def test_soak_simulation_replay_loop() -> None:
    async def _executor(_session: dict, _job: ExecutionJob) -> tuple[str, bool]:
        return ("ok", False)

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)

    turns = 300
    for idx in range(turns):
        result = await plane.submit_turn(session_id="soak-loop", user_input=f"turn-{idx}")
        assert result[0] in {"ok", ""}

    assert len(plane._scheduler._jobs) == 0
    assert len(plane._scheduler._pending_job_ids) == 0

    completed_events = [
        e for e in plane.ledger_events() if str(e.get("type") or "") == "JOB_COMPLETED"
    ]
    assert len(completed_events) >= turns
    terminal_states = {_terminal_state_from_event(e) for e in completed_events}
    assert TurnTerminalState.SUCCESS.value in terminal_states or TurnTerminalState.RECOVERED.value in terminal_states
