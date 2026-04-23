import asyncio

import pytest

from dadbot.core.control_plane import (
    ExecutionControlPlane,
    ExecutionJob,
    InMemoryExecutionLedger,
    LedgerReader,
    LedgerWriter,
    Scheduler,
    SessionRegistry,
)
from dadbot.core.recovery_manager import RecoveryManager
from dadbot.core.session_store import SessionStore


def test_control_plane_idempotency_dedupes_concurrent_requests():
    registry = SessionRegistry()
    calls = {"count": 0}

    async def _kernel_execute(_session, _job):
        calls["count"] += 1
        await asyncio.sleep(0.01)
        return ("ok", False)

    control_plane = ExecutionControlPlane(registry=registry, kernel_executor=_kernel_execute)

    async def _run():
        return await asyncio.gather(
            control_plane.submit_turn(
                session_id="s1",
                user_input="hi",
                metadata={"request_id": "req-1", "trace_id": "t1"},
            ),
            control_plane.submit_turn(
                session_id="s1",
                user_input="hi again",
                metadata={"request_id": "req-1", "trace_id": "t1"},
            ),
        )

    results = asyncio.run(_run())
    assert results == [("ok", False), ("ok", False)]
    assert calls["count"] == 1

    queued = [
        event
        for event in control_plane.ledger_events()
        if str(event.get("type") or "") == "JOB_QUEUED"
    ]
    assert len(queued) == 1
    assert queued[0]["payload"]["request_id"] == "req-1"


def test_scheduler_enforces_backpressure_limit():
    registry = SessionRegistry()
    shared_ledger = InMemoryExecutionLedger()
    scheduler = Scheduler(
        registry,
        reader=LedgerReader(shared_ledger),
        writer=LedgerWriter(shared_ledger),
        max_inflight_jobs=1,
    )

    async def _run():
        first = ExecutionJob(session_id="s-backpressure", user_input="first")
        second = ExecutionJob(session_id="s-backpressure", user_input="second")
        await scheduler.register(first)
        with pytest.raises(RuntimeError):
            await scheduler.register(second)

    asyncio.run(_run())


def test_recovery_manager_rebuilds_projection_and_returns_pending_jobs():
    ledger = InMemoryExecutionLedger()
    session_store = SessionStore(projection_only=True)

    ledger.write(
        {
            "type": "SESSION_STATE_UPDATED",
            "session_id": "s-recover",
            "trace_id": "tr",
            "timestamp": 1.0,
            "kernel_step_id": "session_store.apply",
            "payload": {"state": {"x": 1}, "version": 1},
        }
    )
    ledger.write(
        {
            "type": "JOB_QUEUED",
            "session_id": "s-recover",
            "trace_id": "tr",
            "timestamp": 2.0,
            "kernel_step_id": "control_plane.enqueue",
            "payload": {
                "job_id": "job-recover-1",
                "request_id": "req-recover-1",
                "user_input": "hello",
                "attachments": [],
                "metadata": {"trace_id": "tr"},
                "priority": 0,
                "submitted_at": 2.0,
            },
        }
    )

    manager = RecoveryManager(ledger=ledger)
    report = manager.recover(session_store=session_store)

    assert report["ledger_events"] == 2
    assert report["session_count"] == 1
    assert report["session_snapshot_version"] >= 1
    assert len(report["pending_jobs"]) == 1
    assert report["pending_jobs"][0]["job_id"] == "job-recover-1"

    state = session_store.get("s-recover")
    assert state == {"x": 1}
