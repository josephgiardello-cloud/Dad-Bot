from __future__ import annotations

import asyncio

import pytest

from dadbot_system.kernel import KernelTaskManager, SessionRegistry

pytestmark = pytest.mark.asyncio


async def test_kernel_task_manager_records_failure_in_session_event_log() -> None:
    registry = SessionRegistry()
    registry.create("s1")
    manager = KernelTaskManager(registry)

    async def _boom() -> None:
        raise RuntimeError("boom")

    manager.register(name="failing-task", coro=_boom(), session_id="s1")
    outcomes = await manager.await_all()

    assert any(isinstance(item, RuntimeError) for item in outcomes)
    session = registry.get("s1") or {}
    events = list(session.get("event_log") or [])
    assert any(str(event.get("status") or "") == "failed" for event in events)


async def test_kernel_task_manager_await_session_ignores_unrelated_pending_tasks() -> None:
    registry = SessionRegistry()
    registry.create("s-a")
    manager = KernelTaskManager(registry)

    release = asyncio.Event()

    async def _long_running() -> None:
        await release.wait()

    async def _session_task() -> str:
        return "ok"

    manager.register(name="global-long-running", coro=_long_running())
    manager.register(name="session-task", coro=_session_task(), session_id="s-a")

    outcomes = await manager.await_session("s-a")
    assert outcomes == ["ok"]
    assert manager.pending_count == 1

    release.set()
    await manager.shutdown(cancel_pending=True)


async def test_kernel_task_manager_shutdown_cancels_pending_tasks() -> None:
    registry = SessionRegistry()
    manager = KernelTaskManager(registry)

    blocker = asyncio.Event()

    async def _wait_forever() -> None:
        await blocker.wait()

    manager.register(name="pending", coro=_wait_forever())
    assert manager.pending_count == 1

    await manager.shutdown(cancel_pending=True)
    assert manager.pending_count == 0
