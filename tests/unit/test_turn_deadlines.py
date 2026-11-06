from __future__ import annotations

import asyncio
import threading
import time
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
from dadbot.core.execution_coordinator import ExecutionCoordinator
from dadbot.core.graph import TurnContext
from dadbot.core.graph_pipeline_nodes import ControlPlaneNode
from dadbot.core.kernel_gateway import KernelGateway
from dadbot.core.orchestrator import DadBotOrchestrator, _normalize_timeout_seconds
from dadbot.core.runtime_errors import TransientExecutionError
from dadbot.core.turn_mixin import DadBotTurnMixin
from dadbot.utils.async_bridge import run_coro_in_thread


def test_response_engine_does_not_block_loop_and_cancelled_node_does_not_finalize():
    async def run():
        started = threading.Event()
        release = threading.Event()
        finalized = Mock()
        caller = threading.get_ident()

        def execute(*_args):
            assert threading.get_ident() != caller
            started.set()
            assert release.wait(2)
            return "answer", False

        bot = SimpleNamespace(
            turn_orchestrator=SimpleNamespace(
                control_plane=SimpleNamespace(execute_from_graph_context=execute),
            ),
            reply_finalization=SimpleNamespace(finalize=finalized),
        )
        registry = {"agent_service": SimpleNamespace(bot=bot)}
        context = TurnContext(user_input="hello")
        task = asyncio.create_task(ControlPlaneNode().execute(registry, context))
        try:
            async with asyncio.timeout(1):
                while not started.is_set():
                    await asyncio.sleep(0.001)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            finalized.assert_not_called()
            assert "response_authority" not in context.state
        finally:
            release.set()

    asyncio.run(run())


@pytest.mark.parametrize("idle", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_coordinator_cancels_stalled_drain_or_idle_wait(idle, cancel):
    async def run():
        cleaned = asyncio.Event()
        entered = asyncio.Event()
        scope = ContextVar("drain_scope", default="parent")

        async def stall():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        async def drain(_executor):
            scope.set("draining")
            if idle:
                return False
            await stall()

        async def wait_for_work(**_kwargs):
            await stall()

        future = asyncio.get_running_loop().create_future()
        scheduler = SimpleNamespace(drain_once=drain, wait_for_work=wait_for_work, _jobs={})
        coordinator = ExecutionCoordinator()
        task = asyncio.create_task(coordinator.drain_scheduler_until_resolved(
            future=future, job=SimpleNamespace(job_id="stalled", metadata={}),
            session_key="s", trace_token="trace", deadline=time.time() + (1 if cancel else 0.03),
            scheduler=scheduler, kernel_executor=Mock(), lifecycle_projection=SimpleNamespace(get=lambda _id: None),
            mutate_runtime_plan=Mock(), emit_runtime_stream_event=Mock(), emit_progress_snapshot=Mock(),
        ))
        await asyncio.wait_for(entered.wait(), 1)
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await asyncio.wait_for(task, 1)
        assert future.cancelled()
        assert cleaned.is_set()
        assert not [item for item in asyncio.all_tasks() if item is not asyncio.current_task() and not item.done()]

    asyncio.run(run())


def test_coordinator_preserves_contextvars_on_success():
    async def run():
        scope = ContextVar("core_scope", default="before")
        future = asyncio.get_running_loop().create_future()

        async def drain(_executor):
            scope.set("after")
            future.set_result(("ok", False))
            return True

        await ExecutionCoordinator().drain_scheduler_until_resolved(
            future=future, job=SimpleNamespace(job_id="done", metadata={}),
            session_key="s", trace_token="trace", deadline=time.time() + 1,
            scheduler=SimpleNamespace(drain_once=drain), kernel_executor=Mock(),
            lifecycle_projection=Mock(), mutate_runtime_plan=Mock(),
            emit_runtime_stream_event=Mock(), emit_progress_snapshot=Mock(),
        )
        assert scope.get() == "after"

    asyncio.run(run())


def test_control_plane_timeout_cleans_scheduler_and_allows_next_turn():
    async def run():
        cleaned = asyncio.Event()
        timeouts = []

        async def execute(_session, job):
            timeouts.append(job.metadata["timeout_seconds"])
            if job.user_input == "stall":
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaned.set()
            return "second turn works", False

        control_plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=execute)
        with pytest.raises(TimeoutError):
            await control_plane.submit_turn(
                session_id="deadline-test", user_input="stall", timeout_seconds=0.03,
            )
        assert cleaned.is_set()
        with KernelGateway.open_scope():
            assert not control_plane.scheduler._jobs
        result = await control_plane.submit_turn(
            session_id="deadline-test", user_input="next",
        )
        assert result[0]
        assert timeouts == [0.03, 120.0]
        with KernelGateway.open_scope():
            assert not control_plane.scheduler._jobs

    asyncio.run(run())


def test_control_plane_external_cancellation_cleans_dedupe_slot():
    async def run():
        entered = asyncio.Event()
        cleaned = asyncio.Event()

        async def execute(_session, _job):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        control_plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=execute)
        task = asyncio.create_task(control_plane.submit_turn(
            session_id="cancel-test", user_input="stall",
            metadata={"request_id": "cancel-request"}, timeout_seconds=1,
        ))
        await asyncio.wait_for(entered.wait(), 1)
        dedupe_future, = control_plane._inflight_by_request.values()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned.is_set()
        assert dedupe_future.cancelled()
        assert not control_plane._inflight_by_request
        with KernelGateway.open_scope():
            assert not control_plane.scheduler._jobs

    asyncio.run(run())


def test_executor_timeout_cleans_active_context_without_finalizing(monkeypatch):
    from dadbot.core import orchestrator as module

    async def run():
        context = SimpleNamespace(trace_id="trace", metadata={})
        job = SimpleNamespace(user_input="stall", attachments=[], session_id="s", job_id="job",
                              metadata={"timeout_seconds": 0.03})
        orchestrator = DadBotOrchestrator.__new__(DadBotOrchestrator)
        orchestrator.bot = SimpleNamespace()
        orchestrator._job_builder = SimpleNamespace(build=lambda **_kwargs: context)
        for name in (
            "_attach_runtime_ledger_emit", "_hydrate_context_session_state", "_check_manifest_drift",
            "_prepare_execution_mode_from_checkpoint", "_run_subconscious_retrieval",
            "_stamp_determinism_metadata", "_emit_sovereign_event", "_analyze_behavioral_reflection",
            "_analyze_composite_friction", "_finalize_orchestrator_success", "_run_shadow_observation",
        ):
            setattr(orchestrator, name, Mock())
        cleaned = asyncio.Event()

        async def graph(**_kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        orchestrator._run_graph_with_trace_binding = graph
        monkeypatch.setattr(module, "_build_determinism_manifest", lambda: {})
        with pytest.raises(TransientExecutionError, match="exceeded 0.03s"):
            await orchestrator._execute_job({}, job)
        assert cleaned.is_set()
        assert orchestrator.bot._active_turn_context is None
        orchestrator._finalize_orchestrator_success.assert_not_called()
        orchestrator._run_shadow_observation.assert_not_called()

    asyncio.run(run())
    assert _normalize_timeout_seconds(None) == 120


def test_thread_bridge_timeout_returns_without_waiting_for_worker_shutdown():
    started = threading.Event()
    release = threading.Event()
    cleaned = threading.Event()

    async def blocked():
        started.set()
        try:
            release.wait(2)
            await asyncio.sleep(0)
        finally:
            cleaned.set()

    before = time.monotonic()
    try:
        with pytest.raises(TransientExecutionError, match="thread bridge exceeded"):
            run_coro_in_thread(blocked(), timeout_seconds=0.03)
        assert time.monotonic() - before < 0.5
        assert started.is_set()
    finally:
        release.set()
    assert cleaned.wait(1)


def test_thread_bridge_preserves_context_results_and_inner_timeout():
    scope = ContextVar("bridge_scope", default="missing")
    token = scope.set("caller")

    async def read_scope():
        return scope.get()

    async def inner_timeout():
        raise TimeoutError("inner operation timed out")

    try:
        assert run_coro_in_thread(read_scope()) == "caller"
        with pytest.raises(TimeoutError, match="inner operation timed out"):
            run_coro_in_thread(inner_timeout())
    finally:
        scope.reset(token)


def test_thread_bridge_requests_cooperative_cancellation():
    cleaned = threading.Event()

    async def blocked():
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    with pytest.raises(TransientExecutionError):
        run_coro_in_thread(blocked(), timeout_seconds=0.1)
    assert cleaned.wait(1)


def test_turn_lock_timeout_and_release_after_failure(monkeypatch):
    from dadbot.core import turn_mixin

    monkeypatch.setattr(turn_mixin, "TURN_LOCK_TIMEOUT_SECONDS", 0.03)
    lock = threading.Lock()
    bot = SimpleNamespace(_turn_execution_lock=lock)
    lock.acquire()
    try:
        with pytest.raises(TransientExecutionError, match="lock held"):
            DadBotTurnMixin._run_graph_turn_sync(bot, "hello")
    finally:
        lock.release()

    async def fail(*_args, **_kwargs):
        raise RuntimeError("graph failed")

    bot._run_graph_turn_async = fail
    with pytest.raises(RuntimeError, match="graph failed"):
        DadBotTurnMixin._run_graph_turn_sync(bot, "hello")
    assert lock.acquire(blocking=False)
    lock.release()
