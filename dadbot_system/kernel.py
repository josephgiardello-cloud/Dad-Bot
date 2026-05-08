"""dadbot_system.kernel — Kernel/Client split: core scheduling primitives.

Four components that decouple API surface from execution engine:

  SessionRegistry  — ownership + isolation boundary
  ExecutionJob     — unit of work (every turn becomes a job)
  Scheduler        — async background loop; the actual "platform switch"
  ControlPlane     — orchestrator facade; what the API calls
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Type alias for the coroutine the Scheduler calls per job.
KernelExecuteFn = Callable[[dict[str, Any], dict[str, Any]], Awaitable[Any]]


# ──────────────────────────────────────────────────────────────────────────────
# (1) SessionRegistry — ownership + isolation boundary
# ──────────────────────────────────────────────────────────────────────────────


class SessionRegistry:
    """Source of truth for runtime session lifecycle.

    Each entry holds lightweight control-plane state — an asyncio Lock for
    serialised per-session execution, status, and an event log.  The
    DadBotOrchestrator continues to own the heavier session runtime state;
    this registry sits above it and provides the isolation guarantee.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    def create(self, session_id: str) -> dict[str, Any]:
        """Create and return the session entry (idempotent)."""
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "session_id": session_id,
                "state": {},
                "status": "active",
                "event_log": [],
                "lock": asyncio.Lock(),
            }
        return self._sessions[session_id]

    def get(self, session_id: str) -> dict[str, Any] | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str) -> dict[str, Any]:
        return self.get(session_id) or self.create(session_id)

    def terminate(self, session_id: str) -> None:
        entry = self._sessions.get(session_id)
        if entry is not None:
            entry["status"] = "terminated"

    def list_sessions(self) -> list[str]:
        return list(self._sessions)

    def __len__(self) -> int:
        return len(self._sessions)


# ──────────────────────────────────────────────────────────────────────────────
# (2) ExecutionJob — unit of work
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ExecutionJob:
    """Every turn becomes a job, not a direct call.

    This is what decouples API from execution: the API submits a job and
    awaits ``_result_future``; the Scheduler drives actual execution.
    """

    session_id: str
    input_payload: dict[str, Any]
    priority: int = 0
    # Resolved by Scheduler once kernel_execute_fn completes.
    _result_future: asyncio.Future | None = field(default=None, repr=False, compare=False)


@dataclass
class KernelTask:
    """Registered async task tracked by the kernel task manager."""

    task_id: str
    name: str
    task: asyncio.Task[Any]
    session_id: str | None = None
    created_at: float = field(default_factory=time.time)


class KernelTaskManager:
    """Tracks background asyncio tasks and ingests completion/failure outcomes.

    This prevents silent "Future exception was never retrieved" leaks by ensuring
    all task outcomes are observed and recorded through a deterministic manager.
    """

    def __init__(self, registry: SessionRegistry) -> None:
        self._registry = registry
        self._tasks: dict[str, KernelTask] = {}
        self._counter = 0

    def register(
        self,
        *,
        name: str,
        coro: Coroutine[Any, Any, Any],
        session_id: str | None = None,
    ) -> asyncio.Task[Any]:
        """Create and register a task; completion is always observed via callback."""
        self._counter += 1
        task_id = f"kt-{self._counter:08d}"
        task = asyncio.create_task(coro)
        record = KernelTask(
            task_id=task_id,
            name=str(name or "unnamed"),
            task=task,
            session_id=str(session_id).strip() if session_id else None,
        )
        self._tasks[task_id] = record
        self._append_event(record, status="submitted")

        def _on_done(done_task: asyncio.Task[Any], *, _task_id: str = task_id) -> None:
            current = self._tasks.get(_task_id)
            if current is None:
                return
            try:
                if done_task.cancelled():
                    self._append_event(current, status="cancelled")
                else:
                    exc = done_task.exception()
                    if exc is None:
                        self._append_event(current, status="completed")
                    else:
                        self._append_event(current, status="failed", error=str(exc))
            except Exception as callback_exc:  # noqa: BLE001  # pragma: no cover - defensive callback guard
                self._append_event(current, status="failed", error=f"callback-error: {callback_exc}")
            finally:
                self._tasks.pop(_task_id, None)

        task.add_done_callback(_on_done)
        return task

    async def await_all(self) -> list[Any]:
        """Await currently registered tasks and ingest all exceptions."""
        pending = [item.task for item in list(self._tasks.values()) if not item.task.done()]
        if not pending:
            return []
        # Gather with return_exceptions=True ensures all failures are retrieved.
        return await asyncio.gather(*pending, return_exceptions=True)

    async def await_session(self, session_id: str) -> list[Any]:
        """Await pending tasks registered for one session only."""
        session_key = str(session_id or "").strip()
        if not session_key:
            return []
        pending = [
            item.task
            for item in list(self._tasks.values())
            if item.session_id == session_key and not item.task.done()
        ]
        if not pending:
            return []
        return await asyncio.gather(*pending, return_exceptions=True)

    async def shutdown(self, *, cancel_pending: bool = True) -> None:
        """Stop all tracked tasks deterministically and ingest their outcomes."""
        if cancel_pending:
            for item in list(self._tasks.values()):
                if not item.task.done():
                    item.task.cancel()
        await self.await_all()

    @property
    def pending_count(self) -> int:
        return sum(1 for item in self._tasks.values() if not item.task.done())

    def _append_event(self, task: KernelTask, *, status: str, error: str | None = None) -> None:
        payload = {
            "event": "kernel_task",
            "task_id": task.task_id,
            "name": task.name,
            "session_id": task.session_id,
            "status": status,
            "error": str(error or ""),
            "created_at": float(task.created_at),
            "recorded_at": time.time(),
        }
        logger.debug("Kernel task event: %s", payload)
        if task.session_id:
            session = self._registry.get(task.session_id)
            if session is not None:
                session.setdefault("event_log", []).append(payload)


# ──────────────────────────────────────────────────────────────────────────────
# (3) Scheduler — the actual "platform switch"
# ──────────────────────────────────────────────────────────────────────────────


class Scheduler:
    """Async background loop that drains the job queue and calls the kernel.

    Turns are serialised per session via the per-session asyncio.Lock from
    SessionRegistry.  Concurrent sessions run in parallel; turns within a
    session are strictly ordered.

    Usage::

        scheduler = Scheduler(registry)
        asyncio.create_task(scheduler.run(kernel_execute_fn))
    """

    def __init__(self, registry: SessionRegistry) -> None:
        self.queue: asyncio.Queue[ExecutionJob] = asyncio.Queue()
        self.registry = registry
        self._running = False

    async def submit(self, job: ExecutionJob) -> None:
        await self.queue.put(job)

    async def run(self, kernel_execute_fn: KernelExecuteFn) -> None:
        self._running = True
        logger.info("Scheduler started")
        while self._running:
            try:
                job = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except TimeoutError:
                continue

            session = self.registry.get_or_create(job.session_id)

            if session["status"] != "active":
                _reject(job, RuntimeError(f"Session {job.session_id!r} is {session['status']!r}"))
                self.queue.task_done()
                continue

            async with session["lock"]:
                try:
                    result = await kernel_execute_fn(session, job.input_payload)
                    _resolve(job, result)
                except Exception as exc:  # pragma: no cover
                    logger.exception("Kernel execution failed for session %s", job.session_id)
                    _reject(job, exc)

            self.queue.task_done()

        logger.info("Scheduler stopped")

    def stop(self) -> None:
        self._running = False


def _resolve(job: ExecutionJob, result: Any) -> None:
    if job._result_future is not None and not job._result_future.done():
        job._result_future.set_result(result)


def _reject(job: ExecutionJob, exc: Exception) -> None:
    if job._result_future is not None and not job._result_future.done():
        job._result_future.set_exception(exc)


# ──────────────────────────────────────────────────────────────────────────────
# (4) ControlPlane — orchestrator facade
# ──────────────────────────────────────────────────────────────────────────────


class ControlPlane:
    """What the API calls.  Owns Scheduler + SessionRegistry.

    ``submit_turn()`` returns an ``asyncio.Future`` that resolves to the
    value returned by the ``kernel_execute_fn`` once the Scheduler processes
    the job.  The API endpoint awaits the future (with its own timeout).
    """

    def __init__(self, scheduler: Scheduler, registry: SessionRegistry, task_manager: KernelTaskManager) -> None:
        self.scheduler = scheduler
        self.registry = registry
        self.task_manager = task_manager

    async def create_session(self, session_id: str) -> dict[str, Any]:
        return self.registry.create(session_id)

    async def submit_turn(self, session_id: str, payload: dict[str, Any]) -> asyncio.Future:
        """Submit a turn job and return a Future that resolves to the result dict."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        job = ExecutionJob(
            session_id=session_id,
            input_payload=payload,
            _result_future=future,
        )
        await self.scheduler.submit(job)
        return future

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        return self.registry.get(session_id)

    def terminate_session(self, session_id: str) -> None:
        self.registry.terminate(session_id)


# ──────────────────────────────────────────────────────────────────────────────
# Factory helper
# ──────────────────────────────────────────────────────────────────────────────


def build_control_plane() -> ControlPlane:
    """Return a ready-to-use ControlPlane backed by a fresh SessionRegistry."""
    registry = SessionRegistry()
    scheduler = Scheduler(registry)
    task_manager = KernelTaskManager(registry)
    return ControlPlane(scheduler=scheduler, registry=registry, task_manager=task_manager)
