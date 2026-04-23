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
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

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

    async def run(self, kernel_execute_fn: KernelExecuteFn) -> None:  # noqa: C901
        self._running = True
        logger.info("Scheduler started")
        while self._running:
            try:
                job = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
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

    def __init__(self, scheduler: Scheduler, registry: SessionRegistry) -> None:
        self.scheduler = scheduler
        self.registry = registry

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
    return ControlPlane(scheduler=scheduler, registry=registry)
