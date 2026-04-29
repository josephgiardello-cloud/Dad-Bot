from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from uuid import uuid4

from dadbot.contracts import AttachmentList, FinalizedTurnResult
from dadbot.core.execution_boundary import ControlPlaneExecutionBoundary
from dadbot.core.execution_lease import ExecutionLease, LeaseConflictError
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_reader import LedgerReader
from dadbot.core.ledger_writer import LedgerWriter
from dadbot.core.observability import get_exporter, get_metrics, get_tracer
from dadbot.core.recovery_manager import RecoveryManager
from dadbot.core.session_store import SessionStore


@dataclass(slots=True)
class ExecutionJob:
    session_id: str
    user_input: str
    attachments: AttachmentList | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    job_id: str = field(default_factory=lambda: f"job-{uuid4().hex}")


class SessionRegistry:
    """Simple in-memory session registry used by the scheduler."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    def bind(self, session_id: str) -> dict[str, Any]:
        sid = str(session_id or "default")
        session = self._sessions.get(sid)
        if session is None:
            session = {"session_id": sid, "state": {}}
            self._sessions[sid] = session
        return session


class Scheduler:
    """Single-node async scheduler with lease-aware drain semantics."""

    def __init__(
        self,
        registry: SessionRegistry,
        *,
        reader: LedgerReader,
        writer: LedgerWriter,
        max_inflight_jobs: int = 16,
        execution_lease: ExecutionLease | None = None,
        worker_id: str = "worker-1",
        execution_token: str = "",
        enable_observability: bool = True,
    ) -> None:
        self.registry = registry
        self.reader = reader
        self.writer = writer
        self.max_inflight_jobs = int(max_inflight_jobs)
        self.execution_lease = execution_lease or ExecutionLease()
        self.worker_id = str(worker_id or "worker-1")
        self.execution_token = str(execution_token or "")
        self.enable_observability = bool(enable_observability)

        self._jobs: dict[str, tuple[ExecutionJob, asyncio.Future[FinalizedTurnResult]]] = {}
        self._pending_job_ids: list[str] = []

    async def register(self, job: ExecutionJob) -> asyncio.Future[FinalizedTurnResult]:
        if len(self._jobs) >= self.max_inflight_jobs:
            raise RuntimeError("backpressure: max inflight jobs reached")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[FinalizedTurnResult] = loop.create_future()
        self._jobs[job.job_id] = (job, future)
        self._pending_job_ids.append(job.job_id)
        self.writer.append_job_queued(job)
        return future

    async def drain_once(
        self,
        executor: Callable[[dict[str, Any], ExecutionJob], Awaitable[FinalizedTurnResult]],
    ) -> bool:
        if not self._pending_job_ids:
            return False

        job_id = self._pending_job_ids.pop(0)
        job_pair = self._jobs.get(job_id)
        if job_pair is None:
            return False
        job, future = job_pair

        lease_acquired = False
        started_at = time.perf_counter()

        try:
            self.execution_lease.acquire(session_id=job.session_id, owner_id=self.worker_id, ttl_seconds=30.0)
            lease_acquired = True
        except LeaseConflictError:
            self._pending_job_ids.append(job_id)
            return False

        try:
            self.writer.append_job_started(job)
            session = self.registry.bind(job.session_id)
            tracer = get_tracer()
            with tracer.span("scheduler.drain_once"):
                if self.execution_token:
                    with ControlPlaneExecutionBoundary.bind(self.execution_token):
                        result = await executor(session, job)
                else:
                    result = await executor(session, job)
            self.writer.append_job_completed(job, result)
            if not future.done():
                future.set_result(result)
            if self.enable_observability:
                metrics = get_metrics()
                metrics.increment("scheduler.job.completed")
                metrics.observe("scheduler.job.latency_ms", (time.perf_counter() - started_at) * 1000.0)
                get_exporter().export(
                    {
                        "event": "job.completed",
                        "job_id": job.job_id,
                        "session_id": job.session_id,
                    }
                )
            return True
        except Exception as exc:
            self.writer.append_job_failed(job, str(exc))
            if not future.done():
                future.set_exception(exc)
            if self.enable_observability:
                metrics = get_metrics()
                metrics.increment("scheduler.job.failed")
                metrics.observe("scheduler.job.latency_ms", (time.perf_counter() - started_at) * 1000.0)
                get_exporter().export(
                    {
                        "event": "job.failed",
                        "job_id": job.job_id,
                        "session_id": job.session_id,
                        "error": str(exc),
                    }
                )
            raise
        finally:
            self._jobs.pop(job_id, None)
            if lease_acquired:
                self.execution_lease.release(session_id=job.session_id, owner_id=self.worker_id)


class ExecutionControlPlane:
    """Execution boundary around scheduler, lease, ledger, and recovery."""

    def __init__(
        self,
        *,
        registry: SessionRegistry,
        kernel_executor: Callable[[dict[str, Any], ExecutionJob], Awaitable[FinalizedTurnResult]],
        max_inflight_jobs: int = 16,
        execution_lease: ExecutionLease | None = None,
        worker_id: str = "worker-1",
        graph: Any | None = None,
        enable_observability: bool = True,
    ) -> None:
        self.registry = registry
        self.kernel_executor = kernel_executor
        self.execution_token = f"exec-{uuid4().hex}"
        self.ledger = ExecutionLedger()
        self.ledger_writer = LedgerWriter(self.ledger)
        self.ledger_reader = LedgerReader(self.ledger)
        self.execution_lease = execution_lease or ExecutionLease()
        self.scheduler = Scheduler(
            registry,
            reader=self.ledger_reader,
            writer=self.ledger_writer,
            max_inflight_jobs=max_inflight_jobs,
            execution_lease=self.execution_lease,
            worker_id=worker_id,
            execution_token=self.execution_token,
            enable_observability=enable_observability,
        )
        self.recovery = RecoveryManager(self.ledger)

        self.graph = graph
        if self.graph is not None and callable(getattr(self.graph, "set_required_execution_token", None)):
            self.graph.set_required_execution_token(self.execution_token)
        if self.graph is not None and callable(getattr(self.graph, "set_execution_witness_emitter", None)):
            self.graph.set_execution_witness_emitter(self._emit_execution_witness)

    def _emit_execution_witness(self, component: str, turn_context: Any) -> None:
        trace_id = str(getattr(turn_context, "trace_id", "") or "")
        session_id = str((getattr(turn_context, "metadata", {}) or {}).get("session_id") or "")
        self.ledger_writer.append_runtime_witness(component=component, trace_id=trace_id, session_id=session_id)

    async def submit_turn(
        self,
        *,
        session_id: str,
        user_input: str,
        attachments: AttachmentList | None = None,
        metadata: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> FinalizedTurnResult:
        session_key = str(session_id or "default")
        job = ExecutionJob(
            session_id=session_key,
            user_input=str(user_input or ""),
            attachments=attachments,
            metadata=dict(metadata or {}),
        )

        self.ledger_writer.append_session_bound(session_key)
        self.ledger_writer.append_job_submitted(job)

        future = await self.scheduler.register(job)

        # Single-process execution path: drain immediately.
        try:
            await self.scheduler.drain_once(self.kernel_executor)
        except Exception:
            # Scheduler stores the failure on the per-job future; consume it so
            # asyncio does not report "Future exception was never retrieved".
            if future.done():
                _ = future.exception()
            raise

        if timeout_seconds is None:
            return await future
        return await asyncio.wait_for(future, timeout=float(timeout_seconds or 30.0))

    def ledger_events(self) -> list[dict[str, Any]]:
        return self.ledger.read()

    def boot_reconcile(self) -> dict[str, Any]:
        store = SessionStore(ledger=self.ledger, projection_only=True)
        return self.recovery.recover(session_store=store)
