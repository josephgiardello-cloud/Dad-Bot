from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from dadbot.contracts import AttachmentList, FinalizedTurnResult
from dadbot.core.durable_checkpoint import DurableCheckpoint
from dadbot.core.execution_lease import ExecutionLease, LeaseConflictError
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.execution_boundary import ControlPlaneExecutionBoundary
from dadbot.core.idempotency_boundary import IdempotencyBoundary
from dadbot.core.invariant_gate import InvariantGate
from dadbot.core.ledger_reader import LedgerReader
from dadbot.core.ledger_writer import LedgerWriter
from dadbot.core.observability import CorrelationContext, get_metrics, get_exporter
from dadbot.core.recovery_manager import RecoveryManager
from dadbot.core.session_store import SessionStore


KernelExecutor = Callable[[dict[str, Any], "ExecutionJob"], Awaitable[FinalizedTurnResult]]
InMemoryExecutionLedger = ExecutionLedger


class SessionRegistry:
    """Runtime owner for session lifecycle and isolation boundaries."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    def create(self, session_id: str) -> dict[str, Any]:
        normalized = str(session_id or "default").strip() or "default"
        existing = self._sessions.get(normalized)
        if existing is not None:
            return existing

        session = {
            "session_id": normalized,
            "state": {},
            "status": "active",
            "event_log": [],
            "lock": asyncio.Lock(),
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._sessions[normalized] = session
        return session

    def get(self, session_id: str) -> dict[str, Any] | None:
        normalized = str(session_id or "default").strip() or "default"
        return self._sessions.get(normalized)

    def get_or_create(self, session_id: str) -> dict[str, Any]:
        return self.get(session_id) or self.create(session_id)

    def terminate(self, session_id: str) -> None:
        session = self.get_or_create(session_id)
        session["status"] = "terminated"
        session["updated_at"] = time.time()
        session.setdefault("event_log", []).append(
            {
                "event": "session.terminated",
                "session_id": session["session_id"],
                "occurred_at": time.time(),
            }
        )


@dataclass(slots=True)
class ExecutionJob:
    """Unit of work submitted to the global scheduler boundary."""

    session_id: str
    user_input: str
    attachments: AttachmentList | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""
    priority: int = 0
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    submitted_at: float = field(default_factory=time.time)


class Scheduler:
    """Global arbitration boundary for turn execution.

    Pulls pending work from the ledger reader (authoritative history) and
    guarantees serialized execution per session via a session lock.
    """

    def __init__(
        self,
        registry: SessionRegistry,
        *,
        reader: LedgerReader,
        writer: LedgerWriter,
        session_store: SessionStore | None = None,
        max_inflight_jobs: int = 1000,
        execution_lease: ExecutionLease | None = None,
        worker_id: str | None = None,
        gate: InvariantGate | None = None,
    ) -> None:
        self.registry = registry
        self._reader = reader
        self._writer = writer
        self._session_store = session_store
        self._jobs: dict[str, tuple[ExecutionJob, asyncio.Future]] = {}
        self._pending_job_ids: deque[str] = deque()
        self._drain_lock = asyncio.Lock()
        self._max_inflight_jobs = max(1, int(max_inflight_jobs or 1000))
        self._execution_lease = execution_lease
        self._worker_id: str = str(worker_id or uuid.uuid4().hex)
        self._gate = gate or InvariantGate()

    async def register(self, job: ExecutionJob) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        if len(self._jobs) >= self._max_inflight_jobs:
            raise RuntimeError("Scheduler backpressure: max inflight jobs reached")
        self._jobs[job.job_id] = (job, future)
        return future

    def _refresh_pending_from_ledger(self) -> None:
        seen = set(self._pending_job_ids)
        while True:
            event = self._reader.next_unprocessed_event(job_type="JOB_QUEUED")
            if event is None:
                break
            payload = dict(event.get("payload") or {})
            job_id = str(payload.get("job_id") or "").strip()
            if job_id and job_id in self._jobs and job_id not in seen:
                self._pending_job_ids.append(job_id)
                seen.add(job_id)

        for job_id in self._reader.get_pending_jobs():
            if job_id in self._jobs and job_id not in seen:
                self._pending_job_ids.append(job_id)
                seen.add(job_id)

    async def drain_once(self, executor: KernelExecutor) -> bool:
        async with self._drain_lock:
            self._refresh_pending_from_ledger()
            if not self._pending_job_ids:
                return False
            next_job_id = self._pending_job_ids.popleft()

        pair = self._jobs.get(next_job_id)
        if pair is None:
            return False
        job, future = pair

        self._writer.append_execution_witness(
            component="scheduler.drain_once",
            session_id=str(job.session_id),
            trace_id=str(getattr(job, "metadata", {}).get("trace_id") or ""),
            correlation_id=str(getattr(job, "metadata", {}).get("correlation_id") or ""),
            payload={"job_id": str(job.job_id)},
        )

        session = self.registry.get_or_create(job.session_id)
        if str(session.get("status") or "active") != "active":
            self._writer.append_job_failed(job, f"Session {job.session_id!r} is terminated")
            if not future.done():
                future.set_exception(RuntimeError(f"Session {job.session_id!r} is terminated"))
            self._jobs.pop(job.job_id, None)
            return True

        # Lease enforcement: acquire ownership before executing.
        # If a different worker already holds the lease, re-queue and yield.
        lease_token: dict | None = None
        if self._execution_lease is not None:
            try:
                lease_token = self._execution_lease.acquire(
                    session_id=job.session_id,
                    owner_id=self._worker_id,
                )
            except LeaseConflictError:
                # Re-enqueue to retry after other worker releases.
                async with self._drain_lock:
                    self._pending_job_ids.appendleft(job.job_id)
                return False

        # InvariantGate: hard-fail if execution pre-conditions are violated.
        self._gate.validate_job(session, job)

        async with session["lock"]:
            self._writer.append_job_started(job)
            session["updated_at"] = time.time()
            session.setdefault("event_log", []).append(
                {
                    "event": "job.started",
                    "job_id": job.job_id,
                    "session_id": job.session_id,
                    "occurred_at": time.time(),
                }
            )
            job_start_time = time.monotonic()
            try:
                result = await executor(session, job)
                elapsed_ms = (time.monotonic() - job_start_time) * 1000.0
                completed_event = self._writer.append_job_completed(job, result)
                if not future.done():
                    future.set_result(result)
                if self._session_store is not None:
                    self._session_store.apply_event(completed_event)
                session.setdefault("event_log", []).append(
                    {
                        "event": "job.completed",
                        "job_id": job.job_id,
                        "session_id": job.session_id,
                        "occurred_at": time.time(),
                    }
                )
                get_metrics().increment("scheduler.job.completed")
                get_metrics().observe("scheduler.job.latency_ms", elapsed_ms)
                get_exporter().export({
                    "event": "job.completed",
                    "job_id": job.job_id,
                    "session_id": job.session_id,
                    "latency_ms": elapsed_ms,
                })
            except Exception as exc:
                elapsed_ms = (time.monotonic() - job_start_time) * 1000.0
                failed_event = self._writer.append_job_failed(job, str(exc))
                if not future.done():
                    future.set_exception(exc)
                if self._session_store is not None:
                    self._session_store.apply_event(failed_event)
                session.setdefault("event_log", []).append(
                    {
                        "event": "job.failed",
                        "job_id": job.job_id,
                        "session_id": job.session_id,
                        "error": str(exc),
                        "occurred_at": time.time(),
                    }
                )
                get_metrics().increment("scheduler.job.failed")
                get_metrics().observe("scheduler.job.latency_ms", elapsed_ms)
            finally:
                self._jobs.pop(job.job_id, None)

        # Release session lease after execution completes.
        if self._execution_lease is not None and lease_token is not None:
            self._execution_lease.release(session_id=job.session_id, owner_id=self._worker_id)

        return True

    async def submit_and_wait(
        self,
        job: ExecutionJob,
        *,
        executor: KernelExecutor,
        timeout_seconds: float | None = None,
    ) -> FinalizedTurnResult:
        future = await self.register(job)
        deadline = None
        if timeout_seconds is not None:
            deadline = time.monotonic() + max(0.1, float(timeout_seconds))

        while not future.done():
            drained = await self.drain_once(executor)
            if not drained:
                await asyncio.sleep(0)
            if deadline is not None and time.monotonic() >= deadline and not future.done():
                raise TimeoutError("Timed out waiting for scheduled turn")

        return await future


class ExecutionControlPlane:
    """Unified execution owner over kernel, graph, tools, and policy.

    API/entrypoints submit turns to this control plane. The control plane is the
    only structure allowed to schedule graph execution, providing a single
    arbitration layer for lifecycle + scheduling.
    """

    def __init__(
        self,
        *,
        registry: SessionRegistry,
        kernel_executor: KernelExecutor,
        ledger: InMemoryExecutionLedger | None = None,
        scheduler: Scheduler | None = None,
        idempotency: IdempotencyBoundary | None = None,
        execution_lease: ExecutionLease | None = None,
        worker_id: str | None = None,
    ) -> None:
        self.registry = registry
        self._kernel_executor = kernel_executor
        self.ledger = ledger or InMemoryExecutionLedger()
        self.ledger_writer = LedgerWriter(self.ledger)
        self.ledger_reader = LedgerReader(self.ledger)
        self.session_store = SessionStore(ledger=self.ledger, projection_only=True)
        self.idempotency = idempotency or IdempotencyBoundary()
        self.recovery = RecoveryManager(ledger=self.ledger)
        self.checkpoint = DurableCheckpoint(ledger=self.ledger)
        self.execution_lease = execution_lease or ExecutionLease()
        self._worker_id = str(worker_id or uuid.uuid4().hex)
        self._execution_token = uuid.uuid4().hex
        self.scheduler = scheduler or Scheduler(
            registry,
            reader=self.ledger_reader,
            writer=self.ledger_writer,
            session_store=self.session_store,
            execution_lease=self.execution_lease,
            worker_id=self._worker_id,
            gate=InvariantGate(),
        )

    @property
    def execution_token(self) -> str:
        return self._execution_token

    async def _kernel_executor_with_boundary(self, session: dict[str, Any], job: ExecutionJob) -> FinalizedTurnResult:
        with ControlPlaneExecutionBoundary.bind(self._execution_token):
            return await self._kernel_executor(session, job)

    def ledger_events(self) -> list[dict[str, Any]]:
        return self.ledger.snapshot()

    async def create_session(self, session_id: str) -> dict[str, Any]:
        return self.registry.create(session_id)

    def recover_runtime_state(self) -> dict[str, Any]:
        return self.recovery.recover(session_store=self.session_store)

    def boot_reconcile(self) -> dict[str, Any]:
        """Run startup reconciliation gate.  Must be called before accepting turns."""
        return self.recovery.boot_reconcile(
            session_store=self.session_store,
            checkpoint=self.checkpoint,
        )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        return self.registry.get(session_id)

    def terminate_session(self, session_id: str) -> None:
        self.registry.terminate(session_id)

    async def submit_turn(
        self,
        *,
        session_id: str,
        user_input: str,
        attachments: AttachmentList | None = None,
        metadata: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> FinalizedTurnResult:
        metadata_payload = dict(metadata or {})
        # Ensure correlation + trace are present at scheduler job creation so
        # all subsequent ledger lifecycle events can be causally linked.
        correlation_id = str(
            metadata_payload.get("correlation_id")
            or CorrelationContext.current()
            or CorrelationContext.ensure()
            or ""
        ).strip()
        trace_id = str(
            metadata_payload.get("trace_id")
            or correlation_id
            or uuid.uuid4().hex
        ).strip()
        metadata_payload["correlation_id"] = correlation_id
        metadata_payload["trace_id"] = trace_id

        self.ledger_writer.append_execution_witness(
            component="control_plane.submit_turn",
            session_id=str(session_id or "default"),
            trace_id=trace_id,
            correlation_id=correlation_id,
            payload={
                "request_id": str(metadata_payload.get("request_id") or ""),
            },
        )

        request_id = str(
            metadata_payload.get("request_id")
            or metadata_payload.get("idempotency_key")
            or ""
        ).strip()
        shared_result_future: asyncio.Future | None = None

        if request_id:
            status, cached, inflight = self.idempotency.acquire_or_get(
                session_id=session_id,
                request_id=request_id,
                loop=asyncio.get_running_loop(),
            )
            if status == "cached":
                return cached
            if status == "inflight" and inflight is not None:
                return await inflight
            if status == "acquired":
                shared_result_future = inflight

        session = self.registry.get_or_create(session_id)
        job = ExecutionJob(
            session_id=session_id,
            user_input=str(user_input or ""),
            attachments=list(attachments or []),
            metadata=metadata_payload,
            request_id=request_id,
        )

        # Mandatory pre-scheduling ledger gates.
        self.ledger_writer.append_job_submitted(job)
        self.ledger_writer.append_session_bound(session.get("session_id") or session_id, job.job_id)

        future = await self.scheduler.register(job)

        # Mandatory post-admission ledger gate.
        self.ledger_writer.append_job_queued(job)

        deadline = None
        if timeout_seconds is not None:
            deadline = time.monotonic() + max(0.1, float(timeout_seconds))

        while not future.done():
            drained = await self.scheduler.drain_once(self._kernel_executor_with_boundary)
            if not drained:
                await asyncio.sleep(0)
            if deadline is not None and time.monotonic() >= deadline and not future.done():
                raise TimeoutError("Timed out waiting for scheduled turn")

        try:
            result = await future
        except Exception as error:
            if request_id:
                self.idempotency.store_error(session_id=session_id, request_id=request_id, error=error)
            raise

        if request_id:
            self.idempotency.store_result(session_id=session_id, request_id=request_id, result=result)
            if shared_result_future is not None and not shared_result_future.done():
                shared_result_future.set_result(result)
        return result
