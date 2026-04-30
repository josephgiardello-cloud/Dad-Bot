from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from dadbot.contracts import AttachmentList, FinalizedTurnResult
from dadbot.core.execution_boundary import ControlPlaneExecutionBoundary
from dadbot.core.execution_lease import ExecutionLease, LeaseConflictError
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.execution_ledger_memory import InMemoryExecutionLedger
from dadbot.core.ledger_reader import LedgerReader
from dadbot.core.ledger_writer_adapter import LedgerWriterAdapter
from dadbot.core.observability import get_exporter, get_metrics, get_tracer
from dadbot.core.recovery_manager import RecoveryManager
from dadbot.core.session_store import SessionStore


@dataclass(slots=True)
class ExecutionJob:
    session_id: str
    user_input: str
    attachments: AttachmentList | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""
    job_id: str = field(default_factory=lambda: f"job-{uuid4().hex}")

    def __post_init__(self) -> None:
        metadata = dict(self.metadata or {})
        trace_id = str(self.trace_id or metadata.get("trace_id") or "").strip()
        if not trace_id:
            trace_id = f"tr-{uuid4().hex}"
        metadata["trace_id"] = trace_id
        self.metadata = metadata
        self.trace_id = trace_id


@dataclass(slots=True)
class SchedulerOptions:
    max_inflight_jobs: int = 16
    worker_id: str = "worker-1"
    execution_token: str = ""
    enable_observability: bool = True
    execution_lease: ExecutionLease | None = None


@dataclass(slots=True)
class ControlPlaneOptions:
    max_inflight_jobs: int = 16
    worker_id: str = "worker-1"
    enable_observability: bool = True
    execution_lease: ExecutionLease | None = None
    ledger: ExecutionLedger | None = None
    scheduler: Scheduler | None = None


class SessionRegistry:
    """Simple in-memory session registry used by the scheduler."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._terminated: set[str] = set()

    def bind(self, session_id: str) -> dict[str, Any]:
        sid = str(session_id or "default")
        session = self._sessions.get(sid)
        if session is None:
            session = {"session_id": sid, "state": {}}
            self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> dict[str, Any] | None:
        return self._sessions.get(str(session_id or "default"))

    def get_or_create(self, session_id: str) -> dict[str, Any]:
        return self.bind(session_id)

    async def create_session(self, session_id: str) -> dict[str, Any]:
        sid = str(session_id or "default")
        self._terminated.discard(sid)
        return self.bind(sid)

    def terminate_session(self, session_id: str) -> None:
        self._terminated.add(str(session_id or "default"))

    def is_terminated(self, session_id: str) -> bool:
        return str(session_id or "default") in self._terminated


class Scheduler:
    """Single-node async scheduler with lease-aware drain semantics."""

    def __init__(
        self,
        registry: SessionRegistry,
        *,
        reader: LedgerReader,
        writer: LedgerWriter,
        options: SchedulerOptions | None = None,
        **legacy_options: Any,
    ) -> None:
        resolved_options = self._resolve_options(options, legacy_options)
        self.registry = registry
        self.reader = reader
        self.writer = writer
        self.max_inflight_jobs = int(resolved_options.max_inflight_jobs)
        self.execution_lease = resolved_options.execution_lease or ExecutionLease()
        self.worker_id = str(resolved_options.worker_id or "worker-1")
        self.execution_token = str(resolved_options.execution_token or "")
        self.enable_observability = bool(resolved_options.enable_observability)

        self._jobs: dict[
            str,
            tuple[ExecutionJob, asyncio.Future[FinalizedTurnResult]],
        ] = {}
        self._pending_job_ids: list[str] = []

    @staticmethod
    def _resolve_options(
        options: SchedulerOptions | None,
        legacy_options: dict[str, Any],
    ) -> SchedulerOptions:
        resolved = options or SchedulerOptions()
        if "max_inflight_jobs" in legacy_options:
            resolved.max_inflight_jobs = int(legacy_options["max_inflight_jobs"])
        if "execution_lease" in legacy_options:
            resolved.execution_lease = legacy_options["execution_lease"]
        if "worker_id" in legacy_options:
            resolved.worker_id = str(legacy_options["worker_id"] or "worker-1")
        if "execution_token" in legacy_options:
            resolved.execution_token = str(legacy_options["execution_token"] or "")
        if "enable_observability" in legacy_options:
            resolved.enable_observability = bool(legacy_options["enable_observability"])
        return resolved

    async def _execute_with_boundary(
        self,
        executor: Callable[[dict[str, Any], ExecutionJob], Awaitable[FinalizedTurnResult]],
        session: dict[str, Any],
        job: ExecutionJob,
    ) -> FinalizedTurnResult:
        if self.execution_token:
            with ControlPlaneExecutionBoundary.bind(self.execution_token):
                return await executor(session, job)
        return await executor(session, job)

    @staticmethod
    def _resolve_future(
        future: asyncio.Future[FinalizedTurnResult],
        *,
        result: FinalizedTurnResult | None = None,
        error: Exception | None = None,
    ) -> None:
        if future.done():
            return
        if error is not None:
            future.set_exception(error)
            return
        if result is not None:
            future.set_result(result)

    def _record_job_observability(
        self,
        *,
        event: str,
        job: ExecutionJob,
        started_at: float,
        error: str = "",
    ) -> None:
        if not self.enable_observability:
            return
        metrics = get_metrics()
        metrics.increment(f"scheduler.job.{event}")
        metrics.observe(
            "scheduler.job.latency_ms",
            (time.perf_counter() - started_at) * 1000.0,
        )
        payload = {
            "event": f"job.{event}",
            "job_id": job.job_id,
            "session_id": job.session_id,
        }
        if error:
            payload["error"] = error
        get_exporter().export(payload)

    async def register(self, job: ExecutionJob) -> asyncio.Future[FinalizedTurnResult]:
        if len(self._jobs) >= self.max_inflight_jobs:
            raise RuntimeError("backpressure: max inflight jobs reached")
        assert str(job.trace_id or "").strip(), "Missing trace_id at scheduler register"

        loop = asyncio.get_running_loop()
        future: asyncio.Future[FinalizedTurnResult] = loop.create_future()
        self._jobs[job.job_id] = (job, future)
        self._pending_job_ids.append(job.job_id)
        self.writer.append_job_queued(job)
        return future

    async def drain_once(
        self,
        executor: Callable[
            [dict[str, Any], ExecutionJob],
            Awaitable[FinalizedTurnResult],
        ],
    ) -> bool:
        if not self._pending_job_ids:
            return False

        job_id = self._pending_job_ids.pop(0)
        job_pair = self._jobs.get(job_id)
        if job_pair is None:
            return False
        job, future = job_pair
        assert str(job.trace_id or "").strip(), "Missing trace_id at scheduler drain"

        lease_acquired = False
        started_at = time.perf_counter()

        try:
            self.execution_lease.acquire(
                session_id=job.session_id,
                owner_id=self.worker_id,
                ttl_seconds=30.0,
            )
            lease_acquired = True
        except LeaseConflictError:
            self._pending_job_ids.append(job_id)
            return False

        try:
            self.writer.append_job_started(job)
            session = self.registry.bind(job.session_id)
            tracer = get_tracer()
            with tracer.span("scheduler.drain_once"):
                result = await self._execute_with_boundary(executor, session, job)
            self.writer.append_job_completed(job, result)
            self._resolve_future(future, result=result)
            self._record_job_observability(
                event="completed",
                job=job,
                started_at=started_at,
            )
            return True
        except Exception as exc:
            self.writer.append_job_failed(job, str(exc))
            self._resolve_future(future, error=exc)
            self._record_job_observability(
                event="failed",
                job=job,
                started_at=started_at,
                error=str(exc),
            )
            raise
        finally:
            if future.done():
                self._jobs.pop(job_id, None)
            if lease_acquired:
                self.execution_lease.release(
                    session_id=job.session_id,
                    owner_id=self.worker_id,
                )


class ExecutionControlPlane:
    """Execution boundary around scheduler, lease, ledger, and recovery."""

    def __init__(
        self,
        *,
        registry: SessionRegistry,
        kernel_executor: Callable[
            [dict[str, Any], ExecutionJob],
            Awaitable[FinalizedTurnResult],
        ],
        graph: Any | None = None,
        options: ControlPlaneOptions | None = None,
        **legacy_options: Any,
    ) -> None:
        resolved_options = self._resolve_options(options, legacy_options)
        self.registry = registry
        self.kernel_executor = kernel_executor
        self.execution_token = f"exec-{uuid4().hex}"
        self.ledger = resolved_options.ledger or InMemoryExecutionLedger()
        self.ledger_writer = LedgerWriterAdapter(self.ledger)
        self.ledger_reader = LedgerReader(self.ledger)
        self.execution_lease = resolved_options.execution_lease or ExecutionLease()
        scheduler_options = SchedulerOptions(
            max_inflight_jobs=resolved_options.max_inflight_jobs,
            worker_id=resolved_options.worker_id,
            execution_token=self.execution_token,
            enable_observability=resolved_options.enable_observability,
            execution_lease=self.execution_lease,
        )
        self.scheduler = resolved_options.scheduler or Scheduler(
            registry,
            reader=self.ledger_reader,
            writer=self.ledger_writer,
            options=scheduler_options,
        )
        self.recovery = RecoveryManager(self.ledger)
        self.graph = graph
        self._inflight_by_request: dict[
            tuple[str, str],
            asyncio.Future[FinalizedTurnResult],
        ] = {}

        if self.graph is not None and callable(
            getattr(self.graph, "set_required_execution_token", None),
        ):
            self.graph.set_required_execution_token(self.execution_token)
        if self.graph is not None and callable(
            getattr(self.graph, "set_execution_witness_emitter", None),
        ):
            self.graph.set_execution_witness_emitter(self._emit_execution_witness)

    @staticmethod
    def _resolve_options(
        options: ControlPlaneOptions | None,
        legacy_options: dict[str, Any],
    ) -> ControlPlaneOptions:
        resolved = options or ControlPlaneOptions()
        if "max_inflight_jobs" in legacy_options:
            resolved.max_inflight_jobs = int(legacy_options["max_inflight_jobs"])
        if "execution_lease" in legacy_options:
            resolved.execution_lease = legacy_options["execution_lease"]
        if "worker_id" in legacy_options:
            resolved.worker_id = str(legacy_options["worker_id"] or "worker-1")
        if "enable_observability" in legacy_options:
            resolved.enable_observability = bool(legacy_options["enable_observability"])
        if "ledger" in legacy_options:
            resolved.ledger = legacy_options["ledger"]
        if "scheduler" in legacy_options:
            resolved.scheduler = legacy_options["scheduler"]
        return resolved

    async def create_session(self, session_id: str) -> dict[str, Any]:
        return await self.registry.create_session(session_id)

    def terminate_session(self, session_id: str) -> None:
        self.registry.terminate_session(session_id)

    def _emit_execution_witness(self, component: str, turn_context: Any) -> None:
        trace_id = str(getattr(turn_context, "trace_id", "") or "")
        session_id = str(
            (getattr(turn_context, "metadata", {}) or {}).get("session_id") or "",
        )
        self.ledger_writer.append_runtime_witness(
            component=component,
            trace_id=trace_id,
            session_id=session_id,
        )

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
        if self.registry.is_terminated(session_key):
            raise RuntimeError(f"session {session_key!r} has been terminated")

        md = dict(metadata or {})
        trace_id = str(md.get("trace_id") or "").strip() or f"tr-{uuid4().hex}"
        md["trace_id"] = trace_id
        assert trace_id, "Missing trace_id at control plane entry"
        request_id = str(md.get("request_id") or "")
        inflight_key = (session_key, request_id) if request_id else None
        if inflight_key is not None:
            existing = self._inflight_by_request.get(inflight_key)
            if existing is not None:
                return await existing

        job = ExecutionJob(
            session_id=session_key,
            user_input=str(user_input or ""),
            attachments=attachments,
            metadata=md,
            trace_id=trace_id,
        )

        self.ledger_writer.append_job_submitted(job)
        self.ledger_writer.append_session_bound(
            session_key,
            job.job_id,
            trace_id=job.trace_id,
            kernel_step_id="control_plane.bind_session",
        )
        future = await self.scheduler.register(job)
        if inflight_key is not None:
            self._inflight_by_request[inflight_key] = future

        deadline = time.monotonic() + float(timeout_seconds or 30.0)
        try:
            while not future.done():
                drained = await self.scheduler.drain_once(self.kernel_executor)
                if not drained:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "submit_turn timed out waiting for scheduler",
                        )
                    await asyncio.sleep(0.01)
            return await asyncio.wait_for(
                future,
                timeout=max(0.001, deadline - time.monotonic()),
            )
        finally:
            if inflight_key is not None:
                self._inflight_by_request.pop(inflight_key, None)

    def ledger_events(self) -> list[dict[str, Any]]:
        return self.ledger.read()

    def boot_reconcile(self) -> dict[str, Any]:
        store = SessionStore(ledger=self.ledger, projection_only=True)
        return self.recovery.boot_reconcile(session_store=store)
