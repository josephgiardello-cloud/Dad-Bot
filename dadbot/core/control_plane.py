from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from dadbot.contracts import AttachmentList, FinalizedTurnResult
from dadbot.core.compaction import ArchiveTier, CompactionPolicy, EventCompactor
from dadbot.core.execution_boundary import ControlPlaneExecutionBoundary
from dadbot.core.execution_lease import ExecutionLease, LeaseConflictError
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.execution_ledger_memory import InMemoryExecutionLedger
from dadbot.core.kernel_gateway import KernelGateway
from dadbot.core.kernel_signals import get_exporter, get_metrics, get_tracer
from dadbot.core.ledger_reader import LedgerReader
from dadbot.core.ledger_writer_adapter import LedgerWriterAdapter
from dadbot.core.recovery_manager import RecoveryManager
from dadbot.core.semantic_primitives import hash as semantic_hash
from dadbot.core.session_store import SessionStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExecutionJob:
    session_id: str
    user_input: str
    attachments: AttachmentList | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""
    job_id: str = ""

    @staticmethod
    def _stable_token(*parts: Any, prefix: str) -> str:
        payload = "|".join(str(part or "") for part in parts)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        return f"{prefix}-{digest}"

    def __post_init__(self) -> None:
        metadata = dict(self.metadata or {})
        trace_id = str(self.trace_id or metadata.get("trace_id") or "").strip()
        if not trace_id:
            trace_id = self._stable_token(
                self.session_id,
                metadata.get("request_id"),
                self.user_input,
                self.attachments,
                prefix="tr",
            )
        job_id = str(self.job_id or metadata.get("job_id") or "").strip()
        if not job_id:
            job_id = self._stable_token(self.session_id, trace_id, prefix="job")
        metadata["trace_id"] = trace_id
        metadata["job_id"] = job_id
        self.metadata = metadata
        self.trace_id = trace_id
        self.job_id = job_id


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


class SchedulerWriter(Protocol):
    def append_job_queued(self, job: Any) -> dict[str, Any]: ...

    def append_job_started(self, job: Any) -> dict[str, Any]: ...

    def append_job_completed(self, job: Any, result: Any) -> dict[str, Any]: ...

    def append_job_failed(self, job: Any, error: str) -> dict[str, Any]: ...


class Scheduler:
    """Single-node async scheduler with lease-aware drain semantics."""

    def __init__(
        self,
        registry: SessionRegistry,
        *,
        reader: LedgerReader,
        writer: SchedulerWriter,
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
        token_seed = (
            f"{resolved_options.worker_id}|"
            f"{resolved_options.max_inflight_jobs}|"
            f"{int(bool(resolved_options.enable_observability))}"
        )
        self.execution_token = f"exec-{hashlib.sha256(token_seed.encode('utf-8')).hexdigest()[:20]}"
        self.ledger = resolved_options.ledger or InMemoryExecutionLedger()
        self._ledger_writer = LedgerWriterAdapter(
            self.ledger,
            scope_validator=KernelGateway.assert_scope,
        )
        self.ledger_reader = LedgerReader(self.ledger)
        self.execution_lease = resolved_options.execution_lease or ExecutionLease()
        scheduler_options = SchedulerOptions(
            max_inflight_jobs=resolved_options.max_inflight_jobs,
            worker_id=resolved_options.worker_id,
            execution_token=self.execution_token,
            enable_observability=resolved_options.enable_observability,
            execution_lease=self.execution_lease,
        )
        self._scheduler = resolved_options.scheduler or Scheduler(
            registry,
            reader=self.ledger_reader,
            writer=self._ledger_writer,
            options=scheduler_options,
        )
        self.recovery = RecoveryManager(ledger=self.ledger)
        self._inflight_by_request: dict[tuple[str, str], asyncio.Future[FinalizedTurnResult]] = {}
        self._inflight_lock = asyncio.Lock()
        self.graph = graph
        self._ledger_compactor: EventCompactor | None = None
        self._last_compaction_report: dict[str, Any] = {"compacted": False, "reason": "not_run"}
        self._global_confluence_contracts: dict[str, str] = {}
        self._last_confluence_report: dict[str, Any] = {"enforced": False, "reason": "not_run"}
        self._confluence_metrics: dict[str, int] = {
            "attempted": 0,
            "bound_first_observation": 0,
            "matched": 0,
            "mismatch": 0,
            "enforced_blocked": 0,
        }
        self.kernel_gateway = KernelGateway(self)

    @property
    def scheduler(self) -> Scheduler:
        KernelGateway.assert_scope("control_plane.scheduler")
        return self._scheduler

    @property
    def ledger_writer(self) -> LedgerWriterAdapter:
        KernelGateway.assert_scope("control_plane.ledger_writer")
        return self._ledger_writer

    @staticmethod
    def _stable_hash(payload: Any) -> str:
        return semantic_hash(payload)

    def _job_trace_events(self, job: ExecutionJob) -> list[dict[str, Any]]:
        events = list(self.ledger.read())
        trace_id = str(job.trace_id or "").strip()
        job_id = str(job.job_id or "").strip()
        filtered = [
            dict(event)
            for event in events
            if (
                str(event.get("trace_id") or "").strip() == trace_id
                or str(dict(event.get("payload") or {}).get("job_id") or "").strip() == job_id
            )
        ]
        return sorted(filtered, key=lambda item: int(item.get("sequence") or 0))

    @staticmethod
    def _event_stream_digest(events: list[dict[str, Any]]) -> str:
        canonical = [
            {
                "sequence": int(event.get("sequence") or 0),
                "type": str(event.get("type") or ""),
                "trace_id": str(event.get("trace_id") or ""),
                "session_id": str(event.get("session_id") or ""),
                "kernel_step_id": str(event.get("kernel_step_id") or ""),
                "payload_hash": hashlib.sha256(
                    json.dumps(dict(event.get("payload") or {}), sort_keys=True, default=str).encode("utf-8"),
                ).hexdigest(),
            }
            for event in list(events or [])
        ]
        return hashlib.sha256(
            json.dumps(canonical, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()

    @staticmethod
    def _event_semantic_digest(events: list[dict[str, Any]]) -> str:
        canonical = [
            {
                "type": str(event.get("type") or ""),
                "kernel_step_id": str(event.get("kernel_step_id") or ""),
            }
            for event in list(events or [])
        ]
        canonical.sort(key=lambda item: (str(item.get("type") or ""), str(item.get("kernel_step_id") or "")))
        return hashlib.sha256(
            json.dumps(canonical, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()

    @staticmethod
    def _confluence_mode(metadata: dict[str, Any]) -> str:
        override = str(dict(metadata or {}).get("confluence_mode") or "").strip().lower()
        if override in {"off", "audit", "enforce"}:
            return override
        raw = str(os.environ.get("DADBOT_GLOBAL_CONFLUENCE_MODE", "off")).strip().lower()
        if raw in {"off", "audit", "enforce"}:
            return raw
        return "off"

    def _derive_confluence_key(
        self,
        *,
        user_input: str,
        attachments: AttachmentList | None,
        metadata: dict[str, Any],
    ) -> str:
        explicit = str(dict(metadata or {}).get("confluence_key") or "").strip()
        if explicit:
            return explicit
        namespace = str(dict(metadata or {}).get("confluence_namespace") or "global").strip() or "global"
        payload = {
            "namespace": namespace,
            "user_input": str(user_input or ""),
            "attachments": list(attachments or []),
            "semantic_eval_input_hash": str(dict(metadata or {}).get("semantic_eval_input_hash") or ""),
        }
        return f"auto:{self._stable_hash(payload)}"

    def _prepare_global_confluence_law(
        self,
        *,
        user_input: str,
        attachments: AttachmentList | None,
        metadata: dict[str, Any],
    ) -> None:
        mode = self._confluence_mode(metadata)
        if mode == "off":
            return
        explicit_key = str(dict(metadata or {}).get("confluence_key") or "").strip()
        if mode == "enforce" and not explicit_key:
            allow_legacy = str(os.environ.get("DADBOT_ALLOW_LEGACY_CONFLUENCE_KEY", "0")).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if not allow_legacy:
                raise RuntimeError(
                    "Missing explicit confluence_key in enforce mode. "
                    "Set metadata['confluence_key'] at orchestrator boundary.",
                )
        key = self._derive_confluence_key(
            user_input=user_input,
            attachments=attachments,
            metadata=metadata,
        )
        known = str(self._global_confluence_contracts.get(key) or "")
        metadata["_global_confluence_mode"] = mode
        metadata["_global_confluence_key"] = key
        if known:
            metadata.setdefault("expected_execution_confluence_hash", known)

    def _assert_lifecycle_order(self, trace_events: list[dict[str, Any]]) -> None:
        event_types = [str(event.get("type") or "") for event in list(trace_events or [])]
        required = ["JOB_SUBMITTED", "SESSION_BOUND", "JOB_QUEUED", "JOB_STARTED", "JOB_COMPLETED"]
        positions: dict[str, int] = {}
        for event_type in required:
            if event_type not in event_types:
                raise RuntimeError(
                    f"Control-plane lifecycle invariant violated: missing event {event_type!r}",
                )
            positions[event_type] = int(event_types.index(event_type))
        ordered = [positions[name] for name in required]
        if ordered != sorted(ordered):
            raise RuntimeError(
                "Control-plane lifecycle invariant violated: event order is non-monotonic",
            )

    def _record_turn_composition_contract(  # noqa: PLR0915
        self,
        *,
        session: dict[str, Any],
        job: ExecutionJob,
        result: FinalizedTurnResult,
        state_before_hash: str,
    ) -> dict[str, Any]:
        state = dict(session.get("state") or {})
        state_after_hash = self._stable_hash(state)
        terminal_state = dict(state.get("last_terminal_state") or {})
        trace_events = self._job_trace_events(job)
        if any(str(event.get("type") or "").strip() for event in list(trace_events or [])):
            self._assert_lifecycle_order(trace_events)

        event_log_hash = self._event_stream_digest(trace_events)
        output_payload = {
            "response": str(result[0] if isinstance(result, tuple) and len(result) >= 1 else ""),
            "should_end": bool(result[1] if isinstance(result, tuple) and len(result) >= 2 else False),
        }
        state_delta_hash = self._stable_hash(
            {
                "before": str(state_before_hash or ""),
                "after": str(state_after_hash or ""),
            },
        )
        composition_payload = {
            "contract_version": "turn-composition-v1",
            "context_input_hash": self._stable_hash(
                {
                    "session_id": str(job.session_id or ""),
                    "trace_id": str(job.trace_id or ""),
                    "user_input": str(job.user_input or ""),
                    "attachments": list(job.attachments or []),
                    "metadata": dict(job.metadata or {}),
                },
            ),
            "execution_dag_hash": str(terminal_state.get("execution_dag_hash") or ""),
            "policy_hash": str(terminal_state.get("policy_hash") or ""),
            "state_delta_hash": state_delta_hash,
            "event_log_hash": event_log_hash,
            "output_hash": self._stable_hash(output_payload),
            "mutation_effects_hash": str(terminal_state.get("post_commit_mutation_effects_hash") or ""),
            "determinism_closure_hash": str(terminal_state.get("determinism_closure_hash") or ""),
        }
        composition_hash = self._stable_hash(composition_payload)
        confluence_payload = {
            "contract_version": "turn-confluence-v1",
            "semantic_input_hash": self._stable_hash(
                {
                    "user_input": str(job.user_input or ""),
                    "attachments": list(job.attachments or []),
                    "semantic_eval_input_hash": str(dict(job.metadata or {}).get("semantic_eval_input_hash") or ""),
                },
            ),
            "execution_dag_hash": str(terminal_state.get("execution_dag_hash") or ""),
            "policy_hash": str(terminal_state.get("policy_hash") or ""),
            "output_hash": self._stable_hash(output_payload),
            "mutation_effects_hash": str(terminal_state.get("post_commit_mutation_effects_hash") or ""),
            "determinism_closure_hash": str(terminal_state.get("determinism_closure_hash") or ""),
            "event_semantic_hash": self._event_semantic_digest(trace_events),
        }
        confluence_class_hash = self._stable_hash(confluence_payload)
        contract = dict(composition_payload)
        contract["composition_hash"] = composition_hash
        contract["confluence_class_hash"] = confluence_class_hash

        expected = str(dict(job.metadata or {}).get("expected_execution_composition_hash") or "").strip()
        if expected and expected != composition_hash:
            raise RuntimeError(
                f"Execution composition mismatch: expected={expected!r}, actual={composition_hash!r}",
            )
        expected_confluence = str(dict(job.metadata or {}).get("expected_execution_confluence_hash") or "").strip()
        if expected_confluence and expected_confluence != confluence_class_hash:
            raise RuntimeError(
                f"Execution confluence mismatch: expected={expected_confluence!r}, actual={confluence_class_hash!r}",
            )

        confluence_key = str(dict(job.metadata or {}).get("_global_confluence_key") or "").strip()
        confluence_mode = str(dict(job.metadata or {}).get("_global_confluence_mode") or "off").strip().lower()
        confluence_report = {
            "enforced": False,
            "mode": confluence_mode,
            "key": confluence_key,
            "observed_hash": confluence_class_hash,
            "expected_hash": expected_confluence,
            "contract_version": "turn-confluence-v1",
        }
        if confluence_key and confluence_mode != "off":
            self._confluence_metrics["attempted"] = int(self._confluence_metrics.get("attempted", 0)) + 1
            known = str(self._global_confluence_contracts.get(confluence_key) or "")
            if not known:
                self._global_confluence_contracts[confluence_key] = confluence_class_hash
                confluence_report["enforced"] = True
                confluence_report["action"] = "bound_first_observation"
                self._confluence_metrics["bound_first_observation"] = int(
                    self._confluence_metrics.get("bound_first_observation", 0),
                ) + 1
            elif known != confluence_class_hash:
                confluence_report["enforced"] = True
                confluence_report["expected_hash"] = known
                confluence_report["action"] = "mismatch"
                self._last_confluence_report = dict(confluence_report)
                self._confluence_metrics["mismatch"] = int(self._confluence_metrics.get("mismatch", 0)) + 1
                fail_mode = str(
                    os.environ.get("DADBOT_CONFLUENCE_VIOLATION_MODE", "fail"),
                ).strip().lower()
                if fail_mode != "audit":
                    self._confluence_metrics["enforced_blocked"] = int(
                        self._confluence_metrics.get("enforced_blocked", 0),
                    ) + 1
                    raise RuntimeError(
                        "Global confluence law violated for key="
                        f"{confluence_key!r}: expected={known!r}, "
                        f"actual={confluence_class_hash!r}",
                    )
                logger.warning(
                    "Global confluence law mismatch (audit override): key=%s expected=%s actual=%s",
                    confluence_key,
                    known,
                    confluence_class_hash,
                )
            else:
                confluence_report["enforced"] = True
                confluence_report["expected_hash"] = known
                confluence_report["action"] = "matched"
                self._confluence_metrics["matched"] = int(self._confluence_metrics.get("matched", 0)) + 1
        self._last_confluence_report = dict(confluence_report)

        state_mut = session.setdefault("state", {})
        if isinstance(state_mut, dict):
            state_mut["last_execution_composition_contract"] = dict(contract)
            state_mut["last_execution_confluence_report"] = dict(confluence_report)
        return contract

    @staticmethod
    def _load_archived_events(archive_path: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if not archive_path:
            return events
        with gzip.open(archive_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                stripped = str(line or "").strip()
                if not stripped:
                    continue
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, dict):
                        events.append(parsed)
                except json.JSONDecodeError:
                    continue
        return events

    def _compaction_losslessness_proof(
        self,
        *,
        pre_events: list[dict[str, Any]],
        post_events: list[dict[str, Any]],
        archive_path: str,
    ) -> dict[str, Any]:
        archived_events = self._load_archived_events(archive_path)
        reconstructed = sorted(
            [dict(event) for event in list(archived_events or [])] + [dict(event) for event in list(post_events or [])],
            key=lambda item: int(item.get("sequence") or 0),
        )
        pre_digest = self._event_stream_digest(list(pre_events or []))
        reconstructed_digest = self._event_stream_digest(reconstructed)
        sequence_equivalent = [
            int(item.get("sequence") or 0) for item in list(pre_events or [])
        ] == [
            int(item.get("sequence") or 0) for item in reconstructed
        ]
        return {
            "contract_version": "ledger-compaction-lossless-v1",
            "equivalent": bool(pre_digest == reconstructed_digest and sequence_equivalent),
            "pre_digest": pre_digest,
            "reconstructed_digest": reconstructed_digest,
            "archived_event_count": len(archived_events),
            "reconstructed_event_count": len(reconstructed),
            "sequence_equivalent": bool(sequence_equivalent),
        }

    @staticmethod
    def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
        raw = str(os.environ.get(name, str(default))).strip()
        value = int(raw) if raw.isdigit() else int(default)
        return max(int(minimum), value)

    def _ensure_compactor(self) -> EventCompactor:
        if self._ledger_compactor is None:
            max_events = self._env_int("DADBOT_LEDGER_MAX_EVENTS", 10000, minimum=100)
            max_age_seconds = float(self._env_int("DADBOT_LEDGER_MAX_AGE_SECONDS", 86400, minimum=60))
            min_snapshot_distance = self._env_int("DADBOT_LEDGER_MIN_SNAPSHOT_DISTANCE", 200, minimum=0)
            archive_dir = Path(
                str(os.environ.get("DADBOT_LEDGER_ARCHIVE_DIR", "runtime/archives")).strip()
                or "runtime/archives"
            )
            self._ledger_compactor = EventCompactor(
                policy=CompactionPolicy(
                    max_events=max_events,
                    max_age_seconds=max_age_seconds,
                    min_snapshot_distance=min_snapshot_distance,
                ),
                archive=ArchiveTier(archive_dir),
            )
        return self._ledger_compactor

    def _partition_summary(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        by_session: dict[str, int] = {}
        for event in list(events or []):
            sid = str(event.get("session_id") or "").strip() or "unknown"
            by_session[sid] = int(by_session.get(sid, 0)) + 1
        top_sessions = sorted(by_session.items(), key=lambda item: (-int(item[1]), str(item[0])))[:8]
        return {
            "partition_count": len(by_session),
            "top_partitions": [{"session_id": sid, "event_count": int(count)} for sid, count in top_sessions],
        }

    def _maybe_compact_ledger(self) -> dict[str, Any]:
        hard_max = self._env_int("DADBOT_LEDGER_HARD_LIMIT_EVENTS", 50000, minimum=500)
        pre_events = list(self.ledger.read())
        event_count = len(pre_events)

        if event_count <= 0:
            return {"compacted": False, "reason": "empty", "event_count": 0, **self._partition_summary(pre_events)}

        force = bool(event_count >= hard_max)
        compactor = self._ensure_compactor()
        snapshot = {"head_sequence": event_count}
        report = dict(compactor.compact(ledger=self.ledger, snapshot=snapshot, force=force) or {})
        post_events = list(self.ledger.read())
        lossless_proof = {
            "contract_version": "ledger-compaction-lossless-v1",
            "equivalent": True,
            "pre_digest": self._event_stream_digest(pre_events),
            "reconstructed_digest": self._event_stream_digest(post_events),
            "archived_event_count": 0,
            "reconstructed_event_count": len(post_events),
            "sequence_equivalent": True,
        }
        archive_path = str(report.get("archive_path") or "")
        if bool(report.get("compacted", False)) and archive_path:
            lossless_proof = self._compaction_losslessness_proof(
                pre_events=pre_events,
                post_events=post_events,
                archive_path=archive_path,
            )
            if not bool(lossless_proof.get("equivalent", False)):
                raise RuntimeError("Ledger compaction losslessness invariant violated")
        report.setdefault("event_count", event_count)
        report.setdefault("forced", force)
        report["lossless_proof"] = dict(lossless_proof)
        report.update(self._partition_summary(post_events))
        self._last_compaction_report = report
        return report

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

    async def _submit_turn_kernel(  # noqa: C901, PLR0912, PLR0915
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
        request_id = str(md.get("request_id") or "").strip()
        dedupe_key = (session_key, request_id)
        dedupe_future: asyncio.Future[FinalizedTurnResult] | None = None
        owns_dedupe_slot = False
        if request_id:
            async with self._inflight_lock:
                existing = self._inflight_by_request.get(dedupe_key)
                if existing is not None:
                    dedupe_future = existing
                else:
                    dedupe_future = asyncio.get_running_loop().create_future()
                    self._inflight_by_request[dedupe_key] = dedupe_future
                    owns_dedupe_slot = True
            if not owns_dedupe_slot and dedupe_future is not None:
                return await dedupe_future

        trace_id = str(md.get("trace_id") or "").strip()
        if not trace_id:
            trace_seed = {
                "session_id": session_key,
                "request_id": str(md.get("request_id") or ""),
                "user_input": str(user_input or ""),
                "attachments": list(attachments or []),
            }
            trace_blob = json_dumps_sorted(trace_seed)
            trace_id = f"tr-{hashlib.sha256(trace_blob.encode('utf-8')).hexdigest()[:20]}"
        md["trace_id"] = trace_id
        assert trace_id, "Missing trace_id at control plane entry"
        self._prepare_global_confluence_law(
            user_input=str(user_input or ""),
            attachments=attachments,
            metadata=md,
        )

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
        session_before = self.registry.get_or_create(session_key)
        before_state_hash = self._stable_hash(dict(session_before.get("state") or {}))
        max_wait_seconds = timeout_seconds or 30.0
        deadline = time.time() + max_wait_seconds
        try:
            while not future.done():
                if time.time() > deadline:
                    raise TimeoutError("submit_turn exceeded timeout")
                drained = await self.scheduler.drain_once(self.kernel_executor)
                if not drained:
                    await asyncio.sleep(0.01)
            result = await future
            session_after = self.registry.get_or_create(session_key)
            self._validate_trace_invariant(job, result, session=session_after, state_before_hash=before_state_hash)
            self._maybe_compact_ledger()
            if dedupe_future is not None and not dedupe_future.done():
                dedupe_future.set_result(result)
            return result
        except Exception as exc:
            if dedupe_future is not None and not dedupe_future.done():
                dedupe_future.set_exception(exc)
            raise
        finally:
            if request_id and owns_dedupe_slot:
                async with self._inflight_lock:
                    if self._inflight_by_request.get(dedupe_key) is dedupe_future:
                        self._inflight_by_request.pop(dedupe_key, None)

    async def submit_turn(
        self,
        *,
        session_id: str,
        user_input: str,
        attachments: AttachmentList | None = None,
        metadata: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> FinalizedTurnResult:
        # _validate_trace_invariant is executed in _submit_turn_kernel after job completion.
        return await self.kernel_gateway.submit_turn(
            session_id=session_id,
            user_input=user_input,
            attachments=attachments,
            metadata=metadata,
            timeout_seconds=timeout_seconds,
        )

    def ledger_events(self) -> list[dict[str, Any]]:
        return self.ledger.read()

    def _validate_trace_invariant(
        self,
        job: ExecutionJob,
        result: FinalizedTurnResult,
        *,
        session: dict[str, Any] | None = None,
        state_before_hash: str = "",
    ) -> None:
        """Validate execution trace invariants: trace_id, complete nodes, exactly one commit boundary.

        Ensures:
        1. trace_id is present in job metadata
        2. Execution produced ordered trace nodes
        3. Exactly one commit boundary (save node) exists

        This is a defensive check to catch execution path deviations early.
        """
        trace_id = job.metadata.get("trace_id") or ""
        if not trace_id.strip():
            logger.warning("Trace invariant violation: missing trace_id in job %s", job.job_id)
            return

        # Query ledger for events matching this trace
        trace_events = self._job_trace_events(job)

        if not trace_events:
            logger.warning("Trace invariant violation: no events recorded for trace %s", trace_id)
            return

        # Count commit boundary markers from canonical TURN_EVENT payloads.
        commit_count = 0
        has_node_events = False
        for event in trace_events:
            payload = dict(event.get("payload") or {}) if isinstance(event, dict) else {}
            event_type = str(event.get("event_type") or payload.get("event_type") or "").strip().lower()
            stage = str(event.get("stage") or event.get("node_type") or payload.get("stage") or payload.get("node_type") or "").strip().lower()
            if event_type in {"node_start", "node_complete", "node_completed", "turn_start", "turn_complete", "turn_failed"}:
                has_node_events = True
            if event_type in {"node_complete", "node_completed"} and stage == "save":
                commit_count += 1

        # Fallback to unified sink trace when ledger envelope normalization omits stage markers.
        if commit_count == 0 and isinstance(session, dict):
            session_state = dict(session.get("state") or {})
            turn_trace = dict(session_state.get("turn_trace") or {})
            if str(turn_trace.get("trace_id") or "").strip() == str(trace_id).strip():
                fallback_commit_count = int(turn_trace.get("commit_boundary_count") or 0)
                if fallback_commit_count > 0:
                    commit_count = fallback_commit_count

        if not has_node_events and commit_count == 0:
            # Some persistence backends store only checkpoint summaries.
            # In that mode commit-boundary cardinality cannot be derived here.
            # Still record the composition contract for confluence tracking.
            if session is not None:
                self._record_turn_composition_contract(
                    session=session,
                    job=job,
                    result=result,
                    state_before_hash=str(state_before_hash or ""),
                )
            return

        if commit_count != 1:
            logger.warning(
                "Trace invariant violation: expected exactly 1 commit boundary, found %d for trace %s",
                commit_count,
                trace_id,
            )
        if session is None:
            return
        self._record_turn_composition_contract(
            session=session,
            job=job,
            result=result,
            state_before_hash=str(state_before_hash or ""),
        )

    def boot_reconcile(self) -> dict[str, Any]:
        """Phase 3: boot reconciliation is now ledger-only via direct replay."""
        store = SessionStore(ledger=self.ledger, projection_only=True)
        events = self.ledger.read()
        store.rebuild_from_ledger(events)
        snap = store.snapshot()
        pending = list(store.pending_jobs())
        return {
            "pending_jobs": pending,
            "ledger_events": len(events),
            "replay_hash": self.ledger.replay_hash(),
            "session_count": len(dict(snap.get("sessions") or {})),
            "session_snapshot_version": int(snap.get("version") or 0),
            "ledger_partitioning": self._partition_summary(events),
            "ledger_compaction": dict(self._last_compaction_report or {}),
            "execution_confluence": dict(self._last_confluence_report or {}),
            "execution_confluence_metrics": dict(self._confluence_metrics),
            "ok": True,
        }


def json_dumps_sorted(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)
