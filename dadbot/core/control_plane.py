from __future__ import annotations

import asyncio
import contextlib
import gzip
import hashlib
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, cast

from dadbot.contracts import AttachmentList, FinalizedTurnResult
from dadbot.core.contracts.lifecycle_events import (
    Claimed,
    Completed,
    Failed,
    LeaseExpired,
    LeaseRenewed,
    Redelivered,
    Released,
)
from dadbot.core.compaction import ArchiveTier, CompactionPolicy, EventCompactor
from dadbot.core.autonomous_goal_daemon import AutonomousGoalDaemon
from dadbot.core.belief_state_engine import BeliefStateEngine
from dadbot.core.compositional_tool_planner import CompositionalToolPlanner
from dadbot.core.behavior_alignment_trainer import BehaviorAlignmentTrainer
from dadbot.core.contract_evaluator import validate_sovereign_ledger_transition
from dadbot.core.cognitive_policy_engine import CognitivePolicyEngine
from dadbot.core.control_plane_projection import ExecutionProjection
from dadbot.core.control_plane_reducer import ExecutionState as ReducedExecutionState
from dadbot.core.control_plane_reducer import ExecutionStatus, lease_expired
from dadbot.core.core_state import InputEvent, project_views, transition
from dadbot.core.execution_boundary import ControlPlaneExecutionBoundary
from dadbot.core.execution_context import (
    close_core_state_scope,
)
from dadbot.core.execution_lease import ExecutionLease, LeaseConflictError
from dadbot.core.execution_resource_budget import BackpressureSignal, resolve_dynamic_compute_ticket
from dadbot.core.execution_context import get_active_core_state
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.execution_ledger_memory import InMemoryExecutionLedger
from dadbot.core.execution_result_unified import (
    _TERMINAL_STATUS_VALUES,
    build_unified_execution_result,
    ensure_unified_execution_result,
    mark_unified_execution_failure,
    mark_unified_execution_success,
)
from dadbot.core.failure_taxonomy import classify_failure
from dadbot.core.distributed_correctness import (
    DistributedCorrectnessModel,
    NodeRole,
)
from dadbot.core.memory_set_invariants import (
    MemorySetInvariantViolation,
)
from dadbot.core.runtime_errors import (
    AuthorityViolation,
    ExecutionStageError,
    InvariantViolation,
    PersistenceFailure,
    ReplayMismatch,
)
from dadbot.core.runtime_contracts import (
    validate_decision_explanation_contract,
    validate_trace_event_contract,
)
from dadbot.core.kernel_gateway import KernelGateway
from dadbot.core.kernel_signals import get_exporter, get_metrics, get_tracer
from dadbot.core.ledger_reader import LedgerReader
from dadbot.core.ledger_writer import LedgerWriter
from dadbot.core.ledger_writer_adapter import LedgerWriterAdapter
from dadbot.core.ledger_coordinator import LedgerCoordinator
from dadbot.core.invariant_enforcer import InvariantEnforcer
from dadbot.core.recovery_manager import RecoveryManager
from dadbot.core.runtime_types import ToolSpec
from dadbot.core.execution_coordinator import ExecutionCoordinator
from dadbot.core.hypothesis_engine import MultiHypothesisEngine
from dadbot.core.memory_hierarchy_manager import MemoryHierarchyManager
from dadbot.core.multi_agent_swarm import MultiAgentSwarm
from dadbot.core.semantic_memory_graph import SemanticMemoryGraph
from dadbot.core.semantic_primitives import hash as semantic_hash
from dadbot.core.semantic_safety_engine import SemanticSafetyEngine
from dadbot.core.session_planning_optimizer import SessionPlanningOptimizer
from dadbot.core.session_store import SessionStore
from dadbot.core.interactive_cognition_ui import InteractiveCognitionUI
from dadbot.core.tool_routing_engine import ToolRoutingEngine
from dadbot.core.tool_ecosystem_hub import ToolEcosystemHub
from dadbot.core.tool_self_model import ToolSelfModel
from dadbot.core.adaptation_engine import AdaptationEngine
from dadbot.core.phase_closure_runtime import PhaseClosureRuntime
from dadbot.core.response_engine import ResponseEngine
from dadbot.core.topology_runtime import TopologyRuntime, TopologyValidationResult
from dadbot.core._control_plane_reconciliation import ReconciliationMixin
from dadbot.core._control_plane_compaction import CompactionMixin
from dadbot.core.planning_utils import (
    build_semantic_memory_candidates as _build_semantic_memory_candidates,
    mutate_runtime_plan as _mutate_runtime_plan,
    normalize_tool_runtime_contract as _normalize_tool_runtime_contract,
    runtime_plan_intent as _runtime_plan_intent,
    validate_tool_runtime_contract as _validate_tool_runtime_contract,
)
from dadbot.core.memory_ranking import rank_semantic_memory_items as _rank_semantic_memory_items
from dadbot.core.persistence_coordinator import PersistenceCoordinator
from dadbot.core.control_plane_feedback import (
    _response_influence_share,
    _synthesize_reward_feedback,
    _update_response_engine_drift_monitor,
)
from dadbot.core.control_plane_progress import (
    _write_progress_event,
)
from dadbot.core.control_plane_lifecycle import (
    ExecutionLifecycleState,
    SchedulerExceptionMapper,
    TurnTerminalState,
    _apply_projection_execution_state,
    _assert_lifecycle_emission_transition,
    _build_sovereign_transition_states,
    _coerce_lifecycle_state,
    _lifecycle_state_from_projection,
    _resolved_execution_mode,
    _target_lifecycle_state_for_event,
)
from dadbot.core.control_plane_submit_phase import (
    _append_submit_turn_phase,
    _assert_submit_turn_phase_boundary,
)
from dadbot.core.control_plane_submission import (
    _initialize_submit_scope_impl,
    _prepare_submit_register_phase_impl,
    _register_submit_job_impl,
)
from dadbot.core.control_plane_planning import (
    _prepare_submit_runtime_planning_impl,
)
from dadbot.core.control_plane_submit_execution import (
    _run_submit_execution_phase_impl,
)
from dadbot.core.control_plane_post_turn import (
    _apply_submit_success_postprocessing_impl,
)
from dadbot.core.control_plane_submit_failure import (
    _record_submit_exception_impl,
)
from dadbot.core.control_plane_submit_metadata import (
    _prepare_submit_metadata_impl,
)
from dadbot.core.control_plane_trace_invariants import (
    _fallback_commit_count_from_session_impl,
    _trace_event_invariant_counts_impl,
    _validate_trace_invariant_impl,
)
from dadbot.core.control_plane_composition_contracts import (
    _build_composition_payload_impl,
    _build_confluence_payload_impl,
    _confluence_config_from_metadata_impl,
    _enforce_global_confluence_law_impl,
    _expected_hashes_from_metadata_impl,
    _record_turn_composition_contract_impl,
    _result_output_payload_impl,
    _validate_composition_expectations_impl,
)

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class _FailurePolicyStrategy:
    event_type: str


_FAILURE_POLICY_STRATEGIES: dict[str, _FailurePolicyStrategy] = {
    "quarantine": _FailurePolicyStrategy(event_type="JOB_QUARANTINED"),
    "reconcile": _FailurePolicyStrategy(event_type="JOB_RECONCILE_REQUIRED"),
    "manual_retry": _FailurePolicyStrategy(event_type="JOB_MANUAL_RETRY_REQUIRED"),
}


_SCHEDULER_EXCEPTION_MAPPER = SchedulerExceptionMapper()
def _classify_execution_failure(exc: BaseException) -> dict[str, Any]:
    failure = classify_failure(exc)
    if isinstance(exc, asyncio.CancelledError):
        # Cancellation is explicitly modeled as retryable but keeps a separate class
        # for operational dashboards.
        failure["failure_class"] = "cancelled"
        failure["failure_source"] = "runtime"
    return failure


def _set_terminal_turn_state(
    job: "ExecutionJob",
    *,
    terminal_state: TurnTerminalState,
    reason: str,
    strict: bool = True,
) -> None:
    metadata = dict(job.metadata or {})
    execution_state = dict(metadata.get("execution_state") or {})
    existing = str(
        execution_state.get("terminal_turn_state") or metadata.get("terminal_turn_state") or "",
    ).strip().upper()
    incoming = str(terminal_state.value or "").strip().upper()
    if existing and existing != incoming:
        if not strict:
            return
        raise InvariantViolation(
            "Terminal turn state transition is not idempotent",
            context={
                "job_id": str(job.job_id or ""),
                "trace_id": str(job.trace_id or ""),
                "existing": existing,
                "incoming": incoming,
            },
        )
    execution_state["terminal_turn_state"] = incoming
    execution_state["terminal_transition_reason"] = str(reason or "")
    metadata["execution_state"] = execution_state
    metadata["terminal_turn_state"] = incoming
    job.metadata = metadata


def _extract_execution_degradations(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    items = metadata.get("turn_ir_degradations")
    if not isinstance(items, list):
        items = metadata.get("ir_degradations")
    if not isinstance(items, list):
        return []
    return [dict(item) for item in items if isinstance(item, dict)]


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


class SchedulerProtocol(Protocol):
    """GAP 3: Explicit scheduler boundary for ControlPlane.

    ControlPlane depends only on this interface, not on the concrete Scheduler
    implementation.  Any object satisfying these three methods can be injected,
    which enforces the scheduler/control-plane boundary structurally.
    """

    worker_id: str
    lease_ttl_seconds: float

    async def register(self, job: ExecutionJob) -> asyncio.Future[FinalizedTurnResult]: ...
    async def drain_once(
        self,
        executor: Callable[[dict[str, Any], ExecutionJob], Awaitable[FinalizedTurnResult]],
    ) -> bool: ...
    async def wait_for_work(self, *, timeout_seconds: float | None = None) -> bool: ...


@dataclass(slots=True)
class SchedulerOptions:
    max_inflight_jobs: int = 16
    worker_id: str = "worker-1"
    execution_token: str = ""
    enable_observability: bool = True
    projection: ExecutionProjection | None = None
    execution_lease: ExecutionLease | None = None
    lease_ttl_seconds: float = 30.0
    redelivery_retry_interval_seconds: float = 0.05
    fairness_aging_rate: float = 5.0
    tenant_balance_weight: float = 1.0
    on_runtime_claim_guard: Callable[[str], None] | None = None
    on_runtime_lease_sync: Callable[[str], None] | None = None


@dataclass(slots=True)
class ControlPlaneOptions:
    max_inflight_jobs: int = 16
    worker_id: str = "worker-1"
    enable_observability: bool = True
    lease_ttl_seconds: float = 30.0
    redelivery_retry_interval_seconds: float = 0.05
    ledger: ExecutionLedger | None = None
    scheduler: SchedulerProtocol | None = None
    stream_sink: Callable[[dict[str, Any]], None] | None = None


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


class DurableReconcileQueue:
    """Ledger-backed reconcile queue boundary.

    Queue intent is persisted as ``JOB_RECONCILE_REQUIRED`` events; this helper
    centralizes enqueue, pending projection, and bounded consume semantics.
    """

    def __init__(
        self,
        *,
        ledger: ExecutionLedger,
        write_event: Callable[..., dict[str, Any]] | None,
        request_is_ambiguous: Callable[[str, str], bool],
        effect_is_ambiguous: Callable[[str, str], bool],
    ) -> None:
        self._ledger = ledger
        self._write_event = write_event
        self._request_is_ambiguous = request_is_ambiguous
        self._effect_is_ambiguous = effect_is_ambiguous

    def enqueue_required(
        self,
        *,
        session_id: str,
        trace_token: str,
        request_id: str,
        effect_id: str,
        reason: str,
    ) -> None:
        if not callable(self._write_event):
            return
        self._write_event(
            event_type="JOB_RECONCILE_REQUIRED",
            session_id=str(session_id or "default"),
            trace_id=str(trace_token or "").strip(),
            kernel_step_id="control_plane.reconcile_required",
            payload={
                "request_id": str(request_id or "").strip(),
                "effect_id": str(effect_id or "").strip(),
                "reason": str(reason or "ambiguous_effect_state"),
            },
            committed=False,
        )

    def _extract_reconcile_event_fields(self, event: dict[str, Any]) -> tuple[str, str, str, str]:
        sid = str(event.get("session_id") or "default").strip() or "default"
        payload = dict(event.get("payload") or {})
        rid = str(payload.get("request_id") or "").strip()
        eid = str(payload.get("effect_id") or "").strip()
        reason = str(payload.get("reason") or "").strip()
        return sid, rid, eid, reason

    def _entry_needs_reconciliation(self, session_id: str, request_id: str, effect_id: str) -> bool:
        if request_id and self._request_is_ambiguous(session_id, request_id):
            return True
        if effect_id and self._effect_is_ambiguous(session_id, effect_id):
            return True
        return False

    def pending_entries(self) -> list[dict[str, Any]]:
        pending: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for event in self._ledger.read():
            if str(event.get("type") or "") != "JOB_RECONCILE_REQUIRED":
                continue
            sid, rid, eid, reason = self._extract_reconcile_event_fields(event)
            if not rid and not eid:
                continue
            key = (sid, rid, eid)
            if key in seen:
                continue
            seen.add(key)
            if not self._entry_needs_reconciliation(sid, rid, eid):
                continue
            pending.append(
                {
                    "session_id": sid,
                    "request_id": rid,
                    "effect_id": eid,
                    "reason": reason or "queued_reconcile_required",
                },
            )
        return pending

    def consume(
        self,
        *,
        enabled: bool,
        max_items: int,
        max_rounds: int,
        mode: str,
        apply: Callable[..., dict[str, Any]],
    ) -> dict[str, Any]:
        pending = self.pending_entries()
        if not enabled:
            return {
                "enabled": False,
                "mode": mode,
                "max_items": max_items,
                "max_rounds": max_rounds,
                "queued": len(pending),
                "attempted": 0,
                "applied": 0,
                "failed": 0,
                "remaining": len(pending),
                "converged": True,
            }

        attempted = 0
        applied = 0
        failed = 0
        rounds = 0
        converged = True
        seen_signatures: set[str] = set()
        while attempted < max_items and rounds < max_rounds:
            rounds += 1
            pending = self.pending_entries()
            if not pending:
                break
            signature = "|".join(
                sorted(
                    f"{str(item.get('session_id') or 'default')}::{str(item.get('request_id') or '')}::{str(item.get('effect_id') or '')}"
                    for item in pending
                ),
            )
            if signature in seen_signatures:
                converged = False
                break
            seen_signatures.add(signature)

            round_applied = 0
            budget = max_items - attempted
            for item in pending[:budget]:
                attempted += 1
                try:
                    report = apply(
                        session_id=str(item.get("session_id") or "default"),
                        request_id=str(item.get("request_id") or ""),
                        effect_id=str(item.get("effect_id") or ""),
                        reason=str(item.get("reason") or "queued_reconcile_required"),
                        mode=mode,
                    )
                except Exception:
                    failed += 1
                    continue
                if bool(report.get("applied")):
                    applied += 1
                    round_applied += 1
            if round_applied <= 0:
                converged = False
                break

        remaining = len(self.pending_entries())
        return {
            "enabled": True,
            "mode": mode,
            "max_items": max_items,
            "max_rounds": max_rounds,
            "queued": len(pending),
            "attempted": attempted,
            "applied": applied,
            "failed": failed,
            "remaining": remaining,
            "converged": bool(converged and remaining == 0),
        }


class SchedulerWriter(Protocol):
    def append_job_queued(self, job: Any) -> dict[str, Any]: ...

    def append_job_started(self, job: Any) -> dict[str, Any]: ...

    def append_job_completed(self, job: Any, result: Any) -> dict[str, Any]: ...

    def append_job_failed(
        self,
        job: Any,
        error: Any,
        *,
        failure: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def append_execution_lifecycle(
        self,
        event: Any,
        *,
        session_id: str,
        trace_token: str,
        step_key: str,
        committed: bool = False,
    ) -> dict[str, Any]: ...

    def append_effect_begin(
        self,
        *,
        session_id: str,
        trace_token: str,
        effect_id: str,
        request_id: str = "",
        step_key: str = "scheduler.execute.effect.begin",
    ) -> dict[str, Any]: ...

    def append_effect_commit(
        self,
        *,
        session_id: str,
        trace_token: str,
        effect_id: str,
        request_id: str = "",
        step_key: str = "scheduler.execute.effect.commit",
    ) -> dict[str, Any]: ...


class Scheduler:
    """Single-node async scheduler with lifecycle-projection drain semantics."""

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
        self.projection = resolved_options.projection or ExecutionProjection()
        self.worker_id = str(resolved_options.worker_id or "worker-1")
        self.execution_token = str(resolved_options.execution_token or "")
        self.enable_observability = bool(resolved_options.enable_observability)
        self.execution_lease = resolved_options.execution_lease
        self.lease_ttl_seconds = max(0.001, float(resolved_options.lease_ttl_seconds or 30.0))
        self.redelivery_retry_interval_seconds = max(
            0.001,
            float(resolved_options.redelivery_retry_interval_seconds or 0.05),
        )
        self.fairness_aging_rate = max(0.0, float(resolved_options.fairness_aging_rate or 0.0))
        self.tenant_balance_weight = max(0.0, float(resolved_options.tenant_balance_weight or 0.0))
        self._on_runtime_claim_guard = resolved_options.on_runtime_claim_guard
        self._on_runtime_lease_sync = resolved_options.on_runtime_lease_sync

        self._jobs: dict[
            str,
            tuple[ExecutionJob, asyncio.Future[FinalizedTurnResult]],
        ] = {}
        self._invariant_enforcer = InvariantEnforcer()
        self._pending_job_ids: list[str] = []
        self._work_event: asyncio.Event | None = None
        self._work_event_loop: asyncio.AbstractEventLoop | None = None

    def _ensure_work_event(self) -> asyncio.Event:
        loop = asyncio.get_running_loop()
        if self._work_event is None or self._work_event_loop is not loop:
            self._work_event = asyncio.Event()
            self._work_event_loop = loop
        return self._work_event

    def _notify_work_available(self) -> None:
        self._ensure_work_event().set()

    def _apply_scheduler_membership_rule(self) -> None:
        """Option A invariant: scheduler queue contains non-terminal jobs only."""
        if not self._pending_job_ids:
            return
        retained: list[str] = []
        for job_id in list(self._pending_job_ids):
            pair = self._jobs.get(job_id)
            if pair is None:
                continue
            _job, future = pair
            projected = self.projection.get(job_id)
            if projected is not None and projected.status in {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED}:
                self._jobs.pop(job_id, None)
                if not future.done():
                    future.set_exception(
                        RuntimeError(
                            "scheduler membership invariant: terminal job cannot remain pending",
                        ),
                    )
                continue
            retained.append(job_id)
        self._pending_job_ids = retained

    async def wait_for_work(self, *, timeout_seconds: float | None = None) -> bool:
        self._apply_scheduler_membership_rule()
        now = time.monotonic()
        if self._ready_pending_job_ids(now):
            return True

        work_event = self._ensure_work_event()
        work_event.clear()
        next_ready_delay = self._next_ready_delay(now)

        if timeout_seconds is not None and timeout_seconds <= 0.0:
            return bool(self._ready_pending_job_ids(time.monotonic()))

        try:
            wait_timeout = timeout_seconds
            if next_ready_delay is not None:
                wait_timeout = next_ready_delay if wait_timeout is None else min(wait_timeout, next_ready_delay)
            if wait_timeout is None:
                await work_event.wait()
            else:
                await asyncio.wait_for(work_event.wait(), timeout=wait_timeout)
        except TimeoutError:
            return bool(self._ready_pending_job_ids(time.monotonic()))
        return bool(self._ready_pending_job_ids(time.monotonic()))

    def _job_ready(self, job: ExecutionJob, *, now: float) -> bool:
        projected = self.projection.get(job.job_id)
        _apply_projection_execution_state(job, projected)
        return bool(
            self.projection.get_runnable(
                now=datetime.fromtimestamp(now),
                execution_ids=[job.job_id],
            )
        )

    def _ready_pending_job_ids(self, now: float) -> list[str]:
        ready: list[str] = []
        for job_id in list(self._pending_job_ids):
            pair = self._jobs.get(job_id)
            if pair is None:
                continue
            job, _future = pair
            if self._job_ready(job, now=now):
                ready.append(job_id)
        return ready

    def _next_ready_delay(self, now: float) -> float | None:
        self._apply_scheduler_membership_rule()
        active_expiries: list[float] = []
        for job_id in list(self._pending_job_ids):
            pair = self._jobs.get(job_id)
            if pair is None:
                continue
            job, _future = pair
            state = self.projection.get(job_id)
            if state is not None and state.lease_expiry is not None:
                expiry_ts = state.lease_expiry.timestamp()
                if expiry_ts > now:
                    active_expiries.append(expiry_ts - now)
        if not active_expiries:
            return None
        return max(0.0, min(active_expiries))

    def _claim_order_key(self, job_id: str) -> tuple[float, str, int, str]:
        pair = self._jobs.get(job_id)
        if pair is None:
            return (float("inf"), "", 0, str(job_id or ""))
        job, _future = pair
        claim_order = dict(dict(job.metadata or {}).get("claim_order") or {})
        timestamp = float(claim_order.get("timestamp") or 0.0)
        worker_id = str(claim_order.get("worker_id") or self.worker_id or "")
        lease_epoch = int(claim_order.get("lease_epoch") or 0)
        return (timestamp, worker_id, lease_epoch, str(job_id or ""))

    def _pop_next_ready_job_id(self) -> str | None:
        self._apply_scheduler_membership_rule()
        now = time.monotonic()
        ordered_pending = sorted(set(self._pending_job_ids), key=lambda job_id: self._fairness_key(job_id, now))
        for job_id in ordered_pending:
            pair = self._jobs.get(job_id)
            if pair is None:
                self._pending_job_ids = [pending_id for pending_id in self._pending_job_ids if pending_id != job_id]
                continue
            job, _future = pair
            if not self._job_ready(job, now=now):
                continue
            with contextlib.suppress(ValueError):
                self._pending_job_ids.remove(job_id)
            return job_id
        return None

    def _fairness_key(self, job_id: str, now: float) -> tuple[float, float, float, str, int, str]:
        pair = self._jobs.get(job_id)
        if pair is None:
            return (float("inf"), float("inf"), float("inf"), "", 0, str(job_id or ""))
        job, _future = pair
        metadata = dict(job.metadata or {})
        scheduling = dict(metadata.get("scheduling") or {})
        raw_priority = scheduling.get("priority")
        if raw_priority is None:
            raw_priority = metadata.get("priority")
        base_priority = float(100.0 if raw_priority is None else raw_priority)
        raw_submitted_monotonic = scheduling.get("submitted_monotonic")
        submitted_monotonic = float(now if raw_submitted_monotonic is None else raw_submitted_monotonic)
        # Use whole-second age buckets to avoid clock jitter changing deterministic order.
        age_seconds = float(max(0, int(now - submitted_monotonic)))
        effective_priority = max(0.0, base_priority - (age_seconds * self.fairness_aging_rate))
        tenant_id = str(metadata.get("tenant_id") or "global")
        tenant_pending_count = 0.0
        for pending_job_id in self._pending_job_ids:
            pending_pair = self._jobs.get(pending_job_id)
            if pending_pair is None:
                continue
            pending_job, _pending_future = pending_pair
            pending_tenant_id = str(dict(pending_job.metadata or {}).get("tenant_id") or "global")
            if pending_tenant_id == tenant_id:
                tenant_pending_count += 1.0
        tenant_penalty = max(0.0, tenant_pending_count - 1.0) * self.tenant_balance_weight
        claim_key = self._claim_order_key(job_id)
        return (effective_priority + tenant_penalty, effective_priority, *claim_key)

    def _emit_scheduler_event(
        self,
        *,
        event_type: str,
        job: ExecutionJob,
        payload: dict[str, Any],
        step_key: str,
    ) -> None:
        write_event = getattr(self.writer, "write_event", None)
        if not callable(write_event):
            return
        write_event(
            event_type=event_type,
            session_id=str(job.session_id or "default"),
            trace_id=str(job.trace_id or ""),
            kernel_step_id=step_key,
            payload={
                "job_id": str(job.job_id or ""),
                "request_id": str(dict(job.metadata or {}).get("request_id") or ""),
                "execution_state": dict(payload.get("execution_state") or {}),
                **{key: value for key, value in dict(payload or {}).items() if key != "execution_state"},
            },
            committed=False,
        )

    def _emit_failure_policy_event(self, *, job: ExecutionJob, failure: dict[str, Any]) -> None:
        action = str(failure.get("failure_action") or "").strip().lower()
        strategy = _FAILURE_POLICY_STRATEGIES.get(action)
        if strategy is None:
            return
        self._emit_scheduler_event(
            event_type=strategy.event_type,
            job=job,
            payload={
                "execution_state": dict(job.metadata.get("execution_state") or {}),
                "failure_type": str(failure.get("failure_type") or ""),
                "failure_action": str(failure.get("failure_action") or ""),
                "auto_retry": bool(failure.get("auto_retry", False)),
            },
            step_key="scheduler.execute.failure_policy",
        )

    def _append_lifecycle_event(
        self,
        job: ExecutionJob,
        event: Any,
        *,
        step_key: str,
        committed: bool = False,
    ) -> dict[str, Any]:
        projected = self.projection.get(job.job_id)
        current_state: ExecutionLifecycleState | None = None
        if projected is not None:
            current_state = _coerce_lifecycle_state(_lifecycle_state_from_projection(projected))
        _assert_lifecycle_emission_transition(
            execution_id=str(job.job_id or ""),
            event=event,
            current_state=current_state,
        )
        before_state = current_state or ExecutionLifecycleState.SUBMITTED
        after_state = _target_lifecycle_state_for_event(event, current_state=before_state)
        before_transition, after_transition = _build_sovereign_transition_states(
            job=job,
            before_state=before_state,
            after_state=after_state,
        )
        validate_sovereign_ledger_transition(before_transition, after_transition)

        metadata = dict(job.metadata or {})
        execution_state = dict(metadata.get("execution_state") or {})
        execution_state["causal_step_count"] = int(after_transition.get("causal_step_count") or 0)
        execution_state["invariance_hash"] = str(after_transition.get("invariance_hash") or "")
        if after_state == ExecutionLifecycleState.COMPLETED:
            execution_state["turn_truth_ok"] = bool(after_transition.get("turn_truth_ok"))
        metadata["execution_state"] = execution_state
        job.metadata = metadata

        payload = self.writer.append_execution_lifecycle(
            event,
            session_id=str(job.session_id or "default"),
            trace_token=str(job.trace_id or ""),
            step_key=step_key,
            committed=committed,
        )
        state = self.projection.apply(event)
        _apply_projection_execution_state(job, state)
        return payload

    @staticmethod
    def _resolve_options(
        options: SchedulerOptions | None,
        legacy_options: dict[str, Any],
    ) -> SchedulerOptions:
        resolved = options or SchedulerOptions()
        if "max_inflight_jobs" in legacy_options:
            resolved.max_inflight_jobs = int(legacy_options["max_inflight_jobs"])
        if "projection" in legacy_options:
            resolved.projection = legacy_options["projection"]
        if "worker_id" in legacy_options:
            resolved.worker_id = str(legacy_options["worker_id"] or "worker-1")
        if "execution_token" in legacy_options:
            resolved.execution_token = str(legacy_options["execution_token"] or "")
        if "enable_observability" in legacy_options:
            resolved.enable_observability = bool(legacy_options["enable_observability"])
        if "execution_lease" in legacy_options:
            resolved.execution_lease = legacy_options["execution_lease"]
        if "lease_ttl_seconds" in legacy_options:
            resolved.lease_ttl_seconds = float(legacy_options["lease_ttl_seconds"])
        if "redelivery_retry_interval_seconds" in legacy_options:
            resolved.redelivery_retry_interval_seconds = float(
                legacy_options["redelivery_retry_interval_seconds"],
            )
        if "fairness_aging_rate" in legacy_options:
            resolved.fairness_aging_rate = float(legacy_options["fairness_aging_rate"])
        if "tenant_balance_weight" in legacy_options:
            resolved.tenant_balance_weight = float(legacy_options["tenant_balance_weight"])
        if "on_runtime_claim_guard" in legacy_options:
            resolved.on_runtime_claim_guard = legacy_options["on_runtime_claim_guard"]
        if "on_runtime_lease_sync" in legacy_options:
            resolved.on_runtime_lease_sync = legacy_options["on_runtime_lease_sync"]
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
        error: BaseException | None = None,
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
        failure: dict[str, Any] | None = None,
    ) -> None:
        if not self.enable_observability:
            return
        metrics = get_metrics()
        metrics.increment(f"scheduler.job.{event}")
        metrics.observe(
            "scheduler.job.latency_ms",
            (time.perf_counter() - started_at) * 1000.0,
        )
        payload: dict[str, Any] = self._observability_payload(event=event, job=job, error=error)
        self._attach_observability_failure(payload=payload, job=job, failure=failure)
        get_exporter().export(payload)

    @staticmethod
    def _observability_payload(*, event: str, job: ExecutionJob, error: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event": f"job.{event}",
            "job_id": job.job_id,
            "session_id": job.session_id,
        }
        if error:
            payload["error"] = error
        return payload

    @staticmethod
    def _unified_failure_payload(failure_view: dict[str, Any]) -> dict[str, Any]:
        return {
            "failure_class": str(failure_view.get("class") or ""),
            "failure_source": str(failure_view.get("source") or ""),
            "retryable": bool(failure_view.get("retryable", False)),
            "error_type": str(failure_view.get("type") or ""),
            "message": str(failure_view.get("message") or ""),
            "class": str(failure_view.get("class") or ""),
            "source": str(failure_view.get("source") or ""),
            "type": str(failure_view.get("type") or ""),
        }

    @staticmethod
    def _merge_legacy_failure_fields(target: dict[str, Any], failure: dict[str, Any]) -> None:
        target["failure_type"] = str(failure.get("failure_type") or "")
        target["failure_action"] = str(failure.get("failure_action") or "")
        target["auto_retry"] = bool(failure.get("auto_retry", False))

    def _attach_observability_failure(
        self,
        *,
        payload: dict[str, Any],
        job: ExecutionJob,
        failure: dict[str, Any] | None,
    ) -> None:
        execution_result = ensure_unified_execution_result(
            dict(getattr(job, "metadata", {}).get("execution_result") or {}),
        )
        failure_view = dict(execution_result.get("failure") or {})
        has_failure = bool(
            str(failure_view.get("class") or "")
            or str(failure_view.get("type") or "")
            or str(failure_view.get("message") or ""),
        )
        if has_failure:
            payload["failure"] = self._unified_failure_payload(failure_view)
            if isinstance(failure, dict) and bool(failure):
                self._merge_legacy_failure_fields(payload["failure"], failure)
            return
        if isinstance(failure, dict) and bool(failure):
            # Legacy fallback for older callers; unified execution_result is authoritative.
            payload["failure"] = dict(failure)

    async def register(self, job: ExecutionJob) -> asyncio.Future[FinalizedTurnResult]:
        if len(self._jobs) >= self.max_inflight_jobs:
            raise BackpressureSignal(
                reason="max inflight jobs reached",
                retry_after_ms=self.redelivery_retry_interval_seconds * 1000.0,
                trace_id=job.trace_id,
            )
        assert str(job.trace_id or "").strip(), "Missing trace_id at scheduler register"

        loop = asyncio.get_running_loop()
        future: asyncio.Future[FinalizedTurnResult] = loop.create_future()
        _apply_projection_execution_state(job, self.projection.get(job.job_id))
        job.metadata.setdefault("execution_state", {})["last_transition_reason"] = "scheduler.register"
        job.metadata.setdefault("claim_order", {})
        job.metadata.setdefault("scheduling", {})
        job.metadata["claim_order"].setdefault(
            "timestamp",
            float(dict(job.metadata or {}).get("submitted_timestamp") or 0.0),
        )
        job.metadata["claim_order"].setdefault(
            "worker_id",
            str(self.worker_id or "worker-1"),
        )
        job.metadata["claim_order"].setdefault(
            "lease_epoch",
            int(dict(job.metadata.get("execution_state") or {}).get("redelivery_count") or 0),
        )
        raw_priority = dict(job.metadata or {}).get("priority")
        priority_value = 100.0 if raw_priority is None else float(raw_priority)
        job.metadata["scheduling"].setdefault(
            "submitted_monotonic",
            float(time.monotonic()),
        )
        job.metadata["scheduling"].setdefault(
            "priority",
            priority_value,
        )
        job.metadata.setdefault("tenant_id", str(dict(job.metadata or {}).get("tenant_id") or "global"))
        job.metadata["execution_mode"] = _resolved_execution_mode(job)
        self._jobs[job.job_id] = (job, future)
        self._pending_job_ids.append(job.job_id)
        self._notify_work_available()
        self.writer.append_job_queued(job)
        return future

    def _release_runtime_lease(self, *, session_id: str) -> None:
        if self.execution_lease is not None:
            with contextlib.suppress(Exception):
                self.execution_lease.release(session_id=str(session_id or "default"), owner_id=self.worker_id)
        if callable(self._on_runtime_lease_sync):
            self._on_runtime_lease_sync("")

    def _emit_redelivery_events(
        self,
        *,
        job: ExecutionJob,
        projected: ReducedExecutionState,
        now: datetime,
    ) -> None:
        if projected.owner and lease_expired(projected, now=now):
            prior_owner = str(projected.owner or "")
            self._append_lifecycle_event(
                job,
                LeaseExpired(
                    execution_id=job.job_id,
                    occurred_at=now,
                    worker_id=projected.owner,
                ),
                step_key="scheduler.lease_expired",
            )
            self._append_lifecycle_event(
                job,
                Redelivered(
                    execution_id=job.job_id,
                    occurred_at=now,
                    previous_worker_id=prior_owner,
                    new_worker_id=self.worker_id,
                ),
                step_key="scheduler.redelivery",
            )
            self._emit_scheduler_event(
                event_type="JOB_REDELIVERY_SCHEDULED",
                job=job,
                payload={
                    "execution_state": dict(job.metadata.get("execution_state") or {}),
                    "lease_owner": str(projected.owner or ""),
                },
                step_key="scheduler.redelivery",
            )
        elif projected.attempt_count <= 0:
            prior_owner = str(dict(job.metadata.get("execution_state") or {}).get("last_worker_id") or "")
            if prior_owner:
                self._append_lifecycle_event(
                    job,
                    Redelivered(
                        execution_id=job.job_id,
                        occurred_at=now,
                        previous_worker_id=prior_owner,
                        new_worker_id=self.worker_id,
                    ),
                    step_key="scheduler.redelivery.external",
                )

    def _claim_job_for_execution(
        self,
        *,
        job_id: str,
        job: ExecutionJob,
        projected: ReducedExecutionState,
        now: datetime,
    ) -> bool:
        if callable(self._on_runtime_claim_guard):
            self._on_runtime_claim_guard("scheduler_claim")
        lease_token: dict[str, Any] | None = None
        if self.execution_lease is not None:
            lease_token = self.execution_lease.acquire(
                session_id=str(job.session_id or "default"),
                owner_id=self.worker_id,
                ttl_seconds=self.lease_ttl_seconds,
            )
        if projected.owner and not lease_expired(projected, now=now):
            self._release_runtime_lease(session_id=job.session_id)
            self._pending_job_ids.append(job_id)
            self._notify_work_available()
            return False
        self._emit_redelivery_events(job=job, projected=projected, now=now)

        self._append_lifecycle_event(
            job,
            Claimed(
                execution_id=job.job_id,
                occurred_at=now,
                worker_id=self.worker_id,
                lease_expiry=now + timedelta(seconds=self.lease_ttl_seconds),
            ),
            step_key="scheduler.claim",
        )
        if lease_token is not None:
            job.metadata.setdefault("lease_fence", {})
            job.metadata["lease_fence"]["fencing_token"] = int(lease_token.get("fencing_token") or 0)
            job.metadata["lease_fence"]["lease_id"] = str(lease_token.get("lease_id") or "")
        fence_token = int(lease_token.get("fencing_token") or 0) if lease_token is not None else 0
        if callable(self._on_runtime_lease_sync):
            self._on_runtime_lease_sync(f"{self.execution_token}:{job.job_id}:{fence_token}")
        return True

    def _record_scheduler_success(
        self,
        *,
        job: ExecutionJob,
        future: asyncio.Future[FinalizedTurnResult],
        result: FinalizedTurnResult,
        started_at: float,
    ) -> None:
        current_execution_result = ensure_unified_execution_result(
            dict(job.metadata.get("execution_result") or {}),
        )
        current_execution_result = mark_unified_execution_success(
            cast(dict[str, Any], current_execution_result),
            response=str(result[0] if isinstance(result, tuple) and len(result) >= 1 else ""),
            should_end=bool(result[1] if isinstance(result, tuple) and len(result) >= 2 else False),
        )
        job.metadata["execution_result"] = current_execution_result
        self._append_lifecycle_event(
            job,
            Completed(
                execution_id=job.job_id,
                occurred_at=datetime.now(),
                result_ref=f"job:{job.job_id}:result",
            ),
            step_key="scheduler.execute.complete",
            committed=True,
        )
        effect_id = str(dict(job.metadata or {}).get("effect_id") or "").strip()
        request_id = str(dict(job.metadata or {}).get("request_id") or "").strip()
        if effect_id:
            self.writer.append_effect_commit(
                session_id=str(job.session_id or "default"),
                trace_token=str(job.trace_id or ""),
                effect_id=effect_id,
                request_id=request_id,
                step_key="scheduler.execute.effect.commit",
            )
        _set_terminal_turn_state(
            job,
            terminal_state=_SCHEDULER_EXCEPTION_MAPPER.from_success(
                recovered=(_resolved_execution_mode(job) == "recovery"),
            ),
            reason="scheduler.execute.complete",
        )
        self.writer.append_job_completed(job, result)
        self._resolve_future(future, result=result)
        self._release_runtime_lease(session_id=job.session_id)
        self._record_job_observability(
            event="completed",
            job=job,
            started_at=started_at,
        )

    def _record_scheduler_failure(
        self,
        *,
        job: ExecutionJob,
        future: asyncio.Future[FinalizedTurnResult],
        exc: BaseException,
        started_at: float,
    ) -> None:
        failure = _classify_execution_failure(exc)
        current_execution_result = mark_unified_execution_failure(
            cast(dict[str, Any], build_unified_execution_result()),
            failure_class=str(failure.get("failure_class") or "runtime_exception"),
            failure_source=str(failure.get("failure_source") or "execution"),
            retryable=bool(failure.get("retryable", False)),
            exception_type=str(failure.get("exception_type") or type(exc).__name__),
            message=str(exc),
        )
        job.metadata["execution_result"] = current_execution_result
        execution_state = dict(job.metadata.get("execution_state") or {})
        execution_state["failure_type"] = str(failure.get("failure_type") or "")
        execution_state["failure_action"] = str(failure.get("failure_action") or "")
        execution_state["auto_retry"] = bool(failure.get("auto_retry", False))
        execution_state["last_transition_reason"] = (
            f"scheduler.execute.failed:{str(failure.get('failure_action') or 'unknown')}"
        )
        job.metadata["execution_state"] = execution_state
        _set_terminal_turn_state(
            job,
            terminal_state=_SCHEDULER_EXCEPTION_MAPPER.from_exception(exc),
            reason=f"scheduler.execute.failed:{type(exc).__name__}",
        )
        self._emit_failure_policy_event(job=job, failure=failure)
        self._append_lifecycle_event(
            job,
            Failed(
                execution_id=job.job_id,
                occurred_at=datetime.now(),
                error_ref=f"{type(exc).__name__}:{str(exc)}",
            ),
            step_key="scheduler.execute.failed",
            committed=True,
        )
        self.writer.append_job_failed(job, exc, failure=failure)
        self._resolve_future(future, error=exc)
        self._release_runtime_lease(session_id=job.session_id)
        self._record_job_observability(
            event="failed",
            job=job,
            started_at=started_at,
            error=str(exc),
            failure=failure,
        )

    def _maybe_begin_effect(self, job: ExecutionJob) -> None:
        effect_id = str(dict(job.metadata or {}).get("effect_id") or "").strip()
        request_id = str(dict(job.metadata or {}).get("request_id") or "").strip()
        if effect_id:
            self.writer.append_effect_begin(
                session_id=str(job.session_id or "default"),
                trace_token=str(job.trace_id or ""),
                effect_id=effect_id,
                request_id=request_id,
                step_key="scheduler.execute.effect.begin",
            )

    async def _execute_and_record_job(
        self,
        executor: Callable[[dict[str, Any], ExecutionJob], Awaitable[FinalizedTurnResult]],
        job: ExecutionJob,
        future: Any,
        started_at: float,
    ) -> bool:
        job.metadata["execution_mode"] = _resolved_execution_mode(job)
        self._maybe_begin_effect(job)
        self.writer.append_job_started(job)
        session = self.registry.bind(job.session_id)
        tracer = get_tracer()
        with tracer.span("scheduler.drain_once"):
            result = await self._execute_with_boundary(executor, session, job)
        self._invariant_enforcer.enforce_global_turn_invariant_gate(
            session=session,
            execution_result=dict(job.metadata.get("execution_result") or {}),
            trace_id=str(job.trace_id or ""),
        )
        self._record_scheduler_success(
            job=job,
            future=future,
            result=result,
            started_at=started_at,
        )
        return True

    async def drain_once(
        self,
        executor: Callable[
            [dict[str, Any], ExecutionJob],
            Awaitable[FinalizedTurnResult],
        ],
    ) -> bool:
        if not self._pending_job_ids:
            return False

        job_id = self._pop_next_ready_job_id()
        if job_id is None:
            return False
        job_pair = self._jobs.get(job_id)
        if job_pair is None:
            return False
        job, future = job_pair
        assert str(job.trace_id or "").strip(), "Missing trace_id at scheduler drain"

        started_at = time.perf_counter()
        projected = self.projection.get(job.job_id)
        _apply_projection_execution_state(job, projected)
        now = datetime.now()

        if projected is None:
            raise RuntimeError(f"missing lifecycle state for job {job.job_id!r}")

        if projected.status in {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED}:
            self._jobs.pop(job_id, None)
            return False

        try:
            claimed = self._claim_job_for_execution(
                job_id=job_id,
                job=job,
                projected=projected,
                now=now,
            )
            if not claimed:
                return False
        except RuntimeError:
            self._release_runtime_lease(session_id=job.session_id)
            self._pending_job_ids.append(job_id)
            self._notify_work_available()
            return False

        try:
            return await self._execute_and_record_job(executor, job, future, started_at)
        except asyncio.CancelledError as exc:
            logger.exception(
                "scheduler drain cancelled for job_id=%s session_id=%s",
                str(job.job_id or ""),
                str(job.session_id or ""),
            )
            self._record_scheduler_failure(
                job=job,
                future=future,
                exc=exc,
                started_at=started_at,
            )
            raise
        except (
            TimeoutError,
            ExecutionStageError,
            InvariantViolation,
            MemorySetInvariantViolation,
            PersistenceFailure,
            RuntimeError,
        ) as exc:
            logger.exception(
                "scheduler drain failed for job_id=%s session_id=%s",
                str(job.job_id or ""),
                str(job.session_id or ""),
            )
            self._record_scheduler_failure(
                job=job,
                future=future,
                exc=exc,
                started_at=started_at,
            )
            raise
        finally:
            if future.done():
                if not future.cancelled():
                    with contextlib.suppress(BaseException):
                        future.exception()
                self._jobs.pop(job_id, None)


class ExecutionControlPlane(ReconciliationMixin, CompactionMixin):
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
        self._stream_sink = resolved_options.stream_sink
        token_seed = (
            f"{resolved_options.worker_id}|"
            f"{resolved_options.max_inflight_jobs}|"
            f"{int(bool(resolved_options.enable_observability))}"
        )
        self.execution_token = f"exec-{hashlib.sha256(token_seed.encode('utf-8')).hexdigest()[:20]}"
        self.ledger = resolved_options.ledger or InMemoryExecutionLedger()
        self._ledger_coordinator = LedgerCoordinator(
            ledger=self.ledger,
            scope_validator=KernelGateway.assert_scope,
        )
        ledger_writer_gateway = LedgerWriterAdapter(
            self.ledger,
            scope_validator=KernelGateway.assert_scope,
        )
        ledger_runtime = self._ledger_coordinator.build_runtime(
            writer=ledger_writer_gateway,
            reconcile_queue_factory=lambda writer, effect_journal: DurableReconcileQueue(
                ledger=self.ledger,
                write_event=getattr(writer, "write_event", None),
                request_is_ambiguous=lambda session_id, request_id: self._request_has_ambiguous_inflight_effect_state(
                    session_id=session_id,
                    request_id=request_id,
                ),
                effect_is_ambiguous=lambda session_id, effect_id: effect_journal.is_ambiguous(
                    session_id=session_id,
                    effect_id=effect_id,
                ),
            ),
        )
        self._ledger_writer = ledger_runtime.writer
        self._ledger_index = ledger_runtime.index
        self._effect_journal = ledger_runtime.effect_journal
        self.ledger_reader = ledger_runtime.reader
        self._reconcile_queue = ledger_runtime.reconcile_queue
        self._execution_lease = ExecutionLease(default_ttl_seconds=resolved_options.lease_ttl_seconds)
        scheduler_options = SchedulerOptions(
            max_inflight_jobs=resolved_options.max_inflight_jobs,
            worker_id=resolved_options.worker_id,
            execution_token=self.execution_token,
            enable_observability=resolved_options.enable_observability,
            projection=ExecutionProjection(),
            execution_lease=self._execution_lease,
            lease_ttl_seconds=resolved_options.lease_ttl_seconds,
            redelivery_retry_interval_seconds=resolved_options.redelivery_retry_interval_seconds,
            on_runtime_claim_guard=self._scheduler_claim_guard,
            on_runtime_lease_sync=self._scheduler_lease_sync,
        )
        self._scheduler = resolved_options.scheduler or Scheduler(
            registry,
            reader=self.ledger_reader,
            writer=self._ledger_writer,
            options=scheduler_options,
        )
        self.lifecycle_projection = getattr(self._scheduler, "projection", ExecutionProjection())
        self.recovery = RecoveryManager(ledger=self.ledger)
        self._inflight_by_request: dict[tuple[str, str, str], asyncio.Future[FinalizedTurnResult]] = {}
        self._inflight_lock = asyncio.Lock()
        self.graph = graph
        bind_execution_token = getattr(self.graph, "set_required_execution_token", None)
        if callable(bind_execution_token):
            bind_execution_token(self.execution_token)
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
        self._execution_coordinator = ExecutionCoordinator()
        self._persistence_coordinator = PersistenceCoordinator()
        self._stream_event_sequence: int = 0
        self._distributed_correctness = DistributedCorrectnessModel()
        self._distributed_epoch = 1
        # Bootstrap: register local node as initial authority to allow first turn submission.
        # Scheduler lease sync will update this periodically during execution.
        initial_now_ms = self._distributed_now_ms()
        self._distributed_correctness.register_node(
            node_id=self._scheduler.worker_id,
            epoch=int(self._distributed_epoch),
            lease_until_ms=initial_now_ms + int(self._scheduler.lease_ttl_seconds * 1000),
            role=NodeRole.LEADER,
            state_hash=self.execution_token,
        )
        self._cognitive_policy_engine = CognitivePolicyEngine()
        self._semantic_memory_graph = SemanticMemoryGraph()
        self._tool_routing_engine = ToolRoutingEngine()
        self._adaptation_engine = AdaptationEngine()
        self._semantic_safety_engine = SemanticSafetyEngine()
        self._belief_state_engine = BeliefStateEngine()
        self._hypothesis_engine = MultiHypothesisEngine()
        self._memory_hierarchy_manager = MemoryHierarchyManager()
        self._tool_self_model = ToolSelfModel()
        self._compositional_tool_planner = CompositionalToolPlanner()
        self._planning_optimizer = SessionPlanningOptimizer()
        self._autonomous_goal_daemon = AutonomousGoalDaemon()
        self._interactive_cognition_ui = InteractiveCognitionUI()
        self._alignment_trainer = BehaviorAlignmentTrainer()
        self._tool_ecosystem_hub = ToolEcosystemHub()
        self._multi_agent_swarm = MultiAgentSwarm()
        self._response_engine = ResponseEngine()
        self._topology_runtime = TopologyRuntime(strict_mode=True)
        self._last_topology_validation: TopologyValidationResult | None = None
        self._phase_closure_runtime = PhaseClosureRuntime()
        self._autonomous_goal_task: asyncio.Task[None] | None = None
        self._autonomous_goal_stop: asyncio.Event | None = None
        self._sync_distributed_authority(role=NodeRole.LEADER, state_hash=self.execution_token)
        self.kernel_gateway = KernelGateway(self)
        self.bootstrap()

    def _topology_begin_turn(self, *, trace_id: str, session_id: str) -> None:
        self._topology_runtime.begin_turn(
            trace_id=str(trace_id or ""),
            session_id=str(session_id or "default"),
        )

    def _topology_record_node(
        self,
        *,
        node_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._topology_runtime.record_node_entry(
            node_id=str(node_id or ""),
            timestamp_ms=float(time.time() * 1000.0),
            metadata=dict(metadata or {}),
        )

    def _topology_end_turn(self) -> TopologyValidationResult:
        result = self._topology_runtime.end_turn()
        self._last_topology_validation = result
        # "no_active_trace" means a concurrent turn already ended the shared trace
        # (deduplicated concurrent requests land here); treat as a no-op, not a violation.
        if result.details.get("reason") == "no_active_trace":
            return result
        if not result.passed or result.violations_critical > 0:
            raise InvariantViolation(
                "Topology runtime enforcement blocked non-canonical execution path.",
                context={
                    "violations_critical": int(result.violations_critical),
                    "violations_high": int(result.violations_high),
                    "violations_total": int(result.violations_total),
                    "details": dict(result.details or {}),
                },
            )
        return result

    def _pre_execution_contract_gate(
        self,
        *,
        session_id: str,
        user_input: str,
        metadata: dict[str, Any],
    ) -> None:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            raise RuntimeError("pre-execution contract gate requires dict state")

        canonical_state = self._phase_closure_runtime.kernel.ensure_global_canonical_state_schema(dict(state))
        metadata["pre_execution_canonical_state_hash"] = self._stable_hash(canonical_state)

        metadata.setdefault("execution_timestamp_ms", int(time.time() * 1000))
        metadata.setdefault("max_staleness_ms", 2000)

        pre_drift = self._phase_closure_runtime.kernel.detect_drift(
            slot=f"pre_execution:{str(session_id or 'default')}",
            value={"state": canonical_state, "input": str(user_input or "")},
        )
        metadata["drift_precheck"] = dict(pre_drift)
        self._phase_closure_runtime.kernel.register_side_effect(
            effect_type="execution_stage",
            subject="pre_execution",
            payload={"session_id": str(session_id or "default")},
            trace_id=str(metadata.get("trace_id") or ""),
        )

    def _post_planning_pre_tool_contract_gate(
        self,
        *,
        session_id: str,
        metadata: dict[str, Any],
    ) -> None:
        tool_contract = dict(metadata.get("tool_runtime_contract") or {})
        tool_name = str(tool_contract.get("tool_name") or "").strip()
        runtime_plan = dict(metadata.get("runtime_plan") or {})
        uncertainty = float(dict(runtime_plan.get("uncertainty") or {}).get("score") or 0.0)
        strategy = str(runtime_plan.get("strategy") or "").strip().lower()
        uncertainty_action = str(runtime_plan.get("uncertainty_action") or "").strip().lower()

        if tool_name and uncertainty >= 0.75 and (
            strategy in {"task_execution", "direct_answer"}
            or uncertainty_action not in {"branch", "retry", "hedge", "clarify"}
        ):
            raise RuntimeError(
                "Uncertainty enforcement gate: planner must branch/hedge before tool execution",
            )

        memory_entries = [
            dict(item)
            for item in list(metadata.get("world_model_memory_entries") or [])
            if isinstance(item, dict)
        ]
        entity_bindings = [
            dict(item)
            for item in list(metadata.get("world_model_entity_bindings") or [])
            if isinstance(item, dict)
        ]
        self._phase_closure_runtime.world.validate_memory_bindings(
            memory_entries=memory_entries,
            entity_bindings=entity_bindings,
        )

        self._phase_closure_runtime.kernel.register_side_effect(
            effect_type="execution_stage",
            subject="post_planning_pre_tool",
            payload={
                "session_id": str(session_id or "default"),
                "tool_name": tool_name,
                "uncertainty": uncertainty,
            },
            trace_id=str(metadata.get("trace_id") or ""),
        )

    def _post_execution_pre_commit_contract_gate(
        self,
        *,
        session_id: str,
        job: ExecutionJob,
        result: FinalizedTurnResult,
        input_state_hash: str,
    ) -> None:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            raise RuntimeError("post-execution contract gate requires dict state")

        semantic_items = [
            dict(item)
            for item in list(dict(state.get("semantic_memory") or {}).get("items") or [])
            if isinstance(item, dict)
        ]
        memory_entries: list[dict[str, Any]] = []
        for index, item in enumerate(semantic_items, start=1):
            row = dict(item)
            row.setdefault("id", str(row.get("memory_id") or f"mem-{index}"))
            memory_entries.append(row)
        self._phase_closure_runtime.world.validate_memory_bindings(
            memory_entries=memory_entries,
            entity_bindings=[
                dict(item)
                for item in list(job.metadata.get("world_model_entity_bindings") or [])
                if isinstance(item, dict)
            ],
        )

        side_effect_records = [
            dict(record.__dict__)
            for record in self._phase_closure_runtime.kernel.side_effect_registry.records(
                trace_id=str(job.trace_id or ""),
            )
        ]
        post_drift = self._phase_closure_runtime.kernel.detect_drift(
            slot=f"post_execution:{str(session_id or 'default')}",
            value={"state": state, "result": list(result) if isinstance(result, tuple) else result},
        )

        state_transition = {
            "turn_id": str(job.job_id or ""),
            "trace_id": str(job.trace_id or ""),
            "input_state_hash": str(input_state_hash or ""),
            "action": str(dict(job.metadata.get("runtime_plan") or {}).get("strategy") or ""),
            "tool_calls": [
                dict(item)
                for item in list(dict(job.metadata.get("tool_routing_plan") or {}).get("candidates") or [])
                if isinstance(item, dict)
            ][:5],
            "output_state_hash": self._stable_hash(state),
            "side_effects": side_effect_records,
            "timestamp": float(time.time()),
            "drift_postcheck": dict(post_drift),
        }
        transitions = [
            dict(item)
            for item in list(state.get("state_transition_ledger") or [])
            if isinstance(item, dict)
        ]
        transitions.append(state_transition)
        state["state_transition_ledger"] = transitions[-1024:]
        state["last_state_transition"] = dict(state_transition)
        self._merge_session_state(session_id=str(session_id or "default"), state_patch=state)
        self._emit_runtime_stream_event(
            event_type="state.transition.recorded",
            session_id=str(session_id or "default"),
            trace_id=str(job.trace_id or ""),
            payload={
                "turn_id": str(state_transition.get("turn_id") or ""),
                "input_state_hash": str(state_transition.get("input_state_hash") or ""),
                "output_state_hash": str(state_transition.get("output_state_hash") or ""),
                "action": str(state_transition.get("action") or ""),
                "tool_call_count": int(len(list(state_transition.get("tool_calls") or []))),
                "side_effect_count": int(len(list(state_transition.get("side_effects") or []))),
                "timestamp": float(state_transition.get("timestamp") or 0.0),
            },
        )

    def set_turn_steering(self, *, session_id: str, steering: dict[str, Any]) -> dict[str, Any]:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            return {"applied": False, "reason": "no_state"}
        state["pending_turn_steering"] = dict(steering or {})
        self._interactive_cognition_ui.apply_live_control(
            state=state,
            control=dict(steering or {}),
            source="set_turn_steering",
        )
        self._emit_runtime_stream_event(
            event_type="turn.steering.updated",
            session_id=str(session_id or "default"),
            trace_id="",
            payload={"steering": dict(steering or {})},
        )
        return {"applied": True, "steering": dict(steering or {})}

    def update_interactive_plan(
        self,
        *,
        session_id: str,
        trace_id: str,
        edits: dict[str, Any],
        actor: str = "operator",
    ) -> dict[str, Any]:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            return {"updated": False, "reason": "no_state"}
        updated_plan = self._interactive_cognition_ui.apply_plan_edit(
            state=state,
            trace_id=str(trace_id or ""),
            edits=dict(edits or {}),
            actor=str(actor or "operator"),
        )
        return {"updated": True, "plan": dict(updated_plan)}

    def register_external_connector(
        self,
        *,
        session_id: str,
        name: str,
        capabilities: list[str],
        endpoint: str = "",
        health: float = 1.0,
    ) -> dict[str, Any]:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            return {"registered": False, "reason": "no_state"}
        connector = self._tool_ecosystem_hub.register_connector(
            state=state,
            name=str(name or ""),
            capabilities=[str(item) for item in list(capabilities or [])],
            endpoint=str(endpoint or ""),
            health=float(health),
        )
        return {"registered": bool(connector), "connector": dict(connector)}

    def swarm_status(self, *, session_id: str) -> dict[str, Any]:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            return {"available": False, "reason": "no_state"}
        return self._multi_agent_swarm.health_snapshot(state=state)

    def _consume_pending_turn_steering(self, *, session_id: str) -> dict[str, Any]:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            return {}
        pending = dict(state.get("pending_turn_steering") or {})
        if pending:
            state.pop("pending_turn_steering", None)
        return pending

    async def start_autonomous_goal_loop(self, *, interval_seconds: float = 60.0) -> dict[str, Any]:
        interval = max(1.0, float(interval_seconds or 60.0))
        if self._autonomous_goal_task is not None and not self._autonomous_goal_task.done():
            return {"started": False, "reason": "already_running", "interval_seconds": interval}
        self._autonomous_goal_stop = asyncio.Event()

        async def _run_loop() -> None:
            assert self._autonomous_goal_stop is not None
            while not self._autonomous_goal_stop.is_set():
                for session_name in list(getattr(self.registry, "_sessions", {}).keys()):
                    with contextlib.suppress(Exception):
                        self.run_autonomous_goal_cycle(session_id=session_name, source="daemon")
                try:
                    await asyncio.wait_for(self._autonomous_goal_stop.wait(), timeout=interval)
                except TimeoutError:
                    continue

        self._autonomous_goal_task = asyncio.create_task(_run_loop(), name="dadbot.autonomous_goal_loop")
        return {"started": True, "interval_seconds": interval}

    async def stop_autonomous_goal_loop(self) -> dict[str, Any]:
        if self._autonomous_goal_task is None:
            return {"stopped": False, "reason": "not_running"}
        if self._autonomous_goal_stop is not None:
            self._autonomous_goal_stop.set()
        with contextlib.suppress(Exception):
            await self._autonomous_goal_task
        self._autonomous_goal_task = None
        self._autonomous_goal_stop = None
        return {"stopped": True}

    def run_autonomous_goal_cycle(self, *, session_id: str, source: str = "manual") -> dict[str, Any]:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            return {"ran": False, "reason": "no_state"}
        actions = self._autonomous_goal_daemon.next_actions(state=state, max_items=4)
        self._autonomous_goal_daemon.persist_cycle(state=state, actions=actions, source=source)
        self._emit_runtime_stream_event(
            event_type="autonomous.goal.cycle",
            session_id=str(session_id or "default"),
            trace_id="",
            payload={"actions": [dict(item) for item in list(actions or [])], "source": str(source or "manual")},
        )
        return {"ran": True, "actions": [dict(item) for item in list(actions or [])]}

    def _emit_runtime_stream_event(
        self,
        *,
        event_type: str,
        session_id: str,
        trace_id: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._stream_event_sequence += 1
        event = {
            "event_type": str(event_type or "runtime.unknown"),
            "session_id": str(session_id or "default"),
            "trace_id": str(trace_id or ""),
            "timestamp": float(time.time()),
            "sequence": int(self._stream_event_sequence),
            "payload": dict(payload or {}),
        }
        event = dict(validate_trace_event_contract(event))
        self._append_stream_timeline(
            session_id=str(session_id or "default"),
            event=event,
        )
        sink = self._stream_sink
        if callable(sink):
            try:
                sink(event)
            except Exception as exc:
                logger.debug("stream sink failed: %s", exc)
        _write_progress_event(component="runtime_stream", phase=str(event_type or "runtime.unknown"), payload=event)

    def _append_stream_timeline(self, *, session_id: str, event: dict[str, Any]) -> None:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            return
        timeline = list(state.get("execution_timeline") or [])
        timeline.append(dict(event))
        state["execution_timeline"] = timeline[-512:]

    def _merge_session_state(self, *, session_id: str, state_patch: dict[str, Any]) -> dict[str, Any]:
        session = self.registry.get_or_create(str(session_id or "default"))
        existing = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(existing, dict):
            return dict(state_patch or {})
        if state_patch is existing:
            return existing
        existing.update(dict(state_patch or {}))
        session["state"] = existing
        return existing

    def _inject_semantic_memory_context(
        self,
        *,
        session_id: str,
        user_input: str,
        metadata: dict[str, Any],
        limit: int = 5,
    ) -> None:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            metadata["semantic_memory_context"] = []
            return
        semantic_memory = dict(state.get("semantic_memory_projection") or state.get("semantic_memory") or {})
        items = [dict(item) for item in list(semantic_memory.get("items") or []) if isinstance(item, dict)]
        ranked = _rank_semantic_memory_items(
            items=items,
            user_input=str(user_input or ""),
            limit=int(limit),
        )
        metadata["semantic_memory_context"] = ranked

    def _discover_tool_specs(self) -> list[ToolSpec]:
        registry = getattr(self.graph, "tool_registry", None)
        if registry is None:
            return []
        discover = getattr(registry, "discover", None)
        if not callable(discover):
            return []
        try:
            specs = discover()
        except Exception:
            return []
        if not isinstance(specs, list):
            return []
        return [spec for spec in specs if isinstance(spec, ToolSpec)]

    @property
    def scheduler(self) -> SchedulerProtocol:
        KernelGateway.assert_scope("control_plane.scheduler")
        return self._scheduler

    @property
    def ledger_writer(self) -> LedgerWriterAdapter:
        KernelGateway.assert_scope("control_plane.ledger_writer")
        return self._ledger_writer

    @staticmethod
    def _resolve_options(
        options: ControlPlaneOptions | None,
        legacy_options: dict[str, Any],
    ) -> ControlPlaneOptions:
        resolved = options or ControlPlaneOptions()
        if "max_inflight_jobs" in legacy_options:
            resolved.max_inflight_jobs = int(legacy_options["max_inflight_jobs"])
        if "worker_id" in legacy_options:
            resolved.worker_id = str(legacy_options["worker_id"] or "worker-1")
        if "enable_observability" in legacy_options:
            resolved.enable_observability = bool(legacy_options["enable_observability"])
        if "lease_ttl_seconds" in legacy_options:
            resolved.lease_ttl_seconds = float(legacy_options["lease_ttl_seconds"])
        if "redelivery_retry_interval_seconds" in legacy_options:
            resolved.redelivery_retry_interval_seconds = float(
                legacy_options["redelivery_retry_interval_seconds"],
            )
        if "ledger" in legacy_options:
            resolved.ledger = legacy_options["ledger"]
        if "scheduler" in legacy_options:
            resolved.scheduler = legacy_options["scheduler"]
        if "stream_sink" in legacy_options:
            resolved.stream_sink = legacy_options["stream_sink"]
        return resolved

    async def create_session(self, session_id: str) -> dict[str, Any]:
        return await self.registry.create_session(session_id)

    def terminate_session(self, session_id: str) -> None:
        self.registry.terminate_session(session_id)

    def _active_lease_count(self) -> int:
        snapshot = dict(self.lifecycle_projection.snapshot() or {})
        count = 0
        for _execution_id, state in snapshot.items():
            if not isinstance(state, dict):
                continue
            status = str(state.get("status") or "").strip().lower()
            owner = str(state.get("owner") or "").strip()
            if status in {"claimed", "running"} and owner:
                count += 1
        return count

    def _distributed_now_ms(self) -> int:
        return int(time.time() * 1000)

    def _sync_distributed_authority(self, *, role: NodeRole, state_hash: str = "") -> None:
        self._distributed_correctness.register_node(
            node_id=self._scheduler.worker_id,
            epoch=int(self._distributed_epoch),
            lease_until_ms=self._distributed_now_ms() + int(self._scheduler.lease_ttl_seconds * 1000),
            role=role,
            state_hash=str(state_hash or self.execution_token),
        )

    def _scheduler_claim_guard(self, operation: str) -> None:
        self._enforce_distributed_runtime_authority(operation=str(operation or "scheduler_claim"))

    def _scheduler_lease_sync(self, state_hash: str) -> None:
        self._sync_distributed_authority(role=NodeRole.LEADER, state_hash=str(state_hash or ""))

    def _enforce_distributed_runtime_authority(self, *, operation: str) -> None:
        now_ms = self._distributed_now_ms()
        self._distributed_correctness.enforce_no_split_brain(now_ms=now_ms)
        if self._distributed_correctness.validate_authority(
            node_id=self._scheduler.worker_id,
            now_ms=now_ms,
        ):
            return
        authority = self._distributed_correctness.current_authority(now_ms=now_ms)
        raise AuthorityViolation(
            "Distributed correctness violation: non-authoritative runtime path rejected",
            context={
                "operation": str(operation or "unknown"),
                "worker_id": self._scheduler.worker_id,
                "authority": "" if authority is None else authority.node_id,
                "epoch": 0 if authority is None else int(authority.epoch),
                "now_ms": now_ms,
            },
        )

    def distributed_reconciliation_plan(self) -> dict[str, Any]:
        plan = self._distributed_correctness.reconcile(now_ms=self._distributed_now_ms())
        return {
            "authoritative_node": plan.authoritative_node,
            "authoritative_hash": plan.authoritative_hash,
            "divergent_nodes": list(plan.divergent_nodes),
            "converged": bool(plan.converged),
        }

    def _emit_progress_snapshot(
        self,
        *,
        phase: str,
        session_id: str,
        trace_token: str,
        job_id: str,
        future_done: bool,
        completion_expectations: dict[str, bool] | None = None,
        note: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        expectations = dict(completion_expectations or {})
        unmet = [name for name, ok in expectations.items() if not bool(ok)]
        payload = {
            "session_id": str(session_id or "default"),
            "trace_id": str(trace_token or ""),
            "job_id": str(job_id or ""),
            "future_done": bool(future_done),
            "queue_size": int(len(getattr(self._scheduler, "_pending_job_ids", []) or [])),
            "reconciliation_backlog_size": int(len(self._pending_reconcile_required_entries())),
            "active_leases": int(self._active_lease_count()),
            "pending_replay_entries": int(len(self._inflight_by_request)),
            "expected_completion_conditions": expectations,
            "unmet_completion_conditions": unmet,
            "note": str(note or ""),
            "extra": dict(extra or {}),
        }
        _write_progress_event(component="control_plane", phase=phase, payload=payload)
        self._emit_runtime_stream_event(
            event_type="turn.progress",
            session_id=str(session_id or "default"),
            trace_id=str(trace_token or ""),
            payload=payload,
        )

    def _durable_completed_result_for_request(
        self,
        *,
        session_id: str,
        request_id: str,
    ) -> FinalizedTurnResult | None:
        return self._ledger_index.completed_result(
            session_id=session_id,
            request_id=request_id,
        )

    async def _preflight_submit_turn(
        self,
        *,
        session_key: str,
        request_id: str,
        effect_id: str,
        metadata: dict[str, Any],
    ) -> tuple[str, tuple[str, str, str], asyncio.Future[FinalizedTurnResult] | None, bool, FinalizedTurnResult | None]:
        self._emit_progress_snapshot(
            phase="before_replay",
            session_id=session_key,
            trace_token=str(metadata.get("trace_id") or ""),
            job_id="",
            future_done=False,
            note="submit_turn entered",
            extra={"request_id": request_id, "effect_id": effect_id},
        )
        durable_result = self._durable_completed_result_for_request(
            session_id=session_key,
            request_id=request_id,
        )
        durable_short_circuit = self._durable_preflight_result(
            session_key=session_key,
            request_id=request_id,
            effect_id=effect_id,
            metadata=metadata,
            durable_result=durable_result,
        )
        if durable_short_circuit is not None:
            return durable_short_circuit

        effect_id = self._resolve_preflight_effect_id(
            session_key=session_key,
            request_id=request_id,
            effect_id=effect_id,
            metadata=metadata,
        )
        dedupe_key = (session_key, request_id, effect_id)
        dedupe_future, owns_dedupe_slot = await self._acquire_preflight_dedupe_slot(
            request_id=request_id,
            effect_id=effect_id,
            dedupe_key=dedupe_key,
        )
        if not owns_dedupe_slot and dedupe_future is not None:
            return effect_id, dedupe_key, dedupe_future, False, await dedupe_future

        self._assert_preflight_not_ambiguous(
            session_key=session_key,
            request_id=request_id,
            effect_id=effect_id,
            metadata=metadata,
        )

        return effect_id, dedupe_key, dedupe_future, owns_dedupe_slot, None

    def _durable_preflight_result(
        self,
        *,
        session_key: str,
        request_id: str,
        effect_id: str,
        metadata: dict[str, Any],
        durable_result: FinalizedTurnResult | None,
    ) -> tuple[str, tuple[str, str, str], None, bool, FinalizedTurnResult] | None:
        if durable_result is None:
            return None
        self._emit_progress_snapshot(
            phase="before_replay",
            session_id=session_key,
            trace_token=str(metadata.get("trace_id") or ""),
            job_id="",
            future_done=True,
            note="durable completed result returned",
            extra={"request_id": request_id},
        )
        return effect_id, (session_key, request_id, effect_id), None, False, durable_result

    def _resolve_preflight_effect_id(
        self,
        *,
        session_key: str,
        request_id: str,
        effect_id: str,
        metadata: dict[str, Any],
    ) -> str:
        if not effect_id and request_id:
            effect_id = self._effect_journal.derive_effect_id(
                session_id=session_key,
                request_id=request_id,
                trace_id=str(metadata.get("trace_id") or ""),
            )
        if effect_id and self._effect_journal.is_committed(session_id=session_key, effect_id=effect_id):
            raise RuntimeError(
                "Effect already committed but no durable JOB_COMPLETED payload found; "
                "refusing ambiguous replay",
            )
        return effect_id

    async def _acquire_preflight_dedupe_slot(
        self,
        *,
        request_id: str,
        effect_id: str,
        dedupe_key: tuple[str, str, str],
    ) -> tuple[asyncio.Future[FinalizedTurnResult] | None, bool]:
        dedupe_future: asyncio.Future[FinalizedTurnResult] | None = None
        owns_dedupe_slot = False
        if request_id or effect_id:
            async with self._inflight_lock:
                existing = self._inflight_by_request.get(dedupe_key)
                if existing is not None:
                    dedupe_future = existing
                else:
                    dedupe_future = asyncio.get_running_loop().create_future()
                    self._inflight_by_request[dedupe_key] = dedupe_future
                    owns_dedupe_slot = True
        return dedupe_future, owns_dedupe_slot

    def _assert_preflight_not_ambiguous(
        self,
        *,
        session_key: str,
        request_id: str,
        effect_id: str,
        metadata: dict[str, Any],
    ) -> None:
        self._enforce_distributed_runtime_authority(operation="replay_acceptance_gate")
        if self._request_has_ambiguous_inflight_effect_state(
            session_id=session_key,
            request_id=request_id,
        ):
            self._emit_progress_snapshot(
                phase="during_reconciliation",
                session_id=session_key,
                trace_token=str(metadata.get("trace_id") or ""),
                job_id="",
                future_done=False,
                note="ambiguous request inflight detected",
                extra={
                    "request_id": request_id,
                    "effect_id": effect_id,
                    "classification": "replay_to_reconciliation_gate",
                },
            )
            self._emit_reconcile_required_event(
                session_id=session_key,
                trace_token=str(metadata.get("trace_id") or ""),
                request_id=request_id,
                effect_id=effect_id,
                reason="ambiguous_request_inflight",
            )
            raise ReplayMismatch(
                "Ambiguous effect state for request replay: prior JOB_STARTED exists "
                "without terminal event; refusing duplicate execution",
            )

        if effect_id and self._effect_journal.is_ambiguous(session_id=session_key, effect_id=effect_id):
            self._emit_progress_snapshot(
                phase="during_reconciliation",
                session_id=session_key,
                trace_token=str(metadata.get("trace_id") or ""),
                job_id="",
                future_done=False,
                note="ambiguous effect begin without commit detected",
                extra={
                    "request_id": request_id,
                    "effect_id": effect_id,
                    "classification": "replay_to_reconciliation_gate",
                },
            )
            self._emit_reconcile_required_event(
                session_id=session_key,
                trace_token=str(metadata.get("trace_id") or ""),
                request_id=request_id,
                effect_id=effect_id,
                reason="ambiguous_effect_begin_without_commit",
            )
            raise ReplayMismatch(
                "Ambiguous effect state for replay: prior EFFECT_BEGIN exists without EFFECT_COMMIT; "
                "refusing duplicate execution",
            )

    def _resolve_submit_trace_and_effect(
        self,
        *,
        session_key: str,
        user_input: str,
        attachments: AttachmentList | None,
        metadata: dict[str, Any],
        request_id: str,
        effect_id: str,
    ) -> tuple[str, str]:
        trace_id = str(metadata.get("trace_id") or "").strip()
        if not trace_id:
            trace_seed = {
                "session_id": session_key,
                "request_id": str(metadata.get("request_id") or ""),
                "user_input": str(user_input or ""),
                "attachments": list(attachments or []),
            }
            trace_blob = json_dumps_sorted(trace_seed)
            trace_id = f"tr-{hashlib.sha256(trace_blob.encode('utf-8')).hexdigest()[:20]}"
        metadata["trace_id"] = trace_id
        if not effect_id:
            effect_id = self._effect_journal.derive_effect_id(
                session_id=session_key,
                request_id=request_id,
                trace_id=trace_id,
            )

        if not request_id and not str(metadata.get("effect_id") or "").strip():
            candidate_job_id = ExecutionJob._stable_token(session_key, trace_id, prefix="job")
            projected_candidate = self.lifecycle_projection.get(candidate_job_id)
            if projected_candidate is not None and projected_candidate.status in {
                ExecutionStatus.COMPLETED,
                ExecutionStatus.FAILED,
            }:
                previous_trace_id = trace_id
                trace_id = ExecutionJob._stable_token(session_key, trace_id, time.time_ns(), prefix="tr")
                metadata["trace_id"] = trace_id
                effect_id = self._effect_journal.derive_effect_id(
                    session_id=session_key,
                    request_id=request_id,
                    trace_id=trace_id,
                )
                self._emit_progress_snapshot(
                    phase="before_replay",
                    session_id=session_key,
                    trace_token=trace_id,
                    job_id="",
                    future_done=False,
                    note="rotated terminal trace identity for non-idempotent retry",
                    extra={"previous_trace_id": previous_trace_id},
                )
        return trace_id, effect_id

    def _completion_expectations(
        self,
        *,
        job_id: str,
        future_done: bool,
        projection_terminal: bool,
    ) -> dict[str, bool]:
        return self._execution_coordinator.completion_expectations(
            job_id=job_id,
            future_done=future_done,
            projection_terminal=projection_terminal,
            scheduler=self._scheduler,
        )

    async def _drain_scheduler_until_resolved(
        self,
        *,
        future: asyncio.Future[FinalizedTurnResult],
        job: ExecutionJob,
        session_key: str,
        trace_token: str,
        deadline: float,
    ) -> int:
        return await self._execution_coordinator.drain_scheduler_until_resolved(
            future=future,
            job=job,
            session_key=session_key,
            trace_token=trace_token,
            deadline=deadline,
            scheduler=self.scheduler,
            kernel_executor=self.kernel_executor,
            lifecycle_projection=self.lifecycle_projection,
            mutate_runtime_plan=_mutate_runtime_plan,
            emit_runtime_stream_event=self._emit_runtime_stream_event,
            emit_progress_snapshot=self._emit_progress_snapshot,
        )

    def _build_response_engine_context(
        self,
        *,
        job: ExecutionJob,
        session_key: str,
        initial_response: str,
    ) -> Any:
        """Build execution context for ResponseEngine ranking.

        Phase 1: Create context from job metadata and session state for response ranking.
        """
        session = self.registry.get_or_create(session_key)
        session_state_ref = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(session_state_ref, dict):
            session_state_ref = {}

        metadata = dict(job.metadata or {})
        compute_ticket = resolve_dynamic_compute_ticket(
            metadata=metadata,
            user_input=str(job.user_input or ""),
        )
        metadata["dynamic_compute_ticket"] = compute_ticket.to_dict()
        if not isinstance(metadata.get("reward_feedback"), dict):
            pending_feedback = dict(session_state_ref.get("response_learning_pending") or {})
            if pending_feedback:
                synthesized_feedback = _synthesize_reward_feedback(
                    pending_selection=pending_feedback,
                    current_user_input=str(job.user_input or ""),
                    metadata=metadata,
                    session_state=session_state_ref,
                )
                if isinstance(synthesized_feedback, dict) and synthesized_feedback:
                    metadata["reward_feedback"] = synthesized_feedback
                    session_state_ref.pop("response_learning_pending", None)
                else:
                    attempts = int(pending_feedback.get("feedback_attempts") or 0) + 1
                    if attempts >= 3:
                        session_state_ref.pop("response_learning_pending", None)
                    else:
                        pending_feedback["feedback_attempts"] = attempts
                        session_state_ref["response_learning_pending"] = pending_feedback
        else:
            session_state_ref.pop("response_learning_pending", None)

        self._restore_response_engine_runtime_state(session_state_ref)

        session_state = dict(session_state_ref)
        
        # Simple dynamic context object with execution data.
        from types import SimpleNamespace

        context = SimpleNamespace()
        context.user_input = str(job.user_input or "")
        context.initial_response = initial_response
        context.session_state = session_state
        context.job_metadata = metadata
        context.trace_id = str(job.trace_id or "")
        context.compute_ticket = compute_ticket

        context.memory_context = list(metadata.get("memory_context") or [])

        context.persona_traits = list(session_state.get("persona_traits") or [])
        persona_constraints = dict(session_state.get("persona_constraints") or {})
        persona_state = dict(session_state.get("persona_state") or {})
        stability = dict(persona_state.get("identity_stability") or {})
        drift_guard = float(stability.get("drift_guard_level", 0.0) or 0.0)
        if drift_guard >= 0.60:
            required_tone = str(persona_state.get("tone_baseline") or "").strip().lower()
            if required_tone and not str(persona_constraints.get("required_tone") or "").strip().lower():
                persona_constraints["required_tone"] = required_tone
            max_risk = persona_constraints.get("max_risk")
            bounded_risk = 0.45
            if isinstance(max_risk, (int, float)):
                bounded_risk = min(float(max_risk), bounded_risk)
            persona_constraints["max_risk"] = bounded_risk
        context.persona_constraints = persona_constraints
        context.persona_state = dict(session_state.get("persona_state") or {})
        context.emotion_state = dict(session_state.get("emotion_state") or {})
        context.felt_persona_state = dict(session_state.get("felt_persona_state") or {})

        # Allow either explicit user_preferences or profile.preferences path.
        user_preferences = session_state.get("user_preferences")
        if not isinstance(user_preferences, dict):
            profile_blob = session_state.get("profile")
            if isinstance(profile_blob, dict):
                user_preferences = dict(profile_blob.get("preferences") or {})
        context.user_preferences = dict(user_preferences or {})

        context.conversation_trajectory = dict(
            session_state.get("conversation_trajectory")
            or metadata.get("conversation_trajectory")
            or {},
        )
        
        return context

    def _restore_response_engine_runtime_state(self, session_state: dict[str, Any]) -> None:
        if not isinstance(session_state, dict):
            return
        durable_memory = dict(session_state.get("memory_store") or {})
        recent_raw = list(
            session_state.get("response_engine_recent_responses")
            or durable_memory.get("_response_engine_recent_responses")
            or []
        )
        normalized_recent: list[str] = []
        for item in recent_raw:
            text = str(item or "").strip()
            if text:
                normalized_recent.append(text)
        if not normalized_recent:
            history_items = list(session_state.get("history") or [])
            assistant_replies: list[str] = []
            for entry in history_items:
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("role") or "").strip().lower() != "assistant":
                    continue
                reply_text = str(entry.get("content") or "").strip()
                if reply_text:
                    assistant_replies.append(reply_text)
            normalized_recent = assistant_replies[-10:]
        if hasattr(self._response_engine, "_recent_responses"):
            self._response_engine._recent_responses = normalized_recent[-10:]

        stored_weights = dict(
            session_state.get("response_engine_reward_weights")
            or durable_memory.get("_response_engine_reward_weights")
            or {}
        )
        base_weights = dict(getattr(self._response_engine, "_reward_weights", {}) or {})
        if not base_weights:
            return
        bound = float(getattr(self._response_engine, "_reward_weight_bound", 0.25) or 0.25)
        for key, value in stored_weights.items():
            if key not in base_weights or not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            base_weights[key] = float(max(-bound, min(bound, numeric)))
        if hasattr(self._response_engine, "_reward_weights"):
            self._response_engine._reward_weights = base_weights

    def _capture_response_engine_runtime_state(self) -> dict[str, Any]:
        recent = list(getattr(self._response_engine, "_recent_responses", []) or [])
        reward_weights = dict(getattr(self._response_engine, "_reward_weights", {}) or {})
        return {
            "response_engine_recent_responses": [str(item or "") for item in recent[-10:]],
            "response_engine_reward_weights": {
                str(key): float(value)
                for key, value in reward_weights.items()
                if isinstance(value, (int, float))
            },
        }

    @staticmethod
    def _estimate_emotion_signal(*, user_input: str, response_text: str) -> dict[str, float]:
        text = f"{str(user_input or '')} {str(response_text or '')}".lower()
        positive = ("glad", "great", "thank", "good", "helpful", "better")
        negative = ("worried", "anxious", "sad", "stressed", "overwhelmed", "angry")
        connective = ("we", "together", "with you", "i hear", "i'm with you")
        confidence = ("can", "will", "next", "plan", "step")

        valence = 0.5
        if any(token in text for token in positive):
            valence += 0.20
        if any(token in text for token in negative):
            valence -= 0.25

        arousal = 0.35
        arousal += min(text.count("!"), 3) * 0.08
        arousal += min(text.count("?"), 3) * 0.05

        attachment = 0.35
        if any(token in text for token in connective):
            attachment += 0.25

        confidence_score = 0.40
        if any(token in text for token in confidence):
            confidence_score += 0.25

        return {
            "valence": max(0.0, min(1.0, valence)),
            "arousal": max(0.0, min(1.0, arousal)),
            "attachment": max(0.0, min(1.0, attachment)),
            "confidence": max(0.0, min(1.0, confidence_score)),
        }

    def _update_identity_state(
        self,
        *,
        session_state: dict[str, Any],
        user_input: str,
        response_text: str,
        semantic_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        now = float(time.time())
        signal = self._estimate_emotion_signal(user_input=user_input, response_text=response_text)

        memory_bias = min(len(list(semantic_items or [])), 6) * 0.01
        emotion_memory = dict(session_state.get("emotion_state_memory") or {})
        prev_emotion = dict(emotion_memory.get("state") or session_state.get("emotion_state") or {})
        alpha = 0.15

        updated_emotion = {
            "valence": (1.0 - alpha) * float(prev_emotion.get("valence", 0.5) or 0.5)
            + alpha * float(signal["valence"]),
            "arousal": (1.0 - alpha) * float(prev_emotion.get("arousal", 0.35) or 0.35)
            + alpha * float(signal["arousal"]),
            "attachment": (1.0 - alpha) * float(prev_emotion.get("attachment", 0.35) or 0.35)
            + alpha * min(1.0, float(signal["attachment"]) + memory_bias),
            "confidence": (1.0 - alpha) * float(prev_emotion.get("confidence", 0.40) or 0.40)
            + alpha * min(1.0, float(signal["confidence"]) + (memory_bias * 0.5)),
        }
        for key in ("valence", "arousal", "attachment", "confidence"):
            updated_emotion[key] = max(0.0, min(1.0, float(updated_emotion[key])))

        emotion_turns = int(emotion_memory.get("turns", 0) or 0) + 1
        session_state["emotion_state"] = dict(updated_emotion)
        session_state["emotion_state_memory"] = {
            "state": dict(updated_emotion),
            "turns": emotion_turns,
            "updated_at": now,
            "source": "control_plane",
        }

        persona_state = dict(session_state.get("persona_state") or {})
        persona_turns = int(persona_state.get("turns", 0) or 0) + 1
        response_len = int(len(str(response_text or "")))
        verbosity_ema_prev = float(persona_state.get("verbosity_ema", 120.0) or 120.0)
        verbosity_ema = (0.9 * verbosity_ema_prev) + (0.1 * float(response_len))
        if verbosity_ema < 90:
            proposed_style_anchor = "concise"
        elif verbosity_ema > 220:
            proposed_style_anchor = "expanded"
        else:
            proposed_style_anchor = "balanced"

        prior_stability = dict(persona_state.get("identity_stability") or {})
        prior_guard = float(prior_stability.get("drift_guard_level", 0.0) or 0.0)
        drift_anchor = str(persona_state.get("drift_anchor") or proposed_style_anchor)
        style_anchor = proposed_style_anchor
        drift_attempts = int(persona_state.get("drift_attempts", 0) or 0)
        if drift_anchor and prior_guard >= 0.65 and proposed_style_anchor != drift_anchor:
            # In guarded mode, preserve identity anchor against transient style pressure.
            style_anchor = drift_anchor
            drift_attempts += 1

        baseline_prev = float(persona_state.get("emotional_baseline_tendency", 0.5) or 0.5)
        emotional_baseline = (0.92 * baseline_prev) + (0.08 * float(updated_emotion["valence"]))
        tone_baseline = "steady" if float(updated_emotion["arousal"]) < 0.55 else "engaged"

        mismatch = 1 if (drift_anchor and drift_anchor != style_anchor) else 0
        drift_violations = int(persona_state.get("drift_violations", 0) or 0) + mismatch
        stability_score = max(0.0, min(1.0, 1.0 - (0.08 * float(drift_violations))))
        drift_guard_level = 0.40 + (0.50 * stability_score)

        session_state["persona_state"] = {
            "tone_baseline": tone_baseline,
            "conversational_style_anchor": style_anchor,
            "emotional_baseline_tendency": max(0.0, min(1.0, emotional_baseline)),
            "drift_anchor": drift_anchor,
            "verbosity_ema": float(verbosity_ema),
            "drift_attempts": drift_attempts,
            "drift_violations": drift_violations,
            "identity_stability": {
                "score": float(stability_score),
                "drift_guard_level": float(max(0.0, min(1.0, drift_guard_level))),
            },
            "turns": persona_turns,
            "updated_at": now,
            "source": "control_plane",
        }

        return {
            "emotion_state": dict(updated_emotion),
            "persona_state": dict(session_state.get("persona_state") or {}),
        }

    @staticmethod
    def _extract_continuity_markers(*, user_input: str, response_text: str) -> list[str]:
        pool = f"{str(user_input or '')} {str(response_text or '')}".lower()
        markers = [
            "next step",
            "together",
            "steady",
            "plan",
            "outcome",
            "tradeoff",
            "safest",
            "with you",
        ]
        selected = [marker for marker in markers if marker in pool]
        return selected[:4]

    def _update_conversation_trajectory(
        self,
        *,
        session_state: dict[str, Any],
        user_input: str,
        response_text: str,
        identity_state: dict[str, Any],
        felt_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        trajectory = dict(session_state.get("conversation_trajectory") or {})
        persona_state = dict(identity_state.get("persona_state") or session_state.get("persona_state") or {})
        emotion_state = dict(identity_state.get("emotion_state") or session_state.get("emotion_state") or {})
        stability = dict(persona_state.get("identity_stability") or {})
        pressure = 0.35 + (0.55 * float(stability.get("score", 0.5) or 0.5))

        desired_goal = str(trajectory.get("desired_goal") or "").strip().lower()
        if not desired_goal:
            intent = _runtime_plan_intent(str(user_input or ""))
            if intent == "emotional":
                desired_goal = "engage"
            elif intent == "question":
                desired_goal = "clarify"
            else:
                desired_goal = "inform"

        updated = {
            "desired_goal": desired_goal,
            "preferred_tone": str(persona_state.get("tone_baseline") or "steady"),
            "continuity_pressure": float(max(0.0, min(1.0, pressure))),
            "emotional_target": {
                "valence": float(emotion_state.get("valence", 0.5) or 0.5),
                "arousal": float(emotion_state.get("arousal", 0.35) or 0.35),
            },
            "felt_state": dict(felt_state or {}),
            "narrative_phase": str(dict(felt_state or {}).get("narrative_phase") or "steady"),
            "preferred_response_goal": str(dict(felt_state or {}).get("preferred_response_goal") or desired_goal),
            "continuity_markers": self._extract_continuity_markers(
                user_input=str(user_input or ""),
                response_text=str(response_text or ""),
            ),
            "last_user_preview": str(user_input or "")[:180],
            "last_response_preview": str(response_text or "")[:220],
            "updated_at": float(time.time()),
        }
        session_state["conversation_trajectory"] = dict(updated)
        return updated

    def _update_felt_persona_stream(
        self,
        *,
        session_state: dict[str, Any],
        user_input: str,
        response_text: str,
        identity_state: dict[str, Any],
        semantic_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prior = dict(session_state.get("felt_persona_state") or {})
        emotion = dict(identity_state.get("emotion_state") or session_state.get("emotion_state") or {})
        persona = dict(identity_state.get("persona_state") or session_state.get("persona_state") or {})
        stability = dict(persona.get("identity_stability") or {})

        valence = float(emotion.get("valence", 0.5) or 0.5)
        arousal = float(emotion.get("arousal", 0.35) or 0.35)
        confidence = float(emotion.get("confidence", 0.4) or 0.4)
        previous_valence = float(prior.get("last_valence", valence) or valence)
        delta_valence = valence - previous_valence

        momentum_raw = (0.55 * delta_valence) + (0.30 * confidence) - (0.25 * arousal)
        emotional_momentum = (0.78 * float(prior.get("emotional_momentum", 0.0) or 0.0)) + (0.22 * momentum_raw)
        memory_resonance = min(len(list(semantic_items or [])), 8) / 8.0
        blended_resonance = (0.75 * float(prior.get("memory_resonance", 0.0) or 0.0)) + (0.25 * memory_resonance)

        if arousal >= 0.62 and valence < 0.50:
            narrative_phase = "stabilizing"
            preferred_goal = "engage"
            target_stance = "supportive"
        elif emotional_momentum >= 0.08:
            narrative_phase = "building"
            preferred_goal = "inform"
            target_stance = "forward"
        elif emotional_momentum <= -0.06:
            narrative_phase = "repairing"
            preferred_goal = "clarify"
            target_stance = "balanced"
        else:
            narrative_phase = "steady"
            preferred_goal = "clarify"
            target_stance = "balanced"

        coherence_pressure = 0.35 + (0.45 * float(stability.get("score", 0.5) or 0.5)) + (0.20 * blended_resonance)
        coherence_pressure = max(0.0, min(1.0, coherence_pressure))

        updated = {
            "narrative_phase": narrative_phase,
            "preferred_response_goal": preferred_goal,
            "target_stance": target_stance,
            "emotional_momentum": float(max(-1.0, min(1.0, emotional_momentum))),
            "memory_resonance": float(max(0.0, min(1.0, blended_resonance))),
            "coherence_pressure": float(coherence_pressure),
            "style_anchor": str(persona.get("conversational_style_anchor") or "balanced"),
            "last_valence": float(valence),
            "last_user_preview": str(user_input or "")[:140],
            "last_response_preview": str(response_text or "")[:180],
            "updated_at": float(time.time()),
        }
        session_state["felt_persona_state"] = dict(updated)
        return updated

    @staticmethod
    def _apply_continuity_shaping(
        *,
        response_text: str,
        session_state: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        raw = str(response_text or "")
        shaped = " ".join(raw.split())

        persona_state = dict(session_state.get("persona_state") or {})
        emotion_state = dict(session_state.get("emotion_state") or {})
        tone_baseline = str(persona_state.get("tone_baseline") or "steady")
        style_anchor = str(persona_state.get("conversational_style_anchor") or "balanced")
        arousal = float(emotion_state.get("arousal", 0.35) or 0.35)

        # Continuity shaping is style-only: punctuation and pacing, not content meaning.
        while "!!" in shaped:
            shaped = shaped.replace("!!", "!")
        while "??" in shaped:
            shaped = shaped.replace("??", "?")
        if tone_baseline == "steady" and arousal < 0.45 and shaped.endswith("!"):
            shaped = shaped[:-1] + "."
        if style_anchor == "concise" and len(shaped) > 320:
            shaped = shaped[:320].rstrip() + "..."

        continuity = {
            "tone_baseline": tone_baseline,
            "style_anchor": style_anchor,
            "emotional_alignment": {
                "valence": float(emotion_state.get("valence", 0.5) or 0.5),
                "arousal": float(arousal),
                "attachment": float(emotion_state.get("attachment", 0.35) or 0.35),
                "confidence": float(emotion_state.get("confidence", 0.40) or 0.40),
            },
            "continuity_applied": shaped != raw,
        }
        return shaped, continuity

    def _record_canonical_gate_breach(
        self,
        *,
        session_key: str,
        metadata: dict[str, Any] | None,
        gate: str,
        reason: str,
    ) -> None:
        payload = {
            "gate": str(gate or "unknown"),
            "reason": str(reason or "canonical_gate_violation"),
            "timestamp": float(time.time()),
        }
        if isinstance(metadata, dict):
            stream = list(metadata.get("canonical_gate_breaches") or [])
            stream.append(dict(payload))
            metadata["canonical_gate_breaches"] = stream[-32:]
            metadata["canonical_gate_breach_count"] = int(metadata.get("canonical_gate_breach_count") or 0) + 1

        validation_enabled = False
        if isinstance(metadata, dict) and "canonical_gate_validation" in metadata:
            validation_enabled = bool(metadata.get("canonical_gate_validation"))
        else:
            validation_enabled = str(
                os.environ.get("DADBOT_CANONICAL_GATE_VALIDATION", "0"),
            ).strip().lower() in {"1", "true", "yes", "on"}

        if not validation_enabled:
            return

        with contextlib.suppress(Exception):
            session = self.registry.get_or_create(str(session_key or "default"))
            state = session.setdefault("state", {})
            if isinstance(state, dict):
                metrics = dict(state.get("canonical_gate_metrics") or {})
                total = int(metrics.get("total_breaches", 0) or 0) + 1
                per_gate = dict(metrics.get("per_gate") or {})
                per_gate[payload["gate"]] = int(per_gate.get(payload["gate"], 0) or 0) + 1
                metrics["total_breaches"] = total
                metrics["per_gate"] = per_gate
                metrics["last_breach"] = dict(payload)
                state["canonical_gate_metrics"] = metrics

    def execute_from_graph_context(
        self,
        turn_context: Any,
        rich_context: dict[str, Any] | None = None,
    ) -> FinalizedTurnResult:
        from types import SimpleNamespace

        metadata = dict(getattr(turn_context, "metadata", {}) or {})
        state = dict(getattr(turn_context, "state", {}) or {})
        if isinstance(rich_context, dict) and rich_context:
            metadata.setdefault("rich_context", dict(rich_context))

        job = cast(
            ExecutionJob,
            SimpleNamespace(
            user_input=str(getattr(turn_context, "user_input", "") or ""),
            metadata=metadata,
            trace_id=str(getattr(turn_context, "trace_id", "") or ""),
            ),
        )
        session_key = str(
            metadata.get("session_id")
            or dict(metadata.get("control_plane") or {}).get("session_id")
            or getattr(turn_context, "trace_id", "")
            or "default",
        )
        initial_response = str(
            state.get("candidate")
            or state.get("initial_response")
            or getattr(turn_context, "user_input", "")
            or "",
        )

        engine_context = self._build_response_engine_context(
            job=job,
            session_key=session_key,
            initial_response=initial_response,
        )
        ranked_response = self._response_engine.run(engine_context)
        final_response = str(ranked_response or "")
        execution_mode = str(metadata.get("execution_mode") or "").strip().lower()
        strict_trace = bool(metadata.get("strict_trace_invariant", False))
        if strict_trace and execution_mode in {"recovery", "replay"}:
            authoritative = str(initial_response or "").strip()
            if authoritative:
                final_response = authoritative
        final_response = self._stabilize_strict_response_text(
            strict_trace=strict_trace,
            response_text=final_response,
            user_input=str(getattr(turn_context, "user_input", "") or ""),
        )
        if not final_response:
            self._record_canonical_gate_breach(
                session_key=session_key,
                metadata=metadata,
                gate="response_engine_empty_graph_context",
                reason="response_engine produced empty response",
            )
            raise RuntimeError(
                "Canonical response gate violation: response_engine produced empty response."
            )

        telemetry = getattr(engine_context, "response_engine_telemetry", None)
        if isinstance(telemetry, dict) and telemetry:
            metadata["response_engine_telemetry"] = dict(telemetry)

        runtime_state = self._capture_response_engine_runtime_state()
        state.update(runtime_state)
        metadata["response_engine_runtime_state"] = dict(runtime_state)
        memory_store = state.get("memory_store")
        if isinstance(memory_store, dict):
            memory_store["_response_engine_recent_responses"] = list(
                runtime_state.get("response_engine_recent_responses") or []
            )
            memory_store["_response_engine_reward_weights"] = dict(
                runtime_state.get("response_engine_reward_weights") or {}
            )

        metadata["response_authority"] = "response_engine"
        state["response_authority"] = "response_engine"
        if isinstance(telemetry, dict) and telemetry:
            state["response_engine_telemetry"] = dict(telemetry)

        try:
            turn_context.metadata.update(metadata)
            turn_context.state.update(state)
        except Exception:
            pass

        return str(final_response or ""), False

    @staticmethod
    def _stabilize_strict_response_text(
        *,
        strict_trace: bool,
        response_text: str,
        user_input: str,
    ) -> str:
        text = str(response_text or "").strip()
        if not strict_trace:
            return text
        if text.lower().startswith("i hear you:"):
            user = str(user_input or "").strip()
            if user:
                return user
        return text

    def _finalize_submit_success(
        self,
        *,
        job: ExecutionJob,
        result: FinalizedTurnResult,
        session_key: str,
        trace_token: str,
        before_state_hash: str,
        dedupe_future: asyncio.Future[FinalizedTurnResult] | None,
        loop_iterations: int,
    ) -> FinalizedTurnResult:
        finalization = dict(job.metadata.get("submit_finalization") or {})
        if bool(finalization.get("done", False)):
            return (
                str(finalization.get("response") or (result[0] if isinstance(result, tuple) and len(result) >= 1 else "")),
                bool(finalization.get("should_end", result[1] if isinstance(result, tuple) and len(result) >= 2 else False)),
            )

        current_execution_result = ensure_unified_execution_result(
            dict(job.metadata.get("execution_result") or {}),
        )
        # Extract initial response from execution
        initial_response = str(result[0] if isinstance(result, tuple) and len(result) >= 1 else "")
        
        # Phase 1: Run ResponseEngine to rank and select best response
        # ResponseEngine takes the execution context and generates/ranks candidates
        engine_context = self._build_response_engine_context(
            job=job,
            session_key=session_key,
            initial_response=initial_response,
        )
        continuity_state = dict(getattr(engine_context, "session_state", {}) or {})
        try:
            with contextlib.suppress(Exception):
                self._topology_record_node(
                    node_id="response_engine.generate_and_rank",
                    metadata={"trace_id": str(trace_token or "")},
                )
            # PHASE 1: ResponseEngine is SOLE selection authority
            # Only ResponseEngine may influence which response is selected.
            # All other systems are telemetry-only observers.
            ranked_response = self._response_engine.run(engine_context)
            final_response = str(ranked_response or "")
            execution_mode = str(job.metadata.get("execution_mode") or "").strip().lower()
            strict_trace = bool(job.metadata.get("strict_trace_invariant", False))
            if strict_trace and execution_mode in {"recovery", "replay"}:
                authoritative = str(initial_response or "").strip()
                if authoritative:
                    final_response = authoritative
            final_response = self._stabilize_strict_response_text(
                strict_trace=strict_trace,
                response_text=final_response,
                user_input=str(job.user_input or ""),
            )
            if not final_response:
                self._record_canonical_gate_breach(
                    session_key=session_key,
                    metadata=job.metadata,
                    gate="response_engine_empty_finalize",
                    reason="response_engine produced empty response",
                )
                raise RuntimeError(
                    "Canonical response gate violation: response_engine produced empty response."
                )
            telemetry = getattr(engine_context, "response_engine_telemetry", None)
            if isinstance(telemetry, dict) and telemetry:
                job.metadata["response_engine_telemetry"] = telemetry
        except Exception as exc:
            logger.exception("Canonical response gate violation during response_engine ranking")
            self._record_canonical_gate_breach(
                session_key=session_key,
                metadata=job.metadata,
                gate="response_engine_ranking_exception",
                reason=str(type(exc).__name__),
            )
            raise RuntimeError(
                "Canonical response gate violation: response_engine ranking failed."
            ) from exc

        shaped_response, continuity_telemetry = self._apply_continuity_shaping(
            response_text=final_response,
            session_state=continuity_state,
        )
        final_response = str(shaped_response or final_response)
        job.metadata["response_continuity"] = dict(continuity_telemetry)
        
        if current_execution_result.get("status") not in _TERMINAL_STATUS_VALUES:
            current_execution_result = mark_unified_execution_success(
                cast(dict[str, Any], current_execution_result),
                response=final_response,
                should_end=bool(result[1] if isinstance(result, tuple) and len(result) >= 2 else False),
            )
            job.metadata["execution_result"] = current_execution_result
            job.metadata["response_authority"] = "response_engine"
            telemetry_container = dict(job.metadata.get("response_engine_telemetry") or {})
            selected = dict(telemetry_container.get("selected") or {})
            if not selected:
                self._record_canonical_gate_breach(
                    session_key=session_key,
                    metadata=job.metadata,
                    gate="response_engine_missing_selected_metadata",
                    reason="response_engine selected metadata missing",
                )
                raise RuntimeError(
                    "Canonical response gate violation: missing response_engine selected metadata."
                )

            if str(job.metadata.get("response_authority") or "") != "response_engine":
                raise RuntimeError("Final response authority invariant violated: expected response_engine")

            if str(selected.get("source") or "").strip() == "":
                raise RuntimeError("Final response source invariant violated: missing response_engine source")

            shadow_events = list(
                dict(job.metadata.get("response_engine_telemetry") or {}).get("shadow_decision_bus")
                or dict(getattr(engine_context, "job_metadata", {}) or {}).get("shadow_decision_bus")
                or []
            )
            selected_source = str(selected.get("source") or "").strip()
            if not selected_source:
                self._record_canonical_gate_breach(
                    session_key=session_key,
                    metadata=job.metadata,
                    gate="response_engine_selected_source_missing",
                    reason="selected source missing",
                )
                raise RuntimeError(
                    "Canonical response gate violation: selected source missing."
                )
            selected_influence_share = _response_influence_share(selected)
            decision_confidence = max(
                0.0,
                float(selected.get("decision_confidence", telemetry_container.get("decision_confidence", 0.0)) or 0.0),
            )
            selected_score = float(selected.get("selected_score", selected.get("final_score", selected.get("score", 0.0))) or 0.0)
            second_best_score = float(
                selected.get(
                    "second_best_score",
                    telemetry_container.get("second_best_score", selected_score),
                )
                or selected_score,
            )
            selected["influence_share"] = selected_influence_share
            selected["decision_confidence"] = decision_confidence
            selected["selected_score"] = selected_score
            selected["second_best_score"] = second_best_score
            telemetry_container["selected"] = selected
            telemetry_container["decision_confidence"] = decision_confidence
            telemetry_container["selected_score"] = selected_score
            telemetry_container["second_best_score"] = second_best_score
            candidates = list(telemetry_container.get("candidates") or [])
            rejected = [
                {
                    "source": str(item.get("source") or "unknown"),
                    "score": float(item.get("final_score", 0.0) or 0.0),
                    "reason": "ranked_but_not_selected",
                }
                for item in candidates
                if str(item.get("source") or "") != selected_source
            ]
            vetoed = [
                {
                    "source": str(event.get("source") or "unknown"),
                    "type": str(event.get("type") or ""),
                    "content_preview": str(event.get("content_preview") or "")[:280],
                    "reason": str(event.get("reason") or "")[:320],
                    "would_replace": bool(event.get("would_replace", False)),
                    "priority": float(event.get("priority", 0.0) or 0.0),
                    "timestamp": float(event.get("timestamp", 0.0) or 0.0),
                }
                for event in shadow_events
                if str(event.get("type") or "").strip().lower() == "veto"
            ]
            selected_reasoning = dict(selected.get("reasoning") or {})
            if not selected_reasoning:
                components = dict(selected.get("components") or {})
                selected_reasoning = {
                    "safety": "Derived from safety_weight component",
                    "tools": "Derived from tool_weight component",
                    "memory": "Derived from memory_weight component",
                    "coherence": "Derived from coherence_weight component",
                }
                if not components:
                    selected_reasoning = {
                        "safety": "No structured safety rationale available",
                        "tools": "No structured tool rationale available",
                        "memory": "No structured memory rationale available",
                        "coherence": "No structured coherence rationale available",
                    }
            decision_report = {
                "selected": {
                    "source": selected_source,
                    "score": float(selected.get("final_score", selected.get("score", 0.0)) or 0.0),
                    "reason": str(selected.get("reason") or "response_engine_selection"),
                    "decision_confidence": float(decision_confidence),
                    "selected_score": float(selected_score),
                    "second_best_score": float(second_best_score),
                    "influence_share": dict(selected_influence_share),
                    "reasoning": selected_reasoning,
                },
                "rejected": rejected,
                "vetoed": vetoed,
                "reasoning": selected_reasoning,
                "decision_confidence": float(decision_confidence),
                "influence_share": dict(selected_influence_share),
                "final_output": str(final_response or ""),
                "shadow_event_count": len(shadow_events),
                "generated_at": float(time.time()),
            }
            job.metadata["response_engine_decision_report"] = decision_report
            telemetry_container["decision_report"] = dict(decision_report)
            job.metadata["response_engine_telemetry"] = telemetry_container

            # Update result tuple with ranked response
            result = (final_response, result[1] if isinstance(result, tuple) and len(result) >= 2 else False)
        session_after = self.registry.get_or_create(session_key)
        self._post_execution_pre_commit_contract_gate(
            session_id=session_key,
            job=job,
            result=result,
            input_state_hash=before_state_hash,
        )
        self._validate_trace_invariant(job, result, session=session_after, state_before_hash=before_state_hash)
        self._apply_turn_committed_core_state(
            session=session_after,
            job=job,
            result=result,
        )
        if isinstance(session_after, dict):
            state = session_after.setdefault("state", {})
            if isinstance(state, dict):
                selected = dict(dict(job.metadata.get("response_engine_telemetry") or {}).get("selected") or {})
                if selected:
                    state["response_learning_pending"] = {
                        "trace_id": str(trace_token or job.trace_id or ""),
                        "created_at": float(time.time()),
                        "selected": selected,
                        "selected_text_preview": str(
                            dict(job.metadata.get("response_engine_telemetry") or {}).get("selected_text_preview") or ""
                        ),
                        "feedback_attempts": 0,
                    }
                state["response_engine_decision_report"] = dict(
                    job.metadata.get("response_engine_decision_report") or {},
                )
                report = dict(job.metadata.get("response_engine_decision_report") or {})
                selected_for_drift = dict(report.get("selected") or selected or {})
                reasoning_for_drift = dict(report.get("reasoning") or selected_for_drift.get("reasoning") or {})
                drift_monitor = _update_response_engine_drift_monitor(
                    monitor=dict(state.get("response_engine_drift_monitor") or {}),
                    selected=selected_for_drift,
                    selected_reasoning=reasoning_for_drift,
                    shadow_event_count=int(report.get("shadow_event_count", 0) or 0),
                    trace_id=str(trace_token or job.trace_id or ""),
                )
                state["response_engine_drift_monitor"] = drift_monitor
                report["drift_monitor"] = {
                    "window_size": int(drift_monitor.get("window_size", 0) or 0),
                    "rolling_averages": dict(drift_monitor.get("rolling_averages") or {}),
                    "rates": dict(drift_monitor.get("rates") or {}),
                    "trend": dict(drift_monitor.get("trend") or {}),
                    "anomalies": list(drift_monitor.get("anomalies") or []),
                }
                state["response_engine_decision_report"] = report
                runtime_state = self._capture_response_engine_runtime_state()
                state.update(runtime_state)
                memory_store = state.get("memory_store")
                if isinstance(memory_store, dict):
                    memory_store["_response_engine_recent_responses"] = list(
                        runtime_state.get("response_engine_recent_responses") or []
                    )
                    memory_store["_response_engine_reward_weights"] = dict(
                        runtime_state.get("response_engine_reward_weights") or {}
                    )
                job.metadata["response_engine_decision_report"] = report
                job.metadata["response_engine_drift_monitor"] = {
                    "window_size": int(drift_monitor.get("window_size", 0) or 0),
                    "rolling_averages": dict(drift_monitor.get("rolling_averages") or {}),
                    "rates": dict(drift_monitor.get("rates") or {}),
                    "trend": dict(drift_monitor.get("trend") or {}),
                    "anomalies": list(drift_monitor.get("anomalies") or []),
                    "last_entry": dict(drift_monitor.get("last_entry") or {}),
                    "updated_at": float(drift_monitor.get("updated_at", 0.0) or 0.0),
                }
        self._maybe_compact_ledger()
        if dedupe_future is not None and not dedupe_future.done():
            dedupe_future.set_result(result)
        projected = self.lifecycle_projection.get(job.job_id)
        projection_terminal = bool(
            projected is not None and projected.status in {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED}
        )
        self._emit_progress_snapshot(
            phase="during_scheduler_drain",
            session_id=session_key,
            trace_token=trace_token,
            job_id=job.job_id,
            future_done=True,
            completion_expectations=self._completion_expectations(
                job_id=job.job_id,
                future_done=True,
                projection_terminal=projection_terminal,
            ),
            note="submit_turn resolved",
            extra={"loop_iterations": loop_iterations},
        )
        finalized = cast(
            FinalizedTurnResult,
            _apply_submit_success_postprocessing_impl(
                self,
                job=job,
                result=result,
                session_key=session_key,
                trace_token=trace_token,
                loop_iterations=loop_iterations,
            ),
        )
        self._assert_complete_run_contract(
            job=job,
            result=finalized,
            phase="submit_success",
        )
        return finalized

    def _record_submit_exception(
        self,
        *,
        job: ExecutionJob,
        future: asyncio.Future[FinalizedTurnResult],
        dedupe_future: asyncio.Future[FinalizedTurnResult] | None,
        session_key: str,
        trace_token: str,
        exc: BaseException,
    ) -> None:
        _record_submit_exception_impl(
            self,
            job=job,
            future=future,
            dedupe_future=dedupe_future,
            session_key=session_key,
            trace_token=trace_token,
            exc=exc,
            classify_execution_failure=_classify_execution_failure,
            set_terminal_turn_state=_set_terminal_turn_state,
            scheduler_exception_mapper=_SCHEDULER_EXCEPTION_MAPPER,
            mutate_runtime_plan=_mutate_runtime_plan,
        )

    def _prepare_submit_metadata(
        self,
        *,
        metadata: dict[str, Any] | None,
        user_input: str,
        attachments: AttachmentList | None,
        timeout_seconds: float | None,
    ) -> tuple[dict[str, Any], float, str, str]:
        md, resolved_timeout_seconds, request_id, effect_id = _prepare_submit_metadata_impl(
            self,
            metadata=metadata,
            user_input=user_input,
            attachments=attachments,
            timeout_seconds=timeout_seconds,
            normalize_tool_runtime_contract=_normalize_tool_runtime_contract,
            extract_execution_degradations=_extract_execution_degradations,
        )
        return md, float(resolved_timeout_seconds), str(request_id), str(effect_id)

    @staticmethod
    def _confluence_mode(metadata: dict[str, Any]) -> str:
        return "enforce"

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
        explicit_key = str(dict(metadata or {}).get("confluence_key") or "").strip()
        if not explicit_key:
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
        metadata["_global_confluence_mode"] = "enforce"
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

    @staticmethod
    def _result_output_payload(result: FinalizedTurnResult) -> dict[str, Any]:
        return _result_output_payload_impl(result)

    def _build_composition_payload(
        self,
        *,
        job: ExecutionJob,
        terminal_state: dict[str, Any],
        output_payload: dict[str, Any],
        trace_events: list[dict[str, Any]],
        state_before_hash: str,
        state_after_hash: str,
    ) -> dict[str, Any]:
        return _build_composition_payload_impl(
            self,
            job=job,
            terminal_state=terminal_state,
            output_payload=output_payload,
            trace_events=trace_events,
            state_before_hash=state_before_hash,
            state_after_hash=state_after_hash,
        )

    def _build_confluence_payload(
        self,
        *,
        job: ExecutionJob,
        terminal_state: dict[str, Any],
        output_payload: dict[str, Any],
        trace_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return _build_confluence_payload_impl(
            self,
            job=job,
            terminal_state=terminal_state,
            output_payload=output_payload,
            trace_events=trace_events,
        )

    @staticmethod
    def _expected_hashes_from_metadata(metadata: dict[str, Any]) -> tuple[str, str]:
        return _expected_hashes_from_metadata_impl(metadata)

    @staticmethod
    def _confluence_config_from_metadata(metadata: dict[str, Any]) -> tuple[str, str]:
        return _confluence_config_from_metadata_impl(metadata)

    def _validate_composition_expectations(
        self,
        *,
        expected: str,
        expected_confluence: str,
        composition_hash: str,
        confluence_class_hash: str,
    ) -> None:
        _validate_composition_expectations_impl(
            expected=expected,
            expected_confluence=expected_confluence,
            composition_hash=composition_hash,
            confluence_class_hash=confluence_class_hash,
        )

    def _enforce_global_confluence_law(
        self,
        *,
        confluence_key: str,
        confluence_mode: str,
        confluence_class_hash: str,
        expected_confluence: str,
    ) -> dict[str, Any]:
        return _enforce_global_confluence_law_impl(
            self,
            confluence_key=confluence_key,
            confluence_mode=confluence_mode,
            confluence_class_hash=confluence_class_hash,
            expected_confluence=expected_confluence,
            logger=logger,
        )

    def _record_turn_composition_contract(
        self,
        *,
        session: dict[str, Any],
        job: ExecutionJob,
        result: FinalizedTurnResult,
        state_before_hash: str,
    ) -> dict[str, Any]:
        return _record_turn_composition_contract_impl(
            self,
            session=session,
            job=job,
            result=result,
            state_before_hash=state_before_hash,
            logger=logger,
        )

    async def _register_submit_job(
        self,
        *,
        session_key: str,
        user_input: str,
        attachments: AttachmentList | None,
        metadata: dict[str, Any],
        trace_token: str,
    ) -> tuple[ExecutionJob, asyncio.Future[FinalizedTurnResult], float]:
        job, future, submitted_ts = await _register_submit_job_impl(
            self,
            execution_job_type=ExecutionJob,
            session_key=session_key,
            user_input=user_input,
            attachments=attachments,
            metadata=metadata,
            trace_token=trace_token,
        )
        return cast(ExecutionJob, job), cast(asyncio.Future[FinalizedTurnResult], future), float(submitted_ts)

    def _initialize_submit_scope(
        self,
        *,
        session_key: str,
        trace_token: str,
        job_id: str,
        resolved_timeout_seconds: float,
    ) -> tuple[str, float, object]:
        return _initialize_submit_scope_impl(
            self,
            session_key=session_key,
            trace_token=trace_token,
            job_id=job_id,
            resolved_timeout_seconds=resolved_timeout_seconds,
        )

    async def _submit_turn_kernel(
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
        # Refresh distributed authority lease before checking to ensure thin-spine path
        # doesn't fail with expired lease during initial turns.
        self._sync_distributed_authority(role=NodeRole.LEADER, state_hash=self.execution_token)
        self._enforce_distributed_runtime_authority(operation="submit_turn_entry_gate")

        md, resolved_timeout_seconds, request_id, effect_id = self._prepare_submit_metadata(
            metadata=metadata,
            user_input=user_input,
            attachments=attachments,
            timeout_seconds=timeout_seconds,
        )
        md["session_id"] = session_key
        self._pre_execution_contract_gate(
            session_id=session_key,
            user_input=str(user_input or ""),
            metadata=md,
        )
        contract_ok, contract_reason = _validate_tool_runtime_contract(md)
        if not contract_ok:
            raise RuntimeError(f"tool runtime contract validation failed: {contract_reason}")
        _prepare_submit_runtime_planning_impl(
            self,
            session_key=session_key,
            user_input=user_input,
            metadata=md,
        )
        phase_trace: list[tuple[str, float]] = []
        _append_submit_turn_phase(phase_trace, "preflight")
        _assert_submit_turn_phase_boundary(
            phase_trace=phase_trace,
            expected_phase="preflight",
            operation="_preflight_submit_turn",
        )
        effect_id, dedupe_key, dedupe_future, owns_dedupe_slot, immediate_result = await self._preflight_submit_turn(
            session_key=session_key,
            request_id=request_id,
            effect_id=effect_id,
            metadata=md,
        )
        if immediate_result is not None:
            return immediate_result

        _append_submit_turn_phase(phase_trace, "register")
        _assert_submit_turn_phase_boundary(
            phase_trace=phase_trace,
            expected_phase="register",
            operation="_resolve_submit_trace_and_effect",
        )
        trace_id, effect_id = _prepare_submit_register_phase_impl(
            self,
            session_key=session_key,
            user_input=user_input,
            attachments=attachments,
            metadata=md,
            request_id=request_id,
            effect_id=effect_id,
        )
        assert trace_id, "Missing trace_id at control plane entry"
        _assert_submit_turn_phase_boundary(
            phase_trace=phase_trace,
            expected_phase="register",
            operation="_register_submit_job",
        )
        job, future, _submitted_ts = await self._register_submit_job(
            session_key=session_key,
            user_input=user_input,
            attachments=attachments,
            metadata=md,
            trace_token=trace_id,
        )
        self._emit_progress_snapshot(
            phase="during_scheduler_drain",
            session_id=session_key,
            trace_token=trace_id,
            job_id=job.job_id,
            future_done=False,
            note="job registered",
            extra={"request_id": request_id, "effect_id": effect_id},
        )
        _append_submit_turn_phase(phase_trace, "execution")
        _assert_submit_turn_phase_boundary(
            phase_trace=phase_trace,
            expected_phase="execution",
            operation="_initialize_submit_scope",
        )
        before_state_hash, deadline, _cs_token = self._initialize_submit_scope(
            session_key=session_key,
            trace_token=trace_id,
            job_id=job.job_id,
            resolved_timeout_seconds=resolved_timeout_seconds,
        )
        try:
            return await _run_submit_execution_phase_impl(
                self,
                future=future,
                phase_trace=phase_trace,
                job=job,
                session_key=session_key,
                trace_id=trace_id,
                deadline=deadline,
                before_state_hash=before_state_hash,
                dedupe_future=dedupe_future,
            )
        finally:
            if (request_id or effect_id) and owns_dedupe_slot:
                async with self._inflight_lock:
                    if self._inflight_by_request.get(dedupe_key) is dedupe_future:
                        self._inflight_by_request.pop(dedupe_key, None)
            close_core_state_scope(_cs_token)

    def _coerce_projection_lifecycle_state(self, execution_id: str) -> ExecutionLifecycleState | None:
        projected = self.lifecycle_projection.get(str(execution_id or ""))
        if projected is None:
            return None
        return _coerce_lifecycle_state(_lifecycle_state_from_projection(projected))

    @staticmethod
    def _trace_event_invariant_counts(trace_events: list[dict[str, Any]]) -> tuple[int, bool]:
        return _trace_event_invariant_counts_impl(trace_events)

    @staticmethod
    def _fallback_commit_count_from_session(
        *,
        session: dict[str, Any] | None,
        trace_token: str,
    ) -> int:
        return _fallback_commit_count_from_session_impl(session=session, trace_token=trace_token)

    def _record_trace_composition_contract(
        self,
        *,
        session: dict[str, Any] | None,
        job: ExecutionJob,
        result: FinalizedTurnResult,
        state_before_hash: str,
    ) -> None:
        if session is None:
            return
        self._record_turn_composition_contract(
            session=session,
            job=job,
            result=result,
            state_before_hash=str(state_before_hash or ""),
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
        # Structural anchor: _validate_trace_invariant is executed in _submit_turn_kernel.
        assert callable(getattr(self, "_validate_trace_invariant", None))
        return await self.kernel_gateway.submit_turn(
            session_id=session_id,
            user_input=user_input,
            attachments=attachments,
            metadata=metadata,
            timeout_seconds=timeout_seconds,
        )

    def ledger_events(self) -> list[dict[str, Any]]:
        return self.ledger.read()

    def execution_timeline(self, *, session_id: str, limit: int = 128) -> list[dict[str, Any]]:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            return []
        timeline = [dict(item) for item in list(state.get("execution_timeline") or []) if isinstance(item, dict)]
        return timeline[-max(0, int(limit)):]

    def explain_last_decision(self, *, session_id: str) -> dict[str, Any]:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            return dict(
                validate_decision_explanation_contract(
                    {
                        "available": False,
                        "reason": "no_state",
                        "timeline_events": 0,
                        "last_plan": {},
                        "recent_events": [],
                        "semantic_memory_context_size": 0,
                        "active_hypothesis": {},
                        "belief_calibration": {},
                        "tool_self_model": {},
                        "interactive_cognition_ui": {},
                        "alignment_policy": {},
                        "tool_ecosystem": {},
                        "swarm_health": {},
                    },
                ),
            )
        timeline = self.execution_timeline(session_id=session_id, limit=64)
        last_plan = {}
        for event in reversed(timeline):
            if str(event.get("event_type") or "") in {"plan.updated", "plan.created"}:
                last_plan = dict((event.get("payload") or {}).get("runtime_plan") or {})
                break
        explanation = {
            "available": bool(timeline),
            "timeline_events": int(len(timeline)),
            "last_plan": dict(last_plan),
            "recent_events": [str(item.get("event_type") or "") for item in timeline[-8:]],
            "semantic_memory_context_size": int(
                len(list(dict(state.get("semantic_memory") or {}).get("items") or [])),
            ),
            "active_hypothesis": dict(dict(state.get("hypothesis_store") or {}).get("last_active") or {}),
            "belief_calibration": dict(dict(state.get("belief_state") or {}).get("calibration") or {}),
            "tool_self_model": dict(dict(state.get("tool_self_model") or {}).get("tools") or {}),
            "interactive_cognition_ui": self._interactive_cognition_ui.snapshot(state=state, limit=12),
            "alignment_policy": dict(dict(state.get("alignment_trainer") or {}).get("policy") or {}),
            "tool_ecosystem": self._tool_ecosystem_hub.summary(state=state),
            "swarm_health": self._multi_agent_swarm.health_snapshot(state=state),
        }
        return dict(validate_decision_explanation_contract(explanation))

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
        _validate_trace_invariant_impl(
            self,
            job=job,
            result=result,
            session=session,
            state_before_hash=state_before_hash,
            logger=logger,
        )

    def _assert_complete_run_contract(
        self,
        *,
        job: ExecutionJob,
        result: FinalizedTurnResult,
        phase: str,
    ) -> None:
        """Enforce complete-run contract before returning a finalized turn result."""
        trace_id = str(job.trace_id or "").strip()
        job_id = str(job.job_id or "").strip()
        if not trace_id or not job_id:
            raise InvariantViolation(
                "Complete-run contract violated: missing stable execution identity",
                context={"phase": str(phase or ""), "trace_id": trace_id, "job_id": job_id},
            )

        projection = self.lifecycle_projection.get(job_id)
        if projection is None or projection.status not in {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED}:
            raise InvariantViolation(
                "Complete-run contract violated: execution lifecycle is not terminal",
                context={
                    "phase": str(phase or ""),
                    "trace_id": trace_id,
                    "job_id": job_id,
                    "projection_status": "" if projection is None else str(projection.status.value),
                },
            )

        execution_result = ensure_unified_execution_result(dict(job.metadata.get("execution_result") or {}))
        status = str(execution_result.get("status") or "").strip().lower()
        if status not in _TERMINAL_STATUS_VALUES:
            raise InvariantViolation(
                "Complete-run contract violated: execution_result status is not terminal",
                context={
                    "phase": str(phase or ""),
                    "trace_id": trace_id,
                    "job_id": job_id,
                    "execution_result_status": status,
                },
            )

        execution_state = dict(job.metadata.get("execution_state") or {})
        terminal_turn_state = str(
            execution_state.get("terminal_turn_state") or job.metadata.get("terminal_turn_state") or "",
        ).strip().upper()
        allowed_terminal_turn_states = {state.value for state in TurnTerminalState}
        if terminal_turn_state not in allowed_terminal_turn_states:
            raise InvariantViolation(
                "Complete-run contract violated: missing or invalid terminal_turn_state",
                context={
                    "phase": str(phase or ""),
                    "trace_id": trace_id,
                    "job_id": job_id,
                    "terminal_turn_state": terminal_turn_state,
                    "allowed_terminal_turn_states": sorted(allowed_terminal_turn_states),
                },
            )

        response_text = str(result[0] if isinstance(result, tuple) and len(result) >= 1 else "")
        if not response_text:
            raise InvariantViolation(
                "Complete-run contract violated: empty terminal response payload",
                context={"phase": str(phase or ""), "trace_id": trace_id, "job_id": job_id},
            )

    def _apply_turn_committed_core_state(
        self,
        *,
        session: dict[str, Any],
        job: ExecutionJob,
        result: FinalizedTurnResult,
    ) -> None:
        self._persistence_coordinator.apply_turn_committed_core_state(session=session, job=job, result=result)

    def _promote_semantic_memory(
        self,
        *,
        session: dict[str, Any],
        job: ExecutionJob,
        response_text: str,
    ) -> None:
        self._persistence_coordinator.promote_semantic_memory(
            session=session,
            job=job,
            response_text=response_text,
        )

    def bootstrap(self) -> dict[str, Any]:
        events = self.ledger.read()
        self.lifecycle_projection.rebuild_from_ledger(events)
        self._ledger_index.refresh(force=True)
        return {
            "event_count": len(events),
            "execution_lifecycle": self.lifecycle_projection.snapshot(),
        }


def json_dumps_sorted(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)
