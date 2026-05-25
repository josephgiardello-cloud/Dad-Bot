from __future__ import annotations

import asyncio
import time
from enum import StrEnum
from typing import Any

from dadbot.core.contracts.lifecycle_events import (
    Claimed,
    Completed,
    Failed,
    LeaseExpired,
    LeaseRenewed,
    Redelivered,
    Released,
    Submitted,
)
from dadbot.core.control_plane_reducer import ExecutionState as ReducedExecutionState
from dadbot.core.control_plane_reducer import ExecutionStatus
from dadbot.core.execution_mode import ExecutionModeResolver
from dadbot.core.global_transition_invariants import (
    TransitionBoundaryView,
    enforce_global_transition_invariants,
)


class ExecutionLifecycleState(StrEnum):
    SUBMITTED = "submitted"
    QUEUED = "queued"
    RECOVERY_PENDING = "recovery_pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TurnTerminalState(StrEnum):
    SUCCESS = "SUCCESS"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    FAILED = "FAILED"
    RECOVERED = "RECOVERED"


class SchedulerExceptionMapper:
    """Canonical scheduler exception -> terminal turn state mapper."""

    @staticmethod
    def from_exception(exc: BaseException) -> TurnTerminalState:
        if isinstance(exc, asyncio.CancelledError):
            return TurnTerminalState.CANCELLED
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
            return TurnTerminalState.TIMEOUT
        return TurnTerminalState.FAILED

    @staticmethod
    def from_success(*, recovered: bool) -> TurnTerminalState:
        if recovered:
            return TurnTerminalState.RECOVERED
        return TurnTerminalState.SUCCESS


_ALLOWED_LIFECYCLE_TRANSITIONS: dict[ExecutionLifecycleState, frozenset[ExecutionLifecycleState]] = {
    ExecutionLifecycleState.SUBMITTED: frozenset({
        ExecutionLifecycleState.QUEUED,
        ExecutionLifecycleState.RUNNING,
        ExecutionLifecycleState.FAILED,
    }),
    ExecutionLifecycleState.QUEUED: frozenset({
        ExecutionLifecycleState.RECOVERY_PENDING,
        ExecutionLifecycleState.RUNNING,
        ExecutionLifecycleState.FAILED,
    }),
    ExecutionLifecycleState.RECOVERY_PENDING: frozenset({
        ExecutionLifecycleState.QUEUED,
        ExecutionLifecycleState.RUNNING,
        ExecutionLifecycleState.FAILED,
    }),
    ExecutionLifecycleState.RUNNING: frozenset({
        ExecutionLifecycleState.RECOVERY_PENDING,
        ExecutionLifecycleState.COMPLETED,
        ExecutionLifecycleState.FAILED,
    }),
    ExecutionLifecycleState.COMPLETED: frozenset(),
    ExecutionLifecycleState.FAILED: frozenset(),
}


def _coerce_lifecycle_state(value: Any) -> ExecutionLifecycleState:
    raw = str(value or "").strip().lower()
    try:
        return ExecutionLifecycleState(raw)
    except ValueError:
        return ExecutionLifecycleState.SUBMITTED


def _target_lifecycle_state_for_event(
    event: Any,
    *,
    current_state: ExecutionLifecycleState,
) -> ExecutionLifecycleState:
    if isinstance(event, Submitted):
        return ExecutionLifecycleState.SUBMITTED
    if isinstance(event, Claimed):
        return ExecutionLifecycleState.RUNNING
    if isinstance(event, LeaseRenewed):
        return ExecutionLifecycleState.RUNNING
    if isinstance(event, LeaseExpired):
        return ExecutionLifecycleState.RECOVERY_PENDING
    if isinstance(event, Released):
        return ExecutionLifecycleState.QUEUED
    if isinstance(event, Completed):
        return ExecutionLifecycleState.COMPLETED
    if isinstance(event, Failed):
        return ExecutionLifecycleState.FAILED
    if isinstance(event, Redelivered):
        # Redelivery can be a no-op state annotation in lifecycle reducer terms.
        if current_state == ExecutionLifecycleState.RECOVERY_PENDING:
            return ExecutionLifecycleState.QUEUED
        return current_state
    raise RuntimeError(f"unsupported lifecycle event type at emission boundary: {type(event).__name__}")


def _execution_status_for_lifecycle(state: ExecutionLifecycleState) -> str:
    if state == ExecutionLifecycleState.COMPLETED:
        return "completed"
    if state == ExecutionLifecycleState.FAILED:
        return "failed"
    if state == ExecutionLifecycleState.SUBMITTED:
        return "submitted"
    return "running"


def _resolved_execution_mode(job: Any) -> str:
    """Resolve execution mode for telemetry/logging (no checkpoint available)."""
    return ExecutionModeResolver.resolve_for_logging(job)


def _build_sovereign_transition_states(
    *,
    job: Any,
    before_state: ExecutionLifecycleState,
    after_state: ExecutionLifecycleState,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = dict(getattr(job, "metadata", {}) or {})
    execution_state = dict(metadata.get("execution_state") or {})
    before_causal_step_count = int(execution_state.get("causal_step_count") or 0)
    after_causal_step_count = before_causal_step_count + (1 if after_state != before_state else 0)
    invariance_hash = str(
        execution_state.get("invariance_hash")
        or metadata.get("invariance_hash")
        or f"cp:{str(getattr(job, 'job_id', '') or '')}:{after_causal_step_count}",
    )
    after_turn_truth_ok = bool(execution_state.get("turn_truth_ok", after_state != ExecutionLifecycleState.COMPLETED or True))

    before = {
        "session_id": str(getattr(job, "session_id", "default") or "default"),
        "trace_id": str(getattr(job, "trace_id", "unknown-trace") or "unknown-trace"),
        "execution_mode": _resolved_execution_mode(job),
        "execution_state": before_state.value,
        "execution_status": _execution_status_for_lifecycle(before_state),
        "turn_truth_ok": bool(execution_state.get("turn_truth_ok")) if before_state == ExecutionLifecycleState.COMPLETED else None,
        "invariance_hash": str(execution_state.get("invariance_hash") or metadata.get("invariance_hash") or ""),
        "causal_step_count": before_causal_step_count,
        "metadata": {},
    }
    after = {
        "session_id": str(getattr(job, "session_id", "default") or "default"),
        "trace_id": str(getattr(job, "trace_id", "unknown-trace") or "unknown-trace"),
        "execution_mode": _resolved_execution_mode(job),
        "execution_state": after_state.value,
        "execution_status": _execution_status_for_lifecycle(after_state),
        "turn_truth_ok": after_turn_truth_ok if after_state == ExecutionLifecycleState.COMPLETED else None,
        "invariance_hash": invariance_hash,
        "causal_step_count": after_causal_step_count,
        "metadata": {},
    }
    return before, after


def _assert_lifecycle_emission_transition(
    *,
    execution_id: str,
    event: Any,
    current_state: ExecutionLifecycleState | None,
) -> None:
    if isinstance(event, Submitted):
        if current_state is not None:
            raise RuntimeError(
                f"Invalid lifecycle emission transition for {execution_id!r}: "
                "Submitted must be first event",
            )
        return

    if current_state is None:
        raise RuntimeError(
            f"Invalid lifecycle emission transition for {execution_id!r}: "
            f"{type(event).__name__} cannot be emitted before Submitted",
        )

    target_state = _target_lifecycle_state_for_event(event, current_state=current_state)
    if target_state == current_state:
        return
    if target_state not in _ALLOWED_LIFECYCLE_TRANSITIONS[current_state]:
        raise RuntimeError(
            f"Invalid lifecycle emission transition for {execution_id!r}: "
            f"{current_state.value!r} -> {target_state.value!r} via {type(event).__name__}",
        )


def _ensure_execution_state(job: Any) -> dict[str, Any]:
    metadata = dict(getattr(job, "metadata", {}) or {})
    state = dict(metadata.get("execution_state") or {})
    state.setdefault("lifecycle_state", ExecutionLifecycleState.SUBMITTED.value)
    state.setdefault("redelivery_count", 0)
    state.setdefault("lease_conflict_count", 0)
    state.setdefault("last_worker_id", "")
    state.setdefault("last_transition_reason", "")
    state.setdefault("retry_not_before_monotonic", 0.0)
    metadata["execution_state"] = state
    job.metadata = metadata
    return state


def _transition_execution_state(
    job: Any,
    *,
    target: ExecutionLifecycleState,
    reason: str,
    worker_id: str = "",
    retry_not_before_monotonic: float | None = None,
    redelivery_increment: int = 0,
    lease_conflict_increment: int = 0,
) -> dict[str, Any]:
    state = _ensure_execution_state(job)
    current = _coerce_lifecycle_state(state.get("lifecycle_state"))
    before_causal_step_count = int(state.get("causal_step_count") or 0)
    if current != target and target not in _ALLOWED_LIFECYCLE_TRANSITIONS[current]:
        raise RuntimeError(
            "Invalid execution lifecycle transition: "
            f"{current.value!r} -> {target.value!r} for job {getattr(job, 'job_id', '')!r}",
        )
    state["lifecycle_state"] = target.value
    state["last_transition_reason"] = str(reason or "")
    state["last_transition_at"] = float(time.time())
    if worker_id:
        state["last_worker_id"] = str(worker_id)
    if retry_not_before_monotonic is None:
        if target != ExecutionLifecycleState.RECOVERY_PENDING:
            state["retry_not_before_monotonic"] = 0.0
    else:
        state["retry_not_before_monotonic"] = max(0.0, float(retry_not_before_monotonic))
    if redelivery_increment:
        state["redelivery_count"] = int(state.get("redelivery_count") or 0) + int(redelivery_increment)
    if lease_conflict_increment:
        state["lease_conflict_count"] = int(state.get("lease_conflict_count") or 0) + int(lease_conflict_increment)

    state["causal_step_count"] = before_causal_step_count + (1 if target != current else 0)
    enforce_global_transition_invariants(
        TransitionBoundaryView(
            session_id=str(getattr(job, "session_id", "default") or "default"),
            trace_id=str(getattr(job, "trace_id", "unknown-trace") or "unknown-trace"),
            before_state=current.value,
            after_state=target.value,
            before_causal_step_count=before_causal_step_count,
            after_causal_step_count=int(state.get("causal_step_count") or 0),
            turn_truth_ok=None,
            policy_posture=str(state.get("policy_posture") or "moderate"),
            active_fault_count=int(state.get("active_fault_count") or 0),
            metadata={"reason": str(reason or "")},
        ),
    )

    job.metadata["execution_state"] = state
    return dict(state)


def _lifecycle_state_from_projection(state: ReducedExecutionState | None) -> str:
    if state is None:
        return ExecutionLifecycleState.SUBMITTED.value
    if state.status == ExecutionStatus.SUBMITTED:
        return ExecutionLifecycleState.SUBMITTED.value
    if state.status in {ExecutionStatus.CLAIMED, ExecutionStatus.RUNNING}:
        return ExecutionLifecycleState.RUNNING.value
    if state.status == ExecutionStatus.EXPIRED:
        return ExecutionLifecycleState.RECOVERY_PENDING.value
    if state.status == ExecutionStatus.RELEASED:
        return ExecutionLifecycleState.QUEUED.value
    if state.status == ExecutionStatus.COMPLETED:
        return ExecutionLifecycleState.COMPLETED.value
    if state.status == ExecutionStatus.FAILED:
        return ExecutionLifecycleState.FAILED.value
    return ExecutionLifecycleState.SUBMITTED.value


def _apply_projection_execution_state(
    job: Any,
    state: ReducedExecutionState | None,
) -> dict[str, Any]:
    metadata = dict(getattr(job, "metadata", {}) or {})
    prior = dict(metadata.get("execution_state") or {})
    projected = {
        "_derived_from_ledger": True,
        "_derived_from_ledger_reason": "lifecycle_projection",
        "lifecycle_state": _lifecycle_state_from_projection(state),
        "redelivery_count": max(0, int((state.attempt_count if state is not None else 0)) - 1),
        "lease_conflict_count": int(prior.get("lease_conflict_count") or 0),
        "last_worker_id": str((state.owner if state is not None else "") or prior.get("last_worker_id") or ""),
        "last_transition_reason": str(prior.get("last_transition_reason") or "lifecycle_projection"),
        "retry_not_before_monotonic": 0.0,
        "failure_type": str(prior.get("failure_type") or ""),
        "failure_action": str(prior.get("failure_action") or ""),
        "auto_retry": bool(prior.get("auto_retry", False)),
    }
    metadata["execution_state"] = projected
    job.metadata = metadata
    return projected


__all__ = [
    "ExecutionLifecycleState",
    "TurnTerminalState",
    "SchedulerExceptionMapper",
    "_coerce_lifecycle_state",
    "_target_lifecycle_state_for_event",
    "_execution_status_for_lifecycle",
    "_build_sovereign_transition_states",
    "_assert_lifecycle_emission_transition",
    "_ensure_execution_state",
    "_transition_execution_state",
    "_resolved_execution_mode",
    "_lifecycle_state_from_projection",
    "_apply_projection_execution_state",
]