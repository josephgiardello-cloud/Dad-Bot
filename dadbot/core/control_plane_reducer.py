from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Iterable

from dadbot.core.contracts.lifecycle_events import (
    Claimed,
    Completed,
    Failed,
    LeaseExpired,
    LeaseRenewed,
    LifecycleEvent,
    Redelivered,
    Released,
    Submitted,
)


class LifecycleInvariantError(RuntimeError):
    """Raised when a lifecycle event stream violates the canonical contract."""


class ExecutionStatus(StrEnum):
    SUBMITTED = "submitted"
    CLAIMED = "claimed"
    RUNNING = "running"
    EXPIRED = "expired"
    RELEASED = "released"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ExecutionState:
    status: ExecutionStatus
    owner: str | None
    lease_expiry: datetime | None
    attempt_count: int


def lease_expired(state: ExecutionState, *, now: datetime) -> bool:
    return state.lease_expiry is not None and state.lease_expiry <= now


def _reduce_submitted(state: ExecutionState | None, event: Submitted) -> ExecutionState:
    if state is not None:
        raise LifecycleInvariantError("Submitted must be the first lifecycle event")
    return ExecutionState(
        status=ExecutionStatus.SUBMITTED,
        owner=None,
        lease_expiry=None,
        attempt_count=0,
    )


def _require_started(state: ExecutionState | None) -> ExecutionState:
    if state is None:
        raise LifecycleInvariantError("Lifecycle stream must begin with Submitted")
    return state


def _reduce_claimed(state: ExecutionState, event: Claimed) -> ExecutionState:
    if state.status in {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED}:
        raise LifecycleInvariantError("cannot claim a terminal execution")
    if state.owner and state.lease_expiry and state.lease_expiry > event.occurred_at:
        raise LifecycleInvariantError("cannot claim while an active lease exists")
    return ExecutionState(
        status=ExecutionStatus.CLAIMED,
        owner=event.worker_id,
        lease_expiry=event.lease_expiry,
        attempt_count=int(state.attempt_count) + 1,
    )


def _reduce_lease_renewed(state: ExecutionState, event: LeaseRenewed) -> ExecutionState:
    if state.owner != event.worker_id:
        raise LifecycleInvariantError("cannot renew a lease you do not own")
    if state.status not in {ExecutionStatus.CLAIMED, ExecutionStatus.RUNNING}:
        raise LifecycleInvariantError("cannot renew lease when execution is not claimed")
    return ExecutionState(
        status=ExecutionStatus.RUNNING,
        owner=event.worker_id,
        lease_expiry=event.lease_expiry,
        attempt_count=int(state.attempt_count),
    )


def _reduce_lease_expired(state: ExecutionState, event: LeaseExpired) -> ExecutionState:
    if state.owner != event.worker_id:
        raise LifecycleInvariantError("cannot expire a lease for a different worker")
    return ExecutionState(
        status=ExecutionStatus.EXPIRED,
        owner=None,
        lease_expiry=state.lease_expiry,
        attempt_count=int(state.attempt_count),
    )


def _reduce_released(state: ExecutionState, event: Released) -> ExecutionState:
    if state.owner != event.worker_id:
        raise LifecycleInvariantError("cannot release a lease you do not own")
    return ExecutionState(
        status=ExecutionStatus.RELEASED,
        owner=None,
        lease_expiry=None,
        attempt_count=int(state.attempt_count),
    )


def _reduce_completed(state: ExecutionState, event: Completed) -> ExecutionState:
    del event
    if state.owner is None:
        raise LifecycleInvariantError("cannot complete an execution with no active owner")
    return ExecutionState(
        status=ExecutionStatus.COMPLETED,
        owner=None,
        lease_expiry=None,
        attempt_count=int(state.attempt_count),
    )


def _reduce_failed(state: ExecutionState, event: Failed) -> ExecutionState:
    del event
    if state.owner is None:
        raise LifecycleInvariantError("cannot fail an execution with no active owner")
    return ExecutionState(
        status=ExecutionStatus.FAILED,
        owner=None,
        lease_expiry=None,
        attempt_count=int(state.attempt_count),
    )


def _reduce_redelivered(state: ExecutionState, event: Redelivered) -> ExecutionState:
    if event.previous_worker_id == event.new_worker_id:
        raise LifecycleInvariantError("redelivery requires different worker ids")
    return ExecutionState(
        status=state.status,
        owner=state.owner,
        lease_expiry=state.lease_expiry,
        attempt_count=max(int(state.attempt_count), 1),
    )


def _reduce_one(state: ExecutionState | None, event: LifecycleEvent) -> ExecutionState:
    if isinstance(event, Submitted):
        return _reduce_submitted(state, event)

    started_state = _require_started(state)

    if isinstance(event, Claimed):
        return _reduce_claimed(started_state, event)

    if isinstance(event, LeaseRenewed):
        return _reduce_lease_renewed(started_state, event)

    if isinstance(event, LeaseExpired):
        return _reduce_lease_expired(started_state, event)

    if isinstance(event, Released):
        return _reduce_released(started_state, event)

    if isinstance(event, Completed):
        return _reduce_completed(started_state, event)

    if isinstance(event, Failed):
        return _reduce_failed(started_state, event)

    if isinstance(event, Redelivered):
        return _reduce_redelivered(started_state, event)

    raise LifecycleInvariantError(f"unsupported lifecycle event: {type(event).__name__}")


def reduce_execution_lifecycle(events: Iterable[LifecycleEvent]) -> ExecutionState:
    state: ExecutionState | None = None
    for event in list(events or []):
        state = _reduce_one(state, event)
    if state is None:
        raise LifecycleInvariantError("empty lifecycle stream")
    return state
