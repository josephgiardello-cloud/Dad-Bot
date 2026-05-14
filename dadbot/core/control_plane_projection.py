from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from dadbot.core.contracts.lifecycle_events import LifecycleEvent, lifecycle_event_from_ledger_event
from dadbot.core.control_plane_reducer import (
    ExecutionState,
    ExecutionStatus,
    lease_expired,
    reduce_execution_lifecycle,
)


@dataclass(slots=True)
class ExecutionProjection:
    _states: dict[str, ExecutionState]
    _events: dict[str, list[LifecycleEvent]]

    def __init__(self) -> None:
        self._states = {}
        self._events = {}

    def clear(self) -> None:
        self._states.clear()
        self._events.clear()

    def apply(self, event: LifecycleEvent) -> ExecutionState:
        execution_id = str(event.execution_id or "").strip()
        events = list(self._events.get(execution_id) or [])
        events.append(event)
        state = reduce_execution_lifecycle(events)
        self._events[execution_id] = events
        self._states[execution_id] = state
        return state

    def rebuild_from_ledger(self, events: Iterable[dict[str, object]]) -> None:
        self.clear()
        ordered = sorted(
            list(events or []),
            key=lambda item: int((item or {}).get("sequence") or (item or {}).get("_seq") or 0),
        )
        for ledger_event in ordered:
            lifecycle_event = lifecycle_event_from_ledger_event(dict(ledger_event or {}))
            if lifecycle_event is None:
                continue
            self.apply(lifecycle_event)

    def get(self, execution_id: str) -> ExecutionState | None:
        state = self._states.get(str(execution_id or "").strip())
        return deepcopy(state) if state is not None else None

    def get_runnable(self, *, now: datetime, execution_ids: Iterable[str] | None = None) -> list[str]:
        ids = [str(item or "").strip() for item in list(execution_ids or self._states.keys())]
        runnable: list[str] = []
        for execution_id in ids:
            state = self._states.get(execution_id)
            if state is None:
                continue
            if state.status == ExecutionStatus.SUBMITTED:
                runnable.append(execution_id)
                continue
            if state.status == ExecutionStatus.EXPIRED:
                runnable.append(execution_id)
                continue
            if state.status == ExecutionStatus.CLAIMED and lease_expired(state, now=now):
                runnable.append(execution_id)
        return runnable

    def snapshot(self) -> dict[str, dict[str, object]]:
        return {
            execution_id: {
                "status": state.status.value,
                "owner": state.owner,
                "lease_expiry": state.lease_expiry.isoformat() if state.lease_expiry else None,
                "attempt_count": int(state.attempt_count),
            }
            for execution_id, state in sorted(self._states.items())
        }
