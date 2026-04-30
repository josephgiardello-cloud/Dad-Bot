from __future__ import annotations

from copy import deepcopy
from typing import Any

from dadbot.core.execution_ledger import ExecutionLedger


class SessionMutationError(RuntimeError):
    """Raised when a projection-only session store is mutated directly."""


class SessionStore:
    """Projection helper over ledger events for recovery operations."""

    def __init__(
        self,
        ledger: ExecutionLedger | None = None,
        projection_only: bool = False,
    ):
        self.ledger = ledger
        self.projection_only = bool(projection_only)
        self._sessions: dict[str, dict[str, Any]] = {}
        self._version = 0

    def _guard_mutation(self) -> None:
        if self.projection_only:
            raise SessionMutationError(
                "projection-only session store rejects direct mutation",
            )

    def get(self, session_id: str) -> dict[str, Any] | None:
        payload = self._sessions.get(str(session_id or "default"))
        return deepcopy(payload) if payload is not None else None

    def set(self, session_id: str, state: dict[str, Any]) -> None:
        self._guard_mutation()
        self._sessions[str(session_id or "default")] = deepcopy(dict(state or {}))
        self._version += 1

    def delete(self, session_id: str) -> None:
        self._guard_mutation()
        self._sessions.pop(str(session_id or "default"), None)
        self._version += 1

    def snapshot(self) -> dict[str, Any]:
        return {"version": self._version, "sessions": deepcopy(self._sessions)}

    def apply_kernel_mutation(
        self,
        *,
        session_id: str,
        state_patch: dict[str, Any],
        kernel_step_id: str,
        trace_id: str,
    ) -> None:
        state = dict(self._sessions.get(str(session_id or "default")) or {})
        state.update(deepcopy(dict(state_patch or {})))
        self._sessions[str(session_id or "default")] = state
        self._version += 1

    def apply_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        session_id = str(event.get("session_id") or "default")
        payload = dict(event.get("payload") or {})
        if event_type == "SESSION_STATE_UPDATED":
            self.apply_kernel_mutation(
                session_id=session_id,
                state_patch=dict(payload.get("state") or {}),
                kernel_step_id=str(
                    event.get("kernel_step_id") or "session_store.apply",
                ),
                trace_id=str(event.get("trace_id") or ""),
            )
        elif event_type == "SESSION_STATE_DELETED":
            self._sessions.pop(session_id, None)
            self._version += 1
        elif event_type == "JOB_COMPLETED":
            state = dict(self._sessions.get(session_id) or {})
            state["last_result"] = deepcopy(payload.get("result"))
            self._sessions[session_id] = state
            self._version += 1

    def rebuild_from_ledger(self, events: list[dict[str, Any]]) -> None:
        self._sessions = {}
        self._version = 0
        sorted_events = sorted(
            list(events or []),
            key=lambda e: int(e.get("sequence") or e.get("_seq") or 0),
        )
        for event in sorted_events:
            self.apply_event(event)

    def pending_jobs(self) -> list[dict[str, Any]]:
        events = self.ledger.read() if self.ledger is not None else []
        latest_by_job: dict[str, dict[str, Any]] = {}
        for event in events:
            payload = dict(event.get("payload") or {})
            job_id = str(payload.get("job_id") or event.get("job_id") or "")
            if not job_id:
                continue
            latest_by_job[job_id] = dict(event)

        pending: list[dict[str, Any]] = []
        for job_id, event in latest_by_job.items():
            event_type = str(event.get("type") or "")
            payload = dict(event.get("payload") or {})
            if event_type in {"JOB_SUBMITTED", "JOB_QUEUED", "JOB_STARTED"}:
                pending.append(
                    {
                        "job_id": job_id,
                        "session_id": str(event.get("session_id") or "default"),
                        "status": event_type,
                        "request_id": str(payload.get("request_id") or ""),
                    },
                )
        return pending
