from __future__ import annotations

from copy import deepcopy
from typing import Any

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.session_state_codec import apply_session_state_update


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
        self._sessions: dict[tuple[str, str], dict[str, Any]] = {}
        self._version = 0

    @staticmethod
    def _identity(session_id: str, tenant_id: str | None = None) -> tuple[str, str]:
        sid = str(session_id or "").strip()
        if not sid:
            raise ValueError("session_id is required")
        return str(tenant_id or "").strip(), sid

    def _guard_mutation(self) -> None:
        if self.projection_only:
            raise SessionMutationError(
                "projection-only session store rejects direct mutation",
            )

    def get(self, session_id: str, *, tenant_id: str | None = None) -> dict[str, Any] | None:
        payload = self._sessions.get(self._identity(session_id, tenant_id))
        return deepcopy(payload) if payload is not None else None

    def _set(self, session_id: str, state: dict[str, Any], *, tenant_id: str | None = None) -> None:
        self._guard_mutation()
        self._sessions[self._identity(session_id, tenant_id)] = deepcopy(dict(state or {}))
        self._version += 1

    def _delete(self, session_id: str, *, tenant_id: str | None = None) -> None:
        self._guard_mutation()
        self._sessions.pop(self._identity(session_id, tenant_id), None)
        self._version += 1

    def __getattr__(self, name: str) -> Any:
        if name == "set":
            return self._set
        if name == "delete":
            return self._delete
        raise AttributeError(name)

    def snapshot(self) -> dict[str, Any]:
        return {"version": self._version, "sessions": deepcopy(self._sessions)}

    def _apply_kernel_mutation(
        self,
        *,
        session_id: str,
        tenant_id: str | None = None,
        state_patch: dict[str, Any],
        step_key: str,
        trace_token: str,
    ) -> None:
        identity = self._identity(session_id, tenant_id)
        state = dict(self._sessions.get(identity) or {})
        state.update(deepcopy(dict(state_patch or {})))
        self._sessions[identity] = state
        self._version += 1

    def _apply_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        session_id = str(event.get("session_id") or "default")
        tenant_id = str(event.get("tenant_id") or "").strip()
        if not tenant_id:
            return
        payload = dict(event.get("payload") or {})
        payload_tenant_id = str(payload.get("tenant_id") or "").strip()
        if payload_tenant_id and tenant_id and payload_tenant_id != tenant_id:
            raise ValueError("ledger event tenant does not match its payload")
        identity = self._identity(session_id, tenant_id)
        if event_type == "SESSION_STATE_UPDATED":
            self._sessions[identity] = apply_session_state_update(self._sessions.get(identity) or {}, payload)
            self._version += 1
        elif event_type == "SESSION_STATE_DELETED":
            self._sessions.pop(identity, None)
            self._version += 1
        elif event_type == "JOB_COMPLETED":
            state = dict(self._sessions.get(identity) or {})
            state["last_result"] = deepcopy(payload.get("result"))
            self._sessions[identity] = state
            self._version += 1

    def rebuild_from_ledger(self, events: list[dict[str, Any]]) -> None:
        self._sessions = {}
        self._version = 0
        sorted_events = sorted(
            list(events or []),
            key=lambda e: int(e.get("sequence") or e.get("_seq") or 0),
        )
        for event in sorted_events:
            self._apply_event(event)

    def pending_jobs(self) -> list[dict[str, Any]]:
        events = self.ledger.read() if self.ledger is not None else []
        latest_by_job: dict[str, dict[str, Any]] = {}
        reconciled_requests: set[tuple[str, str, str]] = set()
        for event in events:
            tenant_id = str(event.get("tenant_id") or "").strip()
            if not tenant_id:
                continue
            payload = dict(event.get("payload") or {})
            if str(event.get("type") or "") in {"JOB_RECONCILED", "AGENT_LOOP_NO_EFFECT_CONFIRMED"}:
                request_id = str(payload.get("request_id") or "").strip()
                if request_id:
                    reconciled_requests.add(
                        (
                            tenant_id,
                            str(event.get("session_id") or "default").strip() or "default",
                            request_id,
                        ),
                    )
            job_id = str(payload.get("job_id") or event.get("job_id") or "")
            if not job_id:
                continue
            latest_by_job[job_id] = dict(event)

        pending: list[dict[str, Any]] = []
        for job_id, event in latest_by_job.items():
            event_type = str(event.get("type") or "")
            payload = dict(event.get("payload") or {})
            if event_type in {"JOB_SUBMITTED", "JOB_QUEUED", "JOB_STARTED"}:
                request_id = str(payload.get("request_id") or "")
                identity = (
                    str(event.get("tenant_id") or "").strip(),
                    str(event.get("session_id") or "default").strip() or "default",
                    request_id,
                )
                if request_id and identity in reconciled_requests:
                    continue
                pending.append(
                    {
                        "job_id": job_id,
                        "session_id": str(event.get("session_id") or "default"),
                        "tenant_id": str(event.get("tenant_id") or ""),
                        "status": event_type,
                        "request_id": request_id,
                    },
                )
        return pending

    @classmethod
    def projected_from_ledger(
        cls,
        *,
        ledger: ExecutionLedger,
    ) -> SessionStore:
        """Build a projection-only SessionStore derived from ledger events."""
        store = cls(ledger=ledger, projection_only=True)
        store.rebuild_from_ledger(ledger.read())
        return store
