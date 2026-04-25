from threading import RLock
from copy import deepcopy
from typing import Any


class SessionMutationError(RuntimeError):
    pass


class SessionStore:
    """
    In-memory session state store with:
    - thread safety
    - immutable read snapshots
    - optional ledger hook integration
    """

    def __init__(self, ledger=None, *, projection_only: bool = False):
        self._sessions = {}
        self._lock = RLock()
        self._ledger = ledger  # optional execution ledger hook
        self._version = 0
        self._projection_only = bool(projection_only)

    def set_projection_only(self, enabled: bool = True) -> None:
        self._projection_only = bool(enabled)

    def get(self, session_id: str):
        with self._lock:
            state = self._sessions.get(session_id)
            return deepcopy(state) if state is not None else None

    def set(self, session_id: str, state: dict):
        if self._projection_only:
            raise SessionMutationError("SessionStore is projection-only; direct set() is blocked")
        with self._lock:
            self._sessions[session_id] = deepcopy(state)
            self._version += 1

        # optional: emit to ledger if present
        if self._ledger is not None:
            try:
                self._ledger.write({
                    "type": "SESSION_STATE_UPDATED",
                    "session_id": session_id,
                    "trace_id": "",
                    "timestamp": 0,
                    "kernel_step_id": "session_store.set",
                    "payload": {"version": self._version},
                })
            except Exception:
                pass

    def delete(self, session_id: str):
        if self._projection_only:
            raise SessionMutationError("SessionStore is projection-only; direct delete() is blocked")
        with self._lock:
            self._sessions.pop(session_id, None)
            self._version += 1

        if self._ledger is not None:
            try:
                self._ledger.write({
                    "type": "SESSION_STATE_DELETED",
                    "session_id": session_id,
                    "trace_id": "",
                    "timestamp": 0,
                    "kernel_step_id": "session_store.delete",
                    "payload": {"version": self._version},
                })
            except Exception:
                pass

    def list_sessions(self):
        with self._lock:
            return list(self._sessions.keys())

    def apply_kernel_mutation(
        self,
        *,
        session_id: str,
        state_patch: dict[str, Any],
        kernel_step_id: str,
        trace_id: str,
    ) -> None:
        if not str(kernel_step_id or "").strip():
            raise SessionMutationError("kernel_step_id is required")
        if not str(trace_id or "").strip():
            raise SessionMutationError("trace_id is required")

        with self._lock:
            current = dict(self._sessions.get(session_id) or {})
            current.update(deepcopy(dict(state_patch or {})))
            self._sessions[session_id] = current
            self._version += 1

        if self._ledger is not None:
            self._ledger.write(
                {
                    "type": "SESSION_STATE_UPDATED",
                    "session_id": session_id,
                    "trace_id": str(trace_id),
                    "timestamp": 0,
                    "kernel_step_id": str(kernel_step_id),
                    "payload": {
                        "state": deepcopy(dict(state_patch or {})),
                        "version": self._version,
                    },
                }
            )

    def apply_event(self, event: dict[str, Any]) -> None:
        """Apply a ledger event into this store as a read-model projection."""
        event_type = str(event.get("type") or "")
        session_id = str(event.get("session_id") or "").strip()
        payload = dict(event.get("payload") or {})
        if not session_id:
            return

        with self._lock:
            if event_type == "SESSION_STATE_UPDATED":
                state = payload.get("state")
                if isinstance(state, dict):
                    self._sessions[session_id] = deepcopy(state)
            elif event_type == "SESSION_STATE_DELETED":
                self._sessions.pop(session_id, None)
            elif event_type == "JOB_COMPLETED":
                result = payload.get("result")
                existing = dict(self._sessions.get(session_id) or {})
                existing["last_result"] = deepcopy(result)
                self._sessions[session_id] = existing
            self._version += 1

    def rebuild_from_ledger(self, events: list[dict[str, Any]]) -> None:
        """Reconstruct session state exclusively from ordered ledger events."""
        ordered = sorted(
            [dict(event) for event in list(events or []) if isinstance(event, dict)],
            key=lambda event: int(event.get("sequence") or 0),
        )
        with self._lock:
            self._sessions = {}
            self._version = 0
        for event in ordered:
            self.apply_event(event)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "sessions": deepcopy(self._sessions),
                "version": int(self._version),
            }
