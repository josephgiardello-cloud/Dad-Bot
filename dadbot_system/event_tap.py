from __future__ import annotations

from threading import RLock
from typing import Any
from uuid import uuid4

from dadbot.core.kernel_locks import KernelEventTotalityLock


class EventTap:
    """Boundary-only event emitter for deterministic lifecycle observability."""

    def __init__(self, event_store, *, checkpoint_every: int = 25):
        self.store = event_store
        self._lock = RLock()
        self._checkpoint_every = max(1, int(checkpoint_every))
        self._run_id = ""
        self._event_counter = 0

    @property
    def run_id(self) -> str:
        return str(self._run_id or "")

    def begin_run(
        self,
        *,
        session_id: str,
        tenant_id: str,
        run_id: str | None = None,
        contract_version: str = "1.0",
    ) -> str:
        with self._lock:
            resolved_run_id = str(run_id or "").strip()
            if resolved_run_id:
                self.store.start_run(
                    session_id=str(session_id),
                    tenant_id=str(tenant_id),
                    run_id=resolved_run_id,
                    contract_version=str(contract_version),
                )
            else:
                resolved_run_id = self.store.start_run(
                    session_id=str(session_id),
                    tenant_id=str(tenant_id),
                    contract_version=str(contract_version),
                )
            self._run_id = resolved_run_id
            self._event_counter = 0
            return resolved_run_id

    def emit(self, event_type, **payload):
        with self._lock:
            run_id = str(payload.pop("run_id", "") or self._run_id).strip()
            if not run_id:
                raise RuntimeError("EventTap.emit requires active run_id")
            ordering_index = int(self._event_counter) + 1
            payload.setdefault("ordering_index", ordering_index)
            event = {
                "event_id": str(payload.pop("event_id", "") or uuid4().hex),
                "run_id": run_id,
                "type": str(event_type),
                "payload": dict(payload or {}),
                "excluded_from_hash": bool(payload.pop("excluded_from_hash", False))
                if isinstance(payload, dict)
                else False,
            }
            appended = self.store.append(event)
            self._event_counter += 1
            KernelEventTotalityLock.note_event(
                run_id=run_id,
                sequence_id=int(appended.get("sequence_id") or 0),
                event_id=str(appended.get("event_id") or ""),
                event_type=str(appended.get("type") or event_type),
            )
            return appended

    def checkpoint(self, *, state_snapshot: dict[str, Any], state_hash: str = "") -> int:
        with self._lock:
            if not self._run_id:
                raise RuntimeError("EventTap.checkpoint requires active run")
            return int(
                self.store.append_checkpoint(
                    run_id=self._run_id,
                    state=dict(state_snapshot or {}),
                    state_hash=str(state_hash or ""),
                )
            )

    def maybe_checkpoint(self, *, state_snapshot: dict[str, Any], state_hash: str = "") -> int | None:
        with self._lock:
            if self._event_counter > 0 and self._event_counter % self._checkpoint_every == 0:
                return self.checkpoint(state_snapshot=state_snapshot, state_hash=state_hash)
            return None

    def latest_checkpoint(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._run_id:
                return None
            return self.store.latest_checkpoint(self._run_id)

    def events_after_cursor(self, cursor: int) -> list[dict[str, Any]]:
        with self._lock:
            if not self._run_id:
                return []
            events = self.store.list_events_after(self._run_id, int(cursor))
            return [
                {
                    "sequence_id": int(event.sequence_id),
                    "event_id": str(event.event_id),
                    "run_id": str(event.run_id),
                    "type": str(event.event_type),
                    "payload": dict(event.payload),
                    "event_time": str(event.event_time),
                    "event_hash": str(event.event_hash),
                    "previous_event_hash": str(event.previous_event_hash),
                }
                for event in events
            ]
