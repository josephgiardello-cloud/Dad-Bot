"""Snapshot + Restore Engine â€” fast startup via periodic ledger snapshots.

Contract:
  - A snapshot records the full reduced state at a given ledger head sequence.
  - On startup, restore_from_snapshot() loads the snapshot state into
    session_store, then replay_tail() applies any events after the snapshot
    head to bring the store current.
  - This avoids full replay from event 0 on every boot.

Usage:
    engine = SnapshotEngine()
    snapshot = engine.take_snapshot(ledger=ledger, session_store=store)
    # ... after restart ...
    engine.restore_from_snapshot(snapshot, session_store=fresh_store)
    engine.replay_tail(snapshot, ledger=ledger, session_store=fresh_store)
"""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from threading import RLock
from typing import Any

from dadbot.core.event_reducer import CanonicalEventReducer
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_writer import LedgerWriter
from dadbot.core.session_store import SessionStore

_SNAPSHOT_EVENT_TYPE = "ledger_snapshot_projection"


class SnapshotConsistencyError(RuntimeError):
    pass


class SnapshotEngine:
    """Periodic snapshot + tail-replay restore engine."""

    def __init__(self, *, reducer: CanonicalEventReducer | None = None) -> None:
        self._reducer = reducer or CanonicalEventReducer()
        self._lock = RLock()
        self._snapshots: list[dict[str, Any]] = []
        self._ledger_ref: ExecutionLedger | None = None

    # ------------------------------------------------------------------
    # Take
    # ------------------------------------------------------------------

    def take_snapshot(
        self,
        *,
        ledger: ExecutionLedger,
        session_store: SessionStore,
        label: str = "periodic",
    ) -> dict[str, Any]:
        """Snapshot the current ledger head and reduced state.

        Returns the snapshot dict (also stored in engine history).
        """
        events = ledger.read()
        if not events:
            head_sequence = 0
            replay_hash = ""
        else:
            head_sequence = int(events[-1].get("sequence") or 0)
            replay_hash = ledger.replay_hash()

        reduced_state = self._reducer.reduce(events)
        store_snapshot = session_store.snapshot()

        snapshot = {
            "label": str(label or "periodic"),
            "created_at": time.time(),
            "head_sequence": head_sequence,
            "event_count": len(events),
            "replay_hash": str(replay_hash or ""),
            "reduced_state": deepcopy(reduced_state),
            "session_snapshot": deepcopy(store_snapshot),
        }
        snapshot["snapshot_hash"] = self._hash_snapshot(snapshot)

        self._ledger_ref = ledger
        self._append_snapshot_event(ledger=ledger, snapshot=snapshot)

        with self._lock:
            self._snapshots.append(deepcopy(snapshot))

        return deepcopy(snapshot)

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    def restore_from_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        session_store: SessionStore,
    ) -> None:
        """Load snapshot state into session_store.

        After this call session_store contains the state as of snapshot head.
        Call replay_tail() afterwards to apply events that arrived since.
        """
        store_snap = dict(snapshot.get("session_snapshot") or {})
        sessions = dict(store_snap.get("sessions") or {})
        seed_events: list[dict[str, Any]] = []
        for session_id, state in sessions.items():
            if not isinstance(state, dict):
                continue
            seed_events.append(
                {
                    "type": "SESSION_STATE_UPDATED",
                    "session_id": str(session_id or "default"),
                    "kernel_step_id": "snapshot_engine.restore",
                    "trace_id": "snapshot_engine.restore",
                    "payload": {"state": dict(state)},
                },
            )
        session_store.rebuild_from_ledger(seed_events)

    def replay_tail(
        self,
        snapshot: dict[str, Any],
        *,
        ledger: ExecutionLedger,
        session_store: SessionStore,
    ) -> dict[str, Any]:
        """Apply events after snapshot head to bring session_store current.

        Returns a report with the number of tail events applied.
        """
        head_sequence = int(snapshot.get("head_sequence") or 0)
        all_events = ledger.read()
        tail = [event for event in all_events if int(event.get("sequence") or 0) > head_sequence]
        if tail:
            restored_events = [
                event for event in all_events if int(event.get("sequence") or 0) <= head_sequence
            ]
            session_store.rebuild_from_ledger(restored_events + tail)
        return {
            "head_sequence": head_sequence,
            "tail_events_applied": len(tail),
            "final_sequence": int(all_events[-1].get("sequence") or 0) if all_events else 0,
        }

    def verify_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        ledger: ExecutionLedger,
    ) -> dict[str, Any]:
        """Verify snapshot is still consistent with the current ledger.

        Checks that ledger events up to head_sequence produce the same
        reduced state as stored in the snapshot.
        """
        head_sequence = int(snapshot.get("head_sequence") or 0)
        all_events = ledger.read()
        events_at_head = [event for event in all_events if int(event.get("sequence") or 0) <= head_sequence]
        current_reduced = self._reducer.reduce(events_at_head)
        stored_reduced = dict(snapshot.get("reduced_state") or {})

        current_hash = hashlib.sha256(
            json.dumps(current_reduced, sort_keys=True, default=str).encode(),
        ).hexdigest()
        stored_hash = hashlib.sha256(
            json.dumps(stored_reduced, sort_keys=True, default=str).encode(),
        ).hexdigest()

        return {
            "ok": current_hash == stored_hash,
            "head_sequence": head_sequence,
            "current_state_hash": current_hash,
            "stored_state_hash": stored_hash,
        }

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def latest(self) -> dict[str, Any] | None:
        projected = self._history_from_ledger()
        if projected:
            return deepcopy(projected[-1])
        with self._lock:
            return deepcopy(self._snapshots[-1]) if self._snapshots else None

    def history(self) -> list[dict[str, Any]]:
        projected = self._history_from_ledger()
        if projected:
            return deepcopy(projected)
        with self._lock:
            return deepcopy(self._snapshots)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_snapshot(data: dict[str, Any]) -> str:
        serialized = json.dumps(
            {key: value for key, value in sorted(data.items()) if key != "snapshot_hash"},
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def _append_snapshot_event(
        self,
        *,
        ledger: ExecutionLedger,
        snapshot: dict[str, Any],
    ) -> None:
        writer = LedgerWriter(ledger)
        writer.write_event(
            _SNAPSHOT_EVENT_TYPE,
            session_id="system:snapshot_engine",
            trace_id=str(snapshot.get("snapshot_hash") or "snapshot"),
            kernel_step_id="snapshot-engine",
            payload=deepcopy(snapshot),
            committed=False,
        )

    def _history_from_ledger(self) -> list[dict[str, Any]]:
        if self._ledger_ref is None:
            return []
        events = self._ledger_ref.read()
        history: list[dict[str, Any]] = []
        for event in events:
            if str(event.get("type") or "") != _SNAPSHOT_EVENT_TYPE:
                continue
            payload = dict(event.get("payload") or {})
            if not payload:
                continue
            history.append(payload)
        return history
