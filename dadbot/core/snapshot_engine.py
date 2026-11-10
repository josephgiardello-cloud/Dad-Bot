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
_TENANT_SESSION_KEY_PREFIX = "tenant-session:v1:"
_LEGACY_SESSION_KEY_PREFIX = "legacy-session:v1:"
_SESSION_KEY_ENCODING = "tenant-session-json-v1"


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
        snapshot = self._normalize_snapshot_session_keys(snapshot)
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
        session_store.rebuild_from_ledger(self._snapshot_session_events(snapshot))

    @staticmethod
    def _snapshot_session_events(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        snapshot_hash = snapshot.get("snapshot_hash")
        if not isinstance(snapshot_hash, str) or snapshot_hash != SnapshotEngine._hash_snapshot(snapshot):
            raise SnapshotConsistencyError("Snapshot hash is missing or invalid")
        store_snap = snapshot.get("session_snapshot")
        if not isinstance(store_snap, dict) or not isinstance(store_snap.get("sessions"), dict):
            raise SnapshotConsistencyError("Snapshot must contain a session projection")
        encoded_keys = store_snap.get("session_key_encoding") == _SESSION_KEY_ENCODING
        if store_snap.get("session_key_encoding") not in (None, _SESSION_KEY_ENCODING):
            raise SnapshotConsistencyError("Snapshot contains an unsupported session identity encoding")
        sessions = store_snap["sessions"]
        seed_events: list[dict[str, Any]] = []
        for encoded_session_id, state in sessions.items():
            if not isinstance(encoded_session_id, str) or not encoded_session_id or not isinstance(state, dict):
                raise SnapshotConsistencyError("Snapshot contains an invalid session projection")
            tenant_id, session_id = (
                SnapshotEngine._decode_snapshot_session_identity(encoded_session_id)
                if encoded_keys
                else (None, encoded_session_id)
            )
            seed_events.append(
                {
                    "type": "SESSION_STATE_UPDATED",
                    "session_id": session_id,
                    **({"tenant_id": tenant_id} if tenant_id is not None else {}),
                    "kernel_step_id": "snapshot_engine.restore",
                    "trace_id": "snapshot_engine.restore",
                    "payload": {"state": dict(state)},
                },
            )
        return seed_events

    @staticmethod
    def _decode_snapshot_session_identity(encoded_identity: str) -> tuple[str | None, str]:
        if encoded_identity.startswith(_TENANT_SESSION_KEY_PREFIX):
            try:
                identity = json.loads(encoded_identity[len(_TENANT_SESSION_KEY_PREFIX):])
            except json.JSONDecodeError as exc:
                raise SnapshotConsistencyError("Snapshot contains an invalid tenant session identity") from exc
            if (
                not isinstance(identity, list)
                or len(identity) != 2
                or not all(isinstance(part, str) for part in identity)
                or not identity[0]
                or not identity[1]
            ):
                raise SnapshotConsistencyError("Snapshot contains an invalid tenant session identity")
            tenant_id, session_id = identity
            return tenant_id, session_id
        if encoded_identity.startswith(_LEGACY_SESSION_KEY_PREFIX):
            try:
                session_id = json.loads(encoded_identity[len(_LEGACY_SESSION_KEY_PREFIX):])
            except json.JSONDecodeError as exc:
                raise SnapshotConsistencyError("Snapshot contains an invalid legacy session identity") from exc
            if not isinstance(session_id, str) or not session_id:
                raise SnapshotConsistencyError("Snapshot contains an invalid legacy session identity")
            return None, session_id
        return None, encoded_identity

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
        session_store.rebuild_from_ledger(self._snapshot_session_events(snapshot) + tail)
        return {
            "head_sequence": head_sequence,
            "tail_events_applied": len(tail),
            "final_sequence": max(
                head_sequence,
                max((int(event.get("sequence") or 0) for event in tail), default=0),
            ),
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
        snapshot_hash = snapshot.get("snapshot_hash")
        snapshot_hash_ok = isinstance(snapshot_hash, str) and snapshot_hash == self._hash_snapshot(snapshot)
        prefix_present = head_sequence == 0 or any(
            int(event.get("sequence") or 0) == head_sequence for event in events_at_head
        )
        try:
            snapshot_events = self._snapshot_session_events(snapshot)
        except SnapshotConsistencyError:
            session_projection_ok = False
        else:
            snapshot_store = SessionStore()
            snapshot_store.rebuild_from_ledger(snapshot_events)
            replayed_store = SessionStore()
            replayed_store.rebuild_from_ledger(events_at_head)
            session_projection_ok = snapshot_store.snapshot()["sessions"] == replayed_store.snapshot()["sessions"]

        return {
            "ok": current_hash == stored_hash and snapshot_hash_ok and prefix_present and session_projection_ok,
            "head_sequence": head_sequence,
            "current_state_hash": current_hash,
            "stored_state_hash": stored_hash,
            "snapshot_hash_ok": snapshot_hash_ok,
            "prefix_present": prefix_present,
            "session_projection_ok": session_projection_ok,
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
        normalized = SnapshotEngine._normalize_snapshot_session_keys(data)
        serialized = json.dumps(
            {key: value for key, value in sorted(normalized.items()) if key != "snapshot_hash"},
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    @staticmethod
    def _normalize_snapshot_session_keys(data: dict[str, Any]) -> dict[str, Any]:
        normalized = deepcopy(data)
        store_snapshot = normalized.get("session_snapshot")
        if not isinstance(store_snapshot, dict) or not isinstance(store_snapshot.get("sessions"), dict):
            return normalized
        encoding = store_snapshot.get("session_key_encoding")
        if encoding == _SESSION_KEY_ENCODING:
            if any(not isinstance(identity, str) for identity in store_snapshot["sessions"]):
                raise SnapshotConsistencyError("Encoded snapshot session identities must be strings")
            return normalized
        if encoding is not None:
            raise SnapshotConsistencyError("Snapshot contains an unsupported session identity encoding")

        sessions: dict[str, Any] = {}
        for identity, state in store_snapshot["sessions"].items():
            if isinstance(identity, tuple) and len(identity) == 2 and all(isinstance(part, str) for part in identity):
                encoded_identity = _TENANT_SESSION_KEY_PREFIX + json.dumps(list(identity), separators=(",", ":"))
            elif isinstance(identity, str):
                encoded_identity = (
                    _LEGACY_SESSION_KEY_PREFIX + json.dumps(identity, separators=(",", ":"))
                    if identity.startswith((_TENANT_SESSION_KEY_PREFIX, _LEGACY_SESSION_KEY_PREFIX))
                    else identity
                )
            else:
                raise SnapshotConsistencyError("Snapshot contains an invalid session identity")
            if encoded_identity in sessions:
                raise SnapshotConsistencyError("Snapshot contains duplicate serialized session identities")
            sessions[encoded_identity] = state

        store_snapshot["sessions"] = sessions
        store_snapshot["session_key_encoding"] = _SESSION_KEY_ENCODING
        return normalized

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
