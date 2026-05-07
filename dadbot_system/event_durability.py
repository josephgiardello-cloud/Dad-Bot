from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .security import EncryptedJsonCodec


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


@dataclass(slots=True, frozen=True)
class DurableEvent:
    sequence_id: int
    event_id: str
    run_id: str
    event_type: str
    event_time: str
    payload: dict[str, Any]
    payload_hash: str
    previous_event_hash: str
    event_hash: str
    excluded_from_hash: bool = False


class SQLiteEventDurabilityStore:
    """Append-only run/event/checkpoint store used for deterministic recovery.

    This layer is intentionally minimal and strictly append-first:
    - events are immutable once written
    - checkpoints are immutable snapshots linked to event sequence IDs
    - recovery replays persisted history, not recomputed heuristics
    """

    def __init__(self, db_path: str | Path, *, encryption_key: str = ""):
        self.db_path = str(db_path)
        self._codec = EncryptedJsonCodec(encryption_key) if str(encryption_key or "").strip() else None
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _encode_payload(self, payload: dict[str, Any]) -> str:
        if self._codec is None:
            return json.dumps(payload, sort_keys=True, default=str)
        return self._codec.encode(payload)

    def _decode_payload(self, payload_json: str) -> dict[str, Any]:
        if self._codec is None:
            return dict(json.loads(str(payload_json)))
        return self._codec.decode(str(payload_json))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    contract_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_time TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    previous_event_hash TEXT NOT NULL DEFAULT '',
                    event_hash TEXT NOT NULL,
                    excluded_from_hash INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_sequence_id INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    state_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id),
                    FOREIGN KEY (event_sequence_id) REFERENCES events(sequence_id)
                )
                """
            )
            self._ensure_column(conn, "events", "event_id", "TEXT")
            self._ensure_column(conn, "events", "previous_event_hash", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "events", "event_hash", "TEXT")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_events_event_id ON events(event_id)")

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        names = {str(row[1]) for row in rows}
        if column not in names:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    @staticmethod
    def _event_hash_payload(
        *,
        run_id: str,
        event_id: str,
        event_type: str,
        event_time: str,
        payload_hash: str,
        previous_event_hash: str,
    ) -> str:
        return _stable_hash(
            {
                "run_id": str(run_id),
                "event_id": str(event_id),
                "event_type": str(event_type),
                "event_time": str(event_time),
                "payload_hash": str(payload_hash),
                "previous_event_hash": str(previous_event_hash or ""),
            }
        )

    def _latest_event_hash(self, conn: sqlite3.Connection, run_id: str) -> str:
        row = conn.execute(
            "SELECT event_hash FROM events WHERE run_id = ? ORDER BY sequence_id DESC LIMIT 1",
            (str(run_id),),
        ).fetchone()
        return str(row[0] if row and row[0] is not None else "")

    def _row_to_event(self, row: sqlite3.Row) -> DurableEvent:
        return DurableEvent(
            sequence_id=int(row["sequence_id"]),
            event_id=str(row["event_id"] or ""),
            run_id=str(row["run_id"]),
            event_type=str(row["event_type"]),
            event_time=str(row["event_time"]),
            payload=self._decode_payload(str(row["payload_json"] or "")),
            payload_hash=str(row["payload_hash"]),
            previous_event_hash=str(row["previous_event_hash"] or ""),
            event_hash=str(row["event_hash"] or ""),
            excluded_from_hash=bool(row["excluded_from_hash"]),
        )

    def start_run(
        self,
        *,
        session_id: str,
        tenant_id: str,
        contract_version: str = "1.0",
        run_id: str | None = None,
    ) -> str:
        now = _utc_now_iso()
        resolved_run_id = str(run_id or uuid4().hex)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, session_id, tenant_id, status, contract_version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_run_id,
                    str(session_id),
                    str(tenant_id),
                    "running",
                    str(contract_version),
                    now,
                    now,
                ),
            )
        return resolved_run_id

    def append_event(
        self,
        *,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
        excluded_from_hash: bool = False,
        event_time: str | None = None,
        event_id: str | None = None,
    ) -> DurableEvent:
        encoded_payload = dict(payload or {})
        payload_hash = _stable_hash(encoded_payload)
        ts = str(event_time or _utc_now_iso())
        resolved_event_id = str(event_id or uuid4().hex)
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT sequence_id, event_id, run_id, event_type, event_time, payload_json, payload_hash,
                       previous_event_hash, event_hash, excluded_from_hash
                FROM events
                WHERE event_id = ?
                """,
                (resolved_event_id,),
            ).fetchone()
            if existing is not None:
                return self._row_to_event(existing)

            previous_event_hash = self._latest_event_hash(conn, str(run_id))
            event_hash = self._event_hash_payload(
                run_id=str(run_id),
                event_id=resolved_event_id,
                event_type=str(event_type),
                event_time=ts,
                payload_hash=payload_hash,
                previous_event_hash=previous_event_hash,
            )
            cursor = conn.execute(
                """
                INSERT INTO events (
                    event_id,
                    run_id,
                    event_type,
                    event_time,
                    payload_json,
                    payload_hash,
                    previous_event_hash,
                    event_hash,
                    excluded_from_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_event_id,
                    str(run_id),
                    str(event_type),
                    ts,
                    self._encode_payload(encoded_payload),
                    payload_hash,
                    previous_event_hash,
                    event_hash,
                    1 if excluded_from_hash else 0,
                ),
            )
            sequence_id = int(cursor.lastrowid)
        return DurableEvent(
            sequence_id=sequence_id,
            event_id=resolved_event_id,
            run_id=str(run_id),
            event_type=str(event_type),
            event_time=ts,
            payload=encoded_payload,
            payload_hash=payload_hash,
            previous_event_hash=previous_event_hash,
            event_hash=event_hash,
            excluded_from_hash=bool(excluded_from_hash),
        )

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = dict(event or {})
        run_id = str(payload.get("run_id") or "").strip()
        if not run_id:
            raise ValueError("event.run_id is required")
        event_type = str(payload.get("type") or payload.get("event_type") or "").strip()
        if not event_type:
            raise ValueError("event.type is required")
        result = self.append_event(
            run_id=run_id,
            event_type=event_type,
            payload=dict(payload.get("payload") or {}),
            excluded_from_hash=bool(payload.get("excluded_from_hash", False)),
            event_time=str(payload.get("event_time") or "") or None,
            event_id=str(payload.get("event_id") or "") or None,
        )
        return {
            "sequence_id": int(result.sequence_id),
            "event_id": str(result.event_id),
            "run_id": str(result.run_id),
            "type": str(result.event_type),
            "event_time": str(result.event_time),
            "payload": dict(result.payload),
            "payload_hash": str(result.payload_hash),
            "previous_event_hash": str(result.previous_event_hash),
            "event_hash": str(result.event_hash),
            "excluded_from_hash": bool(result.excluded_from_hash),
        }

    def list_events(self, run_id: str) -> list[DurableEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT sequence_id, event_id, run_id, event_type, event_time, payload_json, payload_hash, "
                "previous_event_hash, event_hash, excluded_from_hash "
                "FROM events WHERE run_id = ? ORDER BY sequence_id ASC",
                (str(run_id),),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def list_events_after(self, run_id: str, sequence_id: int) -> list[DurableEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT sequence_id, event_id, run_id, event_type, event_time, payload_json, payload_hash, "
                "previous_event_hash, event_hash, excluded_from_hash "
                "FROM events WHERE run_id = ? AND sequence_id > ? ORDER BY sequence_id ASC",
                (str(run_id), int(sequence_id)),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def append_checkpoint(
        self,
        *,
        run_id: str,
        state: dict[str, Any],
        event_sequence_id: int | None = None,
        state_hash: str | None = None,
    ) -> int:
        serialized_state = dict(state or {})
        resolved_state_hash = str(state_hash or "").strip() or _stable_hash(serialized_state)
        with self._connect() as conn:
            if event_sequence_id is None:
                row = conn.execute(
                    "SELECT COALESCE(MAX(sequence_id), 0) AS max_sequence_id FROM events WHERE run_id = ?",
                    (str(run_id),),
                ).fetchone()
                event_sequence_id = int(row["max_sequence_id"] if row else 0)

            cursor = conn.execute(
                """
                INSERT INTO checkpoints (run_id, event_sequence_id, state_json, state_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    int(event_sequence_id),
                    self._encode_payload(serialized_state),
                    resolved_state_hash,
                    _utc_now_iso(),
                ),
            )
            checkpoint_id = int(cursor.lastrowid)
        return checkpoint_id

    def latest_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT checkpoint_id, run_id, event_sequence_id, state_json, state_hash, created_at
                FROM checkpoints
                WHERE run_id = ?
                ORDER BY checkpoint_id DESC
                LIMIT 1
                """,
                (str(run_id),),
            ).fetchone()
        if row is None:
            return None
        return {
            "checkpoint_id": int(row["checkpoint_id"]),
            "run_id": str(row["run_id"]),
            "event_sequence_id": int(row["event_sequence_id"]),
            "state": self._decode_payload(str(row["state_json"] or "")),
            "state_hash": str(row["state_hash"]),
            "created_at": str(row["created_at"]),
        }

    def recover_from_checkpoint(
        self,
        *,
        run_id: str,
        reducer: Callable[[dict[str, Any], DurableEvent], dict[str, Any]],
        initial_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        checkpoint = self.latest_checkpoint(run_id)
        state = dict(initial_state or {})
        replay_from = 0
        if checkpoint is not None:
            state = dict(checkpoint["state"])
            replay_from = int(checkpoint["event_sequence_id"])

        for event in self.list_events(run_id):
            if int(event.sequence_id) <= replay_from:
                continue
            state = dict(reducer(dict(state), event))
        return state

    def rebuild_from_events(
        self,
        *,
        run_id: str,
        reducer: Callable[[dict[str, Any], DurableEvent], dict[str, Any]],
        initial_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = dict(initial_state or {})
        for event in self.list_events(run_id):
            state = dict(reducer(dict(state), event))
        return state

    def mark_run_status(self, run_id: str, status: str) -> None:
        self.append_event(
            run_id=str(run_id),
            event_type="RUN_STATUS",
            payload={"status": str(status)},
            excluded_from_hash=False,
        )
