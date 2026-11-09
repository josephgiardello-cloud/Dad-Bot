"""Execution ledger storage in the existing PostgreSQL state-store tables."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from dadbot.core.ledger_backend import (
    LedgerBackend,
    WALCorruptionError,
    _coerce_event_record,
    _validate_append_head,
)

if TYPE_CHECKING:
    from dadbot.contracts import SovereignEvent
    from dadbot_system.state import PostgresStateStore


class PostgresLedgerBackend(LedgerBackend):
    """Experimental trusted-local ledger; not supported for authenticated production.

    The state store's existing session and event tables hold the head and
    records. The head row serializes all cooperating writers of this stream.
    See ``docs/decisions/postgres-authenticated-boot.md`` before enabling it.
    """

    def __init__(self, store: PostgresStateStore, *, stream: str) -> None:
        stream_name = str(stream).strip()
        if not stream_name:
            raise ValueError("A nonempty PostgreSQL ledger stream is required")
        self._store = store
        self._stream = f"execution-ledger:{stream_name}"
        self._head_key = f"{self._stream}:head"

    def _read_events(self, cursor: Any) -> list[dict[str, Any]]:
        from psycopg import sql

        cursor.execute(
            sql.SQL(
                "SELECT payload FROM {} WHERE session_id = %s "
                "ORDER BY (payload->>'sequence')::bigint"
            ).format(sql.Identifier(self._store._event_table)),
            (self._stream,),
        )
        events = [row[0] for row in cursor.fetchall()]
        if any(not isinstance(event, dict) for event in events):
            raise WALCorruptionError("PostgreSQL ledger contains a non-object record")
        return events

    @staticmethod
    def _validate_head(head: Any, events: list[dict[str, Any]]) -> None:
        if not isinstance(head, dict):
            raise WALCorruptionError("PostgreSQL ledger head is missing or invalid")
        if set(head) != {"count", "sequence", "chain_hash"}:
            raise WALCorruptionError("PostgreSQL ledger head has invalid fields")
        last = events[-1] if events else {}
        expected = {
            "count": len(events),
            "sequence": last.get("sequence", 0),
            "chain_hash": last.get("chain_hash", ""),
        }
        if head != expected:
            raise WALCorruptionError("PostgreSQL ledger head does not match persisted records")

    def append(self, event: dict[str, Any] | SovereignEvent, *, committed: bool = False) -> None:
        from psycopg import sql

        payload = _coerce_event_record(event)
        if not isinstance(payload.get("sequence"), int) or not payload.get("chain_hash"):
            raise ValueError("PostgreSQL execution ledger requires sealed, sequenced events")
        empty_head = {"count": 0, "sequence": 0, "chain_hash": ""}
        with self._store._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SET LOCAL synchronous_commit = on")
            cursor.execute(
                sql.SQL("INSERT INTO {} (session_id, payload) VALUES (%s, %s) ON CONFLICT DO NOTHING").format(
                    sql.Identifier(self._store._session_table)
                ),
                (self._head_key, json.dumps(empty_head)),
            )
            cursor.execute(
                sql.SQL("SELECT payload FROM {} WHERE session_id = %s FOR UPDATE").format(
                    sql.Identifier(self._store._session_table)
                ),
                (self._head_key,),
            )
            row = cursor.fetchone()
            events = self._read_events(cursor)
            self._validate_head(row[0] if row else None, events)
            _validate_append_head(events, payload)
            event_key = "execution-ledger:" + hashlib.sha256(
                f"{self._stream}:{payload['sequence']}".encode("utf-8")
            ).hexdigest()
            cursor.execute(
                sql.SQL("INSERT INTO {} (event_id, session_id, payload) VALUES (%s, %s, %s)").format(
                    sql.Identifier(self._store._event_table)
                ),
                (event_key, self._stream, json.dumps(payload, default=str)),
            )
            head = {
                "count": len(events) + 1,
                "sequence": payload["sequence"],
                "chain_hash": payload["chain_hash"],
            }
            cursor.execute(
                sql.SQL("UPDATE {} SET payload = %s WHERE session_id = %s").format(
                    sql.Identifier(self._store._session_table)
                ),
                (json.dumps(head), self._head_key),
            )
            connection.commit()

    def load(self) -> list[dict[str, Any]]:
        from psycopg import sql

        with self._store._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            cursor.execute(
                sql.SQL("SELECT payload FROM {} WHERE session_id = %s").format(
                    sql.Identifier(self._store._session_table)
                ),
                (self._head_key,),
            )
            row = cursor.fetchone()
            events = self._read_events(cursor)
            if row is None and not events:
                return []
            self._validate_head(row[0] if row else None, events)
            return events

    def close(self) -> None:
        pass
