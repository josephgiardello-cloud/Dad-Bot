from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


class AsyncMemoryLedgerPersistence(Protocol):
    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str) -> None: ...

    async def rpush(self, key: str, value: str) -> None: ...

    async def lrange(self, key: str, start: int, end: int) -> list[str]: ...


class InMemoryAsyncMemoryLedgerPersistence:
    """Small async persistence adapter for tests/local wiring."""

    def __init__(self) -> None:
        self._kv: dict[str, str] = {}
        self._lists: dict[str, list[str]] = {}

    async def get(self, key: str) -> str | None:
        return self._kv.get(str(key))

    async def set(self, key: str, value: str) -> None:
        self._kv[str(key)] = str(value)

    async def rpush(self, key: str, value: str) -> None:
        bucket = self._lists.setdefault(str(key), [])
        bucket.append(str(value))

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        values = list(self._lists.get(str(key), []))
        if not values:
            return []

        normalized_start = max(int(start), 0)
        normalized_end = len(values) - 1 if int(end) < 0 else int(end)
        if normalized_start > normalized_end:
            return []
        return values[normalized_start : normalized_end + 1]


class SQLiteAsyncMemoryLedgerPersistence:
    """Async sqlite-backed persistence adapter for memory ledger state."""

    def __init__(self, db_path: str) -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dadbot_memory_kv (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """,
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dadbot_memory_list (
                    key TEXT NOT NULL,
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    value TEXT NOT NULL
                )
                """,
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_dadbot_memory_list_key_seq
                ON dadbot_memory_list(key, seq)
                """,
            )
            conn.commit()
        finally:
            conn.close()

    def _get_sync(self, key: str) -> str | None:
        conn = sqlite3.connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT value FROM dadbot_memory_kv WHERE key = ?",
                (str(key),),
            ).fetchone()
            return None if row is None else str(row[0])
        finally:
            conn.close()

    def _set_sync(self, key: str, value: str) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                """
                INSERT INTO dadbot_memory_kv(key, value)
                VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(key), str(value)),
            )
            conn.commit()
        finally:
            conn.close()

    def _rpush_sync(self, key: str, value: str) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "INSERT INTO dadbot_memory_list(key, value) VALUES(?, ?)",
                (str(key), str(value)),
            )
            conn.commit()
        finally:
            conn.close()

    def _lrange_sync(self, key: str, start: int, end: int) -> list[str]:
        conn = sqlite3.connect(self._db_path)
        try:
            rows = conn.execute(
                "SELECT value FROM dadbot_memory_list WHERE key = ? ORDER BY seq ASC",
                (str(key),),
            ).fetchall()
            values = [str(row[0]) for row in rows]
            if not values:
                return []

            normalized_start = max(int(start), 0)
            normalized_end = len(values) - 1 if int(end) < 0 else int(end)
            if normalized_start > normalized_end:
                return []
            return values[normalized_start : normalized_end + 1]
        finally:
            conn.close()

    async def get(self, key: str) -> str | None:
        return await asyncio.to_thread(self._get_sync, str(key))

    async def set(self, key: str, value: str) -> None:
        await asyncio.to_thread(self._set_sync, str(key), str(value))

    async def rpush(self, key: str, value: str) -> None:
        await asyncio.to_thread(self._rpush_sync, str(key), str(value))

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        return await asyncio.to_thread(self._lrange_sync, str(key), int(start), int(end))


class MemoryLedger:
    """Append-only tamper-evident memory event ledger."""

    LAST_HASH_KEY = "memory:ledger:last_hash"
    LEDGER_KEY = "memory:ledger"

    def __init__(self, persistence: AsyncMemoryLedgerPersistence) -> None:
        self.persistence = persistence

    def _compute_hash(self, payload: dict[str, Any], prev_hash: str | None = None) -> str:
        canonical = json.dumps(
            {"payload": payload, "prev_hash": prev_hash},
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    async def append_memory_event(self, event: dict[str, Any]) -> dict[str, Any]:
        prev_hash = await self.persistence.get(self.LAST_HASH_KEY)
        ts = datetime.now(timezone.utc)
        entry = {
            "entry_id": f"mem_{int(ts.timestamp() * 1000)}",
            "ts": ts.isoformat().replace("+00:00", "Z"),
            "payload": dict(event or {}),
            "prev_hash": prev_hash,
        }
        entry["entry_hash"] = self._compute_hash(entry["payload"], prev_hash)

        raw_entry = json.dumps(entry, separators=(",", ":"), sort_keys=True)
        await self.persistence.rpush(self.LEDGER_KEY, raw_entry)
        await self.persistence.set(self.LAST_HASH_KEY, str(entry["entry_hash"]))
        return entry

    async def entries(self) -> list[dict[str, Any]]:
        rows = await self.persistence.lrange(self.LEDGER_KEY, 0, -1)
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                parsed = json.loads(row)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
        return out

    async def verify_chain(self) -> bool:
        prev_hash: str | None = None
        for entry in await self.entries():
            payload = dict(entry.get("payload") or {})
            expected_prev = entry.get("prev_hash")
            if expected_prev != prev_hash:
                return False
            recomputed = self._compute_hash(payload, prev_hash)
            if recomputed != entry.get("entry_hash"):
                return False
            prev_hash = str(entry.get("entry_hash") or "") or None
        return True
