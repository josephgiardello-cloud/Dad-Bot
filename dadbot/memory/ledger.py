from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


def _entry_hash(payload: dict[str, Any], prev_hash: str | None) -> str:
    canonical = json.dumps(
        {"payload": payload, "prev_hash": prev_hash},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _new_entry(payload: dict[str, Any], prev_hash: str | None) -> dict[str, Any]:
    ts = datetime.now(UTC)
    return {
        "entry_id": f"mem_{uuid.uuid4().hex}",
        "ts": ts.isoformat().replace("+00:00", "Z"),
        "payload": payload,
        "prev_hash": prev_hash,
        "entry_hash": _entry_hash(payload, prev_hash),
    }


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

    async def append_ledger_entry(self, event: dict[str, Any]) -> dict[str, Any]:
        entry = _new_entry(event, self._kv.get(MemoryLedger.LAST_HASH_KEY))
        self._lists.setdefault(MemoryLedger.LEDGER_KEY, []).append(
            json.dumps(entry, separators=(",", ":"), sort_keys=True),
        )
        self._kv[MemoryLedger.LAST_HASH_KEY] = entry["entry_hash"]
        return entry

    async def read_ledger(self) -> tuple[list[str], str | None]:
        return list(self._lists.get(MemoryLedger.LEDGER_KEY, [])), self._kv.get(MemoryLedger.LAST_HASH_KEY)


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

    def _append_ledger_entry_sync(self, event: dict[str, Any]) -> dict[str, Any]:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT value FROM dadbot_memory_kv WHERE key = ?",
                (MemoryLedger.LAST_HASH_KEY,),
            ).fetchone()
            prev_hash = None if row is None else str(row[0])
            tail = conn.execute(
                "SELECT value FROM dadbot_memory_list WHERE key = ? ORDER BY seq DESC LIMIT 1",
                (MemoryLedger.LEDGER_KEY,),
            ).fetchone()
            if tail is None:
                tail_hash = None
            else:
                last_entry = MemoryLedger._parse_entries([str(tail[0])])[0]
                tail_hash = last_entry.get("entry_hash")
                if tail_hash != _entry_hash(last_entry["payload"], last_entry.get("prev_hash")):
                    raise ValueError("Memory ledger last entry has an invalid hash")
            if tail_hash != prev_hash:
                raise ValueError("Memory ledger head does not match the last entry")
            entry = _new_entry(event, prev_hash)
            conn.execute(
                "INSERT INTO dadbot_memory_list(key, value) VALUES(?, ?)",
                (MemoryLedger.LEDGER_KEY, json.dumps(entry, separators=(",", ":"), sort_keys=True)),
            )
            conn.execute(
                "INSERT INTO dadbot_memory_kv(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (MemoryLedger.LAST_HASH_KEY, entry["entry_hash"]),
            )
            conn.commit()
        finally:
            conn.close()
        return entry

    def _read_ledger_sync(self) -> tuple[list[str], str | None]:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("BEGIN")
            rows = conn.execute(
                "SELECT value FROM dadbot_memory_list WHERE key = ? ORDER BY seq ASC",
                (MemoryLedger.LEDGER_KEY,),
            ).fetchall()
            head = conn.execute(
                "SELECT value FROM dadbot_memory_kv WHERE key = ?",
                (MemoryLedger.LAST_HASH_KEY,),
            ).fetchone()
        finally:
            conn.close()
        return [str(row[0]) for row in rows], None if head is None else str(head[0])

    async def get(self, key: str) -> str | None:
        return await asyncio.to_thread(self._get_sync, str(key))

    async def set(self, key: str, value: str) -> None:
        await asyncio.to_thread(self._set_sync, str(key), str(value))

    async def rpush(self, key: str, value: str) -> None:
        await asyncio.to_thread(self._rpush_sync, str(key), str(value))

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        return await asyncio.to_thread(self._lrange_sync, str(key), int(start), int(end))

    async def append_ledger_entry(self, event: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._append_ledger_entry_sync, event)

    async def read_ledger(self) -> tuple[list[str], str | None]:
        return await asyncio.to_thread(self._read_ledger_sync)


class MemoryLedger:
    """Append-only tamper-evident memory event ledger."""

    LAST_HASH_KEY = "memory:ledger:last_hash"
    LEDGER_KEY = "memory:ledger"

    def __init__(self, persistence: AsyncMemoryLedgerPersistence) -> None:
        self.persistence = persistence

    def _compute_hash(self, payload: dict[str, Any], prev_hash: str | None = None) -> str:
        return _entry_hash(payload, prev_hash)

    async def append_memory_event(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = dict(event or {})
        atomic_append = getattr(self.persistence, "append_ledger_entry", None)
        if callable(atomic_append):
            return await atomic_append(payload)

        prev_hash = await self.persistence.get(self.LAST_HASH_KEY)
        entry = _new_entry(payload, prev_hash)

        raw_entry = json.dumps(entry, separators=(",", ":"), sort_keys=True)
        await self.persistence.rpush(self.LEDGER_KEY, raw_entry)
        await self.persistence.set(self.LAST_HASH_KEY, str(entry["entry_hash"]))
        return entry

    @staticmethod
    def _parse_entries(rows: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            try:
                parsed = json.loads(row)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Malformed memory ledger row {index}") from exc
            if not isinstance(parsed, dict) or not isinstance(parsed.get("payload"), dict):
                raise ValueError(f"Malformed memory ledger row {index}")
            out.append(parsed)
        return out

    async def entries(self) -> list[dict[str, Any]]:
        rows = await self.persistence.lrange(self.LEDGER_KEY, 0, -1)
        return self._parse_entries(rows)

    async def verify_chain(self) -> bool:
        read_ledger = getattr(self.persistence, "read_ledger", None)
        if callable(read_ledger):
            rows, head = await read_ledger()
        else:
            rows = await self.persistence.lrange(self.LEDGER_KEY, 0, -1)
            head = await self.persistence.get(self.LAST_HASH_KEY)
        try:
            entries = self._parse_entries(rows)
        except ValueError:
            return False
        prev_hash: str | None = None
        for entry in entries:
            payload = entry["payload"]
            expected_prev = entry.get("prev_hash")
            if expected_prev != prev_hash:
                return False
            recomputed = self._compute_hash(payload, prev_hash)
            if recomputed != entry.get("entry_hash"):
                return False
            prev_hash = str(entry.get("entry_hash") or "") or None
        return prev_hash == head
