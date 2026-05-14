from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Protocol


class AsyncMemoryLedgerPersistence(Protocol):
    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str) -> None: ...

    async def rpush(self, key: str, value: str) -> None: ...

    async def lrange(self, key: str, start: int, end: int) -> list[str]: ...


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
