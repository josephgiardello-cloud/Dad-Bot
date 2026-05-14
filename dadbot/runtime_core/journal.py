from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

from .models import Event


class EventJournal(Protocol):
    def append(self, event: Event) -> None: ...

    def replay(self) -> list[Event]: ...


class FileEventJournal:
    """Append-only JSONL journal with hash-chained records for replay."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(self, event: Event) -> None:
        previous_hash = self._last_hash()
        payload = {
            "id": str(event.id),
            "type": str(event.type),
            "thread_id": str(event.thread_id),
            "timestamp": float(event.timestamp),
            "payload": dict(event.payload or {}),
            "previous_hash": previous_hash,
        }
        payload["record_hash"] = self._hash_payload(payload)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")
            handle.flush()

    def replay(self) -> list[Event]:
        events: list[Event] = []
        previous_hash = ""
        with self.path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = str(raw_line or "").strip()
                if not line:
                    continue
                record = json.loads(line)
                if str(record.get("previous_hash") or "") != previous_hash:
                    raise RuntimeError("Runtime event journal hash chain is broken")
                expected_hash = self._hash_payload(
                    {key: value for key, value in record.items() if key != "record_hash"},
                )
                actual_hash = str(record.get("record_hash") or "")
                if expected_hash != actual_hash:
                    raise RuntimeError("Runtime event journal record hash mismatch")
                previous_hash = actual_hash
                events.append(
                    Event(
                        id=str(record.get("id") or ""),
                        type=str(record.get("type") or "user_message"),
                        thread_id=str(record.get("thread_id") or "default"),
                        timestamp=float(record.get("timestamp") or 0.0),
                        payload=dict(record.get("payload") or {}),
                    ),
                )
        return events

    def _last_hash(self) -> str:
        last_hash = ""
        with self.path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = str(raw_line or "").strip()
                if not line:
                    continue
                record = json.loads(line)
                last_hash = str(record.get("record_hash") or "")
        return last_hash

    @staticmethod
    def _hash_payload(payload: dict) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode(
                "utf-8",
            ),
        ).hexdigest()
