"""LedgerBackend abstraction â€” pluggable durability tier for ExecutionLedger.

Two implementations:
  InMemoryLedgerBackend  â€” current behavior; no durability guarantee.
  FileWALLedgerBackend   â€” JSONL append-only WAL with optional fsync semantics.

Usage:
    from dadbot.core.ledger_backend import FileWALLedgerBackend
    ledger = ExecutionLedger(backend=FileWALLedgerBackend("runtime/ledger.wal"))
"""
from __future__ import annotations

import json
import os
import threading
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Abstract contract
# ---------------------------------------------------------------------------

class LedgerBackend(ABC):
    """Pluggable storage tier for ExecutionLedger events."""

    @abstractmethod
    def append(self, event: dict[str, Any], *, committed: bool = False) -> None:
        """Persist event.  When committed=True, must not return until durable."""

    @abstractmethod
    def load(self) -> list[dict[str, Any]]:
        """Return all persisted events in append order."""

    @abstractmethod
    def close(self) -> None:
        """Release any file handles or connections."""


# ---------------------------------------------------------------------------
# In-memory (existing behaviour â€” default)
# ---------------------------------------------------------------------------

class InMemoryLedgerBackend(LedgerBackend):
    """Volatile in-memory backend.  Correct but not crash-durable."""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    def append(self, event: dict[str, Any], *, committed: bool = False) -> None:
        with self._lock:
            self._events.append(deepcopy(event))

    def load(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._events)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# File WAL backend â€” append-only JSONL with fsync commit semantics
# ---------------------------------------------------------------------------

class FileWALLedgerBackend(LedgerBackend):
    """Append-only JSONL Write-Ahead Log backed by a local file.

    Durability contract:
    - Every call to append() writes one line to the WAL file and flushes the OS
      buffer.
    - When committed=True (or when the event type is in COMMITTED_TYPES), the
      flush is followed by os.fsync() â€” the write is not returned until the OS
      confirms the bytes hit the storage device.
    - load() replays all valid JSONL lines; corrupt lines (invalid JSON) are
      skipped with a warning, not raised, so the system can still boot.

    Crash recovery:
    - On startup, call ExecutionLedger.load_from_backend() to repopulate the
      in-memory sequence from the file.
    """

    #: Event types that always trigger an fsync regardless of the committed flag.
    COMMITTED_TYPES: frozenset[str] = frozenset({
        "JOB_QUEUED",
        "JOB_STARTED",
        "JOB_COMPLETED",
        "JOB_FAILED",
        "SESSION_STATE_UPDATED",
    })

    def __init__(self, path: str | Path, *, fsync: bool = True) -> None:
        self._path = Path(path)
        self._fsync_enabled = bool(fsync)
        self._lock = threading.RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Create the file if it doesn't exist.
        if not self._path.exists():
            self._path.touch()

    def append(self, event: dict[str, Any], *, committed: bool = False) -> None:
        line = json.dumps(event, default=str) + "\n"
        should_fsync = self._fsync_enabled and (
            committed or str(event.get("type") or "") in self.COMMITTED_TYPES
        )
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                if should_fsync:
                    os.fsync(handle.fileno())

    def load(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self._path.exists():
                return []
            events: list[dict[str, Any]] = []
            with open(self._path, "r", encoding="utf-8") as handle:
                for line_number, raw in enumerate(handle, start=1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        events.append(json.loads(raw))
                    except json.JSONDecodeError as exc:
                        # Tolerate a partial write at the tail â€” skip and continue.
                        import warnings
                        warnings.warn(
                            f"FileWALLedgerBackend: skipping corrupt line {line_number} in "
                            f"{self._path}: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
            return events

    def close(self) -> None:
        pass  # File handles are opened per-write; nothing to release.

    def path(self) -> Path:
        return self._path


# ---------------------------------------------------------------------------
# CRC-checksummed WAL backend (Tier 0 â€” crash-safe corruption detection)
# ---------------------------------------------------------------------------

class CRCFileWALLedgerBackend(LedgerBackend):
    """WAL backend that encodes every line with a CRC-32 checksum.

    Line format: ``<crc32_hex_8chars> <json_payload>\\n``

    On load, lines with a bad or missing CRC are silently skipped with a
    ``RuntimeWarning`` â€” the system can still boot from a partially-corrupt
    file.  A plain-JSON fallback is attempted for legacy lines that pre-date
    the CRC format.

    This backend satisfies Tier 0 item 1: corruption detection at the storage
    layer with partial-write recovery.
    """

    COMMITTED_TYPES: frozenset[str] = FileWALLedgerBackend.COMMITTED_TYPES

    def __init__(self, path: str | Path, *, fsync: bool = True) -> None:
        self._path = Path(path)
        self._fsync_enabled = bool(fsync)
        self._lock = threading.RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()

    def append(self, event: dict[str, Any], *, committed: bool = False) -> None:
        from dadbot.core.durability import CRC32LineCodec
        line = CRC32LineCodec.encode(event)
        should_fsync = self._fsync_enabled and (
            committed or str(event.get("type") or "") in self.COMMITTED_TYPES
        )
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                if should_fsync:
                    os.fsync(handle.fileno())

    def load(self) -> list[dict[str, Any]]:
        from dadbot.core.durability import CRC32LineCodec
        with self._lock:
            if not self._path.exists():
                return []
            events: list[dict[str, Any]] = []
            with open(self._path, "r", encoding="utf-8") as handle:
                for line_number, raw in enumerate(handle, start=1):
                    if not raw.strip():
                        continue
                    event = CRC32LineCodec.decode(raw)
                    if event is None:
                        import warnings
                        warnings.warn(
                            f"CRCFileWALLedgerBackend: corrupt/partial line {line_number} "
                            f"in {self._path} â€” skipping",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        continue
                    events.append(event)
            return events

    def close(self) -> None:
        pass

    @property
    def path(self) -> Path:
        return self._path


# ---------------------------------------------------------------------------
# Sequence validator (Tier 1 item 4 â€” backend-enforced ordering)
# ---------------------------------------------------------------------------

class SequenceValidator:
    """Validates that events loaded from a backend have monotonically increasing
    sequence numbers.

    Used during ``ExecutionLedger.load_from_backend()`` to detect backend
    reordering or partial writes that produced an inconsistent sequence.
    """

    @staticmethod
    def validate(events: list[dict[str, Any]]) -> dict[str, Any]:
        """Check sequence monotonicity.

        Returns a report dict with ``ok`` bool and ``violations`` list.
        """
        violations: list[str] = []
        prev_seq: int | None = None
        for i, event in enumerate(events):
            seq = event.get("sequence")
            if seq is None:
                continue
            seq = int(seq)
            if prev_seq is not None and seq <= prev_seq:
                violations.append(
                    f"Event #{i} sequence {seq} is not > previous {prev_seq} "
                    f"(type={event.get('type')!r})"
                )
            prev_seq = seq
        return {
            "ok": len(violations) == 0,
            "violations": violations,
            "event_count": len(events),
        }


# ---------------------------------------------------------------------------
# Consistency-mode wrappers (Tier 3 item 11)
# ---------------------------------------------------------------------------

class StrongConsistencyBackend(LedgerBackend):
    """Wrapper that forces fsync on *every* append, regardless of event type.

    Provides strong durability at the cost of higher write latency.
    Suitable for critical audit-trail scenarios where no event can be lost.
    """

    def __init__(self, inner: LedgerBackend) -> None:
        if not isinstance(inner, FileWALLedgerBackend) and not isinstance(inner, CRCFileWALLedgerBackend):
            raise TypeError(
                "StrongConsistencyBackend requires a file-based backend "
                "(FileWALLedgerBackend or CRCFileWALLedgerBackend)"
            )
        self._inner = inner

    def append(self, event: dict[str, Any], *, committed: bool = False) -> None:
        self._inner.append(event, committed=True)  # Always committed.

    def load(self) -> list[dict[str, Any]]:
        return self._inner.load()

    def close(self) -> None:
        self._inner.close()


class EventualConsistencyBackend(LedgerBackend):
    """Wrapper that buffers writes and flushes periodically or on commit.

    Provides lower write latency at the cost of potential data loss if the
    process crashes before a flush.  Suitable for low-priority telemetry
    events where some loss is acceptable.

    The buffer is also flushed automatically when a ``committed=True`` event
    arrives (commits act as flush barriers).
    """

    def __init__(
        self,
        inner: LedgerBackend,
        *,
        buffer_size: int = 100,
    ) -> None:
        self._inner = inner
        self._buffer_size = max(1, int(buffer_size))
        self._buffer: list[tuple[dict[str, Any], bool]] = []
        self._lock = threading.RLock()

    def append(self, event: dict[str, Any], *, committed: bool = False) -> None:
        with self._lock:
            self._buffer.append((deepcopy(event), committed))
            if committed or len(self._buffer) >= self._buffer_size:
                self._flush_locked()

    def flush(self) -> int:
        """Flush buffer to inner backend.  Returns number of events flushed."""
        with self._lock:
            return self._flush_locked()

    def _flush_locked(self) -> int:
        count = len(self._buffer)
        for ev, committed in self._buffer:
            self._inner.append(ev, committed=committed)
        self._buffer.clear()
        return count

    def load(self) -> list[dict[str, Any]]:
        with self._lock:
            self._flush_locked()
        return self._inner.load()

    def close(self) -> None:
        self.flush()
        self._inner.close()

    @property
    def buffered_count(self) -> int:
        with self._lock:
            return len(self._buffer)


class BatchWriteBackend(LedgerBackend):
    """Wrapper that accumulates events and writes them in batches.

    Each call to ``append()`` is buffered.  The batch is written when:
      - the buffer reaches ``batch_size`` events, OR
      - ``flush()`` is called explicitly, OR
      - a ``committed=True`` event arrives.

    This reduces the number of I/O operations when many events are written
    in rapid succession (e.g. bulk import or replay).
    """

    def __init__(
        self,
        inner: LedgerBackend,
        *,
        batch_size: int = 50,
    ) -> None:
        self._inner      = inner
        self._batch_size = max(1, int(batch_size))
        self._buffer:    list[tuple[dict[str, Any], bool]] = []
        self._lock       = threading.RLock()

    def append(self, event: dict[str, Any], *, committed: bool = False) -> None:
        with self._lock:
            self._buffer.append((deepcopy(event), committed))
            if committed or len(self._buffer) >= self._batch_size:
                self._flush_locked()

    def flush(self) -> int:
        with self._lock:
            return self._flush_locked()

    def _flush_locked(self) -> int:
        count = len(self._buffer)
        for ev, committed in self._buffer:
            self._inner.append(ev, committed=committed)
        self._buffer.clear()
        return count

    def load(self) -> list[dict[str, Any]]:
        with self._lock:
            self._flush_locked()
        return self._inner.load()

    def close(self) -> None:
        self.flush()
        self._inner.close()

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._buffer)
