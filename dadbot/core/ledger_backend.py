"""LedgerBackend abstraction â€” pluggable durability tier for ExecutionLedger.

Two implementations:
  InMemoryLedgerBackend  â€” current behavior; no durability guarantee.
  FileWALLedgerBackend   â€” JSONL append-only WAL with optional fsync semantics.

Usage:
    from dadbot.core.ledger_backend import FileWALLedgerBackend
    ledger = ExecutionLedger(backend=FileWALLedgerBackend("runtime/ledger.wal"))
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import os
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from dadbot.contracts import SovereignEvent


class WALCorruptionError(RuntimeError):
    """The WAL cannot be replayed safely without operator recovery."""


class LedgerBackendRecoveryRequired(RuntimeError):
    """A buffered write has an uncertain outcome and must not be retried."""


class LedgerHeadConflict(RuntimeError):
    """The ledger changed since the caller read its head."""


def _require_safe_buffer_flush(uncertain: bool) -> None:
    if uncertain:
        raise LedgerBackendRecoveryRequired(
            "Buffered ledger flush outcome is uncertain; preserve the buffer and reconcile the inner backend",
        )


def _require_complete_wal_tail(path: Path) -> None:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell():
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                raise WALCorruptionError(f"Incomplete WAL tail in {path}; recovery is required before appending")


def _validate_append_head(
    persisted: list[dict[str, Any]], event: dict[str, Any], *, verified_prefix: bool = False,
) -> None:
    from dadbot.core.execution_ledger import _verify_persisted_chain
    from dadbot.core.tenant_memory_ledger import validate_tenant_event_seal

    validate_tenant_event_seal(event)
    if "chain_hash" not in event:
        if any("chain_hash" in record for record in persisted):
            raise LedgerHeadConflict(
                "Unchained events cannot be appended to a chained execution ledger",
            )
        return
    if verified_prefix and persisted and persisted[-1].get("chain_hash"):
        violations = _verify_persisted_chain(
            [event], previous_chain_hash=persisted[-1]["chain_hash"], chain_started=True,
        )
        sequence_tail = next((record for record in reversed(persisted) if record.get("sequence") is not None), None)
        sequence_report = SequenceValidator.validate([sequence_tail, event] if sequence_tail is not None else [event])
    else:
        violations = _verify_persisted_chain([*persisted, event])
        sequence_report = SequenceValidator.validate([*persisted, event])
    if violations or not sequence_report["ok"]:
        raise LedgerHeadConflict(
            "Execution ledger head changed or is invalid; reload and verify before retrying",
        )
    if event.get("type") == "TENANT_MEMORY_COMMITTED":
        from dadbot.core.tenant_memory_ledger import replay_all_tenant_memory

        replay_all_tenant_memory([*persisted, event])


class _WALAppendVerifier:
    """Reuse a sealed prefix only after checking all current bytes under the writer lock."""

    def __init__(self) -> None:
        self._snapshot: tuple[bytes, list[dict[str, Any]], bytes] | None = None

    def load_prefix(
        self, path: Path, load: Callable[[], list[dict[str, Any]]], *, reuse: bool,
    ) -> tuple[list[dict[str, Any]], bytes, bool]:
        snapshot = self._snapshot
        self._snapshot = None
        prefix_bytes = path.read_bytes()
        digest = hashlib.sha256(prefix_bytes).digest()
        unchanged = reuse and snapshot is not None and snapshot[0] == digest
        events = snapshot[1] if unchanged and snapshot is not None else load()
        return events, prefix_bytes, unchanged

    def validate(
        self, path: Path, load: Callable[[], list[dict[str, Any]]], event: dict[str, Any], *, reuse: bool,
    ) -> tuple[list[dict[str, Any]], bytes]:
        events, prefix_bytes, unchanged = self.load_prefix(path, load, reuse=reuse)
        _validate_append_head(events, event, verified_prefix=unchanged)
        return events, prefix_bytes

    def record(
        self, prefix_bytes: bytes, line: str, events: list[dict[str, Any]], event: dict[str, Any], *, reuse: bool,
    ) -> bytes | None:
        if reuse and event.get("chain_hash"):
            events.append(deepcopy(event))
            digest = hashlib.sha256(prefix_bytes)
            prefix_digest = digest.digest()
            digest.update(line.encode("utf-8"))
            self._snapshot = (digest.digest(), events, prefix_digest)
            return self._snapshot[0]
        return None


def _coerce_event_record(event: dict[str, Any] | SovereignEvent) -> dict[str, Any]:
    """Convert event to dict, handling Pydantic v1 and v2."""
    if hasattr(event, "model_dump"):
        # Pydantic v2: check if mode parameter is supported
        sig = inspect.signature(event.model_dump)
        if "mode" in sig.parameters:
            payload = event.model_dump(mode="json")
        else:
            # Pydantic v1 fallback
            payload = event.model_dump()
        return dict(payload)
    return dict(event)

# ---------------------------------------------------------------------------
# Abstract contract
# ---------------------------------------------------------------------------


class LedgerBackend(ABC):
    """Pluggable storage tier for ExecutionLedger events."""

    @abstractmethod
    def append(self, event: dict[str, Any] | SovereignEvent, *, committed: bool = False) -> None:
        """Persist event.  When committed=True, must not return until durable."""

    def append_events(
        self,
        events: list[dict[str, Any]] | list[SovereignEvent],
        *,
        committed: bool = False,
    ) -> int:
        count = 0
        for event in list(events or []):
            self.append(event, committed=committed)
            count += 1
        return count

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

    def append(self, event: dict[str, Any] | SovereignEvent, *, committed: bool = False) -> None:
        with self._lock:
            self._events.append(deepcopy(_coerce_event_record(event)))

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
    - load() refuses malformed records and incomplete tails rather than
      replaying a state with missing events.

    Crash recovery:
    - On startup, call ExecutionLedger.load_from_backend() to repopulate the
      in-memory sequence from the file.
    """

    #: Event types that always trigger an fsync regardless of the committed flag.
    COMMITTED_TYPES: frozenset[str] = frozenset(
        {
            "JOB_QUEUED",
            "JOB_STARTED",
            "JOB_COMPLETED",
            "JOB_FAILED",
            "SESSION_STATE_UPDATED",
        },
    )

    def __init__(
        self, path: str | Path, *, fsync: bool = True,
        security_mode: str | None = None, tenant_id: str | None = None,
    ) -> None:
        if security_mode not in (None, "AUTHENTICATED", "TRUSTED_LOCAL"):
            raise ValueError("Invalid WAL security mode")
        if security_mode == "AUTHENTICATED":
            from dadbot_system.delegation import require_tenant_authority
            from dadbot_system.execution_identity import require_tenant_id

            tenant_id = require_tenant_id(tenant_id)
            require_tenant_authority(tenant_id)
        self._path = Path(path)
        self._security_mode = security_mode
        self._tenant_id = tenant_id
        self._known_binding: dict[str, Any] | None = None
        self._fsync_enabled = bool(fsync)
        self._lock = threading.RLock()
        self._append_verifier = _WALAppendVerifier()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        from dadbot.core.durability import FileLockMutex
        from dadbot_system.delegation import tenant_mutation_fence

        fence = (
            tenant_mutation_fence(tenant_id, required=True)
            if security_mode == "AUTHENTICATED" and tenant_id is not None
            else nullcontext()
        )
        with fence, FileLockMutex(self._path.with_name(self._path.name + ".lock")).locked():
            if not self._path.exists():
                if self._binding_path.exists():
                    raise WALCorruptionError("Execution WAL is missing but its binding exists")
                if self._authenticated_marker_path.exists():
                    raise WALCorruptionError("Execution WAL is missing but its authenticated marker exists")
                if any(self._path.parent.glob(self._path.name + ".tenant-*")):
                    raise WALCorruptionError("Execution WAL is missing but tenant establishment evidence exists")
                if security_mode == "AUTHENTICATED":
                    self._create_authenticated_wal(tenant_id)
                else:
                    self._path.touch()
            self._read_binding()

    @property
    def _binding_path(self) -> Path:
        return self._path.with_name(self._path.name + ".binding")

    @property
    def _authenticated_marker_path(self) -> Path:
        return self._path.with_name(self._path.name + ".authenticated")

    def _create_authenticated_wal(self, tenant_id: str | None) -> None:
        from dadbot_system.delegation import _execution_authority, require_tenant_authority
        from dadbot_system.execution_identity import require_tenant_id
        from dadbot_system.security import AuthorizationError

        owner = require_tenant_id(tenant_id)
        require_tenant_authority(owner)
        authority = _execution_authority.get()
        if authority is None:
            raise AuthorizationError("Delegated execution authority is missing")
        store, _ = authority
        binding = {
            "mode": "AUTHENTICATED",
            "tenant_id": owner,
            "storage_id": uuid4().hex,
            "version": 2,
            "authority_id": store.storage_identity(),
        }
        payload = json.dumps(binding, sort_keys=True, separators=(",", ":"))
        signature = store._signature("wal-binding-v1:" + payload)
        header = json.dumps({"wal_binding": binding, "signature": signature}) + "\n"
        with self._authenticated_marker_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(binding["storage_id"] + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        with self._binding_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(header)
            handle.flush()
            os.fsync(handle.fileno())
        with self._path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(header)
            handle.flush()
            os.fsync(handle.fileno())

    def _read_binding(self) -> bool:
        if not self._path.exists():
            raise WALCorruptionError("Execution WAL is missing")
        with self._path.open("r", encoding="utf-8") as handle:
            first = handle.readline()
        try:
            record = json.loads(first) if first else None
        except ValueError as exc:
            raise WALCorruptionError("Invalid WAL header or first record") from exc
        if not isinstance(record, dict) or "wal_binding" not in record:
            if self._security_mode == "AUTHENTICATED" or self._known_binding is not None or self._binding_path.exists():
                raise WALCorruptionError("Legacy unbound WAL requires an explicit ownership transition")
            if self._authenticated_marker_path.exists():
                raise WALCorruptionError("Authenticated WAL security binding is missing")
            return False
        if not self._binding_path.exists():
            raise WALCorruptionError("Execution WAL binding anchor is missing")
        try:
            with self._binding_path.open("r", encoding="utf-8") as handle:
                anchor = json.load(handle)
        except (OSError, ValueError) as exc:
            raise WALCorruptionError("Execution WAL binding anchor is invalid") from exc
        if anchor != record:
            raise WALCorruptionError("Execution WAL binding does not match its anchor")
        self._verify_binding(record)
        return True

    def _verify_binding(self, record: dict[str, Any]) -> None:
        from dadbot_system.delegation import _execution_authority, require_tenant_authority
        from dadbot_system.security import AuthorizationError

        binding = record["wal_binding"]
        if not isinstance(binding, dict) or set(binding) != {
            "mode", "tenant_id", "storage_id", "version", "authority_id"
        } or binding.get("mode") != "AUTHENTICATED" or binding.get("version") != 2:
            raise WALCorruptionError("Invalid WAL security binding")
        if not isinstance(binding.get("storage_id"), str) or len(binding["storage_id"]) != 32:
            raise WALCorruptionError("Invalid WAL storage identity")
        if self._security_mode == "TRUSTED_LOCAL":
            raise AuthorizationError("Authenticated WAL cannot be opened in trusted-local mode")
        owner = binding.get("tenant_id")
        if not isinstance(owner, str):
            raise WALCorruptionError("Invalid WAL tenant binding")
        if self._tenant_id is not None and self._tenant_id != owner:
            raise AuthorizationError("WAL tenant binding mismatch")
        require_tenant_authority(owner)
        authority = _execution_authority.get()
        if authority is None:
            raise AuthorizationError("Delegated execution authority is missing")
        store, _ = authority
        if store.storage_identity() != binding.get("authority_id"):
            raise AuthorizationError("WAL authority store mismatch")
        payload = json.dumps(binding, sort_keys=True, separators=(",", ":"))
        signature = record.get("signature")
        if not isinstance(signature, str) or not hmac.compare_digest(
            signature, store._signature("wal-binding-v1:" + payload)
        ):
            raise WALCorruptionError("Invalid WAL binding signature")
        if self._known_binding is not None and self._known_binding != binding:
            raise WALCorruptionError("WAL binding changed")
        self._known_binding = dict(binding)

    def append(self, event: dict[str, Any] | SovereignEvent, *, committed: bool = False) -> None:
        serialized = _coerce_event_record(event)
        self._append_authorized(serialized, committed=committed)

    def _append_authorized(self, serialized: dict[str, Any], *, committed: bool) -> None:
        from dadbot_system.delegation import _execution_authority, tenant_mutation_fence

        authority = _execution_authority.get()
        fence = tenant_mutation_fence(authority[1].tenant_id, required=True) if authority is not None else nullcontext()
        with fence:
            if "wal_binding" in serialized:
                raise WALCorruptionError("WAL binding cannot be appended as an event")
            line = json.dumps(serialized, default=str) + "\n"
            serialized = json.loads(line)
            self._authorize_event(serialized)
            should_fsync = self._fsync_enabled and (
                committed or str(serialized.get("type") or "") in self.COMMITTED_TYPES
            )
            from dadbot.core.durability import FileLockMutex

            with self._lock, FileLockMutex(self._path.with_name(self._path.name + ".lock")).locked():
                self._read_binding()
                _require_complete_wal_tail(self._path)
                reuse = authority is None and type(self) is FileWALLedgerBackend
                events, prefix_bytes = self._append_verifier.validate(
                    self._path, self._load_unlocked, serialized, reuse=reuse,
                )
                with open(self._path, "a", encoding="utf-8", newline="") as handle:
                    handle.write(line)
                    handle.flush()
                    if should_fsync:
                        os.fsync(handle.fileno())
                self._append_verifier.record(prefix_bytes, line, events, serialized, reuse=reuse)

    @staticmethod
    def _authorize_event(event: dict[str, Any]) -> None:
        from dadbot_system.delegation import _execution_authority, require_tenant_authority
        from dadbot_system.security import AuthorizationError

        authority = _execution_authority.get()
        if authority is None:
            return
        _, task = authority
        require_tenant_authority(task.tenant_id)
        payload = event.get("payload")
        tenant = payload.get("tenant_id") if isinstance(payload, dict) else None
        if event.get("type") == "TENANT_MEMORY_COMMITTED" and not isinstance(tenant, str):
            raise AuthorizationError("Tenant memory event lacks a tenant identity")
        if tenant is not None:
            require_tenant_authority(tenant)
        tenant = event.get("tenant_id")
        if tenant is not None:
            require_tenant_authority(tenant)

    def load(self) -> list[dict[str, Any]]:
        from dadbot.core.durability import FileLockMutex

        with self._lock, FileLockMutex(self._path.with_name(self._path.name + ".lock")).locked():
            return self._load_unlocked()

    def load_if_changed(self, digest: bytes | None) -> tuple[bytes | None, list[dict[str, Any]] | None]:
        """Reuse a snapshot only when all persisted bytes and live authority still match."""
        from dadbot.core.durability import FileLockMutex

        with self._lock, FileLockMutex(self._path.with_name(self._path.name + ".lock")).locked():
            return self._load_if_changed_unlocked(digest)

    def _load_if_changed_unlocked(self, digest: bytes | None) -> tuple[bytes, list[dict[str, Any]] | None]:
        from dadbot_system.delegation import _execution_authority

        self._authorize_event({})
        self._read_binding()
        current = hashlib.sha256(self._path.read_bytes()).digest()
        if _execution_authority.get() is not None:
            return current, self._load_unlocked()
        return current, None if digest == current else self._load_unlocked()

    def _load_unlocked(self) -> list[dict[str, Any]]:
        self._authorize_event({})
        bound = self._read_binding()
        if not self._path.exists():
            return []
        events: list[dict[str, Any]] = []
        with open(self._path, encoding="utf-8", newline="") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.endswith("\n"):
                    raise WALCorruptionError(f"Incomplete WAL tail at line {line_number} in {self._path}")
                line = raw_line.strip()
                if not line:
                    raise WALCorruptionError(f"Empty WAL record at line {line_number} in {self._path}")
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise WALCorruptionError(f"Corrupt WAL record at line {line_number} in {self._path}") from exc
                if not isinstance(event, dict):
                    raise WALCorruptionError(f"Invalid WAL record at line {line_number} in {self._path}")
                if line_number == 1 and bound:
                    continue
                if "wal_binding" in event:
                    raise WALCorruptionError("Unexpected WAL binding record")
                self._authorize_event(event)
                events.append(event)
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

    Corrupt records and incomplete tails fail closed. A plain-JSON fallback
    remains available for legacy lines that pre-date the CRC format.

    This backend satisfies Tier 0 item 1: corruption detection at the storage
    layer with partial-write recovery.
    """

    COMMITTED_TYPES: frozenset[str] = FileWALLedgerBackend.COMMITTED_TYPES

    def __init__(self, path: str | Path, *, fsync: bool = True) -> None:
        self._path = Path(path)
        self._fsync_enabled = bool(fsync)
        self._lock = threading.RLock()
        self._append_verifier = _WALAppendVerifier()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()

    def append(self, event: dict[str, Any] | SovereignEvent, *, committed: bool = False) -> None:
        from dadbot.core.durability import CRC32LineCodec

        serialized = json.loads(json.dumps(_coerce_event_record(event), default=str))
        line = CRC32LineCodec.encode(serialized)
        should_fsync = self._fsync_enabled and (
            committed or str(serialized.get("type") or "") in self.COMMITTED_TYPES
        )
        from dadbot.core.durability import FileLockMutex

        with self._lock, FileLockMutex(self._path.with_name(self._path.name + ".lock")).locked():
            _require_complete_wal_tail(self._path)
            from dadbot_system.delegation import _execution_authority

            reuse = _execution_authority.get() is None and type(self) is CRCFileWALLedgerBackend
            events, prefix_bytes = self._append_verifier.validate(
                self._path, self._load_unlocked, serialized, reuse=reuse,
            )
            with open(self._path, "a", encoding="utf-8", newline="") as handle:
                handle.write(line)
                handle.flush()
                if should_fsync:
                    os.fsync(handle.fileno())
            self._append_verifier.record(prefix_bytes, line, events, serialized, reuse=reuse)

    def load(self) -> list[dict[str, Any]]:
        from dadbot.core.durability import FileLockMutex

        with self._lock, FileLockMutex(self._path.with_name(self._path.name + ".lock")).locked():
            return self._load_unlocked()

    def load_if_changed(self, digest: bytes | None) -> tuple[bytes | None, list[dict[str, Any]] | None]:
        from dadbot.core.durability import FileLockMutex

        with self._lock, FileLockMutex(self._path.with_name(self._path.name + ".lock")).locked():
            current = hashlib.sha256(self._path.read_bytes()).digest()
            return current, None if digest == current else self._load_unlocked()

    def _load_unlocked(self) -> list[dict[str, Any]]:
        from dadbot.core.durability import CRC32LineCodec

        if not self._path.exists():
            return []
        events: list[dict[str, Any]] = []
        with open(self._path, encoding="utf-8", newline="") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.endswith("\n"):
                    raise WALCorruptionError(f"Incomplete WAL tail at line {line_number} in {self._path}")
                event = CRC32LineCodec.decode(raw)
                if not isinstance(event, dict):
                    raise WALCorruptionError(f"Corrupt WAL record at line {line_number} in {self._path}")
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
                    f"Event #{i} sequence {seq} is not > previous {prev_seq} (type={event.get('type')!r})",
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
        if not isinstance(inner, FileWALLedgerBackend) and not isinstance(
            inner,
            CRCFileWALLedgerBackend,
        ):
            raise TypeError(
                "StrongConsistencyBackend requires a file-based backend "
                "(FileWALLedgerBackend or CRCFileWALLedgerBackend)",
            )
        self._inner = inner

    def append(self, event: dict[str, Any] | SovereignEvent, *, committed: bool = False) -> None:
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
        self._flush_uncertain = False

    def append(self, event: dict[str, Any] | SovereignEvent, *, committed: bool = False) -> None:
        with self._lock:
            _require_safe_buffer_flush(self._flush_uncertain)
            self._buffer.append((deepcopy(_coerce_event_record(event)), committed))
            if committed or len(self._buffer) >= self._buffer_size:
                self._flush_locked()

    def flush(self) -> int:
        """Flush buffer to inner backend.  Returns number of events flushed."""
        with self._lock:
            return self._flush_locked()

    def _flush_locked(self) -> int:
        _require_safe_buffer_flush(self._flush_uncertain)
        count = len(self._buffer)
        self._flush_uncertain = True
        for ev, committed in self._buffer:
            self._inner.append(ev, committed=committed)
        self._buffer.clear()
        self._flush_uncertain = False
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
        self._inner = inner
        self._batch_size = max(1, int(batch_size))
        self._buffer: list[tuple[dict[str, Any], bool]] = []
        self._lock = threading.RLock()
        self._flush_uncertain = False

    def append(self, event: dict[str, Any] | SovereignEvent, *, committed: bool = False) -> None:
        with self._lock:
            _require_safe_buffer_flush(self._flush_uncertain)
            self._buffer.append((deepcopy(_coerce_event_record(event)), committed))
            if committed or len(self._buffer) >= self._batch_size:
                self._flush_locked()

    def flush(self) -> int:
        with self._lock:
            return self._flush_locked()

    def _flush_locked(self) -> int:
        _require_safe_buffer_flush(self._flush_uncertain)
        count = len(self._buffer)
        self._flush_uncertain = True
        for ev, committed in self._buffer:
            self._inner.append(ev, committed=committed)
        self._buffer.clear()
        self._flush_uncertain = False
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
