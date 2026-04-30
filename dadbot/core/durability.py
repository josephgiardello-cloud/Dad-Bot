"""Crash-safe write semantics and transactional write units.

CRC32LineCodec:
  Encodes/decodes JSONL lines with a CRC-32 integrity checksum so that
  partial writes (crash mid-line) are detected and discarded on reload.

AtomicWriteUnit:
  Groups multiple ledger writes, session updates, and checkpoint saves into a
  single commit boundary.  Uses UNIT_BEGIN / UNIT_COMMIT sentinel events so
  the replay path can discard uncommitted units on restart.

FileLockMutex:
  Process-level mutual exclusion via an OS-level lock file.  Uses
  os.O_CREAT | os.O_EXCL for atomic creation (portable on Windows + POSIX).
  Embeds a fencing token (PID + UUID) so stale locks from crashed processes
  can be detected and evicted.
"""

from __future__ import annotations

import binascii
import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any

# Sentinel event types used by AtomicWriteUnit.
UNIT_BEGIN_TYPE: str = "__UNIT_BEGIN__"
UNIT_COMMIT_TYPE: str = "__UNIT_COMMIT__"
DEBUG_LOCKS: bool = str(os.getenv("DADBOT_DEBUG_LOCKS", "0")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
logger = logging.getLogger(__name__)


def _debug_lock_event(event: str, *, path: Path) -> None:
    # Keep lock diagnostics latent and low-noise unless explicitly enabled.
    if DEBUG_LOCKS or logger.isEnabledFor(logging.DEBUG):
        logger.debug(event, extra={"path": str(path)})


# ---------------------------------------------------------------------------
# CRC-32 line codec
# ---------------------------------------------------------------------------


class CRC32LineCodec:
    """Encode/decode JSONL lines with CRC-32 checksums for corruption detection.

    Line format: ``<crc32_hex_8chars> <json_payload>\n``

    - On encode: compute CRC-32 of the serialised payload, prepend as 8-char hex.
    - On decode: verify CRC; return None if corrupt/partial (caller should skip).
    - Backward-compat: lines without a CRC prefix (8-char hex + space) are
      treated as plain JSON â€” this allows reading old WAL files.
    """

    @staticmethod
    def encode(event: dict[str, Any]) -> str:
        payload = json.dumps(event, default=str)
        crc = binascii.crc32(payload.encode("utf-8")) & 0xFFFF_FFFF
        return f"{crc:08x} {payload}\n"

    @staticmethod
    def decode(line: str) -> dict[str, Any] | None:
        """Parse a line.  Returns None if corrupt or empty."""
        stripped = line.strip()
        if not stripped:
            return None

        # Try CRC format first.
        parts = stripped.split(" ", 1)
        if len(parts) == 2 and len(parts[0]) == 8:
            crc_hex, payload_str = parts
            try:
                expected = int(crc_hex, 16)
                actual = binascii.crc32(payload_str.encode("utf-8")) & 0xFFFF_FFFF
                if actual != expected:
                    return None
                return json.loads(payload_str)
            except (ValueError, json.JSONDecodeError):
                pass

        # Fallback: plain JSON (legacy lines without CRC).
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def is_corrupt(line: str) -> bool:
        return CRC32LineCodec.decode(line) is None and bool(line.strip())


# ---------------------------------------------------------------------------
# Process-level file lock (cross-process mutual exclusion)
# ---------------------------------------------------------------------------


class FileLockMutex:
    """Process-level mutual exclusion using an OS-level lock file.

    The lock file contains a JSON record with ``pid``, ``token``, and
    ``acquired_at``.  If the lock file exists but the owning PID is no longer
    alive, the lock is considered stale and is evicted automatically.

    Note: This provides *advisory* locking â€” it protects well-behaved
    processes.  It does NOT replace a distributed coordinator (etcd,
    PostgreSQL advisory locks) for strong multi-node guarantees.
    """

    DEFAULT_STALE_SECONDS: float = 60.0

    def __init__(
        self,
        lock_path: str | Path,
        *,
        stale_after_seconds: float = DEFAULT_STALE_SECONDS,
    ) -> None:
        self._path = Path(lock_path)
        self._stale_after = max(1.0, float(stale_after_seconds))
        self._token: str = ""
        self._lock = RLock()

    # ------------------------------------------------------------------
    # Acquire / release
    # ------------------------------------------------------------------

    def acquire(self, *, timeout_seconds: float = 5.0) -> str:
        """Acquire the lock.  Returns the fencing token.

        Raises RuntimeError if the lock cannot be acquired within timeout.
        """
        _debug_lock_event("lock_acquire_start", path=self._path)
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        self._path.parent.mkdir(parents=True, exist_ok=True)

        while time.monotonic() < deadline:
            # Evict stale lock if detected.
            self._evict_if_stale()

            token = uuid.uuid4().hex
            record = json.dumps(
                {
                    "pid": os.getpid(),
                    "token": token,
                    "acquired_at": time.time(),
                },
            )

            try:
                fd = os.open(
                    str(self._path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                try:
                    os.write(fd, record.encode("utf-8"))
                finally:
                    os.close(fd)
                with self._lock:
                    self._token = token
                _debug_lock_event("lock_acquired", path=self._path)
                return token
            except FileExistsError:
                time.sleep(0.05)

        raise RuntimeError(
            f"FileLockMutex: could not acquire lock at {self._path} within {timeout_seconds:.1f}s",
        )

    def release(self, token: str) -> bool:
        """Release the lock only if we hold it (token matches).  Returns True on success."""
        with self._lock:
            if self._token != token:
                return False
            try:
                current = self._read_lock_record()
                if current and current.get("token") == token:
                    self._path.unlink(missing_ok=True)
                self._token = ""
                _debug_lock_event("lock_released", path=self._path)
                return True
            except OSError:
                return False

    @contextmanager
    def locked(self, *, timeout_seconds: float = 5.0):
        """Context manager: acquire on enter, release on exit."""
        token = self.acquire(timeout_seconds=timeout_seconds)
        try:
            yield token
        finally:
            self.release(token)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_lock_record(self) -> dict[str, Any] | None:
        try:
            raw = self._path.read_text(encoding="utf-8")
            return json.loads(raw)
        except (OSError, json.JSONDecodeError):
            return None

    def _evict_if_stale(self) -> bool:
        record = self._read_lock_record()
        if record is None:
            return False
        acquired_at = float(record.get("acquired_at") or 0)
        if (time.time() - acquired_at) < self._stale_after:
            # Also check if PID is alive.
            pid = int(record.get("pid") or 0)
            if pid > 0 and _pid_is_alive(pid):
                return False
        # Stale â€” remove it.
        try:
            self._path.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    @property
    def is_held(self) -> bool:
        record = self._read_lock_record()
        if record is None:
            return False
        acquired_at = float(record.get("acquired_at") or 0)
        if (time.time() - acquired_at) >= self._stale_after:
            return False
        pid = int(record.get("pid") or 0)
        return pid > 0 and _pid_is_alive(pid)

    @property
    def fencing_token(self) -> str:
        with self._lock:
            return self._token


def _pid_is_alive(pid: int) -> bool:
    """Return True if a process with that PID is still running.

    POSIX: uses ``os.kill(pid, 0)`` (existence check, signal never delivered).
    Windows: uses Win32 ``OpenProcess``/``GetExitCodeProcess`` via ctypes.
    ``os.kill(pid, 0)`` maps to ``CTRL_C_EVENT`` (value 0) on Windows and
    would send Ctrl+C to the entire process group — never use it on Windows.
    """
    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION,
            False,
            pid,
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong(0)
            if ctypes.windll.kernel32.GetExitCodeProcess(
                handle,
                ctypes.byref(exit_code),
            ):
                return exit_code.value == STILL_ACTIVE
            return False
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # POSIX: signal 0 checks existence without delivering.
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


# ---------------------------------------------------------------------------
# Atomic write unit â€” transactional multi-step ledger operations
# ---------------------------------------------------------------------------


class AtomicWriteUnit:
    """Transactional write boundary spanning ledger + session + checkpoint.

    Wraps writes in UNIT_BEGIN / UNIT_COMMIT sentinel events.  On replay,
    ``filter_committed()`` discards events from units that never committed
    (interrupted by a crash between BEGIN and COMMIT).

    Usage::

        unit = AtomicWriteUnit(writer, session_store=store, checkpoint=ckpt)
        with unit.transaction() as txn:
            txn.write_event(event_type="JOB_QUEUED", session_id="s1",
                            kernel_step_id="scheduler.enqueue")
            txn.apply_session("s1", {"status": "running"})
            txn.save_checkpoint("post-enqueue")
        # UNIT_COMMIT is written on clean exit; rollback is implicit on exception.
    """

    def __init__(
        self,
        ledger_writer,
        *,
        session_store=None,
        checkpoint=None,
    ) -> None:
        self._writer = ledger_writer
        self._session_store = session_store
        self._checkpoint = checkpoint

    def transaction(self) -> _Transaction:
        return _Transaction(self._writer, self._session_store, self._checkpoint)

    @staticmethod
    def filter_committed(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Replay filter: removes events from incomplete (uncommitted) write units.

        Call this on the event list returned by ``load_from_backend()`` before
        restoring in-memory state to discard partial transactions.
        """
        committed_units: set[str] = set()
        began_units: set[str] = set()

        for event in events:
            etype = event.get("type", "")
            uid = (event.get("payload") or {}).get("unit_id", "")
            if etype == UNIT_COMMIT_TYPE and uid:
                committed_units.add(uid)
            elif etype == UNIT_BEGIN_TYPE and uid:
                began_units.add(uid)

        orphaned = began_units - committed_units
        if not orphaned:
            return events

        result = []
        for event in events:
            etype = event.get("type", "")
            # Sentinel events for orphaned units are dropped.
            if etype in (UNIT_BEGIN_TYPE, UNIT_COMMIT_TYPE):
                if (event.get("payload") or {}).get("unit_id", "") in orphaned:
                    continue
            # Payload events tagged with orphaned unit_id are dropped.
            elif (event.get("payload") or {}).get("_unit_id", "") in orphaned:
                continue
            result.append(event)
        return result


class _Transaction:
    """Context manager returned by AtomicWriteUnit.transaction()."""

    def __init__(self, writer, session_store, checkpoint) -> None:
        self._writer = writer
        self._session_store = session_store
        self._checkpoint = checkpoint
        self._unit_id = uuid.uuid4().hex
        self._committed = False

    def __enter__(self) -> _Transaction:
        self._writer.write_event(
            event_type=UNIT_BEGIN_TYPE,
            session_id="__system__",
            kernel_step_id="atomic_write_unit.begin",
            trace_id=self._unit_id,
            payload={"unit_id": self._unit_id, "ts": time.time()},
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self._writer.write_event(
                event_type=UNIT_COMMIT_TYPE,
                session_id="__system__",
                kernel_step_id="atomic_write_unit.commit",
                trace_id=self._unit_id,
                payload={"unit_id": self._unit_id, "ts": time.time()},
            )
            self._committed = True
        # On exception: no UNIT_COMMIT written.
        # filter_committed() on next load will discard the orphaned events.

    def write_event(self, **kwargs) -> dict[str, Any]:
        """Write an event tagged with this transaction's unit_id."""
        payload = dict(kwargs.pop("payload", None) or {})
        payload["_unit_id"] = self._unit_id
        # Propagate transaction identity as trace lineage when caller omits it.
        if "trace_id" not in kwargs:
            kwargs["trace_id"] = self._unit_id
        return self._writer.write_event(**kwargs, payload=payload)

    def apply_session(self, session_id: str, mutation: dict[str, Any]) -> None:
        if self._session_store is not None:
            self._session_store.apply_event({"session_id": session_id, **mutation})

    def save_checkpoint(self, label: str = "") -> None:
        if self._checkpoint is not None:
            self._checkpoint.save(label=label or "atomic_write_unit")

    @property
    def committed(self) -> bool:
        return self._committed

    @property
    def unit_id(self) -> str:
        return self._unit_id
