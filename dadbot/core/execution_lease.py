from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from threading import RLock
from typing import Any


class LeaseConflictError(RuntimeError):
    pass


class ExecutionLease:
    """Per-session single-writer ownership lease.

    Contract:
    - Only one holder (worker) may hold the lease for a given session at a time.
    - Leases have a TTL; stale leases are automatically expired before acquisition.
    - The owner must pass its lease_id when releasing or renewing to prevent
      unintentional release by a different worker.
    - Kernel/scheduler checks require_session_lease() before executing a job;
      this raises LeaseConflictError if the session is already owned by another holder.
    """

    DEFAULT_TTL_SECONDS: float = 30.0

    def __init__(self, *, default_ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self._lock = RLock()
        self._leases: dict[str, dict[str, Any]] = {}
        self._default_ttl = max(1.0, float(default_ttl_seconds or self.DEFAULT_TTL_SECONDS))
        self._fencing_counters: dict[str, int] = {}  # session_id â†’ monotonic token

    def _next_fencing_token(self, session_id: str) -> int:
        """Return a monotonically increasing fencing token for session_id.

        The token increments on every new acquisition (not on renewal).
        Callers can embed the token in write operations to detect and reject
        writes from stale lease holders.
        """
        current = self._fencing_counters.get(session_id, 0) + 1
        self._fencing_counters[session_id] = current
        return current

    def fencing_token_for(self, session_id: str) -> int | None:
        """Return the current fencing token for session_id, or None if not held."""
        with self._lock:
            lease = self._leases.get(str(session_id or "").strip())
            if lease is None:
                return None
            return lease.get("fencing_token")

    # ------------------------------------------------------------------
    # Acquire / Release / Renew
    # ------------------------------------------------------------------

    def acquire(
        self,
        *,
        session_id: str,
        owner_id: str | None = None,
        ttl_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Acquire the execution lease for session_id.

        If a non-expired lease already exists for a *different* owner, raises LeaseConflictError.
        If the caller is renewing its own lease, the TTL is extended.
        """
        normalized = str(session_id or "").strip()
        if not normalized:
            raise ValueError("session_id must not be empty")

        holder = str(owner_id or uuid.uuid4().hex)
        ttl = float(ttl_seconds or self._default_ttl)
        now = time.monotonic()
        expires_at = now + ttl

        with self._lock:
            existing = self._leases.get(normalized)
            if existing is not None:
                if existing["expires_at"] > now:
                    if existing["owner_id"] != holder:
                        raise LeaseConflictError(
                            f"Session {normalized!r} already owned by {existing['owner_id']!r}; "
                            f"lease expires in {existing['expires_at'] - now:.1f}s"
                        )
                    # Renew in-place.
                    existing["expires_at"] = expires_at
                    existing["renewed_at"] = now
                    existing["ttl_seconds"] = ttl
                    return dict(existing)

            lease = {
                "session_id": normalized,
                "owner_id": holder,
                "lease_id": uuid.uuid4().hex,
                "acquired_at": now,
                "renewed_at": now,
                "expires_at": expires_at,
                "ttl_seconds": ttl,
                "fencing_token": self._next_fencing_token(normalized),
            }
            self._leases[normalized] = lease
            return dict(lease)

    def release(self, *, session_id: str, owner_id: str) -> bool:
        """Release the lease. Returns True if released, False if already gone/expired."""
        normalized = str(session_id or "").strip()
        with self._lock:
            existing = self._leases.get(normalized)
            if existing is None:
                return False
            if existing["owner_id"] != str(owner_id or ""):
                return False
            del self._leases[normalized]
            return True

    def renew(self, *, session_id: str, owner_id: str, ttl_seconds: float | None = None) -> dict[str, Any]:
        """Extend the lease TTL without releasing and reacquiring."""
        return self.acquire(session_id=session_id, owner_id=owner_id, ttl_seconds=ttl_seconds)

    # ------------------------------------------------------------------
    # Enforcement
    # ------------------------------------------------------------------

    def require_session_lease(self, *, session_id: str, owner_id: str) -> None:
        """Raise LeaseConflictError if session_id is owned by a different worker."""
        normalized = str(session_id or "").strip()
        now = time.monotonic()
        with self._lock:
            existing = self._leases.get(normalized)
            if existing is None:
                return  # no lease held â€” caller may proceed
            if existing["expires_at"] <= now:
                # Expired lease â€” evict and allow
                del self._leases[normalized]
                return
            if existing["owner_id"] != str(owner_id or ""):
                raise LeaseConflictError(
                    f"Execution ownership conflict for session {normalized!r}: "
                    f"held by {existing['owner_id']!r}, attempted by {str(owner_id)!r}"
                )

    def evict_expired(self) -> list[str]:
        """Remove all expired leases and return their session IDs."""
        now = time.monotonic()
        with self._lock:
            expired = [
                session_id
                for session_id, lease in list(self._leases.items())
                if lease["expires_at"] <= now
            ]
            for session_id in expired:
                del self._leases[session_id]
        return expired

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def is_held(self, *, session_id: str) -> bool:
        normalized = str(session_id or "").strip()
        now = time.monotonic()
        with self._lock:
            existing = self._leases.get(normalized)
            return existing is not None and existing["expires_at"] > now

    def owner_of(self, *, session_id: str) -> str | None:
        normalized = str(session_id or "").strip()
        now = time.monotonic()
        with self._lock:
            existing = self._leases.get(normalized)
            if existing is None or existing["expires_at"] <= now:
                return None
            return str(existing["owner_id"])

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            active = [
                {
                    "session_id": lease["session_id"],
                    "owner_id": lease["owner_id"],
                    "ttl_remaining": max(0.0, lease["expires_at"] - now),
                    "fencing_token": lease.get("fencing_token"),
                }
                for lease in self._leases.values()
                if lease["expires_at"] > now
            ]
        return {"active_lease_count": len(active), "leases": active}


# ---------------------------------------------------------------------------
# Worker identity â€” persistent process ID across restarts
# ---------------------------------------------------------------------------

class WorkerIdentity:
    """Persists a stable worker identity (UUID) across process restarts.

    On first startup, generates a UUID and writes it to ``identity_path``.
    On subsequent startups, loads the existing UUID.

    Also detects crashed previous workers by comparing the stored PID against
    the live process table.  If the previous PID is no longer alive, the
    identity file is refreshed with the current PID.

    Usage::

        identity = WorkerIdentity("runtime/.worker_identity.json")
        worker_id = identity.worker_id  # stable UUID string
        if identity.previous_crashed:
            logger.warning("Previous worker %s crashed", identity.previous_worker_id)
    """

    def __init__(self, identity_path: str | Path) -> None:
        self._path = Path(identity_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._worker_id: str = ""
        self._previous_worker_id: str = ""
        self._previous_crashed: bool = False
        self._load_or_create()

    def _load_or_create(self) -> None:
        existing = self._read()
        if existing:
            prev_pid = int(existing.get("pid") or 0)
            if prev_pid and prev_pid != os.getpid() and not _pid_is_alive(prev_pid):
                # Previous worker crashed â€” preserve its ID for diagnostics.
                self._previous_worker_id = str(existing.get("worker_id") or "")
                self._previous_crashed = True
            elif prev_pid and prev_pid == os.getpid():
                # Same process (e.g. test re-use) â€” reuse identity.
                self._worker_id = str(existing.get("worker_id") or "")
                return
            elif prev_pid and _pid_is_alive(prev_pid):
                # Another live process owns this identity â€” generate a new one.
                self._worker_id = uuid.uuid4().hex
                self._write()
                return

        if not self._worker_id:
            self._worker_id = str(existing.get("worker_id") or "") if existing else ""
        if not self._worker_id:
            self._worker_id = uuid.uuid4().hex
        self._write()

    def _read(self) -> dict[str, Any] | None:
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write(self) -> None:
        record = {
            "worker_id": self._worker_id,
            "pid": os.getpid(),
            "started_at": time.time(),
        }
        self._path.write_text(json.dumps(record), encoding="utf-8")

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def previous_worker_id(self) -> str:
        return self._previous_worker_id

    @property
    def previous_crashed(self) -> bool:
        return self._previous_crashed


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False
