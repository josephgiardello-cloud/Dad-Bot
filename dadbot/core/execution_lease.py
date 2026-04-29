from __future__ import annotations

import threading
import time
from typing import Any


class LeaseConflictError(RuntimeError):
    """Raised when a different worker already owns the active lease."""


class ExecutionLease:
    """Session-scoped in-memory lease with fencing tokens."""

    def __init__(self) -> None:
        self._leases: dict[str, dict[str, Any]] = {}
        self._fencing_counters: dict[str, int] = {}
        self._lock = threading.RLock()

    def _expired(self, lease: dict[str, Any], now: float) -> bool:
        return float(lease.get("expires_at", 0.0) or 0.0) <= now

    def acquire(self, session_id: str, owner_id: str, ttl_seconds: float = 30.0) -> dict[str, Any]:
        sid = str(session_id or "").strip() or "default"
        owner = str(owner_id or "").strip() or "worker"
        ttl = float(ttl_seconds or 30.0)
        now = time.monotonic()

        with self._lock:
            current = self._leases.get(sid)
            if current and not self._expired(current, now):
                current_owner = str(current.get("owner_id") or "")
                if current_owner and current_owner != owner:
                    raise LeaseConflictError(
                        f"lease conflict: session_id={sid!r} owned_by={current_owner!r} requested_by={owner!r}"
                    )

            if current and str(current.get("owner_id") or "") == owner and not self._expired(current, now):
                token = int(current.get("fencing_token") or 1)
            else:
                token = int(self._fencing_counters.get(sid, 0)) + 1
                self._fencing_counters[sid] = token

            lease = {
                "session_id": sid,
                "owner_id": owner,
                "ttl_seconds": ttl,
                "acquired_at": now,
                "expires_at": now + max(0.001, ttl),
                "fencing_token": token,
            }
            self._leases[sid] = lease
            return dict(lease)

    def release(self, session_id: str, owner_id: str) -> bool:
        sid = str(session_id or "").strip() or "default"
        owner = str(owner_id or "").strip() or "worker"
        with self._lock:
            current = self._leases.get(sid)
            if not current:
                return False
            if str(current.get("owner_id") or "") != owner:
                return False
            self._leases.pop(sid, None)
            return True
