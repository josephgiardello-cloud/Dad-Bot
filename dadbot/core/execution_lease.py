from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from dadbot.core.tenant_session import resolve_tenant_id


class LeaseConflictError(RuntimeError):
    """Raised when a different worker already owns the active lease."""


class WorkerIdentity:
    """Persist a stable worker id to disk for multi-process lease tests."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            data = json.loads(self._path.read_text(encoding="utf-8") or "{}")
        else:
            data = {
                "worker_id": f"worker-{uuid4().hex}",
                "pid": __import__("os").getpid(),
            }
            self._path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        self.worker_id = str(data.get("worker_id") or f"worker-{uuid4().hex}")


class ExecutionLease:
    """Session-scoped in-memory lease with fencing tokens."""

    DEFAULT_TTL_SECONDS = 30.0

    def __init__(self, default_ttl_seconds: float | None = None) -> None:
        self.default_ttl_seconds = float(
            default_ttl_seconds or self.DEFAULT_TTL_SECONDS,
        )
        self._leases: dict[tuple[str, str], dict[str, Any]] = {}
        self._fencing_counters: dict[tuple[str, str], int] = {}
        self._lock = threading.RLock()

    def _normalize_session(self, session_id: str) -> str:
        sid = str(session_id or "").strip()
        if not sid:
            raise ValueError("session_id must be non-empty")
        return sid

    def _identity(self, session_id: str, tenant_id: str | None = None) -> tuple[str, str]:
        return resolve_tenant_id(tenant_id, default_tenant_id="default"), self._normalize_session(session_id)

    def _expired(self, lease: dict[str, Any], now: float) -> bool:
        return float(lease.get("expires_at", 0.0) or 0.0) <= now

    def acquire(
        self,
        session_id: str,
        owner_id: str,
        ttl_seconds: float | None = None,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        tenant, sid = self._identity(session_id, tenant_id)
        identity = (tenant, sid)
        owner = str(owner_id or "").strip() or "worker"
        ttl = float(ttl_seconds or self.default_ttl_seconds)
        now = time.monotonic()

        with self._lock:
            current = self._leases.get(identity)
            if current and not self._expired(current, now):
                current_owner = str(current.get("owner_id") or "")
                if current_owner and current_owner != owner:
                    raise LeaseConflictError(
                        f"lease conflict: session_id={sid!r} owned_by={current_owner!r} requested_by={owner!r}",
                    )

            if current and str(current.get("owner_id") or "") == owner and not self._expired(current, now):
                token = int(current.get("fencing_token") or 1)
                lease_id = str(current.get("lease_id") or f"lease-{uuid4().hex}")
            else:
                token = int(self._fencing_counters.get(identity, 0)) + 1
                self._fencing_counters[identity] = token
                lease_id = f"lease-{uuid4().hex}"

            lease = {
                "session_id": sid,
                "tenant_id": tenant,
                "owner_id": owner,
                "lease_id": lease_id,
                "ttl_seconds": ttl,
                "acquired_at": now,
                "expires_at": now + max(0.001, ttl),
                "fencing_token": token,
            }
            self._leases[identity] = lease
            return dict(lease)

    def renew(
        self,
        session_id: str,
        owner_id: str,
        ttl_seconds: float | None = None,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        tenant, sid = self._identity(session_id, tenant_id)
        identity = (tenant, sid)
        owner = str(owner_id or "").strip() or "worker"
        now = time.monotonic()
        with self._lock:
            current = self._leases.get(identity)
            if not current or self._expired(current, now):
                raise LeaseConflictError(
                    f"cannot renew expired or missing lease for {sid!r}",
                )
            if str(current.get("owner_id") or "") != owner:
                raise LeaseConflictError(
                    f"lease conflict: session_id={sid!r} requested_by={owner!r}",
                )
            ttl = float(
                ttl_seconds or current.get("ttl_seconds") or self.default_ttl_seconds,
            )
            current["expires_at"] = now + max(0.001, ttl)
            current["ttl_seconds"] = ttl
            return dict(current)

    def release(self, session_id: str, owner_id: str, *, tenant_id: str | None = None) -> bool:
        identity = self._identity(session_id, tenant_id)
        owner = str(owner_id or "").strip() or "worker"
        with self._lock:
            current = self._leases.get(identity)
            if not current:
                return False
            if str(current.get("owner_id") or "") != owner:
                return False
            self._leases.pop(identity, None)
            return True

    def owner_of(self, session_id: str, *, tenant_id: str | None = None) -> str | None:
        identity = self._identity(session_id, tenant_id)
        with self._lock:
            lease = self._leases.get(identity)
            if not lease or self._expired(lease, time.monotonic()):
                return None
            return str(lease.get("owner_id") or "") or None

    def lease_for(self, session_id: str, *, tenant_id: str | None = None) -> dict[str, Any] | None:
        identity = self._identity(session_id, tenant_id)
        with self._lock:
            lease = self._leases.get(identity)
            if not lease or self._expired(lease, time.monotonic()):
                return None
            return dict(lease)

    def is_held(self, session_id: str, *, tenant_id: str | None = None) -> bool:
        return self.owner_of(session_id, tenant_id=tenant_id) is not None

    def require_session_lease(self, session_id: str, owner_id: str, *, tenant_id: str | None = None) -> None:
        owner = self.owner_of(session_id, tenant_id=tenant_id)
        requested = str(owner_id or "").strip() or "worker"
        if owner is None or owner == requested:
            return
        raise LeaseConflictError(
            f"lease conflict: session_id={session_id!r} owned_by={owner!r} requested_by={requested!r}",
        )

    def evict_expired(self) -> list[str]:
        now = time.monotonic()
        evicted: list[str] = []
        with self._lock:
            for identity, lease in list(self._leases.items()):
                if self._expired(lease, now):
                    evicted.append(identity[1])
                    self._leases.pop(identity, None)
        return evicted

    def snapshot(self) -> dict[str, Any]:
        self.evict_expired()
        with self._lock:
            leases = [dict(lease) for _, lease in sorted(self._leases.items())]
        return {"active_lease_count": len(leases), "leases": leases}

    def fencing_token_for(self, session_id: str, *, tenant_id: str | None = None) -> int | None:
        identity = self._identity(session_id, tenant_id)
        with self._lock:
            lease = self._leases.get(identity)
            if not lease or self._expired(lease, time.monotonic()):
                return None
            return int(lease.get("fencing_token") or 0)
