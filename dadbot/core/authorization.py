"""Session-level authorization and tenant isolation.

Design
------
This is a capability-based, in-process authorization layer.  It protects
well-behaved code from accidental cross-session or cross-tenant writes.

Components
----------
Capability / CapabilitySet â€” named operations (READ, WRITE, EXECUTE, ADMIN).
TenantBoundary             â€” session IDs must carry a registered tenant prefix.
SessionAuthorizationPolicy â€” maps session_id â†’ CapabilitySet.
CapabilityToken            â€” HMAC-SHA256 signed bearer token (short-lived).
authorize_write            â€” convenience guard for LedgerWriter integration.

Security note
-------------
This is NOT a replacement for a production authentication system (OAuth,
mTLS, JWT service).  It protects well-behaved in-process callers.  For
multi-process / multi-tenant deployments, layer a proper AuthN/AuthZ service
on top and use CapabilityToken as an internal delegation mechanism.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from enum import Enum
from threading import RLock
from typing import Any


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

class Capability(str, Enum):
    READ    = "read"
    WRITE   = "write"
    EXECUTE = "execute"
    ADMIN   = "admin"


class CapabilitySet:
    """Immutable set of granted capabilities.

    ADMIN grants all other capabilities implicitly.
    """

    def __init__(self, *capabilities: Capability) -> None:
        self._caps: frozenset[Capability] = frozenset(capabilities)

    def has(self, cap: Capability) -> bool:
        return Capability.ADMIN in self._caps or cap in self._caps

    def __contains__(self, item: object) -> bool:
        if isinstance(item, Capability):
            return self.has(item)
        return False

    def __repr__(self) -> str:
        names = sorted(c.value for c in self._caps)
        return f"CapabilitySet({', '.join(names)})"

    @classmethod
    def empty(cls) -> "CapabilitySet":
        return cls()

    @classmethod
    def read_only(cls) -> "CapabilitySet":
        return cls(Capability.READ)

    @classmethod
    def read_write(cls) -> "CapabilitySet":
        return cls(Capability.READ, Capability.WRITE)

    @classmethod
    def full(cls) -> "CapabilitySet":
        return cls(Capability.READ, Capability.WRITE, Capability.EXECUTE, Capability.ADMIN)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class AuthorizationError(RuntimeError):
    """Raised when a caller attempts an operation without the required capability."""


# ---------------------------------------------------------------------------
# Tenant boundary
# ---------------------------------------------------------------------------

class TenantBoundary:
    """Enforces that session IDs carry a recognized tenant prefix.

    Prevents accidental cross-tenant writes by rejecting session IDs that
    don't start with any of the registered tenant prefixes.

    Usage::

        boundary = TenantBoundary(allowed_prefixes={"acme-", "beta-"})
        boundary.validate_session("acme-user-001")   # ok
        boundary.validate_session("evil-user")       # raises AuthorizationError
    """

    def __init__(
        self,
        allowed_prefixes: set[str] | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self._prefixes: frozenset[str] = frozenset(allowed_prefixes or set())
        self._enabled = bool(enabled)

    def validate_session(self, session_id: str) -> None:
        if not self._enabled or not self._prefixes:
            return
        sid = str(session_id or "")
        if not any(sid.startswith(p) for p in self._prefixes):
            raise AuthorizationError(
                f"Session {session_id!r} does not belong to any allowed tenant "
                f"prefix: {sorted(self._prefixes)}"
            )

    def add_prefix(self, prefix: str) -> None:
        self._prefixes = self._prefixes | {str(prefix)}

    def remove_prefix(self, prefix: str) -> None:
        self._prefixes = self._prefixes - {str(prefix)}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def prefixes(self) -> frozenset[str]:
        return self._prefixes


# ---------------------------------------------------------------------------
# Session authorization policy
# ---------------------------------------------------------------------------

class SessionAuthorizationPolicy:
    """Maps session IDs to granted capability sets.

    Usage::

        policy = SessionAuthorizationPolicy()
        policy.grant("session-001", CapabilitySet.read_write())
        policy.grant("admin-session", CapabilitySet.full())

        policy.require("session-001", Capability.WRITE)  # ok
        policy.require("session-001", Capability.ADMIN)  # raises AuthorizationError
    """

    def __init__(
        self,
        *,
        default_caps: CapabilitySet | None = None,
        strict: bool = False,
    ) -> None:
        """
        Args:
            default_caps: Capabilities for sessions not explicitly registered.
                          Defaults to read+write.  Ignored when strict=True.
            strict: When True, unregistered sessions get no capabilities at all.
        """
        self._lock         = RLock()
        self._grants:      dict[str, CapabilitySet] = {}
        self._default_caps = (
            CapabilitySet.empty()
            if strict
            else (default_caps or CapabilitySet.read_write())
        )
        self._strict = bool(strict)

    def grant(self, session_id: str, caps: CapabilitySet) -> None:
        with self._lock:
            self._grants[str(session_id)] = caps

    def revoke(self, session_id: str) -> None:
        with self._lock:
            self._grants.pop(str(session_id), None)

    def caps_for(self, session_id: str) -> CapabilitySet:
        with self._lock:
            return self._grants.get(str(session_id), self._default_caps)

    def require(self, session_id: str, capability: Capability) -> None:
        """Raise AuthorizationError if the session lacks capability."""
        if not self.caps_for(session_id).has(capability):
            raise AuthorizationError(
                f"Session {session_id!r} lacks capability {capability.value!r}"
            )

    def check(self, session_id: str, capability: Capability) -> bool:
        return self.caps_for(session_id).has(capability)

    @property
    def strict(self) -> bool:
        return self._strict


# ---------------------------------------------------------------------------
# HMAC-signed capability token
# ---------------------------------------------------------------------------

class CapabilityToken:
    """HMAC-SHA256 signed bearer token granting a capability for a session.

    Tokens are bound to a specific ``session_id`` and ``capability`` and
    expire after ``ttl_seconds``.

    Security note:
        Uses a random 32-byte secret by default.  For multi-process deployments
        the same secret must be shared (via env var / secrets manager).
        Replace with JWT + asymmetric signing for production multi-service use.
    """

    ALGORITHM = "sha256"

    def __init__(self, secret_key: bytes | None = None) -> None:
        self._key = secret_key or os.urandom(32)

    def issue(
        self,
        *,
        session_id: str,
        capability: Capability,
        ttl_seconds: float = 3600.0,
        issuer: str = "dadbot",
    ) -> str:
        """Issue a signed token string."""
        payload = {
            "session_id": str(session_id),
            "capability": capability.value,
            "issued_at":  time.time(),
            "expires_at": time.time() + float(ttl_seconds),
            "issuer":     str(issuer),
        }
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        sig = hmac.new(self._key, payload_bytes, self.ALGORITHM).hexdigest()
        b64 = base64.urlsafe_b64encode(payload_bytes).decode("ascii")
        return f"{b64}.{sig}"

    def verify(
        self,
        token: str,
        *,
        session_id: str,
        capability: Capability,
    ) -> bool:
        """Verify token.  Returns False (never raises) if invalid or expired."""
        try:
            b64, sig = token.rsplit(".", 1)
            payload_bytes = base64.urlsafe_b64decode(b64 + "==")
            expected_sig = hmac.new(self._key, payload_bytes, self.ALGORITHM).hexdigest()
            if not hmac.compare_digest(sig, expected_sig):
                return False
            payload = json.loads(payload_bytes)
            if payload.get("session_id") != session_id:
                return False
            if payload.get("capability") != capability.value:
                return False
            if float(payload.get("expires_at", 0)) < time.time():
                return False
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Convenience guard for LedgerWriter integration
# ---------------------------------------------------------------------------

def authorize_write(
    policy: SessionAuthorizationPolicy | None,
    session_id: str,
) -> None:
    """Raise AuthorizationError if session cannot write.  No-op if policy is None."""
    if policy is not None:
        policy.require(session_id, Capability.WRITE)
