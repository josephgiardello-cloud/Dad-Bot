from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError:  # pragma: no cover - optional runtime dependency.
    Fernet = None
    InvalidToken = Exception


class ServiceSecurityError(RuntimeError):
    pass


class AuthenticationError(ServiceSecurityError):
    pass


class AuthorizationError(ServiceSecurityError):
    pass


class EncryptionUnavailableError(ServiceSecurityError):
    pass


def _urlsafe_b64encode(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _urlsafe_b64decode(payload: str) -> bytes:
    padding = "=" * ((4 - (len(payload) % 4)) % 4)
    return base64.urlsafe_b64decode(payload + padding)


def _stable_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _normalize_scope_set(values: list[str] | tuple[str, ...] | set[str] | None) -> set[str]:
    return {str(value or "").strip().lower() for value in list(values or []) if str(value or "").strip()}


def _normalize_tool_set(values: list[str] | tuple[str, ...] | set[str] | None) -> set[str]:
    return {str(value or "").strip() for value in list(values or []) if str(value or "").strip()}


@dataclass(slots=True, frozen=True)
class ServicePrincipal:
    subject: str
    tenant_id: str
    scopes: frozenset[str] = field(default_factory=frozenset)
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    issuer: str = "dadbot"
    expires_at: float = 0.0
    token_id: str = ""

    def has_scope(self, scope: str) -> bool:
        normalized = str(scope or "").strip().lower()
        return "admin" in self.scopes or normalized in self.scopes

    def require_scope(self, scope: str) -> None:
        if not self.has_scope(scope):
            raise AuthorizationError(f"principal lacks required scope: {scope}")

    @property
    def rate_limit_key(self) -> str:
        # Keep limiter identity stable across token rotations for the same principal.
        return f"{self.tenant_id}:{self.subject}"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "tenant_id": self.tenant_id,
            "scopes": sorted(self.scopes),
            "allowed_tools": sorted(self.allowed_tools),
            "issuer": self.issuer,
            "expires_at": self.expires_at,
            "token_id": self.token_id,
        }


class ServiceTokenManager:
    ALGORITHM = "sha256"
    VERSION = 1

    def __init__(self, secret_key: str | bytes, *, issuer: str = "dadbot") -> None:
        raw_key = secret_key.encode("utf-8") if isinstance(secret_key, str) else bytes(secret_key)
        if not raw_key:
            raise ValueError("A non-empty service token secret is required")
        self._key = raw_key
        self._issuer = str(issuer or "dadbot")

    def issue(
        self,
        *,
        subject: str,
        tenant_id: str,
        scopes: list[str] | tuple[str, ...] | set[str],
        allowed_tools: list[str] | tuple[str, ...] | set[str] | None = None,
        ttl_seconds: float = 3600.0,
        token_id: str = "",
    ) -> str:
        now = time.time()
        payload = {
            "v": self.VERSION,
            "sub": str(subject or "service-client"),
            "tenant_id": str(tenant_id or "default"),
            "scopes": sorted(_normalize_scope_set(scopes)),
            "allowed_tools": sorted(_normalize_tool_set(allowed_tools)),
            "iat": now,
            "exp": now + max(1.0, float(ttl_seconds)),
            "iss": self._issuer,
            "jti": str(token_id or hashlib.sha256(f"{now}:{subject}:{tenant_id}".encode("utf-8")).hexdigest()[:16]),
        }
        payload_bytes = _stable_json(payload)
        signature = hmac.new(self._key, payload_bytes, self.ALGORITHM).hexdigest()
        return f"{_urlsafe_b64encode(payload_bytes)}.{signature}"

    def verify(self, token: str) -> ServicePrincipal:
        try:
            encoded_payload, signature = str(token or "").rsplit(".", 1)
        except ValueError as exc:
            raise AuthenticationError("Malformed bearer token") from exc

        payload_bytes = _urlsafe_b64decode(encoded_payload)
        expected_signature = hmac.new(self._key, payload_bytes, self.ALGORITHM).hexdigest()
        if not hmac.compare_digest(signature, expected_signature):
            raise AuthenticationError("Invalid bearer token signature")

        payload = dict(json.loads(payload_bytes))
        if int(payload.get("v") or 0) != self.VERSION:
            raise AuthenticationError("Unsupported bearer token version")
        if str(payload.get("iss") or "") != self._issuer:
            raise AuthenticationError("Unexpected bearer token issuer")
        if float(payload.get("exp") or 0.0) < time.time():
            raise AuthenticationError("Bearer token has expired")

        return ServicePrincipal(
            subject=str(payload.get("sub") or "service-client"),
            tenant_id=str(payload.get("tenant_id") or "default"),
            scopes=frozenset(_normalize_scope_set(payload.get("scopes"))),
            allowed_tools=frozenset(_normalize_tool_set(payload.get("allowed_tools"))),
            issuer=str(payload.get("iss") or self._issuer),
            expires_at=float(payload.get("exp") or 0.0),
            token_id=str(payload.get("jti") or ""),
        )


class SlidingWindowRateLimiter:
    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window_seconds = max(1.0, float(window_seconds))
        self._events: dict[str, deque[float]] = {}
        self._lock = Lock()

    def enforce(self, *, key: str, limit: int) -> float:
        max_events = max(1, int(limit))
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            bucket = self._events.setdefault(str(key), deque())
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= max_events:
                return max(1.0, self.window_seconds - (now - bucket[0]))
            bucket.append(now)
        return 0.0


class EncryptedJsonCodec:
    PREFIX = "enc::"

    def __init__(self, secret_key: str | bytes) -> None:
        if Fernet is None:
            raise EncryptionUnavailableError(
                "cryptography is required for encrypted durability; install the service dependencies with cryptography"
            )
        raw_key = secret_key.encode("utf-8") if isinstance(secret_key, str) else bytes(secret_key)
        if not raw_key:
            raise ValueError("A non-empty encryption key is required")
        digest = hashlib.sha256(raw_key).digest()
        self._fernet = Fernet(base64.urlsafe_b64encode(digest))

    @classmethod
    def is_encrypted_blob(cls, value: str) -> bool:
        return str(value or "").startswith(cls.PREFIX)

    def encode(self, payload: dict[str, Any]) -> str:
        plaintext = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        token = self._fernet.encrypt(plaintext).decode("utf-8")
        return f"{self.PREFIX}{token}"

    def decode(self, value: str) -> dict[str, Any]:
        raw_value = str(value or "")
        if not self.is_encrypted_blob(raw_value):
            return dict(json.loads(raw_value or "{}"))
        token = raw_value[len(self.PREFIX) :].encode("utf-8")
        try:
            plaintext = self._fernet.decrypt(token)
        except InvalidToken as exc:  # pragma: no cover - depends on external lib behavior.
            raise AuthenticationError("Unable to decrypt durable payload with the configured key") from exc
        return dict(json.loads(plaintext.decode("utf-8")))