from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import math
import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from .tenant_identity import is_versioned_tenant_id, log_legacy_tenant_id_acceptance, require_tenant_id

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


class TicketStoreUnavailableError(ServiceSecurityError):
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


def _validate_principal_times(payload: dict[str, Any]) -> None:
    for field_name in ("iat", "exp"):
        value = payload.get(field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AuthenticationError(f"Invalid bearer token {field_name}")
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise AuthenticationError(f"Invalid bearer token {field_name}") from exc
        if not finite:
            raise AuthenticationError(f"Invalid bearer token {field_name}")
    if payload["exp"] <= time.time() or payload["exp"] <= payload["iat"]:
        raise AuthenticationError("Bearer token has expired or has an invalid lifetime")


def _validate_principal_claims(payload: dict[str, Any]) -> None:
    for field_name in ("sub", "tenant_id", "jti"):
        value = payload.get(field_name)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise AuthenticationError(f"Invalid bearer token {field_name}")
    if payload["sub"].lower() == "anonymous":
        raise AuthenticationError("Anonymous bearer token principal is not permitted")
    _validate_principal_times(payload)
    for field_name in ("scopes", "allowed_tools"):
        value = payload.get(field_name)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise AuthenticationError(f"Invalid bearer token {field_name}")


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
        if (
            not isinstance(subject, str)
            or not subject.strip()
            or subject != subject.strip()
            or subject.lower() == "anonymous"
        ):
            raise ValueError("A non-anonymous subject is required")
        tenant_id = require_tenant_id(
            tenant_id,
            issuer=self._issuer,
            subject=subject,
            acceptance_path="ServiceTokenManager.issue",
        )
        ttl = float(ttl_seconds)
        if not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("Token lifetime must be finite and positive")
        now = time.time()
        payload = {
            "v": self.VERSION,
            "sub": subject,
            "tenant_id": tenant_id,
            "scopes": sorted(_normalize_scope_set(scopes)),
            "allowed_tools": sorted(_normalize_tool_set(allowed_tools)),
            "iat": now,
            "exp": now + ttl,
            "iss": self._issuer,
            "jti": str(token_id or hashlib.sha256(f"{now}:{subject}:{tenant_id}".encode()).hexdigest()[:16]),
        }
        payload_bytes = _stable_json(payload)
        signature = hmac.new(self._key, payload_bytes, self.ALGORITHM).hexdigest()
        return f"{_urlsafe_b64encode(payload_bytes)}.{signature}"

    def verify(self, token: str) -> ServicePrincipal:
        try:
            encoded_payload, signature = str(token or "").rsplit(".", 1)
            payload_bytes = _urlsafe_b64decode(encoded_payload)
        except (ValueError, binascii.Error) as exc:
            raise AuthenticationError("Malformed bearer token") from exc

        expected_signature = hmac.new(self._key, payload_bytes, self.ALGORITHM).hexdigest()
        if not signature.isascii() or not hmac.compare_digest(signature, expected_signature):
            raise AuthenticationError("Invalid bearer token signature")

        try:
            payload = json.loads(payload_bytes)
        except (ValueError, UnicodeDecodeError) as exc:
            raise AuthenticationError("Malformed bearer token payload") from exc
        if not isinstance(payload, dict):
            raise AuthenticationError("Invalid bearer token payload")
        _validate_principal_claims(payload)
        if type(payload.get("v")) is not int or payload["v"] != self.VERSION:
            raise AuthenticationError("Unsupported bearer token version")
        if str(payload.get("iss") or "") != self._issuer:
            raise AuthenticationError("Unexpected bearer token issuer")
        if float(payload.get("exp") or 0.0) < time.time():
            raise AuthenticationError("Bearer token has expired")

        if not is_versioned_tenant_id(payload["tenant_id"]):
            log_legacy_tenant_id_acceptance(
                payload["tenant_id"],
                issuer=payload["iss"],
                subject=payload["sub"],
                acceptance_path="ServiceTokenManager.verify",
            )

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


class RedisWebSocketTicketStore:
    TTL_SECONDS = 60

    def __init__(self, redis_url: str, *, namespace: str) -> None:
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("WebSocket tickets require the 'service' dependencies (redis)") from exc

        self._client = redis.from_url(redis_url, decode_responses=True, socket_timeout=3.0, socket_connect_timeout=3.0)
        self._redis_error = redis.exceptions.RedisError
        self._prefix = f"dadbot:ws-ticket:{namespace}:"

    def _key(self, ticket: str) -> str:
        return self._prefix + hashlib.sha256(ticket.encode("utf-8")).hexdigest()

    def issue(self, principal: ServicePrincipal, *, path: str) -> tuple[str, float]:
        now = time.time()
        expires_at = min(now + self.TTL_SECONDS, principal.expires_at or now + self.TTL_SECONDS)
        if expires_at <= now:
            raise AuthenticationError("Principal has expired")
        ticket = secrets.token_urlsafe(32)
        payload = _stable_json({"path": path, "expires_at": expires_at, "principal": principal.to_metadata()}).decode(
            "utf-8"
        )
        try:
            stored = self._client.set(self._key(ticket), payload, ex=math.ceil(expires_at - now), nx=True)
        except self._redis_error as exc:
            raise TicketStoreUnavailableError("WebSocket ticket store is unavailable") from exc
        if not stored:
            raise TicketStoreUnavailableError("Unable to allocate WebSocket ticket")
        return ticket, expires_at

    def consume(self, ticket: str, *, path: str) -> ServicePrincipal:
        if not ticket or len(ticket) > 128:
            raise AuthenticationError("Invalid WebSocket ticket")
        try:
            # GETDEL is atomic across API workers; a mismatched attempt also burns the ticket.
            raw = self._client.getdel(self._key(ticket))
        except self._redis_error as exc:
            raise TicketStoreUnavailableError("WebSocket ticket store is unavailable") from exc
        if not isinstance(raw, (str, bytes)):
            raise AuthenticationError("Expired or already used WebSocket ticket")
        try:
            payload = json.loads(raw)
            if payload["path"] != path or float(payload["expires_at"]) <= time.time():
                raise AuthenticationError("Expired or mismatched WebSocket ticket")
            metadata = payload["principal"]
            return ServicePrincipal(
                subject=metadata["subject"],
                tenant_id=metadata["tenant_id"],
                scopes=frozenset(metadata["scopes"]),
                allowed_tools=frozenset(metadata["allowed_tools"]),
                issuer=metadata["issuer"],
                expires_at=float(metadata["expires_at"]),
                token_id=metadata["token_id"],
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise AuthenticationError("Invalid WebSocket ticket data") from exc

    def close(self) -> None:
        self._client.close()


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
