from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

DEFAULT_TENANT_ID = "default"
DEFAULT_CHANNEL_NAME = "chat"


def normalize_tenant_id(value: str | None) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip().lower())
    normalized = normalized.strip("-._")
    return normalized or DEFAULT_TENANT_ID


def normalize_channel_name(value: str | None) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip().lower())
    normalized = normalized.strip("-._")
    return normalized or DEFAULT_CHANNEL_NAME


def extract_gateway_metadata(payload: dict[str, Any], *, channel_name: str | None = None) -> dict[str, Any]:
    gateway = dict(payload.get("gateway") or {})
    channel = normalize_channel_name(
        channel_name or gateway.get("channel") or payload.get("channel") or payload.get("source")
    )
    message_id = str(
        gateway.get("message_id") or payload.get("message_id") or payload.get("external_message_id") or ""
    ).strip()
    user_id = str(gateway.get("user_id") or payload.get("user_id") or payload.get("sender_id") or "").strip()
    user_name = str(gateway.get("user_name") or payload.get("user_name") or payload.get("sender_name") or "").strip()
    conversation_id = str(
        gateway.get("conversation_id") or payload.get("conversation_id") or payload.get("channel_thread_id") or ""
    ).strip()
    received_at = str(gateway.get("received_at") or payload.get("received_at") or "").strip()
    delivery_mode = str(gateway.get("delivery_mode") or payload.get("delivery_mode") or "sync").strip().lower() or "sync"

    resolved = {"channel": channel, "delivery_mode": delivery_mode}
    if message_id:
        resolved["message_id"] = message_id
    if user_id:
        resolved["user_id"] = user_id
    if user_name:
        resolved["user_name"] = user_name
    if conversation_id:
        resolved["conversation_id"] = conversation_id
    if received_at:
        resolved["received_at"] = received_at
    return resolved


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def env_flag(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


def env_csv(name: str, default: list[str]) -> list[str]:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return list(default)
    values = [str(item).strip() for item in str(raw_value).split(",")]
    return [item for item in values if item]


class EventType(str, Enum):
    REQUEST_ACCEPTED = "request.accepted"
    REQUEST_DISPATCHED = "request.dispatched"
    STATE_UPDATED = "state.updated"
    RESPONSE_READY = "response.ready"
    REQUEST_FAILED = "request.failed"
    WORKER_STARTED = "worker.started"
    WORKER_STOPPED = "worker.stopped"
    CIRCUIT_OPENED = "circuit.opened"


@dataclass(slots=True)
class AttachmentPayload:
    type: str
    name: str = ""
    mime_type: str = ""
    data_b64: str = ""
    note: str = ""
    transcript: str = ""
    analysis: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AttachmentPayload:
        return cls(
            type=str(payload.get("type") or "").strip().lower(),
            name=str(payload.get("name") or "").strip(),
            mime_type=str(payload.get("mime_type") or "").strip(),
            data_b64=str(payload.get("data_b64") or payload.get("image_b64") or "").strip(),
            note=str(payload.get("note") or "").strip(),
            transcript=str(payload.get("transcript") or "").strip(),
            analysis=str(payload.get("analysis") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolCapability:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionNode:
    node_id: str
    layer: str
    status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionGraph:
    graph_id: str = field(default_factory=lambda: uuid4().hex)
    nodes: list[ExecutionNode] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["nodes"] = [node.to_dict() for node in self.nodes]
        return payload


@dataclass(slots=True)
class ChatRequest:
    session_id: str
    user_input: str
    tenant_id: str = DEFAULT_TENANT_ID
    attachments: list[AttachmentPayload] = field(default_factory=list)
    requested_model: str = ""
    request_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, session_id: str | None = None) -> ChatRequest:
        attachments = [
            AttachmentPayload.from_dict(item) for item in payload.get("attachments", []) if isinstance(item, dict)
        ]
        metadata = dict(payload.get("metadata") or {})
        gateway = extract_gateway_metadata(payload, channel_name=payload.get("channel"))
        if gateway:
            metadata["gateway"] = gateway
        return cls(
            session_id=session_id or str(payload.get("session_id") or "").strip(),
            user_input=str(payload.get("user_input") or payload.get("message") or "").strip(),
            tenant_id=normalize_tenant_id(payload.get("tenant_id")),
            attachments=attachments,
            requested_model=str(payload.get("requested_model") or payload.get("model") or "").strip(),
            request_id=str(payload.get("request_id") or uuid4().hex),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["attachments"] = [attachment.to_dict() for attachment in self.attachments]
        return payload


@dataclass(slots=True)
class ChatResponse:
    session_id: str
    request_id: str
    reply: str
    tenant_id: str = DEFAULT_TENANT_ID
    should_end: bool = False
    active_model: str = ""
    mood: str = "neutral"
    status: str = "completed"
    created_at: str = field(default_factory=utc_now_iso)
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QueueSettings:
    backend: str = "multiprocessing"
    max_queue_size: int = 256
    result_poll_timeout_seconds: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WorkerSettings:
    worker_count: int = 2
    max_retries: int = 2
    task_timeout_seconds: float = 45.0
    retry_backoff_seconds: float = 0.2
    circuit_breaker_failures: int = 3
    circuit_breaker_reset_seconds: float = 30.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ApiSettings:
    host: str = "127.0.0.1"
    port: int = 8010
    title: str = "Dad Bot API"
    cors_origins: list[str] = field(
        default_factory=lambda: [
            "http://127.0.0.1:8501",
            "http://localhost:8501",
        ]
    )
    version_prefix: str = "/v1"
    legacy_routes_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SecuritySettings:
    auth_required: bool = True
    token_secret: str = ""
    token_issuer: str = "dadbot"
    token_ttl_seconds: int = 3600
    max_requests_per_minute: int = 120
    max_write_requests_per_minute: int = 60
    max_admin_requests_per_minute: int = 20

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PersistenceSettings:
    redis_url: str = ""
    postgres_dsn: str = ""
    vector_store_url: str = ""
    session_table: str = "dadbot_session_state"
    task_table: str = "dadbot_task_state"
    event_table: str = "dadbot_event_log"
    event_encryption_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TelemetrySettings:
    log_level: str = "INFO"
    json_logs: bool = True
    otel_enabled: bool = False
    service_name: str = "dadbot"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ServiceConfig:
    default_model: str = "llama3.2"
    api: ApiSettings = field(default_factory=ApiSettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    queue: QueueSettings = field(default_factory=QueueSettings)
    workers: WorkerSettings = field(default_factory=WorkerSettings)
    persistence: PersistenceSettings = field(default_factory=PersistenceSettings)
    telemetry: TelemetrySettings = field(default_factory=TelemetrySettings)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ServiceConfig:
        return cls(
            default_model=str(payload.get("default_model") or "llama3.2"),
            api=ApiSettings(**dict(payload.get("api") or {})),
            security=SecuritySettings(**dict(payload.get("security") or {})),
            queue=QueueSettings(**dict(payload.get("queue") or {})),
            workers=WorkerSettings(**dict(payload.get("workers") or {})),
            persistence=PersistenceSettings(**dict(payload.get("persistence") or {})),
            telemetry=TelemetrySettings(**dict(payload.get("telemetry") or {})),
        )

    @classmethod
    def from_environment(cls) -> ServiceConfig:
        config = cls()
        config.default_model = str(os.environ.get("DADBOT_DEFAULT_MODEL") or config.default_model)
        config.api.host = str(os.environ.get("DADBOT_API_HOST") or config.api.host)
        config.api.port = env_int("DADBOT_API_PORT", config.api.port)
        config.api.cors_origins = env_csv("DADBOT_API_CORS_ORIGINS", config.api.cors_origins)
        config.api.version_prefix = str(os.environ.get("DADBOT_API_VERSION_PREFIX") or config.api.version_prefix)
        config.api.legacy_routes_enabled = env_flag(
            "DADBOT_API_LEGACY_ROUTES_ENABLED",
            config.api.legacy_routes_enabled,
        )
        config.security.auth_required = env_flag("DADBOT_API_AUTH_REQUIRED", config.security.auth_required)
        config.security.token_secret = str(os.environ.get("DADBOT_API_TOKEN_SECRET") or "").strip()
        config.security.token_issuer = str(os.environ.get("DADBOT_API_TOKEN_ISSUER") or config.security.token_issuer)
        config.security.token_ttl_seconds = env_int("DADBOT_API_TOKEN_TTL_SECONDS", config.security.token_ttl_seconds)
        config.security.max_requests_per_minute = env_int(
            "DADBOT_API_RATE_LIMIT_PER_MINUTE",
            config.security.max_requests_per_minute,
        )
        config.security.max_write_requests_per_minute = env_int(
            "DADBOT_API_WRITE_RATE_LIMIT_PER_MINUTE",
            config.security.max_write_requests_per_minute,
        )
        config.security.max_admin_requests_per_minute = env_int(
            "DADBOT_API_ADMIN_RATE_LIMIT_PER_MINUTE",
            config.security.max_admin_requests_per_minute,
        )
        config.queue.backend = str(os.environ.get("DADBOT_QUEUE_BACKEND") or config.queue.backend)
        config.queue.max_queue_size = env_int("DADBOT_QUEUE_MAX_SIZE", config.queue.max_queue_size)
        config.workers.worker_count = env_int("DADBOT_API_WORKERS", config.workers.worker_count)
        config.workers.max_retries = env_int("DADBOT_MAX_RETRIES", config.workers.max_retries)
        config.persistence.redis_url = str(os.environ.get("DADBOT_REDIS_URL") or "").strip()
        config.persistence.postgres_dsn = str(os.environ.get("DADBOT_POSTGRES_DSN") or "").strip()
        config.persistence.vector_store_url = str(os.environ.get("DADBOT_VECTOR_STORE_URL") or "").strip()
        config.persistence.event_encryption_key = str(os.environ.get("DADBOT_EVENT_ENCRYPTION_KEY") or "").strip()
        config.telemetry.log_level = str(os.environ.get("DADBOT_LOG_LEVEL") or config.telemetry.log_level)
        config.telemetry.json_logs = env_flag("DADBOT_JSON_LOGS", config.telemetry.json_logs)
        config.telemetry.otel_enabled = env_flag("DADBOT_OTEL_ENABLED", config.telemetry.otel_enabled)
        config.telemetry.service_name = str(os.environ.get("DADBOT_SERVICE_NAME") or config.telemetry.service_name)
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_model": self.default_model,
            "api": self.api.to_dict(),
            "security": self.security.to_dict(),
            "queue": self.queue.to_dict(),
            "workers": self.workers.to_dict(),
            "persistence": self.persistence.to_dict(),
            "telemetry": self.telemetry.to_dict(),
        }


@dataclass(slots=True)
class WorkerTask:
    session_id: str
    request: ChatRequest
    tenant_id: str = DEFAULT_TENANT_ID
    session_state: dict[str, Any] = field(default_factory=dict)
    execution_graph: ExecutionGraph | None = None
    accepted_at: str = field(default_factory=utc_now_iso)
    task_id: str = field(default_factory=lambda: uuid4().hex)
    task_kind: str = "chat"
    attempt: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "request": self.request.to_dict(),
            "tenant_id": normalize_tenant_id(self.tenant_id),
            "session_state": dict(self.session_state),
            "execution_graph": self.execution_graph.to_dict() if self.execution_graph is not None else None,
            "accepted_at": self.accepted_at,
            "task_id": self.task_id,
            "task_kind": self.task_kind,
            "attempt": self.attempt,
        }


@dataclass(slots=True)
class WorkerResult:
    task_id: str
    session_id: str
    request_id: str
    status: str
    tenant_id: str = DEFAULT_TENANT_ID
    session_state: dict[str, Any] = field(default_factory=dict)
    response: ChatResponse | None = None
    error: str = ""
    completed_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "status": self.status,
            "tenant_id": normalize_tenant_id(self.tenant_id),
            "session_state": dict(self.session_state),
            "response": self.response.to_dict() if self.response is not None else None,
            "error": self.error,
            "completed_at": self.completed_at,
        }


@dataclass(slots=True)
class EventEnvelope:
    session_id: str
    event_type: EventType
    tenant_id: str = DEFAULT_TENANT_ID
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "event_type": self.event_type.value,
            "tenant_id": normalize_tenant_id(self.tenant_id),
            "payload": dict(self.payload),
            "event_id": self.event_id,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class HealthResponse:
    status: str
    workers: int
    queue_backend: str
    state_backend: str
    service_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
