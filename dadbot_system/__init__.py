"""dadbot_system package — all exports are loaded lazily on first access.

Eager imports here pulled fastapi, redis, and opentelemetry-sdk unconditionally,
adding ~400 ms to every ``import dadbot`` call even when those subsystems were
not needed.  The __getattr__ pattern below keeps the public API identical while
deferring each heavy sub-import until the symbol is actually used.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_GROUPS: dict[str, list[str]] = {
    ".api": ["create_api_app"],
    ".client": ["DadServiceClient", "ServiceChatResult", "ServiceClientConfig"],
    ".kernel": ["ControlPlane", "ExecutionJob", "Scheduler", "SessionRegistry", "build_control_plane"],
    ".contracts": [
        "ApiSettings",
        "AttachmentPayload",
        "ChatRequest",
        "ChatResponse",
        "DEFAULT_TENANT_ID",
        "EventEnvelope",
        "EventType",
        "ExecutionGraph",
        "PersistenceSettings",
        "QueueSettings",
        "ServiceConfig",
        "TelemetrySettings",
        "ToolCapability",
        "WorkerSettings",
        "WorkerTask",
        "WorkerResult",
        "normalize_tenant_id",
    ],
    ".events": ["InMemoryEventBus", "QueueEventBus"],
    ".orchestration": ["DadBotOrchestrator", "ToolRegistry"],
    ".state": [
        "AppStateContainer",
        "CompositeStateStore",
        "InMemoryStateStore",
        "NamespacedStateStore",
        "PostgresStateStore",
        "RedisStateStore",
    ],
    ".telemetry": ["configure_logging", "configure_tracing"],
    ".worker": ["LocalMultiprocessBroker", "WorkerProcessManager"],
}

_NAME_TO_MODULE: dict[str, str] = {name: mod for mod, names in _EXPORT_GROUPS.items() for name in names}

__all__ = list(_NAME_TO_MODULE)


def __getattr__(name: str) -> Any:
    if name in _NAME_TO_MODULE:
        mod = import_module(_NAME_TO_MODULE[name], package=__name__)
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
