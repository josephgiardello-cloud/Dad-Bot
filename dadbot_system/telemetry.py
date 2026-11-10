from __future__ import annotations

import json
import logging
from typing import Any

from .contracts import TelemetrySettings
from .runtime_signals import register_telemetry_adapter

STANDARD_LOG_FIELDS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in STANDARD_LOG_FIELDS and not key.startswith("_")
        }
        if extras:
            payload["fields"] = extras
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(settings: TelemetrySettings | None = None, *, force: bool = False) -> None:
    telemetry = settings or TelemetrySettings()
    root = logging.getLogger()
    if root.handlers and not force:
        return

    handler = logging.StreamHandler()
    if telemetry.json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))

    root.handlers = [handler]
    root.setLevel(getattr(logging, str(telemetry.log_level or "INFO").upper(), logging.INFO))


def configure_tracing(settings: TelemetrySettings | None = None):
    telemetry = settings or TelemetrySettings()
    if not telemetry.otel_enabled:
        return None
    from dadbot.telemetry.otel_bridge import get_trace_provider, install_otel_bridge

    status = install_otel_bridge(service_name=telemetry.service_name)
    if not status["installed"]:
        raise RuntimeError(status["reason"])
    return get_trace_provider()


def start_span(name: str, **attributes: Any):
    from dadbot.telemetry.otel_bridge import start_span as bridge_start_span

    return bridge_start_span(name, **attributes)


def shutdown_tracing() -> None:
    from dadbot.telemetry.otel_bridge import shutdown_otel_bridge

    shutdown_otel_bridge()


def is_trace_provider_available() -> bool:
    from dadbot.telemetry.otel_bridge import get_trace_provider

    return get_trace_provider() is not None


register_telemetry_adapter(
    configure=configure_tracing,
    start_span=start_span,
    shutdown=shutdown_tracing,
    trace_provider_available=is_trace_provider_available,
)
