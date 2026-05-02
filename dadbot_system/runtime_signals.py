from __future__ import annotations

import json
import logging
import importlib
from contextlib import contextmanager
from typing import Any

from .contracts import TelemetrySettings

trace = None
Resource = None
TracerProvider = None


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

    global trace, Resource, TracerProvider
    if trace is None or Resource is None or TracerProvider is None:
        try:
            trace = importlib.import_module("opentelemetry.trace")
            Resource = importlib.import_module("opentelemetry.sdk.resources").Resource
            TracerProvider = importlib.import_module("opentelemetry.sdk.trace").TracerProvider
        except Exception:
            return None

    provider = TracerProvider(resource=Resource.create({"service.name": telemetry.service_name}))
    trace.set_tracer_provider(provider)
    return provider


@contextmanager
def start_span(name: str, **attributes: Any):
    if trace is None:
        yield None
        return

    tracer = trace.get_tracer("dadbot")
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        yield span
