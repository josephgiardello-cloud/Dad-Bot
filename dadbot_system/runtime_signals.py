from __future__ import annotations

import json
import logging
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from .contracts import TelemetrySettings

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

@dataclass(slots=True)
class _TelemetryAdapter:
    configure: Callable[[TelemetrySettings | None], Any] | None = None
    start_span: Callable[..., Any] | None = None
    shutdown: Callable[[], None] | None = None


_adapter = _TelemetryAdapter()


def register_telemetry_adapter(
    *,
    configure: Callable[[TelemetrySettings | None], Any],
    start_span: Callable[..., Any],
    shutdown: Callable[[], None],
) -> None:
    _adapter.configure = configure
    _adapter.start_span = start_span
    _adapter.shutdown = shutdown


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
    if _adapter.configure is None:
        raise RuntimeError("Tracing is enabled but no telemetry adapter is registered")
    return _adapter.configure(telemetry)


def start_span(name: str, **attributes: Any):
    if _adapter.start_span is None:
        return nullcontext(None)
    return _adapter.start_span(name, **attributes)


def shutdown_tracing() -> None:
    if _adapter.shutdown is not None:
        _adapter.shutdown()
