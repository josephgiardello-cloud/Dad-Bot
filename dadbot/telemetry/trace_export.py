"""Opt-in local span export with an explicit data allowlist."""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from threading import Lock
from typing import Any, TextIO

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode

logger = logging.getLogger(__name__)
_SPAN_NAMES = frozenset(
    {
        "scheduler.drain_once",
        "orchestrator.submit_chat",
        "worker.process_task",
        "gen_ai.chat",
        "dadbot.tool.execute",
        "mcp.call_tool",
    }
)
_ENUM_ATTRIBUTES = {
    "gen_ai.operation.name": frozenset({"chat"}),
    "gen_ai.provider.name": frozenset({"ollama"}),
    "mcp.method.name": frozenset({"tools/call"}),
}
_NAME_ATTRIBUTE_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_COUNT_ATTRIBUTES = frozenset({"attempt", "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens"})


def _safe_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in attributes.items():
        if key in _COUNT_ATTRIBUTES and type(value) is int and value >= 0:
            result[key] = value
        elif key in _ENUM_ATTRIBUTES and isinstance(value, str) and value in _ENUM_ATTRIBUTES[key]:
            result[key] = value
        elif key in {"dadbot.tool.name", "dadbot.server.name", "mcp.tool.name", "mcp.server.name"} and (
            isinstance(value, str) and _NAME_ATTRIBUTE_PATTERN.fullmatch(value)
        ):
            result[key] = value
        elif key in {"dadbot.tool.is_error", "mcp.result.is_error"} and type(value) is bool:
            result[key] = value
    return result


class PrivacySafeConsoleSpanExporter(SpanExporter):
    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._lock = Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        records = []
        for span in spans:
            context = span.get_span_context()
            if context is None or not context.is_valid:
                logger.error("Cannot export a span without a valid context")
                return SpanExportResult.FAILURE
            records.append(
                json.dumps(
                    {
                        "name": span.name if span.name in _SPAN_NAMES else "dadbot.operation",
                        "trace_id": f"{context.trace_id:032x}",
                        "span_id": f"{context.span_id:016x}",
                        "parent_span_id": f"{span.parent.span_id:016x}" if span.parent else None,
                        "start_time_ns": span.start_time,
                        "end_time_ns": span.end_time,
                        "status": span.status.status_code.name,
                        "service": span.resource.attributes.get("service.name", "dadbot"),
                        "attributes": _safe_attributes(span.attributes or {}),
                    }
                )
            )
        try:
            with self._lock:
                self._stream.write("\n".join(records) + ("\n" if records else ""))
                self._stream.flush()
        except (OSError, ValueError):
            logger.error("DadBot span export failed")
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._stream.flush()


def create_trace_provider(service_name: str) -> TracerProvider:
    provider = TracerProvider(resource=Resource({"service.name": service_name}))
    provider.add_span_processor(SimpleSpanProcessor(PrivacySafeConsoleSpanExporter()))
    return provider


@contextmanager
def start_span(provider: TracerProvider, name: str, attributes: Mapping[str, Any]):
    tracer = provider.get_tracer("dadbot")
    with tracer.start_as_current_span(
        name if name in _SPAN_NAMES else "dadbot.operation",
        attributes=_safe_attributes(attributes),
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        finally:
            if sys.exc_info()[0] is not None:
                span.set_status(StatusCode.ERROR)
