from __future__ import annotations

import asyncio
import io
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from urllib.request import urlopen

import pytest

from dadbot.core.observability import get_metrics
from dadbot.core.prometheus_bridge import start_prometheus_exporter
from dadbot.telemetry.otel_bridge import OpenTelemetryEventExporter, install_otel_bridge

pytestmark = pytest.mark.unit


def test_install_otel_bridge_returns_status_dict() -> None:
    from dadbot.telemetry.otel_bridge import shutdown_otel_bridge

    try:
        status = install_otel_bridge()
        assert isinstance(status, dict)
        assert "installed" in status
        assert "reason" in status
    finally:
        shutdown_otel_bridge()


def test_prometheus_exporter_serves_metrics_endpoint() -> None:
    metrics = get_metrics()
    metrics.reset()
    metrics.increment("scheduler.job.completed", 2)

    exporter = start_prometheus_exporter(port=0)
    try:
        port = exporter._server.server_port
        with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=1) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        assert "dadbot_scheduler_job_completed" in body
    finally:
        exporter.stop()


def test_otel_event_exporter_forwards_to_emitter() -> None:
    class _Emitter:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def emit(self, payload: dict) -> None:
            self.events.append(dict(payload))

    sink: list[dict] = []
    emitter = _Emitter()
    exporter = OpenTelemetryEventExporter(sink=sink.append, enabled=True, event_emitter=emitter)
    exporter.export({"event": "bridge.probe"})
    assert any(item.get("event") == "bridge.probe" for item in sink)
    assert any(item.get("event") == "bridge.probe" for item in emitter.events)


def test_enabled_tracing_exports_spans_without_private_content(monkeypatch) -> None:
    pytest.importorskip("opentelemetry.sdk.trace")
    from dadbot.telemetry.otel_bridge import shutdown_otel_bridge
    from dadbot_system.contracts import TelemetrySettings
    from dadbot_system.telemetry import configure_tracing, start_span

    shutdown_otel_bridge()
    output = io.StringIO()
    monkeypatch.setattr(sys, "stderr", output)
    provider = configure_tracing(TelemetrySettings(otel_enabled=True))
    try:
        with start_span("worker.process_task", attempt=1, session_id="PRIVATE-SESSION") as span:
            assert span is not None
            span.add_event("PRIVATE-PROMPT", {"response": "PRIVATE-RESPONSE"})
        provider.force_flush()
        lines = output.getvalue().splitlines()
        assert lines, "Enabled tracing must export spans, not just allocate a provider"
        record = json.loads(lines[-1])
        assert record["name"] == "worker.process_task"
        assert record["attributes"] == {"attempt": 1}
        assert "PRIVATE" not in output.getvalue()
    finally:
        shutdown_otel_bridge()


@pytest.fixture
def trace_output(monkeypatch):
    pytest.importorskip("opentelemetry.sdk.trace")
    from dadbot.telemetry.otel_bridge import shutdown_otel_bridge
    from dadbot_system.contracts import TelemetrySettings
    from dadbot_system.telemetry import configure_tracing

    shutdown_otel_bridge()
    output = io.StringIO()
    monkeypatch.setattr(sys, "stderr", output)
    assert configure_tracing(TelemetrySettings(otel_enabled=True)) is not None
    try:
        yield output
    finally:
        shutdown_otel_bridge()


def test_kernel_spans_share_otel_parentage_without_changing_kernel_ids(trace_output):
    from dadbot.core.kernel_signals import get_tracer
    from dadbot.telemetry.otel_bridge import start_span
    from dadbot_system.runtime_signals import start_span as service_start_span

    tracer = get_tracer()
    before = tracer.current_trace_id()
    with service_start_span("orchestrator.submit_chat", session_id="PRIVATE-SESSION"):
        with tracer.span("scheduler.drain_once", trace_token="PRIVATE-TRACE") as kernel_span:
            assert kernel_span.trace_id == "PRIVATE-TRACE"
            assert tracer.current_trace_id() == "PRIVATE-TRACE"
            with start_span("gen_ai.chat", **{"gen_ai.provider.name": "ollama"}):
                pass
    assert tracer.current_trace_id() == before
    records = [json.loads(line) for line in trace_output.getvalue().splitlines()]
    model, kernel, service = records
    assert model["parent_span_id"] == kernel["span_id"]
    assert kernel["parent_span_id"] == service["span_id"]
    assert len({record["trace_id"] for record in records}) == 1
    assert "PRIVATE" not in trace_output.getvalue()


def test_span_errors_propagate_without_exporting_exception_content(trace_output):
    from dadbot.telemetry.otel_bridge import start_span

    with pytest.raises(RuntimeError, match="PRIVATE-ERROR"):
        with start_span("gen_ai.chat"):
            raise RuntimeError("PRIVATE-ERROR")
    record = json.loads(trace_output.getvalue())
    assert record["status"] == "ERROR"
    assert "PRIVATE" not in trace_output.getvalue()


def test_cancellation_still_propagates_through_kernel_and_otel_contexts(trace_output):
    from dadbot.core.kernel_signals import get_tracer

    tracer = get_tracer()
    before = tracer.current_trace_id()
    with pytest.raises(asyncio.CancelledError):
        with tracer.span("scheduler.drain_once"):
            raise asyncio.CancelledError("PRIVATE-CANCELLATION")
    assert tracer.current_trace_id() == before
    assert json.loads(trace_output.getvalue())["status"] == "ERROR"
    assert "PRIVATE" not in trace_output.getvalue()


@pytest.mark.parametrize("stream", [False, True])
def test_real_ollama_transport_boundary_emits_only_safe_metadata(trace_output, monkeypatch, stream):
    from dadbot.managers import runtime_client

    class Client:
        def __init__(self, **kwargs):
            self._client = nullcontext()

        def chat(self, **kwargs):
            response = {"message": {"content": "PRIVATE-RESPONSE"}}
            return iter([response]) if kwargs.get("stream") else response

    monkeypatch.setattr(runtime_client.ollama, "Client", Client)
    response = runtime_client.RuntimeClientManager._call_ollama_sync(
        messages=[{"role": "user", "content": "PRIVATE-PROMPT"}], stream=stream
    )
    if stream:
        assert list(response)
    record = json.loads(trace_output.getvalue())
    assert record["name"] == "gen_ai.chat"
    assert record["attributes"] == {"gen_ai.operation.name": "chat", "gen_ai.provider.name": "ollama"}
    assert "PRIVATE" not in trace_output.getvalue()


def test_mcp_transport_span_never_exports_tool_arguments_or_results(trace_output):
    from dadbot.runtime.mcp.tool_client import MCPToolClient

    class Session:
        async def call_tool(self, name, arguments):
            assert name == "PRIVATE-TOOL"
            assert arguments == {"value": "PRIVATE-ARGUMENT"}
            return SimpleNamespace(structured_content={"value": "PRIVATE-RESULT"}, is_error=True)

    result = asyncio.run(MCPToolClient.call_tool(Session(), "PRIVATE-TOOL", {"value": "PRIVATE-ARGUMENT"}))
    assert result["is_error"] is True
    record = json.loads(trace_output.getvalue())
    assert record["name"] == "mcp.call_tool"
    assert record["attributes"] == {"mcp.method.name": "tools/call", "mcp.result.is_error": True}
    assert "PRIVATE" not in trace_output.getvalue()


def test_exporter_filters_attributes_names_events_and_status_descriptions(trace_output):
    from opentelemetry.trace import Status, StatusCode

    from dadbot.telemetry.otel_bridge import get_trace_provider

    with get_trace_provider().get_tracer("test").start_as_current_span("PRIVATE-NAME") as span:
        span.set_attribute("gen_ai.provider.name", "PRIVATE-PROVIDER")
        span.set_attribute("attempt", "PRIVATE-ATTEMPT")
        span.set_attribute("prompt", "PRIVATE-PROMPT")
        span.add_event("PRIVATE-EVENT")
        span.set_status(Status(StatusCode.ERROR, "PRIVATE-ERROR"))
    record = json.loads(trace_output.getvalue())
    assert record["name"] == "dadbot.operation"
    assert record["attributes"] == {}
    assert record["status"] == "ERROR"
    assert "PRIVATE" not in trace_output.getvalue()


def test_trace_installation_is_idempotent_and_shutdown_flushes(trace_output, monkeypatch):
    from dadbot.telemetry.otel_bridge import get_trace_provider, install_otel_bridge, shutdown_otel_bridge, start_span

    provider = get_trace_provider()
    flushed = []
    original_flush = provider.force_flush

    def flush(*args, **kwargs):
        flushed.append(True)
        return original_flush(*args, **kwargs)

    monkeypatch.setattr(provider, "force_flush", flush)
    assert install_otel_bridge()["installed"]
    assert get_trace_provider() is provider
    with start_span("worker.process_task"):
        pass
    shutdown_otel_bridge()
    assert flushed == [True]
    assert get_trace_provider() is None
    assert len(trace_output.getvalue().splitlines()) == 1


def test_invalid_spans_and_output_failures_are_reported(caplog):
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult

    from dadbot.telemetry.trace_export import PrivacySafeConsoleSpanExporter

    exporter = PrivacySafeConsoleSpanExporter(io.StringIO())
    assert exporter.export([ReadableSpan("PRIVATE", context=None)]) == SpanExportResult.FAILURE

    class BrokenOutput(io.StringIO):
        def write(self, text):
            raise OSError("PRIVATE-OUTPUT-FAILURE")

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(PrivacySafeConsoleSpanExporter(BrokenOutput())))
    try:
        with provider.get_tracer("test").start_as_current_span("gen_ai.chat"):
            pass
    finally:
        provider.shutdown()
    assert "without a valid context" in caplog.text
    assert "DadBot span export failed" in caplog.text
    assert "PRIVATE" not in caplog.text


def test_installing_trace_export_does_not_enable_raw_event_export(monkeypatch):
    pytest.importorskip("opentelemetry.sdk.trace")
    from dadbot.core import kernel_signals
    from dadbot.telemetry.otel_bridge import shutdown_otel_bridge

    shutdown_otel_bridge()
    raw_events = []
    exporter = kernel_signals.EventStreamExporter(sink=raw_events.append, enabled=False)
    monkeypatch.setattr(kernel_signals, "_global_exporter", exporter)
    try:
        assert install_otel_bridge()["installed"]
        assert kernel_signals.get_exporter() is exporter
        exporter.export({"prompt": "PRIVATE"})
        assert raw_events == []
    finally:
        shutdown_otel_bridge()


def test_requested_tracing_without_sdk_is_an_explicit_failure(monkeypatch):
    from dadbot.telemetry import otel_bridge
    from dadbot_system.contracts import TelemetrySettings
    from dadbot_system.telemetry import configure_tracing

    otel_bridge.shutdown_otel_bridge()

    def unavailable(name):
        raise ImportError("synthetic missing SDK")

    monkeypatch.setattr(otel_bridge.importlib, "import_module", unavailable)
    with pytest.raises(RuntimeError, match="opentelemetry unavailable"):
        configure_tracing(TelemetrySettings(otel_enabled=True))


def test_api_lifespan_initializes_and_flushes_its_trace_provider(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from dadbot.telemetry.otel_bridge import get_trace_provider, shutdown_otel_bridge
    from dadbot_system.api import create_api_app
    from dadbot_system.contracts import ServiceConfig
    from dadbot_system.orchestration import DadBotOrchestrator
    from dadbot_system.telemetry import configure_tracing

    shutdown_otel_bridge()
    output = io.StringIO()
    monkeypatch.setattr(sys, "stderr", output)
    config = ServiceConfig.from_dict({"security": {"auth_required": False}, "telemetry": {"otel_enabled": True}})
    assert configure_tracing(config.telemetry) is not None
    with TestClient(create_api_app(DadBotOrchestrator(SimpleNamespace()), config=config)):
        assert get_trace_provider() is not None
    assert get_trace_provider() is None
