from __future__ import annotations

from dadbot.core.observability import (
    configure_exporter,
    EventStreamExporter,
    get_exporter,
    get_tracer,
    set_trace_level,
    TraceLevel,
    TracingContext,
)


def test_minimal_events_are_not_sampled_out() -> None:
    sink: list[dict] = []
    exporter = EventStreamExporter(
        sink=sink.append,
        enabled=True,
        min_level=TraceLevel.MINIMAL,
        sample_rate=0.0,
    )

    exporter.export({"event": "minimal.event"}, level=TraceLevel.MINIMAL)

    assert len(sink) == 1
    assert sink[0]["event"] == "minimal.event"
    assert sink[0]["trace_level"] == "MINIMAL"


def test_debug_events_respect_sampling() -> None:
    sink: list[dict] = []
    exporter = EventStreamExporter(
        sink=sink.append,
        enabled=True,
        min_level=TraceLevel.MINIMAL,
        sample_rate=0.0,
    )

    exporter.export({"event": "debug.event"}, level=TraceLevel.DEBUG)

    assert sink == []


def test_audit_events_bypass_sampling_and_min_level() -> None:
    sink: list[dict] = []
    exporter = EventStreamExporter(
        sink=sink.append,
        enabled=True,
        min_level=TraceLevel.OFF,
        sample_rate=0.0,
    )

    exporter.export({"event": "audit.event"}, level=TraceLevel.AUDIT)

    assert len(sink) == 1
    assert sink[0]["trace_level"] == "AUDIT"


def test_tracer_off_returns_noop_span() -> None:
    tracer = TracingContext(min_level=TraceLevel.OFF)

    with tracer.span("disabled") as span:
        assert TracingContext.current_trace_id() == ""
        assert span.trace_id == ""

    assert TracingContext.current_trace_id() == ""


def test_set_trace_level_updates_global_tracer_and_exporter() -> None:
    try:
        set_trace_level(TraceLevel.DEBUG)
        assert get_tracer().level() == TraceLevel.DEBUG

        sink: list[dict] = []
        configure_exporter(sink=sink.append, enabled=True, min_level=TraceLevel.DEBUG, sample_rate=1.0)
        get_exporter().export({"event": "debug-visible"}, level=TraceLevel.DEBUG)
        assert any(item.get("event") == "debug-visible" for item in sink)
    finally:
        set_trace_level(TraceLevel.MINIMAL)


def test_configure_exporter_sampling_and_level_behavior() -> None:
    sink: list[dict] = []
    configure_exporter(
        sink=sink.append,
        enabled=True,
        min_level=TraceLevel.MINIMAL,
        sample_rate=0.0,
    )

    exporter = get_exporter()
    exporter.export({"event": "minimal-visible"}, level=TraceLevel.MINIMAL)
    exporter.export({"event": "debug-sampled-out"}, level=TraceLevel.DEBUG)

    events = [item.get("event") for item in sink]
    assert "minimal-visible" in events
    assert "debug-sampled-out" not in events
