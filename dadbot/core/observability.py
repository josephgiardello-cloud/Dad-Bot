"""Compatibility shim for legacy observability imports.

The single implementation authority lives in ``dadbot.core.kernel_signals``.
This module intentionally re-exports that surface so older imports continue to
work without maintaining a second observability implementation.
"""

from __future__ import annotations

from dadbot.core.kernel_signals import (
    CorrelationContext,
    EventStreamExporter,
    LogLevel,
    MetricsSink,
    ReplayDebugger,
    Span,
    StructuredLogger,
    TraceLevel,
    TracingContext,
    _NoOpSpan,
    _current_correlation_id,
    _current_span_id,
    _current_trace_id,
    _global_exporter,
    _global_metrics,
    _global_tracer,
    configure_exporter,
    get_exporter,
    get_metrics,
    get_tracer,
    set_trace_level,
)

__all__ = [
    "CorrelationContext",
    "EventStreamExporter",
    "LogLevel",
    "MetricsSink",
    "ReplayDebugger",
    "Span",
    "StructuredLogger",
    "TraceLevel",
    "TracingContext",
    "_NoOpSpan",
    "_current_correlation_id",
    "_current_span_id",
    "_current_trace_id",
    "_global_exporter",
    "_global_metrics",
    "_global_tracer",
    "configure_exporter",
    "get_exporter",
    "get_metrics",
    "get_tracer",
    "set_trace_level",
]