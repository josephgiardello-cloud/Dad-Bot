"""OpenTelemetry overlay bridge for DadBot observability.

Design goal: non-invasive adapter layer.
- Does not modify kernel or graph execution code.
- Wraps existing observability sinks and forwards data to OTel when available.
- Falls back gracefully if opentelemetry is not installed.
"""

from __future__ import annotations

import importlib
import logging
from contextlib import nullcontext
from threading import RLock
from typing import Any

from dadbot.core.kernel_signals import EventStreamExporter, MetricsSink

logger = logging.getLogger(__name__)
_provider: Any = None
_install_lock = RLock()


class OpenTelemetryMetricsSink(MetricsSink):
    """Metrics sink that preserves local counters and mirrors to OTel."""

    def __init__(self, *, meter: Any | None = None) -> None:
        super().__init__()
        self._meter = meter
        self._otel_counters: dict[str, Any] = {}
        self._otel_histograms: dict[str, Any] = {}

    def increment(self, key: str, value: int = 1) -> None:
        super().increment(key, value)
        if self._meter is None:
            return
        metric_key = str(key or "unknown")
        counter = self._otel_counters.get(metric_key)
        if counter is None:
            counter = self._meter.create_counter(metric_key)
            self._otel_counters[metric_key] = counter
        counter.add(max(0, int(value)))

    def observe(self, key: str, value: float) -> None:
        super().observe(key, value)
        if self._meter is None:
            return
        metric_key = str(key or "unknown")
        histogram = self._otel_histograms.get(metric_key)
        if histogram is None:
            histogram = self._meter.create_histogram(metric_key)
            self._otel_histograms[metric_key] = histogram
        histogram.record(float(value))


class OpenTelemetryEventExporter(EventStreamExporter):
    """Event exporter that forwards records to OTel log/event sink if provided."""

    def __init__(self, *, event_emitter: Any | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._event_emitter = event_emitter

    def export(self, record: dict[str, Any], *, level=None) -> None:  # type: ignore[override]
        super().export(record, level=level)
        if self._event_emitter is None:
            return
        try:
            self._event_emitter.emit(dict(record))
        except Exception:
            logger.debug("OpenTelemetry event emission failed", exc_info=True)


def install_otel_bridge(*, service_name: str = "dadbot") -> dict[str, Any]:
    """Install OTel-backed observability adapters if SDK is available.

    Returns:
        dict with installation status and reason for diagnostics.

    """
    global _provider
    with _install_lock:
        if _provider is not None:
            return {"installed": True, "reason": "already installed"}
        try:
            otel_metrics = importlib.import_module("opentelemetry.metrics")
            from dadbot.telemetry.trace_export import create_trace_provider
        except ImportError as exc:
            logger.warning("OpenTelemetry SDK unavailable; install the service dependencies")
            return {"installed": False, "reason": f"opentelemetry unavailable: {exc}"}

        import dadbot.core.kernel_signals as obs

        _provider = create_trace_provider(service_name)
        obs._global_metrics = OpenTelemetryMetricsSink(meter=otel_metrics.get_meter("dadbot.core"))
        obs._span_bridge = start_span
        return {"installed": True, "reason": "ok"}


def get_trace_provider():
    return _provider


def start_span(name: str, **attributes: Any):
    if _provider is None:
        return nullcontext(None)
    from dadbot.telemetry.trace_export import start_span as sdk_start_span

    return sdk_start_span(_provider, name, attributes)


def shutdown_otel_bridge() -> None:
    global _provider
    with _install_lock:
        provider, _provider = _provider, None
        if provider is not None:
            if not provider.force_flush():
                logger.error("DadBot span flush did not complete")
            provider.shutdown()
