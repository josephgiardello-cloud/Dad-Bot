"""OpenTelemetry overlay bridge for DadBot observability.

Design goal: non-invasive adapter layer.
- Does not modify kernel or graph execution code.
- Wraps existing observability sinks and forwards data to OTel when available.
- Falls back gracefully if opentelemetry is not installed.
"""

from __future__ import annotations

import logging
from typing import Any

from dadbot.core.observability import EventStreamExporter, MetricsSink

logger = logging.getLogger(__name__)


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
        except Exception:  # noqa: BLE001
            logger.debug("OpenTelemetry event emission failed", exc_info=True)


def install_otel_bridge() -> dict[str, Any]:
    """Install OTel-backed observability adapters if SDK is available.

    Returns:
        dict with installation status and reason for diagnostics.

    """
    try:
        from opentelemetry import metrics as otel_metrics
    except Exception as exc:  # noqa: BLE001
        return {"installed": False, "reason": f"opentelemetry unavailable: {exc}"}

    try:
        import dadbot.core.observability as obs

        meter = otel_metrics.get_meter("dadbot.core")
        obs._global_metrics = OpenTelemetryMetricsSink(meter=meter)  # type: ignore[attr-defined]
        obs.configure_exporter(enabled=True)
        return {"installed": True, "reason": "ok"}
    except Exception as exc:  # noqa: BLE001
        return {"installed": False, "reason": f"install failed: {exc}"}
