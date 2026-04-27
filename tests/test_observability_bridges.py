from __future__ import annotations

from urllib.request import urlopen

from dadbot.core.observability import get_metrics
from dadbot.core.otel_bridge import OpenTelemetryEventExporter, install_otel_bridge
from dadbot.core.prometheus_bridge import start_prometheus_exporter


def test_install_otel_bridge_returns_status_dict() -> None:
    status = install_otel_bridge()
    assert isinstance(status, dict)
    assert "installed" in status
    assert "reason" in status


def test_prometheus_exporter_serves_metrics_endpoint() -> None:
    metrics = get_metrics()
    metrics.reset()
    metrics.increment("scheduler.job.completed", 2)

    exporter = start_prometheus_exporter(port=0)
    try:
        port = exporter._server.server_port  # test-only access
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
    exporter = OpenTelemetryEventExporter(
        sink=sink.append,
        enabled=True,
        event_emitter=emitter,
    )

    exporter.export({"event": "bridge.probe"})

    assert any(item.get("event") == "bridge.probe" for item in sink)
    assert any(item.get("event") == "bridge.probe" for item in emitter.events)
