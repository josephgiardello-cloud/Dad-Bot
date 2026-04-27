"""Prometheus overlay bridge for DadBot observability.

Non-invasive design:
- Exposes /metrics from existing in-process MetricsSink snapshots.
- Does not alter kernel or graph execution paths.
"""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from dadbot.core.observability import get_metrics


def _normalize_metric_name(name: str) -> str:
    return "dadbot_" + "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name or "metric"))


def _render_prometheus() -> str:
    snapshot = get_metrics().snapshot()
    lines: list[str] = []

    counters = dict(snapshot.get("counters") or {})
    histograms = dict(snapshot.get("histograms") or {})

    for key, value in counters.items():
        metric = _normalize_metric_name(key)
        lines.append(f"# TYPE {metric} counter")
        lines.append(f"{metric} {int(value)}")

    for key, summary in histograms.items():
        metric = _normalize_metric_name(key)
        lines.append(f"# TYPE {metric}_count gauge")
        lines.append(f"{metric}_count {int(summary.get('count') or 0)}")
        lines.append(f"# TYPE {metric}_mean gauge")
        mean_value = summary.get("mean")
        lines.append(f"{metric}_mean {0.0 if mean_value is None else float(mean_value)}")
        lines.append(f"# TYPE {metric}_p99 gauge")
        p99_value = summary.get("p99")
        lines.append(f"{metric}_p99 {0.0 if p99_value is None else float(p99_value)}")

    return "\n".join(lines) + "\n"


class _MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"not found")
            return
        payload = _render_prometheus().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


class PrometheusExporter:
    def __init__(self, host: str = "127.0.0.1", port: int = 9464) -> None:
        self.host = str(host)
        self.port = int(port)
        self._server = ThreadingHTTPServer((self.host, self.port), _MetricsHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def start_prometheus_exporter(host: str = "127.0.0.1", port: int = 9464) -> PrometheusExporter:
    exporter = PrometheusExporter(host=host, port=port)
    exporter.start()
    return exporter
