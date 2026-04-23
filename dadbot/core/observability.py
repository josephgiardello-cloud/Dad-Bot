"""Structured observability layer — trace IDs, metrics, and event stream export.

Three components:
  TracingContext   — thread-local + contextvars trace/span ID propagation.
  MetricsSink      — in-process counter + histogram; swappable for OTel.
  EventStreamExporter — publishes events to stdout JSON or an async queue.

Usage:
    from dadbot.core.observability import get_metrics, get_tracer, EventStreamExporter

    tracer = get_tracer()
    with tracer.span("scheduler.drain_once") as span:
        metrics = get_metrics()
        metrics.increment("job.completed")
        metrics.observe("job.latency_ms", elapsed_ms)
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
from collections import defaultdict
from contextvars import ContextVar
from copy import deepcopy
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Trace / Span IDs
# ---------------------------------------------------------------------------

_current_trace_id: ContextVar[str] = ContextVar("_current_trace_id", default="")
_current_span_id: ContextVar[str] = ContextVar("_current_span_id", default="")


def _new_id(prefix: str = "") -> str:
    import uuid
    return (prefix + uuid.uuid4().hex)[:32]


class Span:
    """Lightweight span that propagates trace/span IDs via ContextVars."""

    def __init__(self, name: str, *, trace_id: str = "", parent_span_id: str = "") -> None:
        self.name = str(name or "unnamed")
        self.trace_id = str(trace_id or _current_trace_id.get() or _new_id("tr"))
        self.span_id = _new_id("sp")
        self.parent_span_id = str(parent_span_id or _current_span_id.get() or "")
        self.started_at: float = time.time()
        self.ended_at: float = 0.0
        self._trace_token = None
        self._span_token = None

    def __enter__(self) -> "Span":
        self._trace_token = _current_trace_id.set(self.trace_id)
        self._span_token = _current_span_id.set(self.span_id)
        return self

    def __exit__(self, *_) -> None:
        self.ended_at = time.time()
        if self._trace_token is not None:
            _current_trace_id.reset(self._trace_token)
        if self._span_token is not None:
            _current_span_id.reset(self._span_token)

    @property
    def duration_ms(self) -> float:
        end = self.ended_at or time.time()
        return (end - self.started_at) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
        }


class TracingContext:
    """Tracer — creates spans and propagates trace IDs."""

    def span(self, name: str, *, trace_id: str = "") -> Span:
        return Span(name, trace_id=trace_id)

    @staticmethod
    def current_trace_id() -> str:
        return _current_trace_id.get() or ""

    @staticmethod
    def current_span_id() -> str:
        return _current_span_id.get() or ""

    @staticmethod
    def ensure_trace_id(trace_id: str = "") -> str:
        """Return existing trace ID or create and install a new one."""
        existing = _current_trace_id.get()
        if existing:
            return existing
        new_id = str(trace_id or _new_id("tr"))
        _current_trace_id.set(new_id)
        return new_id


# ---------------------------------------------------------------------------
# Metrics sink
# ---------------------------------------------------------------------------

class MetricsSink:
    """In-process metrics collector.

    Counters: integer totals (e.g. job.completed, job.failed).
    Histograms: list of float samples (e.g. job.latency_ms).

    Swappable for OpenTelemetry by subclassing and overriding increment/observe.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._counters: dict[str, int] = defaultdict(int)
        self._histograms: dict[str, list[float]] = defaultdict(list)

    def increment(self, key: str, value: int = 1) -> None:
        with self._lock:
            self._counters[str(key or "unknown")] += max(0, int(value))

    def observe(self, key: str, value: float) -> None:
        with self._lock:
            self._histograms[str(key or "unknown")].append(float(value))

    def counter(self, key: str) -> int:
        with self._lock:
            return self._counters.get(str(key or "unknown"), 0)

    def histogram_summary(self, key: str) -> dict[str, Any]:
        with self._lock:
            samples = list(self._histograms.get(str(key or "unknown"), []))
        if not samples:
            return {"count": 0, "min": None, "max": None, "mean": None, "p99": None}
        samples_sorted = sorted(samples)
        count = len(samples_sorted)
        p99_index = max(0, int(count * 0.99) - 1)
        return {
            "count": count,
            "min": samples_sorted[0],
            "max": samples_sorted[-1],
            "mean": sum(samples_sorted) / count,
            "p99": samples_sorted[p99_index],
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "histograms": {
                    key: self.histogram_summary(key)
                    for key in self._histograms
                },
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


# ---------------------------------------------------------------------------
# Event stream exporter
# ---------------------------------------------------------------------------

class EventStreamExporter:
    """Publishes structured events to stdout JSON (default) or an async queue.

    Compatible with OpenTelemetry log exporters — each record is a valid
    JSON object on a single line.
    """

    def __init__(
        self,
        *,
        sink: Any = None,
        enabled: bool = True,
    ) -> None:
        """
        Args:
            sink: a callable(record: dict) or a queue.Queue.  Defaults to
                  sys.stdout JSON line writer.
            enabled: set False to silence all exports (e.g. in tests).
        """
        self._enabled = bool(enabled)
        if sink is None:
            self._sink = self._stdout_sink
        elif isinstance(sink, queue.Queue):
            self._sink = sink.put_nowait
        else:
            self._sink = sink

    def export(self, record: dict[str, Any]) -> None:
        if not self._enabled:
            return
        enriched = {
            "exported_at": time.time(),
            "trace_id": _current_trace_id.get() or "",
            "span_id": _current_span_id.get() or "",
            **record,
        }
        try:
            self._sink(enriched)
        except Exception:
            pass  # Exporter must never crash the runtime.

    @staticmethod
    def _stdout_sink(record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False


# ---------------------------------------------------------------------------
# Global singletons (opt-in — callers import and use these)
# ---------------------------------------------------------------------------

_global_tracer = TracingContext()
_global_metrics = MetricsSink()
_global_exporter = EventStreamExporter(enabled=False)  # silent by default


def get_tracer() -> TracingContext:
    return _global_tracer


def get_metrics() -> MetricsSink:
    return _global_metrics


def get_exporter() -> EventStreamExporter:
    return _global_exporter


def configure_exporter(sink: Any = None, *, enabled: bool = True) -> None:
    """Replace the global event stream exporter."""
    global _global_exporter
    _global_exporter = EventStreamExporter(sink=sink, enabled=enabled)


# ---------------------------------------------------------------------------
# Correlation context (Tier 2 item 9 — end-to-end request correlation)
# ---------------------------------------------------------------------------

_current_correlation_id: ContextVar[str] = ContextVar("_current_correlation_id", default="")


class CorrelationContext:
    """Propagates a correlation ID across scheduler → ledger → kernel.

    A correlation ID represents a single end-to-end user request, surviving
    across multiple spans and component boundaries.

    Usage::

        with CorrelationContext.bind("req-abc123"):
            ledger_writer.write_event(...)   # all writes inherit the correlation ID
    """

    @staticmethod
    def bind(correlation_id: str = "") -> "_CorrelationScope":
        cid = str(correlation_id or _new_id("corr"))
        return _CorrelationScope(cid)

    @staticmethod
    def current() -> str:
        return _current_correlation_id.get() or ""

    @staticmethod
    def ensure() -> str:
        existing = _current_correlation_id.get()
        if existing:
            return existing
        new_cid = _new_id("corr")
        _current_correlation_id.set(new_cid)
        return new_cid


class _CorrelationScope:
    def __init__(self, cid: str) -> None:
        self._cid = cid
        self._token = None

    def __enter__(self) -> "str":
        self._token = _current_correlation_id.set(self._cid)
        return self._cid

    def __exit__(self, *_) -> None:
        if self._token is not None:
            _current_correlation_id.reset(self._token)


# ---------------------------------------------------------------------------
# Structured logger (Tier 2 item 9 — structured logs tied to event IDs)
# ---------------------------------------------------------------------------

class LogLevel(str):
    DEBUG   = "DEBUG"
    INFO    = "INFO"
    WARNING = "WARNING"
    ERROR   = "ERROR"


class StructuredLogger:
    """Emits structured JSON log records correlated with event/session/trace IDs.

    Records include: timestamp, level, message, session_id, event_id,
    trace_id, span_id, correlation_id, and any extra kwargs.

    Compatible with any JSON log aggregator (Loki, Datadog, CloudWatch Logs).

    Usage::

        logger = StructuredLogger("dadbot.scheduler")
        logger.info("job started", session_id="s1", event_id="evt-42")
    """

    def __init__(
        self,
        name: str = "dadbot",
        *,
        sink: Any = None,
        min_level: str = LogLevel.DEBUG,
        enabled: bool = True,
    ) -> None:
        self._name    = str(name)
        self._enabled = bool(enabled)
        self._lock    = threading.RLock()
        self._records: list[dict[str, Any]] = []

        if sink is None:
            self._sink = self._stderr_sink
        elif isinstance(sink, list):
            captured = sink
            self._sink = captured.append
        elif callable(sink):
            self._sink = sink
        else:
            self._sink = self._stderr_sink

        self._LEVELS = {
            LogLevel.DEBUG:   0,
            LogLevel.INFO:    1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR:   3,
        }
        self._min_level_value = self._LEVELS.get(str(min_level), 0)

    def _emit(self, level: str, message: str, **extra: Any) -> dict[str, Any]:
        if not self._enabled:
            return {}
        if self._LEVELS.get(level, 0) < self._min_level_value:
            return {}
        record: dict[str, Any] = {
            "timestamp":      time.time(),
            "level":          level,
            "logger":         self._name,
            "message":        str(message),
            "trace_id":       _current_trace_id.get() or "",
            "span_id":        _current_span_id.get() or "",
            "correlation_id": _current_correlation_id.get() or "",
        }
        record.update(extra)
        with self._lock:
            self._records.append(record)
        try:
            self._sink(record)
        except Exception:
            pass
        return record

    def debug(self, message: str, **extra: Any) -> None:
        self._emit(LogLevel.DEBUG, message, **extra)

    def info(self, message: str, **extra: Any) -> None:
        self._emit(LogLevel.INFO, message, **extra)

    def warning(self, message: str, **extra: Any) -> None:
        self._emit(LogLevel.WARNING, message, **extra)

    def error(self, message: str, **extra: Any) -> None:
        self._emit(LogLevel.ERROR, message, **extra)

    def records(self, *, level: str = "") -> list[dict[str, Any]]:
        with self._lock:
            if level:
                return [r for r in self._records if r.get("level") == level]
            return list(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    @staticmethod
    def _stderr_sink(record: dict[str, Any]) -> None:
        try:
            sys.stderr.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Replay debugger (Tier 2 item 9 — step through execution timeline)
# ---------------------------------------------------------------------------

class ReplayDebugger:
    """Step through ledger events and inspect state at each point.

    Usage::

        debugger = ReplayDebugger(reducer)
        steps = debugger.debug_session("s1", ledger.read())
        for step in steps:
            print(step["event"]["type"], step["state"]["sessions"]["s1"])
    """

    def __init__(self, reducer=None) -> None:
        self._reducer = reducer

    def step_through(
        self,
        events: list[dict[str, Any]],
    ):
        """Generator: yield (event, state_snapshot) for each event in order."""
        accumulated: list[dict[str, Any]] = []
        for event in events:
            accumulated.append(event)
            state = self._reduce(accumulated)
            yield {"event": dict(event), "state": state, "seq": len(accumulated)}

    def debug_session(
        self,
        session_id: str,
        events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return a list of {event, state, seq} steps for a single session."""
        session_events = [
            e for e in events
            if str(e.get("session_id") or "") == str(session_id)
        ]
        return list(self.step_through(session_events))

    def diff_states(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a shallow diff of two state dicts."""
        all_keys = set(before) | set(after)
        diff: dict[str, Any] = {}
        for key in all_keys:
            bv = before.get(key)
            av = after.get(key)
            if bv != av:
                diff[key] = {"before": bv, "after": av}
        return diff

    def _reduce(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        if self._reducer is not None:
            try:
                return self._reducer.reduce(events)
            except Exception:
                pass
        # Fallback: aggregate by session_id.
        state: dict[str, Any] = {"sessions": {}}
        for e in events:
            sid = str(e.get("session_id") or "")
            if sid:
                state["sessions"].setdefault(sid, {})
                if e.get("type") == "JOB_COMPLETED":
                    state["sessions"][sid]["last_result"] = (
                        e.get("payload") or {}
                    ).get("result")
        return state
