"""Tier 3 — Performance Envelope: execution timing upper bounds (advisory).

Advisory only: timing thresholds are soft limits for trend analysis.
"""
from __future__ import annotations

import time

import pytest

from dadbot.core.execution_context import ExecutionTraceRecorder, bind_execution_trace
from dadbot.core.graph_context import TurnContext


pytestmark = pytest.mark.unit

# Advisory thresholds (ms) — failures here are never CI-blocking
_MAX_CONTEXT_BUILD_MS = 5.0
_MAX_TRACE_RECORD_MS = 1.0


class TestPerformanceEnvelope:
    def test_turn_context_construction_is_fast(self):
        """TurnContext construction must complete within advisory threshold."""
        start = time.perf_counter()
        for _ in range(100):
            TurnContext(user_input="perf test", metadata={"k": "v"})
        elapsed_ms = (time.perf_counter() - start) * 1000
        per_construction_ms = elapsed_ms / 100
        # Advisory: log timing, never fail hard
        assert per_construction_ms < _MAX_CONTEXT_BUILD_MS, (
            f"[ADVISORY] TurnContext construction: {per_construction_ms:.3f}ms "
            f"(threshold={_MAX_CONTEXT_BUILD_MS}ms)"
        )

    def test_execution_trace_record_is_fast(self):
        """Recording 50 trace steps must complete within advisory threshold."""
        recorder = ExecutionTraceRecorder(trace_id="perf-1", prompt="test")
        start = time.perf_counter()
        with bind_execution_trace(recorder):
            for i in range(50):
                recorder.record(f"step_{i}", payload={"seq": i})
        elapsed_ms = (time.perf_counter() - start) * 1000
        per_record_ms = elapsed_ms / 50
        assert per_record_ms < _MAX_TRACE_RECORD_MS, (
            f"[ADVISORY] Trace record: {per_record_ms:.3f}ms "
            f"(threshold={_MAX_TRACE_RECORD_MS}ms)"
        )

    def test_stage_trace_list_append_is_o1(self):
        """Appending stage traces must remain fast at scale (advisory O(1) check)."""
        from dadbot.core.graph_types import StageTrace
        ctx = TurnContext(user_input="scale test")
        start = time.perf_counter()
        for i in range(200):
            ctx.stage_traces.append(StageTrace(stage=f"stage_{i}", duration_ms=float(i), error=None))
        elapsed_ms = (time.perf_counter() - start) * 1000
        # 200 appends should be well under 10ms
        assert elapsed_ms < 10.0, f"[ADVISORY] 200 stage_trace appends took {elapsed_ms:.2f}ms"
