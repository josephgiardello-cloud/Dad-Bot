"""Tier 2 — Observability Minimal: execution trace must be emittable and readable.

Warn-only: observability gaps are reported but do not block distribution.
"""
from __future__ import annotations

import pytest

from dadbot.core.execution_context import (
    ExecutionTraceRecorder,
    bind_execution_trace,
    active_execution_trace,
    record_execution_step,
)


pytestmark = pytest.mark.unit


class TestObservabilityMinimal:
    def test_active_trace_is_none_outside_bind_scope(self):
        """No active trace must exist outside a bind_execution_trace scope."""
        assert active_execution_trace() is None

    def test_active_trace_is_accessible_inside_bind_scope(self):
        """Inside bind_execution_trace, active_execution_trace() returns the recorder."""
        recorder = ExecutionTraceRecorder(trace_id="obs-1", prompt="test")
        with bind_execution_trace(recorder):
            assert active_execution_trace() is recorder

    def test_active_trace_is_none_after_bind_scope_exits(self):
        """After bind_execution_trace exits, active_execution_trace() returns None."""
        recorder = ExecutionTraceRecorder(trace_id="obs-2", prompt="test")
        with bind_execution_trace(recorder):
            pass
        assert active_execution_trace() is None

    def test_record_execution_step_appends_to_active_trace(self):
        """record_execution_step must append to the active trace recorder."""
        recorder = ExecutionTraceRecorder(trace_id="obs-3", prompt="test")
        with bind_execution_trace(recorder):
            record_execution_step("memory_load", payload={"key": "v"})
            record_execution_step("model_call", payload={"provider": "ollama"})
        assert len(recorder.steps) == 2
        assert recorder.steps[0]["operation"] == "memory_load"
        assert recorder.steps[1]["operation"] == "model_call"

    def test_recorder_steps_are_copies_not_references(self):
        """recorder.steps must return copies — mutations must not affect the recorder."""
        recorder = ExecutionTraceRecorder(trace_id="obs-4", prompt="test")
        with bind_execution_trace(recorder):
            record_execution_step("step_a")
        steps = recorder.steps
        steps.clear()
        assert len(recorder.steps) == 1

    def test_execution_step_outside_trace_returns_none_without_required(self):
        """record_execution_step returns None outside a trace when required=False."""
        result = record_execution_step("step_x", required=False)
        assert result is None
