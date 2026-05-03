"""Tier 1 — Determinism: trace hash stability and execution order consistency.

Hard-fail: any failure here means the system is not distribution-ready.
"""
from __future__ import annotations

import pytest

from dadbot.core.execution_context import (
    ExecutionTraceRecorder,
    bind_execution_trace,
    build_execution_trace_context,
    derive_execution_trace_hash,
)
from tests.test_execution_snapshot_dag import _fake_context


pytestmark = pytest.mark.unit


def _build_trace_with_steps(*operations: str) -> dict:
    ctx = _fake_context()
    recorder = ExecutionTraceRecorder(trace_id="det-1", prompt="test")
    with bind_execution_trace(recorder):
        for op in operations:
            recorder.record(op, payload={"seq": recorder.steps.__len__()})
    return build_execution_trace_context(context=ctx, result="ok", recorder=recorder)


class TestDeterminism:
    def test_identical_traces_produce_identical_hash(self):
        """Same inputs + same step sequence → identical final_hash."""
        trace_a = _build_trace_with_steps("memory_load", "model_call", "safety_check")
        trace_b = _build_trace_with_steps("memory_load", "model_call", "safety_check")
        assert trace_a["final_hash"] == trace_b["final_hash"]

    def test_different_step_order_produces_different_hash(self):
        """Reordering steps must diverge the hash (order-sensitivity)."""
        trace_a = _build_trace_with_steps("step_a", "step_b")
        trace_b = _build_trace_with_steps("step_b", "step_a")
        assert trace_a["final_hash"] != trace_b["final_hash"]

    def test_extra_step_diverges_hash(self):
        """An additional execution step changes the trace hash."""
        trace_a = _build_trace_with_steps("step_a")
        trace_b = _build_trace_with_steps("step_a", "step_b")
        assert trace_a["final_hash"] != trace_b["final_hash"]

    def test_hash_is_stable_across_calls(self):
        """derive_execution_trace_hash is pure and idempotent."""
        trace = _build_trace_with_steps("model_call")
        h1 = derive_execution_trace_hash(trace)
        h2 = derive_execution_trace_hash(trace)
        assert h1 == h2

    def test_execution_dag_topological_order_matches_step_sequence(self):
        """DAG topological_order must exactly match recorded step sequence."""
        ops = ["memory_load", "model_call", "safety_check", "save"]
        trace = _build_trace_with_steps(*ops)
        dag = trace["execution_dag"]
        topo = dag["topological_order"]
        # Each node id encodes seq and operation; verify ordering is preserved
        for idx, op in enumerate(ops):
            assert f"step:{idx}:{op}" in topo[idx]
