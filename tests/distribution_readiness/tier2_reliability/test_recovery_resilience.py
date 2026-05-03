"""Tier 2 — Recovery Resilience: pipeline resilience after partial failure.

Warn-only: failures here are reported but do not block distribution.
"""
from __future__ import annotations

import pytest

from dadbot.core.graph_context import TurnContext
from dadbot.core.graph_types import StageTrace


pytestmark = pytest.mark.unit


class TestRecoveryResilience:
    def test_turn_context_can_be_constructed_with_minimal_fields(self):
        """TurnContext must always be constructible with only user_input."""
        ctx = TurnContext(user_input="minimal")
        assert ctx.user_input == "minimal"
        assert isinstance(ctx.state, dict)
        assert isinstance(ctx.metadata, dict)
        assert isinstance(ctx.stage_traces, list)

    def test_stage_traces_accumulate_across_stages(self):
        """Stage traces must accumulate, not overwrite, across pipeline stages."""
        ctx = TurnContext(user_input="accumulate")
        ctx.stage_traces.append(StageTrace(stage="temporal", duration_ms=1.2, error=None))
        ctx.stage_traces.append(StageTrace(stage="inference", duration_ms=45.0, error=None))
        assert len(ctx.stage_traces) == 2
        assert ctx.stage_traces[0].stage == "temporal"
        assert ctx.stage_traces[1].stage == "inference"

    def test_stage_trace_with_error_does_not_discard_duration(self):
        """A stage that errored still records its duration for observability."""
        trace = StageTrace(stage="save", duration_ms=12.5, error="disk full")
        assert trace.duration_ms == 12.5
        assert trace.error == "disk full"

    def test_short_circuit_can_be_activated_by_safety_node(self):
        """short_circuit flag can be set true mid-pipeline to halt downstream stages."""
        ctx = TurnContext(user_input="trigger halt")
        ctx.short_circuit = True
        assert ctx.short_circuit is True

    def test_state_survives_partial_update(self):
        """A partial state write (subset of keys) must not erase other keys."""
        ctx = TurnContext(user_input="partial update")
        ctx.state["key_a"] = "value_a"
        ctx.state["key_b"] = "value_b"
        ctx.state.update({"key_b": "updated_b"})
        assert ctx.state["key_a"] == "value_a"
        assert ctx.state["key_b"] == "updated_b"

    def test_trace_id_is_unique_per_turn_context(self):
        """Each TurnContext auto-generates a unique trace_id."""
        ctx_a = TurnContext(user_input="turn a")
        ctx_b = TurnContext(user_input="turn b")
        assert ctx_a.trace_id != ctx_b.trace_id
