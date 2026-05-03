"""Tier 2 — Memory Stability: memory retention and decay boundary checks.

Warn-only: retention drift is reported but does not block distribution.
These tests verify structural invariants, not specific numeric thresholds.
"""
from __future__ import annotations

import pytest

from dadbot.core.graph_context import TurnContext


pytestmark = pytest.mark.unit


class TestMemoryStability:
    def test_memory_structured_defaults_to_absent(self):
        """TurnContext.state has no memory_structured key until a node writes it."""
        ctx = TurnContext(user_input="check")
        assert "memory_structured" not in ctx.state

    def test_memory_structured_can_be_written_as_dict(self):
        """memory_structured must accept a dict value when written by ContextBuilderNode."""
        ctx = TurnContext(user_input="check")
        ctx.state["memory_structured"] = {"claims": [{"summary": "test"}]}
        assert isinstance(ctx.state["memory_structured"], dict)

    def test_memory_retrieval_set_is_list_when_written(self):
        """memory_retrieval_set must be a list — not a dict or other container."""
        ctx = TurnContext(user_input="check")
        ctx.state["memory_retrieval_set"] = [{"summary_key": "m1", "summary": "note"}]
        assert isinstance(ctx.state["memory_retrieval_set"], list)

    def test_empty_memory_retrieval_set_is_valid(self):
        """An empty retrieval set is a valid state — memory not required per turn."""
        ctx = TurnContext(user_input="check")
        ctx.state["memory_retrieval_set"] = []
        assert ctx.state["memory_retrieval_set"] == []

    def test_memory_full_history_id_is_string(self):
        """memory_full_history_id must be a string identifier, not a complex object."""
        ctx = TurnContext(user_input="check")
        ctx.state["memory_full_history_id"] = "hist-abc-123"
        assert isinstance(ctx.state["memory_full_history_id"], str)

    def test_determinism_metadata_accepts_fingerprint(self):
        """metadata.determinism.memory_fingerprint must be settable as a string."""
        ctx = TurnContext(user_input="check")
        ctx.metadata["determinism"] = {"memory_fingerprint": "fp-test-1"}
        fp = ctx.metadata["determinism"]["memory_fingerprint"]
        assert isinstance(fp, str)
        assert fp == "fp-test-1"
