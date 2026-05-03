"""Tier 3 — Static Architecture Scan: god-class and complexity signals (advisory).

Advisory only: violations here are evolution signals, not CI blockers.
"""
from __future__ import annotations

import importlib
import inspect

import pytest


pytestmark = pytest.mark.unit

# Advisory: method count thresholds per class
_GOD_CLASS_METHOD_THRESHOLD = 40


def _public_method_count(cls) -> int:
    return sum(
        1
        for name, member in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("_")
    )


def _all_method_count(cls) -> int:
    return sum(1 for _, member in inspect.getmembers(cls, predicate=inspect.isfunction))


class TestStaticArchitectureScan:
    def test_turn_graph_public_method_count_advisory(self):
        """TurnGraph public method count is tracked as a god-class signal."""
        from dadbot.core.graph import TurnGraph
        count = _public_method_count(TurnGraph)
        if count > _GOD_CLASS_METHOD_THRESHOLD:
            pytest.warns(
                UserWarning,
                match=f"TurnGraph has {count} public methods",
            )
        # Advisory: always passes
        assert count > 0, "TurnGraph must have at least one public method"

    def test_turn_context_is_dataclass(self):
        """TurnContext must be a dataclass — structural invariant for field consistency."""
        import dataclasses
        from dadbot.core.graph_context import TurnContext
        assert dataclasses.is_dataclass(TurnContext)

    def test_execution_trace_context_is_dataclass(self):
        """ExecutionTraceContext must be a dataclass — structural invariant."""
        import dataclasses
        from dadbot.core.execution_context import ExecutionTraceContext
        assert dataclasses.is_dataclass(ExecutionTraceContext)

    def test_node_contract_map_keys_are_lowercase(self):
        """All _NODE_STAGE_CONTRACTS keys must be lowercase — canonical stage names."""
        from dadbot.core.graph import _NODE_STAGE_CONTRACTS
        for key in _NODE_STAGE_CONTRACTS:
            assert key == key.lower(), f"Contract key not lowercase: {key!r}"

    def test_graph_pipeline_nodes_module_is_importable(self):
        """graph_pipeline_nodes must be importable as a standalone module."""
        mod = importlib.import_module("dadbot.core.graph_pipeline_nodes")
        assert mod is not None

    def test_execution_context_module_is_importable(self):
        """execution_context must be importable as a standalone module."""
        mod = importlib.import_module("dadbot.core.execution_context")
        assert mod is not None
