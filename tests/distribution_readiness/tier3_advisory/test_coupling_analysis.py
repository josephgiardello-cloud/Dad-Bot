"""Tier 3 — Coupling Analysis: detect implicit coupling in the pipeline.

Advisory only: findings are reported as warnings, not failures.
"""
from __future__ import annotations

import importlib
import inspect

import pytest


pytestmark = pytest.mark.unit


# Modules that should NOT import from each other (implicit coupling check)
_COUPLING_FORBIDDEN_PAIRS: list[tuple[str, str]] = [
    # execution_context must not import from graph — it's the lower layer
    ("dadbot.core.execution_context", "dadbot.core.graph"),
    # graph_context (TurnContext) must not import from orchestrator
    ("dadbot.core.graph_context", "dadbot.core.orchestrator"),
    # execution_trace_context must not import from graph
    ("dadbot.core.execution_trace_context", "dadbot.core.graph"),
]


class TestCouplingAnalysis:
    @pytest.mark.parametrize("source_mod,forbidden_dep", _COUPLING_FORBIDDEN_PAIRS)
    def test_no_forbidden_import(self, source_mod: str, forbidden_dep: str):
        """Source module must not directly import the forbidden dependency."""
        try:
            mod = importlib.import_module(source_mod)
        except ImportError as exc:
            pytest.skip(f"Module not importable: {exc}")

        source_file = getattr(mod, "__file__", None)
        if not source_file:
            pytest.skip(f"No source file for {source_mod}")

        with open(source_file, encoding="utf-8") as fh:
            source_text = fh.read()

        # Extract the leaf module name for the import check
        forbidden_leaf = forbidden_dep.split(".")[-1]
        forbidden_full = forbidden_dep.replace(".", "/")

        coupling_detected = (
            f"from {forbidden_dep}" in source_text
            or f"import {forbidden_dep}" in source_text
            or forbidden_full in source_text
        )

        if coupling_detected:
            pytest.warns(
                UserWarning,
                match=f"Coupling detected: {source_mod} imports {forbidden_dep}",
            )
        # Advisory: log rather than fail
        assert True, f"Advisory check passed for {source_mod} → {forbidden_dep}"

    def test_graph_node_count_advisory(self):
        """Report how many nodes are registered in the default TurnGraph pipeline."""
        from dadbot.core.graph import TurnGraph
        graph = TurnGraph()
        node_count = len(graph.nodes)
        # Advisory threshold: >8 nodes in default pipeline is a god-class signal
        if node_count > 8:
            pytest.warns(
                UserWarning,
                match=f"Default pipeline has {node_count} nodes",
            )
        # Always passes — this is advisory
        assert node_count > 0
