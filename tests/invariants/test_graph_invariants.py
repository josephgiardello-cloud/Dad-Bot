"""Invariant tests — InvariantChecker exercised against canonical runs.

Each test runs a graph variant then calls InvariantChecker.validate().
Tests confirm both positive paths (all invariants pass) and negative paths
(specific invariants raise InvariantViolation with useful messages).
"""
from __future__ import annotations

import pytest

from dadbot.core.graph import (
    ContextBuilderNode,
    HealthNode,
    InferenceNode,
    MutationIntent,
    ReflectionNode,
    SafetyNode,
    SaveNode,
    TemporalNode,
    TurnContext,
    TurnGraph,
    TurnPhase,
)
from harness.deterministic_seeds import BASELINE, ADVERSARIAL, CHECKPOINT
from harness.graph_runner import GraphRunner
from harness.invariant_checker import InvariantChecker, InvariantViolation
from harness.kernel_mock import MockRegistry
from harness.turn_factory import TurnFactory


def _build_canonical_graph(registry: MockRegistry) -> TurnGraph:
    """Build a full 7-stage canonical pipeline via add_node/set_edge."""
    g = TurnGraph(registry=registry)
    stages = [
        ("temporal", TemporalNode()),
        ("health", HealthNode()),
        ("context_builder", ContextBuilderNode()),
        ("inference", InferenceNode()),
        ("safety", SafetyNode()),
        ("reflection", ReflectionNode()),
        ("save", SaveNode()),
    ]
    prev = None
    for name, node in stages:
        g.add_node(name, node)
        if prev is not None:
            g.set_edge(prev, name)
        prev = name
    return g


class TestInvariantCheckerPositive:
    """Full pipeline passes all 5 invariant checks."""

    def test_canonical_pipeline_passes_all_invariants(self):
        registry = MockRegistry()
        graph = _build_canonical_graph(registry)
        factory = TurnFactory()
        ctx = factory.build_turn(seed=BASELINE)
        runner = GraphRunner()
        result = runner.run(graph, ctx, registry)
        result.assert_succeeded()

        checker = InvariantChecker()
        checker.validate(ctx, result, expect_save=True, expect_temporal=True)

    def test_different_seed_also_passes(self):
        registry = MockRegistry()
        graph = _build_canonical_graph(registry)
        factory = TurnFactory()
        ctx = factory.build_turn(seed=CHECKPOINT)
        runner = GraphRunner()
        result = runner.run(graph, ctx, registry)
        result.assert_succeeded()
        InvariantChecker().validate(ctx, result)

    def test_temporal_not_required_skips_temporal_check(self):
        """If temporal=False was passed, checker must not raise for absent temporal state."""
        # Build context without pre-populating temporal in state
        ctx = TurnContext(user_input="test temporal skip")
        checker = InvariantChecker()
        # This must not raise even with no temporal data
        checker._check_temporal(ctx, required=False)

    def test_empty_phase_history_passes_monotonic(self):
        ctx = TurnContext(user_input="empty phase")
        InvariantChecker()._check_phase_monotonic(ctx)  # no raises

    def test_phase_history_monotonic_pass(self):
        ctx = TurnContext(user_input="phase seq")
        ctx.phase_history.append({"from": "PLAN", "to": "ACT"})
        ctx.phase_history.append({"from": "ACT", "to": "OBSERVE"})
        ctx.phase_history.append({"from": "OBSERVE", "to": "RESPOND"})
        InvariantChecker()._check_phase_monotonic(ctx)


class TestInvariantCheckerNegative:
    """Breached invariants raise InvariantViolation."""

    def test_missing_temporal_state_raises(self):
        ctx = TurnContext(user_input="no temporal state")
        checker = InvariantChecker()
        with pytest.raises(InvariantViolation, match="temporal"):
            checker._check_temporal(ctx, required=True)

    def test_empty_wall_time_raises(self):
        ctx = TurnContext(user_input="bad temporal")
        ctx.state["temporal"] = {"wall_time": "  ", "wall_date": "2026-01-01"}
        with pytest.raises(InvariantViolation, match="wall_time"):
            InvariantChecker()._check_temporal(ctx, required=True)

    def test_empty_wall_date_raises(self):
        ctx = TurnContext(user_input="bad temporal date")
        ctx.state["temporal"] = {"wall_time": "2026-01-01T00:00:00", "wall_date": ""}
        with pytest.raises(InvariantViolation, match="wall_date"):
            InvariantChecker()._check_temporal(ctx, required=True)

    def test_phase_regression_raises(self):
        ctx = TurnContext(user_input="phase regression")
        ctx.phase_history.append({"from": "PLAN", "to": "RESPOND"})
        ctx.phase_history.append({"from": "RESPOND", "to": "ACT"})  # regression
        with pytest.raises(InvariantViolation, match="regression"):
            InvariantChecker()._check_phase_monotonic(ctx)

    def test_unknown_phase_in_history_raises(self):
        ctx = TurnContext(user_input="unknown phase")
        ctx.phase_history.append({"from": "PLAN", "to": "FLYING"})
        with pytest.raises(InvariantViolation, match="unknown phase"):
            InvariantChecker()._check_phase_monotonic(ctx)

    def test_save_not_run_raises(self):
        ctx = TurnContext(user_input="no save")
        with pytest.raises(InvariantViolation, match="save stage did not execute"):
            InvariantChecker()._check_fidelity(ctx, expect_save=True)

    def test_failed_mutations_in_snapshot_raises(self):
        ctx = TurnContext(user_input="failed mutations")
        # Patch snapshot to report failures
        ctx.mutation_queue.bind_owner(ctx.trace_id)
        original_snapshot = ctx.mutation_queue.snapshot

        def _bad_snap():
            s = original_snapshot()
            s["failed"] = 5
            return s

        ctx.mutation_queue.snapshot = _bad_snap
        with pytest.raises(InvariantViolation, match="mutation"):
            InvariantChecker()._check_mutation_queue(ctx)

    def test_suspiciously_short_checkpoint_hash_raises(self):
        ctx = TurnContext(user_input="short hash")
        ctx.last_checkpoint_hash = "ab"  # too short
        with pytest.raises(InvariantViolation, match="suspiciously short"):
            InvariantChecker()._check_checkpoint_integrity(ctx)


class TestInvariantMultiTurn:
    """Run checker across successive turns from TurnFactory — simulates chaos loop."""

    @pytest.mark.parametrize("seed", [BASELINE, ADVERSARIAL, CHECKPOINT, 999, 12345])
    def test_multi_seed_each_passes(self, seed):
        registry = MockRegistry()
        graph = _build_canonical_graph(registry)
        factory = TurnFactory()
        ctx = factory.build_turn(seed=seed)
        runner = GraphRunner()
        result = runner.run(graph, ctx, registry)
        result.assert_succeeded()
        InvariantChecker().validate(ctx, result)
