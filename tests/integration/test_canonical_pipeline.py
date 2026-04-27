"""Integration — canonical 7-stage pipeline end-to-end execution tests.

Validates that every stage runs in the correct order, fidelity flags are
all set to True, and the SaveNode executes exactly once.
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
    TurnGraph,
)
from harness.deterministic_seeds import BASELINE, ADVERSARIAL, CHECKPOINT, REPLAY_A
from harness.graph_runner import GraphRunner
from harness.invariant_checker import InvariantChecker
from harness.kernel_mock import MockRegistry
from harness.turn_factory import TurnFactory


_T = {"wall_time": "2026-01-01T00:00:00", "wall_date": "2026-01-01"}


def _build_canonical(registry: MockRegistry) -> TurnGraph:
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
        if prev:
            g.set_edge(prev, name)
        prev = name
    return g


class TestCanonicalPipelineFullRun:
    def test_pipeline_succeeds(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=BASELINE)
        result = GraphRunner().run(graph, ctx, registry)
        result.assert_succeeded()

    def test_all_fidelity_flags_true_after_run(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=BASELINE)
        GraphRunner().run(graph, ctx, registry)
        f = ctx.fidelity
        assert f.temporal, "temporal fidelity not set"
        assert f.inference, "inference fidelity not set"
        assert f.reflection, "reflection fidelity not set"
        assert f.save, "save fidelity not set"

    def test_full_pipeline_property_true(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=BASELINE)
        GraphRunner().run(graph, ctx, registry)
        assert ctx.fidelity.full_pipeline is True

    def test_save_executed_exactly_once(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=BASELINE)
        GraphRunner().run(graph, ctx, registry)
        save_traces = [t for t in ctx.stage_traces if t.stage == "save"]
        assert len(save_traces) == 1

    def test_stage_traces_contain_all_canonical_stages(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=BASELINE)
        GraphRunner().run(graph, ctx, registry)
        stages = {t.stage for t in ctx.stage_traces}
        for expected in ("temporal", "health", "context_builder", "inference", "safety", "reflection", "save"):
            assert expected in stages, f"Stage {expected!r} absent from traces"

    def test_temporal_state_populated(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=BASELINE)
        GraphRunner().run(graph, ctx, registry)
        assert ctx.state.get("temporal"), "state['temporal'] not populated by TemporalNode"

    def test_health_state_populated(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=BASELINE)
        GraphRunner().run(graph, ctx, registry)
        assert ctx.state.get("health") is not None

    def test_inference_candidate_populated(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=BASELINE)
        GraphRunner().run(graph, ctx, registry)
        assert ctx.state.get("candidate") is not None

    def test_reflection_state_populated(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=BASELINE)
        GraphRunner().run(graph, ctx, registry)
        assert ctx.state.get("reflection") is not None

    @pytest.mark.parametrize("seed", [BASELINE, ADVERSARIAL, CHECKPOINT, REPLAY_A])
    def test_multiple_seeds_all_succeed(self, seed):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=seed)
        result = GraphRunner().run(graph, ctx, registry)
        result.assert_succeeded()
        assert ctx.fidelity.full_pipeline is True


class TestCanonicalPipelineCustomResponse:
    def test_custom_inference_response_in_state(self):
        registry = MockRegistry(response="Dad says hi!")
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=42)
        GraphRunner().run(graph, ctx, registry)
        candidate = ctx.state.get("candidate")
        assert candidate is not None
        if isinstance(candidate, tuple):
            assert candidate[0] == "Dad says hi!"
        else:
            assert "Dad says hi!" in str(candidate)


class TestCanonicalPipelineInvariantSuite:
    """Validate that InvariantChecker passes on every canonical run."""

    @pytest.mark.parametrize("seed", range(10))
    def test_invariants_hold_for_seeds_0_to_9(self, seed):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=seed)
        result = GraphRunner().run(graph, ctx, registry)
        result.assert_succeeded()
        InvariantChecker().validate(ctx, result)
