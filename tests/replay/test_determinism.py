"""Replay — determinism tests: same seed → same everything."""
from __future__ import annotations

import pytest

from harness.deterministic_seeds import BASELINE, REPLAY_A, REPLAY_B, ADVERSARIAL, CHECKPOINT
from harness.turn_factory import TurnFactory

from dadbot.core.graph import (
    ContextBuilderNode,
    HealthNode,
    InferenceNode,
    ReflectionNode,
    SafetyNode,
    SaveNode,
    TemporalNode,
    TurnGraph,
)
from harness.graph_runner import GraphRunner
from harness.kernel_mock import MockRegistry


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


class TestDeterministicContextGeneration:
    """Same seed → same TurnContext observable state."""

    @pytest.mark.parametrize("seed", [BASELINE, REPLAY_A, REPLAY_B, ADVERSARIAL, CHECKPOINT])
    def test_same_seed_same_trace_id(self, seed: int):
        factory = TurnFactory()
        a = factory.build_turn(seed=seed)
        b = factory.build_turn(seed=seed)
        assert a.trace_id == b.trace_id

    @pytest.mark.parametrize("seed", [BASELINE, REPLAY_A, REPLAY_B])
    def test_same_seed_same_user_input(self, seed: int):
        factory = TurnFactory()
        a = factory.build_turn(seed=seed)
        b = factory.build_turn(seed=seed)
        assert a.user_input == b.user_input

    @pytest.mark.parametrize("seed", [BASELINE, REPLAY_A, REPLAY_B])
    def test_same_seed_same_temporal_axis(self, seed: int):
        factory = TurnFactory()
        a = factory.build_turn(seed=seed)
        b = factory.build_turn(seed=seed)
        assert a.temporal.wall_time == b.temporal.wall_time
        assert a.temporal.wall_date == b.temporal.wall_date
        assert a.temporal.epoch_seconds == b.temporal.epoch_seconds

    @pytest.mark.parametrize("seed", [BASELINE, REPLAY_A])
    def test_same_seed_same_context_snapshot_hash(self, seed: int):
        factory = TurnFactory()
        a = factory.build_turn(seed=seed)
        b = factory.build_turn(seed=seed)
        assert factory.context_snapshot_hash(a) == factory.context_snapshot_hash(b)

    @pytest.mark.parametrize("seed", [BASELINE, REPLAY_A, REPLAY_B])
    def test_same_seed_virtual_clock_same_tick_sequence(self, seed: int):
        factory = TurnFactory()
        a = factory.build_turn(seed=seed)
        b = factory.build_turn(seed=seed)
        ticks_a = [a.virtual_clock.tick() for _ in range(10)]
        ticks_b = [b.virtual_clock.tick() for _ in range(10)]
        assert ticks_a == ticks_b

    def test_different_seeds_differ(self):
        factory = TurnFactory()
        a = factory.build_turn(seed=REPLAY_A)
        b = factory.build_turn(seed=REPLAY_B)
        assert a.trace_id != b.trace_id
        assert factory.context_snapshot_hash(a) != factory.context_snapshot_hash(b)


class TestBuildPairDeterminism:
    """TurnFactory.build_pair() returns two structurally identical contexts."""

    @pytest.mark.parametrize("seed", [BASELINE, ADVERSARIAL, 999])
    def test_pair_trace_ids_equal(self, seed: int):
        factory = TurnFactory()
        a, b = factory.build_pair(seed=seed)
        assert a.trace_id == b.trace_id

    @pytest.mark.parametrize("seed", [BASELINE, ADVERSARIAL])
    def test_pair_context_snapshot_hashes_equal(self, seed: int):
        factory = TurnFactory()
        a, b = factory.build_pair(seed=seed)
        assert factory.context_snapshot_hash(a) == factory.context_snapshot_hash(b)

    def test_pair_contexts_are_independent_objects(self):
        factory = TurnFactory()
        a, b = factory.build_pair(seed=BASELINE)
        # Mutating one state must not affect the other
        a.state["marker"] = "from_a"
        assert "marker" not in b.state


class TestReplayPhaseSequence:
    """Same seed → same canonical phase sequence after full graph run."""

    @pytest.mark.parametrize("seed", [BASELINE, REPLAY_A, REPLAY_B])
    def test_same_seed_same_phase_sequence(self, seed: int):
        factory = TurnFactory()

        def _run(s):
            registry = MockRegistry()
            graph = _build_canonical(registry)
            ctx = factory.build_turn(seed=s)
            GraphRunner().run(graph, ctx, registry)
            return [e.get("to") for e in ctx.phase_history]

        seq_a = _run(seed)
        seq_b = _run(seed)
        assert seq_a == seq_b, f"Phase sequences diverged for seed={seed}: {seq_a} vs {seq_b}"

    @pytest.mark.parametrize("seed", [BASELINE, REPLAY_A])
    def test_same_seed_same_stage_traces(self, seed: int):
        factory = TurnFactory()

        def _run(s):
            registry = MockRegistry()
            graph = _build_canonical(registry)
            ctx = factory.build_turn(seed=s)
            GraphRunner().run(graph, ctx, registry)
            return [t.stage for t in ctx.stage_traces]

        stages_a = _run(seed)
        stages_b = _run(seed)
        assert stages_a == stages_b


class TestReplayFidelityDeterminism:
    @pytest.mark.parametrize("seed", [BASELINE, ADVERSARIAL, CHECKPOINT])
    def test_same_seed_same_fidelity_dict(self, seed: int):
        factory = TurnFactory()

        def _run(s):
            registry = MockRegistry()
            graph = _build_canonical(registry)
            ctx = factory.build_turn(seed=s)
            GraphRunner().run(graph, ctx, registry)
            return ctx.fidelity.to_dict()

        assert _run(seed) == _run(seed)
