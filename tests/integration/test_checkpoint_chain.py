"""Integration — checkpoint hash-chain linkage and stage ordering tests."""

from __future__ import annotations

import pytest
from harness.deterministic_seeds import ADVERSARIAL, BASELINE, CHECKPOINT
from harness.graph_runner import GraphRunner
from harness.kernel_mock import MockRegistry
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


class TestCheckpointChainLinkage:
    def test_checkpoints_recorded_after_run(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=BASELINE)
        result = GraphRunner().run(graph, ctx, registry)
        result.assert_succeeded()
        # At least the SaveNode atomic_commit checkpoint should be recorded
        assert len(result.checkpoints) >= 1, "Expected at least 1 checkpoint after run"

    def test_checkpoint_contains_stage_field(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=BASELINE)
        result = GraphRunner().run(graph, ctx, registry)
        for cp in result.checkpoints:
            assert "stage" in cp, f"Checkpoint missing 'stage' field: {cp!r}"

    def test_save_checkpoint_has_atomic_commit_status(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=BASELINE)
        result = GraphRunner().run(graph, ctx, registry)
        save_cps = [cp for cp in result.checkpoints if cp.get("stage") == "save"]
        assert save_cps, "No checkpoint recorded for save stage"
        statuses = [cp.get("status") for cp in save_cps]
        assert "atomic_commit" in statuses, f"Expected atomic_commit status, got {statuses}"

    def test_checkpoint_hash_non_empty(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=BASELINE)
        result = GraphRunner().run(graph, ctx, registry)
        for cp in result.checkpoints:
            h = cp.get("checkpoint_hash") or ""
            assert len(h) >= 8, f"checkpoint_hash too short or absent: {h!r}"

    def test_checkpoint_hashes_differ_across_turns(self):
        """Two different turns must produce different checkpoint hashes."""
        registry_a = MockRegistry()
        registry_b = MockRegistry()
        graph_a = _build_canonical(registry_a)
        graph_b = _build_canonical(registry_b)
        ctx_a = TurnFactory().build_turn(seed=BASELINE)
        ctx_b = TurnFactory().build_turn(seed=ADVERSARIAL)
        result_a = GraphRunner().run(graph_a, ctx_a, registry_a)
        result_b = GraphRunner().run(graph_b, ctx_b, registry_b)
        assert result_a.checkpoint_hashes != result_b.checkpoint_hashes, (
            "Different turns should have different checkpoint hash sequences"
        )

    def test_last_checkpoint_hash_on_context_non_empty(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=CHECKPOINT)
        GraphRunner().run(graph, ctx, registry)
        # last_checkpoint_hash is set during _emit_checkpoint calls
        assert ctx.last_checkpoint_hash, "ctx.last_checkpoint_hash should be non-empty after run"
        assert len(ctx.last_checkpoint_hash) >= 8


class TestCheckpointStageSequence:
    def test_save_checkpoint_is_last(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=BASELINE)
        GraphRunner().run(graph, ctx, registry)
        seq = registry.persistence.checkpoint_sequence()
        # The last checkpoint must be save/atomic_commit
        if seq:
            last_stage, last_status = seq[-1]
            assert last_stage == "save", f"Last checkpoint stage must be 'save', got {last_stage!r}"

    @pytest.mark.parametrize("seed", [BASELINE, CHECKPOINT, ADVERSARIAL])
    def test_checkpoint_chain_all_hashes_present(self, seed):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=seed)
        result = GraphRunner().run(graph, ctx, registry)
        result.assert_succeeded()
        for cp in result.checkpoints:
            assert cp.get("checkpoint_hash"), f"Checkpoint missing hash: {cp!r}"
