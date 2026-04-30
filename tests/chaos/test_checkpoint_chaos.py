"""Chaos — checkpoint tamper detection and partial crash simulation."""

from __future__ import annotations

import pytest
from harness.deterministic_seeds import CHAOS_BASE, CHECKPOINT
from harness.graph_runner import GraphRunner
from harness.invariant_checker import InvariantChecker, InvariantViolation
from harness.kernel_mock import MockPersistenceService, MockRegistry
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


class TestCheckpointTamperDetection:
    def test_tampered_hash_detected_by_invariant_checker(self):
        """If checkpoint_hash is truncated to <8 chars, InvariantChecker must raise."""
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=CHECKPOINT)
        result = GraphRunner().run(graph, ctx, registry)
        result.assert_succeeded()

        # Tamper: set a suspiciously short hash
        ctx.last_checkpoint_hash = "ab"
        with pytest.raises(InvariantViolation, match="suspiciously short"):
            InvariantChecker()._check_checkpoint_integrity(ctx)

    def test_empty_checkpoint_hash_passes_if_no_checkpoints(self):
        """An empty hash is allowed when no checkpoints were ever emitted."""
        from dadbot.core.graph import TurnContext

        ctx = TurnContext(user_input="no checkpoints")
        # No checkpoints recorded, so empty hash should not raise
        InvariantChecker()._check_checkpoint_integrity(ctx)

    def test_after_run_checkpoint_hash_is_valid(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=CHECKPOINT)
        GraphRunner().run(graph, ctx, registry)
        # Post-run: invariant checker must not raise on checkpoint
        InvariantChecker()._check_checkpoint_integrity(ctx)


class TestPartialCrashSimulation:
    def test_finalize_turn_exception_falls_back_to_save_turn(self):
        """If finalize_turn raises, SaveNode must call save_turn as fallback."""
        registry = MockRegistry()

        class _CrashyPersistence(MockPersistenceService):
            def finalize_turn(self, ctx, result):
                raise RuntimeError("simulated finalize crash")

        registry.persistence = _CrashyPersistence()

        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=CHAOS_BASE)
        result = GraphRunner().run(graph, ctx, registry)
        # Run may succeed via fallback save_turn path
        assert result.error is None or registry.persistence.save_turn_calls >= 1, (
            "Expected either no error or save_turn fallback was called"
        )
        if result.error is None:
            assert registry.persistence.save_turn_calls == 1

    def test_health_service_failure_does_not_skip_save(self):
        """If health service fails (returns error dict), SaveNode must still run."""
        registry = MockRegistry()

        original_get = registry.get

        def _patched_get(key, default=None):
            if key == "maintenance_service":
                from types import SimpleNamespace

                return SimpleNamespace(tick=lambda ctx: {"status": "error", "ticks": 0})
            return original_get(key, default)

        registry.get = _patched_get
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=CHAOS_BASE)
        result = GraphRunner().run(graph, ctx, registry)
        # SaveNode must run (fidelity.save == True) even when health returns an error dict
        assert ctx.fidelity.save is True

    def test_reflection_exception_aborts_before_save(self):
        """Current contract: unhandled reflection exceptions abort before SaveNode."""
        registry = MockRegistry()

        original_get = registry.get

        def _patched_get(key, default=None):
            if key == "reflection":
                from types import SimpleNamespace

                def _reflect_that_throws(*args, **kw):
                    raise RuntimeError("reflection exploded")

                return SimpleNamespace(reflect_after_turn=_reflect_that_throws)
            return original_get(key, default)

        registry.get = _patched_get
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=CHAOS_BASE)
        result = GraphRunner().run(graph, ctx, registry)

        assert result.error is not None
        assert "reflection exploded" in str(result.error)

        # Save does not execute when reflection raises a hard exception.
        save_traces = [t for t in ctx.stage_traces if t.stage == "save"]
        assert len(save_traces) == 0

    @pytest.mark.parametrize("seed", range(CHAOS_BASE, CHAOS_BASE + 5))
    def test_repeated_chaos_seeds_all_complete(self, seed):
        """Graph must complete (with or without error) without hanging on any seed."""
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=seed)
        result = GraphRunner().run(graph, ctx, registry)
        # We only assert it returns (no infinite loop) — success is preferred
        assert isinstance(result.elapsed_ms, float)
