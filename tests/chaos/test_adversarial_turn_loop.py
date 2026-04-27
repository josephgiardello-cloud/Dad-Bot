"""Chaos — 50-turn adversarial loop with InvariantChecker after every turn.

Runs N turns (default 50; override with DADBOT_CHAOS_TURNS env var).
After every turn:
  - InvariantChecker validates all 5 invariants
  - Accumulated mutation loss == 0
  - Phase violations == 0
  - Broken checkpoint chains == 0
  - SaveNode executed exactly once

Slow tests are marked with @pytest.mark.slow and excluded from default
CI runs. Set DADBOT_CHAOS_TURNS=1000+ for extended soak testing.
"""
from __future__ import annotations

import os

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
from harness.deterministic_seeds import CHAOS_BASE, MUTATION_FUZZ
from harness.graph_runner import GraphRunner
from harness.invariant_checker import InvariantChecker
from harness.kernel_mock import MockRegistry
from harness.mutation_fuzzer import MutationFuzzer
from harness.turn_factory import TurnFactory


_DEFAULT_TURNS = 50
_TURNS = int(os.environ.get("DADBOT_CHAOS_TURNS", _DEFAULT_TURNS))


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


def _goal_intent(priority: int = 100) -> MutationIntent:
    return MutationIntent(type="goal", payload={"op": "upsert_goal"}, priority=priority, requires_temporal=False)


@pytest.mark.slow
class TestAdversarialTurnLoop:
    def test_n_turn_loop_all_invariants_hold(self):
        """Core chaos loop — N turns, invariants validated after each."""
        checker = InvariantChecker()
        fuzzer = MutationFuzzer()
        runner = GraphRunner()

        total_mutations_queued = 0
        total_mutations_drained = 0
        phase_violations = 0
        broken_chains = 0
        save_deviations = 0

        for turn in range(_TURNS):
            seed = CHAOS_BASE + turn
            registry = MockRegistry()
            graph = _build_canonical(registry)

            # Add a few mutations per turn (fuzzer-derived)
            mut_count = (turn % 5) + 1
            intents = fuzzer.generate_valid(seed=MUTATION_FUZZ + turn, count=mut_count)
            ctx = TurnFactory().build_turn(seed=seed, mutations=intents)
            total_mutations_queued += mut_count

            result = runner.run(graph, ctx, registry)

            # --- Invariant check ---
            try:
                checker.validate(ctx, result, expect_save=True, expect_temporal=True)
            except Exception as exc:
                pytest.fail(f"Invariant violation at turn {turn} (seed={seed}): {exc}")

            # --- Accumulate metrics ---
            total_mutations_drained += len(registry.persistence.drained)
            snap = result.mutation_snapshot

            # Phase violations
            _PHASE_ORDER = ["PLAN", "ACT", "OBSERVE", "RESPOND"]
            prev_idx = -1
            for entry in ctx.phase_history:
                to_raw = entry.get("to", "")
                if to_raw in _PHASE_ORDER:
                    idx = _PHASE_ORDER.index(to_raw)
                    if idx < prev_idx:
                        phase_violations += 1
                    prev_idx = idx

            # Broken checkpoint chains
            for cp in result.checkpoints:
                h = cp.get("checkpoint_hash") or ""
                if h and len(h) < 8:
                    broken_chains += 1

            # SaveNode exactly once
            save_traces = [t for t in ctx.stage_traces if t.stage == "save"]
            if len(save_traces) != 1:
                save_deviations += 1

        # --- Global assertions ---
        assert total_mutations_drained == total_mutations_queued, (
            f"Mutation loss detected: queued={total_mutations_queued}, "
            f"drained={total_mutations_drained}"
        )
        assert phase_violations == 0, f"{phase_violations} phase regression(s) detected"
        assert broken_chains == 0, f"{broken_chains} broken checkpoint chain(s) detected"
        assert save_deviations == 0, (
            f"{save_deviations} turn(s) where SaveNode didn't run exactly once"
        )

    def test_10_turn_quick_chaos(self):
        """Fast subset of the chaos loop for standard CI (10 turns)."""
        checker = InvariantChecker()
        runner = GraphRunner()

        for turn in range(10):
            seed = CHAOS_BASE + 1000 + turn
            registry = MockRegistry()
            graph = _build_canonical(registry)
            ctx = TurnFactory().build_turn(seed=seed, mutations=[_goal_intent()])
            result = runner.run(graph, ctx, registry)
            try:
                checker.validate(ctx, result)
            except Exception as exc:
                pytest.fail(f"Invariant violation at quick chaos turn {turn}: {exc}")

            save_traces = [t for t in ctx.stage_traces if t.stage == "save"]
            assert len(save_traces) == 1, f"SaveNode must run exactly once (turn {turn})"

    def test_mutation_free_loop_no_drain_errors(self):
        """Turns with NO pre-queued mutations must not cause drain failures."""
        runner = GraphRunner()
        for turn in range(20):
            registry = MockRegistry()
            graph = _build_canonical(registry)
            ctx = TurnFactory().build_turn(seed=CHAOS_BASE + 2000 + turn)
            result = runner.run(graph, ctx, registry)
            result.assert_succeeded()
            assert result.mutation_snapshot.get("failed", 0) == 0
