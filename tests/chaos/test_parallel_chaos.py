"""Chaos — parallel node mutation merging: deterministic, no lost updates."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from dadbot.core.graph import (
    TurnContext,
    TurnGraph,
    TemporalNode,
    HealthNode,
    ContextBuilderNode,
    InferenceNode,
    SafetyNode,
    ReflectionNode,
    SaveNode,
)
from harness.deterministic_seeds import CHAOS_BASE, PARALLEL_MERGE
from harness.graph_runner import GraphRunner
from harness.invariant_checker import InvariantChecker
from harness.kernel_mock import MockRegistry
from harness.turn_factory import TurnFactory


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


class TestParallelChaos:
    """Run canonical pipeline many times from different seeds concurrently
    and confirm no interference between runs."""

    def test_concurrent_runs_independent_persistence(self):
        """10 concurrent runs must each produce exactly 1 finalize_call."""
        async def _run_one(seed: int):
            registry = MockRegistry()
            graph = _build_canonical(registry)
            ctx = TurnFactory().build_turn(seed=seed)
            await GraphRunner().run_async(graph, ctx, registry)
            return registry.persistence.finalize_calls

        async def _main():
            results = await asyncio.gather(*[_run_one(s) for s in range(10)])
            return results

        counts = asyncio.run(_main())
        for count in counts:
            assert count == 1, f"Expected 1 finalize_call per run, got {count}"

    def test_concurrent_runs_independent_trace_ids(self):
        """Each concurrent run must produce a distinct trace_id."""
        async def _run_one(seed: int) -> str:
            registry = MockRegistry()
            graph = _build_canonical(registry)
            ctx = TurnFactory().build_turn(seed=seed)
            await GraphRunner().run_async(graph, ctx, registry)
            return ctx.trace_id

        async def _main():
            return await asyncio.gather(*[_run_one(s) for s in range(10)])

        trace_ids = asyncio.run(_main())
        assert len(set(trace_ids)) == 10, "All concurrent runs must have distinct trace_ids"

    def test_concurrent_runs_no_shared_state_corruption(self):
        """State written by one run must not appear in another's state dict."""
        markers = {}

        async def _run_one(seed: int):
            registry = MockRegistry()
            graph = _build_canonical(registry)
            ctx = TurnFactory().build_turn(seed=seed)
            # Write a unique marker to state before run
            ctx.state[f"chaos_marker_{seed}"] = seed
            await GraphRunner().run_async(graph, ctx, registry)
            markers[seed] = dict(ctx.state)

        async def _main():
            await asyncio.gather(*[_run_one(s) for s in range(10)])

        asyncio.run(_main())
        for seed, state in markers.items():
            # Own marker must be present
            assert state.get(f"chaos_marker_{seed}") == seed, f"Seed {seed} lost its own marker"
            # Other markers must not be present
            for other_seed in range(10):
                if other_seed != seed:
                    assert f"chaos_marker_{other_seed}" not in state, (
                        f"State contamination: seed {seed} has marker for seed {other_seed}"
                    )

    def test_parallel_same_seed_produces_same_fidelity(self):
        """Same seed across parallel runs must produce identical fidelity snapshots."""
        async def _run_one(seed: int):
            registry = MockRegistry()
            graph = _build_canonical(registry)
            ctx = TurnFactory().build_turn(seed=seed)
            await GraphRunner().run_async(graph, ctx, registry)
            return ctx.fidelity.to_dict()

        async def _main():
            return await asyncio.gather(*[_run_one(PARALLEL_MERGE) for _ in range(5)])

        results = asyncio.run(_main())
        first = results[0]
        for r in results[1:]:
            assert r == first, "Same seed must produce identical fidelity dict"

    @pytest.mark.parametrize("n_workers", [2, 5, 10])
    def test_n_concurrent_workers_all_pass_invariants(self, n_workers: int):
        async def _run_one(seed: int):
            registry = MockRegistry()
            graph = _build_canonical(registry)
            ctx = TurnFactory().build_turn(seed=seed)
            result = await GraphRunner().run_async(graph, ctx, registry)
            InvariantChecker().validate(ctx, result)

        async def _main():
            await asyncio.gather(*[_run_one(CHAOS_BASE + i) for i in range(n_workers)])

        asyncio.run(_main())
