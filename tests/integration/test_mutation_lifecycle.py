"""Integration — mutation lifecycle: pre-queue, run, verify drain via MockPersistenceService."""

from __future__ import annotations

from harness.deterministic_seeds import BASELINE, MUTATION_FUZZ
from harness.graph_runner import GraphRunner
from harness.kernel_mock import MockRegistry
from harness.mutation_fuzzer import MutationFuzzer
from harness.turn_factory import TurnFactory

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


def _goal_intent(priority: int = 100) -> MutationIntent:
    return MutationIntent(type="goal", payload={"op": "upsert_goal"}, priority=priority, requires_temporal=False)


def _memory_intent() -> MutationIntent:
    return MutationIntent(type="memory", payload={"op": "save_mood_state", "temporal": _T})


def _ledger_intent() -> MutationIntent:
    return MutationIntent(type="ledger", payload={"op": "append_history", "temporal": _T})


class TestMutationLifecycleBasic:
    def test_mutations_drained_by_save_node(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        intents = [_goal_intent(priority=i) for i in range(1, 4)]
        ctx = TurnFactory().build_turn(seed=BASELINE, mutations=intents)
        GraphRunner().run(graph, ctx, registry)

        assert len(registry.persistence.drained) == 3, (
            f"Expected 3 drained mutations, got {len(registry.persistence.drained)}"
        )

    def test_queue_empty_after_run(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=BASELINE, mutations=[_goal_intent()])
        GraphRunner().run(graph, ctx, registry)
        assert ctx.mutation_queue.is_empty()

    def test_finalize_turn_called_exactly_once(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=BASELINE, mutations=[_goal_intent()])
        GraphRunner().run(graph, ctx, registry)
        assert registry.persistence.finalize_calls == 1

    def test_no_mutations_no_drain(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=BASELINE)
        GraphRunner().run(graph, ctx, registry)
        assert len(registry.persistence.drained) == 0


class TestMutationLifecycleBatch:
    def test_fuzzer_generated_mutations_drained_in_priority_order(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        fuzzer = MutationFuzzer()
        intents = fuzzer.generate_valid(seed=MUTATION_FUZZ, count=20)
        ctx = TurnFactory().build_turn(seed=BASELINE, mutations=intents)
        GraphRunner().run(graph, ctx, registry)

        drained = registry.persistence.drained
        assert len(drained) == 20
        priorities = [i.priority for i in drained]
        assert priorities == sorted(priorities), "Drained order must be by ascending priority"

    def test_mixed_kinds_all_drained(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        intents = [_goal_intent(), _memory_intent(), _ledger_intent()]
        ctx = TurnFactory().build_turn(seed=BASELINE, mutations=intents)
        GraphRunner().run(graph, ctx, registry)
        kinds = {i.type.value for i in registry.persistence.drained}
        assert "goal" in kinds
        assert "memory" in kinds
        assert "ledger" in kinds


class TestMutationLifecycleSnapshot:
    def test_snapshot_drained_count_matches(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        intents = [_goal_intent(priority=i) for i in range(1, 6)]
        ctx = TurnFactory().build_turn(seed=BASELINE, mutations=intents)
        runner = GraphRunner()
        result = runner.run(graph, ctx, registry)
        snap = result.mutation_snapshot
        assert snap["drained"] == 5

    def test_snapshot_failed_zero(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=BASELINE, mutations=[_goal_intent()])
        result = GraphRunner().run(graph, ctx, registry)
        assert result.mutation_snapshot.get("failed", 0) == 0

    def test_snapshot_pending_zero_after_run(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=BASELINE, mutations=[_goal_intent()])
        result = GraphRunner().run(graph, ctx, registry)
        assert result.mutation_snapshot.get("pending", 0) == 0
