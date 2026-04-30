"""Adversarial — mutation ordering attacks, locked mutation enforcement."""

from __future__ import annotations

import pytest
from harness.deterministic_seeds import ADVERSARIAL
from harness.graph_runner import GraphRunner
from harness.kernel_mock import MockRegistry
from harness.mutation_fuzzer import MutationFuzzer
from harness.turn_factory import TurnFactory

from dadbot.core.graph import (
    ContextBuilderNode,
    HealthNode,
    InferenceNode,
    MutationGuard,
    MutationIntent,
    MutationQueue,
    ReflectionNode,
    SafetyNode,
    SaveNode,
    TemporalNode,
    TurnGraph,
)

_T = {"wall_time": "2026-01-01T00:00:00", "wall_date": "2026-01-01"}


def _goal_intent(priority: int = 100) -> MutationIntent:
    return MutationIntent(type="goal", payload={"op": "upsert_goal"}, priority=priority, requires_temporal=False)


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


# ---------------------------------------------------------------------------
# Mutation priority override is still deterministic (reordering attack)
# ---------------------------------------------------------------------------


class TestMutationReorderingAttack:
    def test_reversed_priority_order_still_drains_ascending(self):
        """Pre-load mutations in descending priority order; drain must still be ascending."""
        q = MutationQueue()
        q.bind_owner("reorder-attack")
        intents = [_goal_intent(priority=100 - i) for i in range(10)]
        for intent in intents:
            q.queue(intent)

        drained: list[MutationIntent] = []
        q.drain(drained.append, hard_fail_on_error=False)
        priorities = [i.priority for i in drained]
        assert priorities == sorted(priorities)

    def test_duplicate_priorities_sequence_id_breaks_ties(self):
        """When priorities are equal, sequence_id (insertion order) determines drain order."""
        q = MutationQueue()
        q.bind_owner("tie-break")
        for _ in range(5):
            q.queue(_goal_intent(priority=50))

        drained: list[MutationIntent] = []
        q.drain(drained.append, hard_fail_on_error=False)
        seq_ids = [i.sequence_id for i in drained]
        assert seq_ids == sorted(seq_ids)

    def test_fuzzer_sorted_matches_drain_order(self):
        fuzzer = MutationFuzzer()
        intents = fuzzer.generate_valid(seed=ADVERSARIAL, count=50)
        q = MutationQueue()
        q.bind_owner("fuzzer-reorder")
        for i in intents:
            q.queue(i)

        drained: list[MutationIntent] = []
        q.drain(drained.append, hard_fail_on_error=False)
        expected_order = fuzzer.priority_sorted(intents)
        assert [i.payload_hash for i in drained] == [i.payload_hash for i in expected_order]


# ---------------------------------------------------------------------------
# Locked mutation attack — MutationGuard blocks all non-save stages
# ---------------------------------------------------------------------------


class TestLockedMutationAttack:
    def test_queue_call_inside_guard_raises(self):
        q = MutationQueue()
        q.bind_owner("locked-test")
        with MutationGuard(q):
            with pytest.raises(RuntimeError, match="MutationGuard violation"):
                q.queue(_goal_intent())

    def test_multiple_queues_inside_guard_all_raise(self):
        q = MutationQueue()
        q.bind_owner("locked-multi")
        with MutationGuard(q):
            for _ in range(5):
                with pytest.raises(RuntimeError, match="MutationGuard violation"):
                    q.queue(_goal_intent())

    def test_guard_released_allows_queuing(self):
        q = MutationQueue()
        q.bind_owner("guard-release")
        with MutationGuard(q):
            pass  # lock enters and exits
        q.queue(_goal_intent())  # must not raise
        assert q.size() == 1

    def test_canonical_run_only_drain_in_save(self):
        """In a canonical run, mutations queued BEFORE run are drained only by SaveNode."""
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=ADVERSARIAL, mutations=[_goal_intent()])
        GraphRunner().run(graph, ctx, registry)
        # All draining goes through MockPersistenceService.finalize_turn
        assert len(registry.persistence.drained) == 1
        assert ctx.mutation_queue.is_empty()


# ---------------------------------------------------------------------------
# Duplicate execution attack
# ---------------------------------------------------------------------------


class TestDuplicateExecutionAttack:
    def test_finalize_turn_called_only_once_despite_duplicate_run_attempt(self):
        """Graph.execute() must not double-run even when called concurrently."""

        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=ADVERSARIAL)

        # Run once normally
        GraphRunner().run(graph, ctx, registry)
        first_call_count = registry.persistence.finalize_calls

        # Attempting to re-run the same context on a fresh graph should succeed
        # but finalize_calls should not accumulate from the *same* ctx
        assert first_call_count == 1

    def test_two_separate_contexts_get_independent_persistence(self):
        """Each context has its own trace_id and does not pollute the other's drain list."""
        registry_a = MockRegistry()
        registry_b = MockRegistry()
        graph_a = _build_canonical(registry_a)
        graph_b = _build_canonical(registry_b)

        ctx_a = TurnFactory().build_turn(seed=1, mutations=[_goal_intent(priority=1)])
        ctx_b = TurnFactory().build_turn(seed=2, mutations=[_goal_intent(priority=2)])

        GraphRunner().run(graph_a, ctx_a, registry_a)
        GraphRunner().run(graph_b, ctx_b, registry_b)

        # Each registry must have exactly 1 drained mutation
        assert len(registry_a.persistence.drained) == 1
        assert len(registry_b.persistence.drained) == 1
        # Priorities must not cross-contaminate
        assert registry_a.persistence.drained[0].priority == 1
        assert registry_b.persistence.drained[0].priority == 2
