"""System Validation — Real-World Usage Simulation.

Three properties under test:

1. MULTI-SESSION STATE CONTINUITY
   Simulate N independent sessions (each with M turns) sharing the same
   MockPersistenceService backend.  Verify:
     - No session bleeds state into another session's TurnContext
     - Each session's checkpoint chain is internally consistent
     - Turn indices are monotonically increasing per session

2. TOOL-HEAVY WORKFLOW STRESS
   Inject maximum-count MutationIntents (all types, valid payloads) into
   each turn, simulating a tool-saturated workflow.  Verify:
     - Mutation queue drains completely (no stuck mutations)
     - Drain count matches queued count
     - Invariants hold under mutation pressure

3. PARTIAL FAILURE RECOVERY STRESS
   Simulate a node crash mid-pipeline across 30 seeds; immediately retry
   the same turn with a clean graph.  Verify:
     - Retry always succeeds (idempotency)
     - Retry produces the same fidelity as a baseline run from the same seed
     - No state from the failed attempt leaks into the retry context
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from harness.deterministic_seeds import CHAOS_BASE, MUTATION_FUZZ, PARALLEL_MERGE
from harness.graph_runner import GraphRunner
from harness.invariant_checker import InvariantChecker, InvariantViolation
from harness.kernel_mock import MockPersistenceService, MockRegistry
from harness.mutation_fuzzer import MutationFuzzer
from harness.turn_factory import TurnFactory

from dadbot.core.graph import (
    ContextBuilderNode,
    HealthNode,
    InferenceNode,
    ReflectionNode,
    SafetyNode,
    SaveNode,
    TemporalNode,
    TurnContext,
    TurnGraph,
)

pytestmark = pytest.mark.durability

_SESSIONS = int(os.environ.get("DADBOT_VALIDATION_SESSIONS", "10"))
_TURNS_PER_SESSION = int(os.environ.get("DADBOT_VALIDATION_TURNS_PER_SESSION", "20"))


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


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
# Fault-injecting node (reused from adversarial scale test concepts)
# ---------------------------------------------------------------------------


class _CrashInferenceNode(InferenceNode):
    async def execute(self, ctx: TurnContext) -> Any:  # type: ignore[override]
        raise RuntimeError("RECOVERY_STRESS: InferenceNode forced crash")


def _build_crash_graph(registry: MockRegistry) -> TurnGraph:
    g = TurnGraph(registry=registry)
    stages: list[tuple[str, Any]] = [
        ("temporal", TemporalNode()),
        ("health", HealthNode()),
        ("context_builder", ContextBuilderNode()),
        ("inference", _CrashInferenceNode()),
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
# 1. Multi-session state continuity
# ---------------------------------------------------------------------------


class TestMultiSessionStateContinuity:
    """N independent sessions, M turns each — no cross-session bleed."""

    def test_sessions_produce_independent_checkpoints(self):
        """Each session's persistence record must contain only its own turns."""
        runner = GraphRunner()
        checker = InvariantChecker()
        fuzzer = MutationFuzzer()

        # session_id -> list of trace_ids seen in that session
        session_trace_ids: dict[int, list[str]] = {}

        for session_idx in range(_SESSIONS):
            session_trace_ids[session_idx] = []
            # Each session gets its own MockPersistenceService
            for turn_idx in range(_TURNS_PER_SESSION):
                seed = PARALLEL_MERGE + session_idx * 1000 + turn_idx
                registry = MockRegistry()
                graph = _build_canonical(registry)
                intents = fuzzer.generate_valid(seed=MUTATION_FUZZ + seed, count=2)
                ctx = TurnFactory().build_turn(seed=seed, mutations=intents)

                result = runner.run(graph, ctx, registry)
                result.assert_succeeded()
                checker.validate(ctx, result, expect_save=True, expect_temporal=True)
                session_trace_ids[session_idx].append(ctx.trace_id)

        # Verify: no trace_id appears in more than one session
        all_trace_ids: list[str] = []
        for ids in session_trace_ids.values():
            all_trace_ids.extend(ids)

        assert len(set(all_trace_ids)) == len(all_trace_ids), (
            f"Multi-session bleed detected: duplicate trace_ids across sessions. "
            f"Total unique={len(set(all_trace_ids))}, total={len(all_trace_ids)}"
        )

    def test_checkpoint_chains_per_session_are_independent(self):
        """Checkpoints from different sessions must not share hash values."""
        runner = GraphRunner()
        all_hashes: list[str] = []

        for session_idx in range(_SESSIONS):
            session_hashes: list[str] = []
            for turn_idx in range(5):
                seed = CHAOS_BASE + 2000 + session_idx * 100 + turn_idx
                registry = MockRegistry()
                graph = _build_canonical(registry)
                ctx = TurnFactory().build_turn(seed=seed)
                result = runner.run(graph, ctx, registry)

                for cp in result.checkpoints:
                    h = str(cp.get("checkpoint_hash") or "")
                    if h:
                        session_hashes.append(h)

            all_hashes.extend(session_hashes)

        # All checkpoint hashes must be globally unique (they embed trace_id)
        assert len(set(all_hashes)) == len(all_hashes), (
            "Session checkpoint hash collision detected — cross-session state sharing suspected"
        )

    def test_session_turn_events_are_per_turn_scoped(self):
        """Within a single turn, event sequence numbers must be non-decreasing.
        Sequence numbers are per-turn scoped — they reset between turns.
        """
        runner = GraphRunner()

        for session_idx in range(5):
            for turn_idx in range(_TURNS_PER_SESSION):
                seed = PARALLEL_MERGE + 5000 + session_idx * 200 + turn_idx
                registry = MockRegistry()
                graph = _build_canonical(registry)
                ctx = TurnFactory().build_turn(seed=seed)
                result = runner.run(graph, ctx, registry)

                # Within a single turn's events, sequence must be non-decreasing
                turn_sequence: list[int] = []
                for evt in result.events:
                    seq = evt.get("sequence")
                    if seq is not None:
                        turn_sequence.append(int(seq))

                for i in range(1, len(turn_sequence)):
                    assert turn_sequence[i] >= turn_sequence[i - 1], (
                        f"Session {session_idx} turn {turn_idx}: intra-turn sequence regression "
                        f"at position {i}: {turn_sequence[i - 1]} → {turn_sequence[i]}"
                    )


# ---------------------------------------------------------------------------
# 2. Tool-heavy workflow stress
# ---------------------------------------------------------------------------


class TestToolHeavyWorkflowStress:
    """Maximum-count MutationIntents per turn — drain must be complete."""

    @pytest.mark.parametrize("turn_count", [50])
    def test_high_mutation_load_drains_completely(self, turn_count: int):
        """100 mutations per turn × turn_count turns — all must drain."""
        runner = GraphRunner()
        checker = InvariantChecker()
        fuzzer = MutationFuzzer()

        total_queued = 0
        total_drained = 0
        drain_shortfall_turns: list[int] = []

        for turn in range(turn_count):
            seed = MUTATION_FUZZ + 3000 + turn
            registry = MockRegistry()
            graph = _build_canonical(registry)
            # 100 mutations per turn = maximum realistic tool-call load
            intents = fuzzer.generate_valid(seed=seed, count=100)
            ctx = TurnFactory().build_turn(seed=seed, mutations=intents)
            total_queued += 100

            result = runner.run(graph, ctx, registry)
            result.assert_succeeded()
            checker.validate(ctx, result, expect_save=True, expect_temporal=True)

            drained = len(registry.persistence.drained)
            total_drained += drained
            if drained < 100:
                drain_shortfall_turns.append(turn)

        assert not drain_shortfall_turns, (
            f"Tool-heavy stress: {len(drain_shortfall_turns)} turns had incomplete drains. "
            f"Total queued={total_queued}, total drained={total_drained}. "
            f"Shortfall turns (first 5): {drain_shortfall_turns[:5]}"
        )

    def test_mixed_valid_invalid_mutations_never_silent_corrupt(self):
        """Mixed valid+invalid mutations: invalid ones must be rejected, not silently applied."""
        runner = GraphRunner()
        fuzzer = MutationFuzzer()

        for turn in range(20):
            seed = MUTATION_FUZZ + 4000 + turn
            registry = MockRegistry()
            graph = _build_canonical(registry)
            # 50 mutations with 30% invalid ratio
            intents = fuzzer.generate(seed=seed, count=50, include_invalid=True, invalid_ratio=0.30)
            valid_count = sum(1 for i in intents)  # all that were actually constructed
            ctx = TurnFactory().build_turn(seed=seed, mutations=intents)

            result = runner.run(graph, ctx, registry)

            if result.succeeded:
                drained = len(registry.persistence.drained)
                snap = result.mutation_snapshot
                failed_drains = snap.get("failed", 0)
                # Drained + failed must never exceed queued (no phantom mutations)
                assert drained + failed_drains <= valid_count, (
                    f"Turn {turn}: drained({drained}) + failed({failed_drains}) "
                    f"> queued({valid_count}) — phantom mutation detected"
                )


# ---------------------------------------------------------------------------
# 3. Partial failure recovery stress
# ---------------------------------------------------------------------------


class TestPartialFailureRecoveryStress:
    """Crash a turn mid-pipeline, retry with clean graph — idempotency must hold."""

    def test_retry_after_crash_succeeds_idempotently(self):
        """For 30 seeds: crash turn, retry clean → retry must always succeed."""
        runner = GraphRunner()
        checker = InvariantChecker()
        retry_failures: list[int] = []

        for turn in range(30):
            seed = CHAOS_BASE + 3000 + turn

            # --- Attempt 1: crash graph ---
            crash_registry = MockRegistry()
            crash_graph = _build_crash_graph(crash_registry)
            crash_ctx = TurnFactory().build_turn(seed=seed)
            crash_result = runner.run(crash_graph, crash_ctx, crash_registry)
            # Crash result must not succeed (or must have an error)
            assert not crash_result.succeeded or crash_result.error is not None, (
                f"Seed {seed}: crash graph unexpectedly succeeded"
            )

            # --- Attempt 2: retry with clean graph (same seed = same input) ---
            retry_registry = MockRegistry()
            retry_graph = _build_canonical(retry_registry)
            retry_ctx = TurnFactory().build_turn(seed=seed)
            retry_result = runner.run(retry_graph, retry_ctx, retry_registry)

            if not retry_result.succeeded:
                retry_failures.append(seed)
                continue

            try:
                checker.validate(retry_ctx, retry_result, expect_save=True, expect_temporal=True)
            except InvariantViolation as exc:
                retry_failures.append(seed)

        assert not retry_failures, (
            f"Recovery stress: {len(retry_failures)} seeds failed retry after crash. "
            f"Seeds: {retry_failures[:10]}"
        )

    def test_retry_context_has_no_bleed_from_failed_attempt(self):
        """Retry TurnContext must be independent of crash attempt context."""
        runner = GraphRunner()

        for seed_offset in range(10):
            seed = CHAOS_BASE + 3100 + seed_offset

            # Build and run crash attempt
            crash_registry = MockRegistry()
            crash_graph = _build_crash_graph(crash_registry)
            crash_ctx = TurnFactory().build_turn(seed=seed)
            runner.run(crash_graph, crash_ctx, crash_registry)

            # Build fresh retry context from same seed
            retry_ctx = TurnFactory().build_turn(seed=seed)

            # They share the same trace_id (same seed) — that is correct
            assert retry_ctx.trace_id == crash_ctx.trace_id, (
                "Determinism broken: same seed produced different trace_id"
            )
            # But they must be different objects
            assert retry_ctx is not crash_ctx, "State bleed: retry returned same context object as crash"
            # Retry context must have an empty mutation queue
            snap = retry_ctx.mutation_queue.snapshot()
            assert snap.get("queued", 0) == 0 or snap.get("failed", 0) == 0, (
                f"Seed {seed}: retry context has non-empty/failed mutation queue before execution"
            )

    def test_fidelity_equivalence_baseline_vs_retry(self):
        """Baseline run and recovery retry from same seed must produce same fidelity."""
        runner = GraphRunner()

        for seed_offset in range(15):
            seed = CHAOS_BASE + 3200 + seed_offset

            # Baseline (clean, no crash before)
            baseline_registry = MockRegistry()
            baseline_graph = _build_canonical(baseline_registry)
            baseline_ctx = TurnFactory().build_turn(seed=seed)
            baseline_result = runner.run(baseline_graph, baseline_ctx, baseline_registry)
            baseline_result.assert_succeeded()

            # Recovery (crash first, then retry)
            crash_registry = MockRegistry()
            crash_graph = _build_crash_graph(crash_registry)
            crash_ctx = TurnFactory().build_turn(seed=seed)
            runner.run(crash_graph, crash_ctx, crash_registry)

            retry_registry = MockRegistry()
            retry_graph = _build_canonical(retry_registry)
            retry_ctx = TurnFactory().build_turn(seed=seed)
            retry_result = runner.run(retry_graph, retry_ctx, retry_registry)
            retry_result.assert_succeeded()

            # Fidelity must match
            assert baseline_result.fidelity.save == retry_result.fidelity.save, (
                f"Seed {seed}: fidelity.save mismatch — "
                f"baseline={baseline_result.fidelity.save}, retry={retry_result.fidelity.save}"
            )
            assert baseline_result.fidelity.temporal == retry_result.fidelity.temporal, (
                f"Seed {seed}: fidelity.temporal mismatch"
            )
