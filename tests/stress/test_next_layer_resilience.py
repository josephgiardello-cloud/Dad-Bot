"""Next-layer stress suite (A-D): degradation, explosion, failure injection, replay perturbation.

This suite targets:
A. Long-horizon degradation under backlog and checkpoint pressure
B. State explosion fuzzing (fan-out, branching, parallel execution)
C. Controlled failure injection (save partial failure, persistence lag, kernel rejections)
D. Replay under perturbation (timing jitter + safe branch reordering)

Most tests are marked ``slow`` and can be scaled via environment variables.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
import tracemalloc
from types import SimpleNamespace
from typing import Any

import pytest
from harness.graph_runner import GraphRunner
from harness.kernel_mock import MockPersistenceService, MockRegistry
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


def _goal_intent(priority: int = 100) -> MutationIntent:
    return MutationIntent(
        type="goal",
        payload={"op": "upsert_goal"},
        priority=priority,
        requires_temporal=False,
    )


def _build_canonical(registry: MockRegistry) -> TurnGraph:
    graph = TurnGraph(registry=registry)
    stages = [
        ("temporal", TemporalNode()),
        ("health", HealthNode()),
        ("context_builder", ContextBuilderNode()),
        ("inference", InferenceNode()),
        ("safety", SafetyNode()),
        ("reflection", ReflectionNode()),
        ("save", SaveNode()),
    ]
    previous = None
    for name, node in stages:
        graph.add_node(name, node)
        if previous:
            graph.set_edge(previous, name)
        previous = name
    return graph


def _phase_monotonic(ctx) -> bool:
    order = ["PLAN", "ACT", "OBSERVE", "RESPOND"]
    previous_index = -1
    for entry in list(ctx.phase_history or []):
        current = str(entry.get("to") or "")
        if current not in order:
            return False
        index = order.index(current)
        if index < previous_index:
            return False
        previous_index = index
    return True


class _BranchNode:
    def __init__(self, name: str, default_jitter_ms: float = 0.0):
        self.name = name
        self._default_jitter_ms = float(default_jitter_ms)

    async def run(self, turn_context):
        jitter_map = dict(turn_context.metadata.get("jitter_ms") or {})
        jitter = float(jitter_map.get(self.name, self._default_jitter_ms))
        if jitter > 0:
            await asyncio.sleep(jitter / 1000.0)
        turn_context.state[f"branch_{self.name}"] = f"ok:{self.name}"
        return turn_context


class _ComposeNode:
    name = "compose"

    async def run(self, turn_context):
        keys = sorted([k for k in turn_context.state if k.startswith("branch_")])
        payload = "|".join(f"{key}={turn_context.state[key]}" for key in keys)
        turn_context.state["safe_result"] = (payload, False)
        return turn_context


class _FlakyPersistence(MockPersistenceService):
    def __init__(self, *, fail_every: int = 2, lag_ms: float = 0.0) -> None:
        super().__init__()
        self._fail_every = max(1, int(fail_every or 1))
        self._lag_ms = max(0.0, float(lag_ms or 0.0))

    def finalize_turn(self, ctx: Any, result: Any) -> Any:
        if self._lag_ms > 0:
            time.sleep(self._lag_ms / 1000.0)
        call_number = self.finalize_calls + 1
        if call_number % self._fail_every == 0:
            self.finalize_calls += 1
            raise RuntimeError("injected finalize_turn partial failure")
        return super().finalize_turn(ctx, result)


class _RejectSpikeKernel:
    """Kernel stub that rejects configured node names at a fixed cadence."""

    def __init__(self, *, reject_every: int = 5, reject_nodes: set[str] | None = None) -> None:
        self._reject_every = max(1, int(reject_every or 1))
        self._reject_nodes = set(reject_nodes or {"inference"})
        self._calls = 0

    async def execute_step(self, turn_context, node_name: str, fn):
        self._calls += 1
        if node_name in self._reject_nodes and self._calls % self._reject_every == 0:
            turn_context.state.setdefault("_kernel_rejections", []).append(node_name)
            return SimpleNamespace(status="rejected", error="")
        await fn()
        return SimpleNamespace(status="ok", error="")


@pytest.mark.slow
def test_long_horizon_degradation_turn_loop_under_pressure():
    """A) 10k+ long-horizon loop with backlog pressure and checkpoint compaction stress."""
    if os.environ.get("DADBOT_ENABLE_LONG_HORIZON", "0") != "1":
        pytest.skip("Set DADBOT_ENABLE_LONG_HORIZON=1 to run 10k+ horizon stress")

    steps = int(os.environ.get("DADBOT_LONG_HORIZON_STEPS", "10000"))
    assert 10_000 <= steps <= 1_000_000, "Use 10k-1M steps for this stress layer"

    registry = MockRegistry()
    graph = _build_canonical(registry)
    runner = GraphRunner()

    latencies_ms: list[float] = []
    queue_depths: list[int] = []
    memory_samples: list[tuple[int, int]] = []
    total_queued = 0

    sample_every = max(1, steps // 20)
    tracemalloc.start()
    for step in range(steps):
        mutation_count = min(512, 8 + step // 200)
        intents = [_goal_intent(priority=(i % 200) + 1) for i in range(mutation_count)]
        ctx = TurnFactory().build_turn(seed=500_000 + step, mutations=intents)

        queued_now = ctx.mutation_queue.size()
        total_queued += queued_now
        queue_depths.append(queued_now)

        result = runner.run(graph, ctx, registry)
        result.assert_succeeded()
        assert _phase_monotonic(ctx), "Phase desync detected in long-horizon loop"
        latencies_ms.append(result.elapsed_ms)

        if step % sample_every == 0 or step == steps - 1:
            _current, peak = tracemalloc.get_traced_memory()
            memory_samples.append((step, peak))
    tracemalloc.stop()

    drained = len(registry.persistence.drained)
    assert drained == total_queued, "Mutation loss detected under long-horizon backlog pressure"

    checkpoints = list(registry.persistence.checkpoints)
    assert checkpoints, "Checkpoint stream unexpectedly empty"
    compacted: dict[tuple[str, str, str], dict[str, Any]] = {}
    for checkpoint in checkpoints:
        key = (
            str(checkpoint.get("trace_id") or ""),
            str(checkpoint.get("stage") or ""),
            str(checkpoint.get("status") or ""),
        )
        compacted[key] = checkpoint
    assert len(compacted) < len(checkpoints), "Compaction stress did not reduce checkpoint cardinality"

    window = max(1, len(latencies_ms) // 10)
    early = statistics.median(latencies_ms[:window])
    late = statistics.median(latencies_ms[-window:])
    divergence_ratio = (late / early) if early > 0 else 1.0
    assert divergence_ratio < float(os.environ.get("DADBOT_MAX_LATENCY_DIVERGENCE", "5.0"))

    first_step, first_peak = memory_samples[0]
    last_step, last_peak = memory_samples[-1]
    slope_bytes_per_turn = (last_peak - first_peak) / max(1, last_step - first_step)
    assert slope_bytes_per_turn <= float(os.environ.get("DADBOT_MAX_MEMORY_SLOPE", "8192"))


@pytest.mark.slow
def test_state_explosion_fuzzing_tracks_growth_and_divergence():
    """B) Increase mutation fan-out, branching, and parallel nodes; verify measured growth signatures."""
    rounds = int(os.environ.get("DADBOT_STATE_EXPLOSION_ROUNDS", "12"))
    scenarios = [
        (16, 4),
        (64, 8),
        (256, 16),
    ]

    queue_growth: list[int] = []
    latency_medians: list[float] = []
    peak_memories: list[int] = []

    for mutation_fanout, branch_count in scenarios:
        registry = MockRegistry()
        graph = TurnGraph(registry=registry)
        graph.add_node("temporal", TemporalNode())
        graph.add_node("fanout", tuple(_BranchNode(name=f"b{i}") for i in range(branch_count)))
        graph.add_node("compose", _ComposeNode())
        graph.add_node("save", SaveNode())
        graph.set_edge("temporal", "fanout")
        graph.set_edge("fanout", "compose")
        graph.set_edge("compose", "save")

        runner = GraphRunner()
        latencies: list[float] = []
        queue_depth = 0

        tracemalloc.start()
        for turn in range(rounds):
            intents = [_goal_intent(priority=(idx % 200) + 1) for idx in range(mutation_fanout)]
            ctx = TurnFactory().build_turn(seed=700_000 + turn + mutation_fanout, mutations=intents)
            queue_depth = max(queue_depth, ctx.mutation_queue.size())
            run = runner.run(graph, ctx, registry)
            run.assert_succeeded()
            assert _phase_monotonic(ctx)
            latencies.append(run.elapsed_ms)

        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        queue_growth.append(queue_depth)
        latency_medians.append(statistics.median(latencies))
        peak_memories.append(peak)

    assert queue_growth == sorted(queue_growth), "Queue growth did not increase with fan-out"
    assert latency_medians[-1] >= latency_medians[0], "Latency did not diverge under higher branching pressure"
    assert peak_memories[-1] >= peak_memories[0], "Memory footprint did not grow with state explosion"


def test_controlled_failure_injection_preserves_recovery_visibility_and_phase_sync():
    """C) Inject save partial failures + lag + kernel rejection spikes and verify safety contracts."""
    # Save partial failure + lag
    registry = MockRegistry()
    registry.persistence = _FlakyPersistence(
        fail_every=2, lag_ms=float(os.environ.get("DADBOT_PERSISTENCE_LAG_MS", "20"))
    )
    graph = _build_canonical(registry)
    runner = GraphRunner()

    total_queued = 0
    visible_pending = 0
    phase_sync_ok = True

    for turn in range(20):
        ctx = TurnFactory().build_turn(seed=800_000 + turn, mutations=[_goal_intent()])
        total_queued += 1
        result = runner.run(graph, ctx, registry)
        assert result.error is None, f"Unexpected hard failure under partial-save injection: {result.error}"
        visible_pending += ctx.mutation_queue.size()
        phase_sync_ok = phase_sync_ok and _phase_monotonic(ctx)

    drained = len(registry.persistence.drained)
    assert drained + visible_pending == total_queued, "Silent mutation loss detected under partial save failures"
    assert registry.persistence.save_turn_calls >= 1, "Fallback save_turn path was never exercised"
    assert phase_sync_ok, "Phase desync detected during failure injection"

    # Kernel rejection spikes
    registry2 = MockRegistry()
    graph2 = _build_canonical(registry2)
    graph2.set_kernel(_RejectSpikeKernel(reject_every=3, reject_nodes={"inference", "safety"}))

    queued2 = 0
    pending2 = 0
    rejected_runs = 0
    for turn in range(15):
        ctx = TurnFactory().build_turn(seed=900_000 + turn, mutations=[_goal_intent()])
        queued2 += 1
        result = runner.run(graph2, ctx, registry2)
        if ctx.state.get("_kernel_rejections"):
            rejected_runs += 1
        # Pipeline may still recover; regardless, queued mutations must remain visible.
        pending2 += ctx.mutation_queue.size()
        assert _phase_monotonic(ctx), "Phase desync detected under kernel rejection spikes"

    drained2 = len(registry2.persistence.drained)
    assert drained2 + pending2 == queued2, "Silent mutation loss detected under kernel rejections"
    assert rejected_runs > 0, "Kernel rejection spike injector did not trigger"


@pytest.mark.slow
def test_replay_under_perturbation_keeps_deterministic_signature():
    """D) Same seed + jitter + safe branch reorder should preserve deterministic output contract."""

    def run_signature(seed: int, *, reverse_branches: bool, jitter_ms: dict[str, float]) -> dict[str, Any]:
        registry = MockRegistry()
        graph = TurnGraph(registry=registry)

        branch_names = [f"p{i}" for i in range(8)]
        branches = [_BranchNode(name=n, default_jitter_ms=0.0) for n in branch_names]
        if reverse_branches:
            branches = list(reversed(branches))

        graph.add_node("temporal", TemporalNode())
        graph.add_node("fanout", tuple(branches))
        graph.add_node("compose", _ComposeNode())
        graph.add_node("save", SaveNode())
        graph.set_edge("temporal", "fanout")
        graph.set_edge("fanout", "compose")
        graph.set_edge("compose", "save")

        ctx = TurnFactory().build_turn(seed=seed, mutations=[_goal_intent(priority=17), _goal_intent(priority=3)])
        ctx.metadata["jitter_ms"] = dict(jitter_ms)
        run = GraphRunner().run(graph, ctx, registry)
        run.assert_succeeded()

        branch_state = {key: value for key, value in sorted(ctx.state.items()) if str(key).startswith("branch_")}
        return {
            "trace_id": ctx.trace_id,
            "phase_history": list(ctx.phase_history),
            "fidelity": ctx.fidelity.to_dict(),
            "mutation_snapshot": ctx.mutation_queue.snapshot(),
            "safe_result": ctx.state.get("safe_result"),
            "determinism_boundary": ctx.determinism_boundary.snapshot(),
            "branch_state": branch_state,
            "checkpoint_hashes": list(run.checkpoint_hashes),
        }

    seed = 1_234_567
    base_jitter = {f"p{i}": float((i % 3) * 2) for i in range(8)}
    perturbed_jitter = {f"p{i}": float(((7 - i) % 4) * 3) for i in range(8)}

    signature_a = run_signature(seed, reverse_branches=False, jitter_ms=base_jitter)
    signature_b = run_signature(seed, reverse_branches=True, jitter_ms=perturbed_jitter)

    assert signature_a == signature_b, "Determinism boundary did not hold under timing jitter + safe branch reordering"
