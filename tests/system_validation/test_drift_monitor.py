"""System Validation — Drift Monitor.

Three properties under test:

1. CANONICAL HASH STABILITY
   Run the canonical graph 10× with the same seed.  The ``context_snapshot_hash``
   must be identical across all 10 runs — non-determinism anywhere in the
   context-building path is a drift event.

2. BOUNDARY GATE AS RUNTIME INVARIANT
   Run ``ci/kernel_boundary_check.py`` via subprocess from within a pytest
   test.  The boundary gate must return exit code 0.  This makes the CI hard-
   fail gate a first-class runtime-verifiable invariant rather than only a
   CI step.

3. INVARIANT STABILITY ACROSS MIXED-SEED LONG RUN
   Run 100 turns with seeds drawn from all named constants in
   ``deterministic_seeds``.  Count violations.  Any violation causes a hard
   fail — the test reports every turn+seed that violated, not just the first.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from harness.deterministic_seeds import (
    ADVERSARIAL,
    BASELINE,
    CHAOS_BASE,
    CHECKPOINT,
    MUTATION_FUZZ,
    PARALLEL_MERGE,
    PHASE_BOUNDARY,
    REPLAY_A,
    REPLAY_B,
    TEMPORAL_FREEZE,
)
from harness.graph_runner import GraphRunner
from harness.invariant_checker import InvariantChecker, InvariantViolation
from harness.kernel_mock import MockRegistry
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
    TurnGraph,
)

pytestmark = pytest.mark.durability

_WORKSPACE_ROOT = Path(__file__).parent.parent.parent

_ALL_NAMED_SEEDS = [
    BASELINE,
    ADVERSARIAL,
    REPLAY_A,
    REPLAY_B,
    CHAOS_BASE,
    MUTATION_FUZZ,
    CHECKPOINT,
    PARALLEL_MERGE,
    TEMPORAL_FREEZE,
    PHASE_BOUNDARY,
]


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
# 1. Canonical hash stability
# ---------------------------------------------------------------------------


class TestCanonicalHashStability:
    """context_snapshot_hash must be identical across repeated runs from the same seed."""

    @pytest.mark.parametrize("seed", _ALL_NAMED_SEEDS)
    def test_context_hash_is_stable_across_10_runs(self, seed: int):
        factory = TurnFactory()
        runner = GraphRunner()
        hashes: list[str] = []

        for run_idx in range(10):
            registry = MockRegistry()
            graph = _build_canonical(registry)
            ctx = TurnFactory().build_turn(seed=seed)
            runner.run(graph, ctx, registry)
            hashes.append(factory.context_snapshot_hash(ctx))

        assert len(set(hashes)) == 1, (
            f"Seed={seed}: context_snapshot_hash drifted across 10 runs. "
            f"Unique hashes: {set(hashes)}"
        )

    def test_different_seeds_produce_different_hashes(self):
        """Sanity check: the hash function is actually discriminating."""
        factory = TurnFactory()
        hashes = {
            seed: factory.context_snapshot_hash(TurnFactory().build_turn(seed=seed))
            for seed in _ALL_NAMED_SEEDS
        }
        # All 10 named seeds must produce distinct hashes
        assert len(set(hashes.values())) == len(_ALL_NAMED_SEEDS), (
            "Hash collision among named seeds — hash function is not discriminating"
        )

    def test_hash_independent_of_run_order(self):
        """Hash must be identical regardless of whether other runs happened before."""
        factory = TurnFactory()

        # Build reference hashes
        reference: dict[int, str] = {}
        for seed in _ALL_NAMED_SEEDS:
            ctx = TurnFactory().build_turn(seed=seed)
            reference[seed] = factory.context_snapshot_hash(ctx)

        # Run a big chaotic block of other turns
        runner = GraphRunner()
        for i in range(50):
            s = CHAOS_BASE + i
            reg = MockRegistry()
            g = _build_canonical(reg)
            c = TurnFactory().build_turn(seed=s)
            runner.run(g, c, reg)

        # Re-check all named seeds — hash must not have changed
        for seed in _ALL_NAMED_SEEDS:
            ctx = TurnFactory().build_turn(seed=seed)
            actual = factory.context_snapshot_hash(ctx)
            assert actual == reference[seed], (
                f"Seed={seed}: hash changed after interleaved runs. "
                f"expected={reference[seed]}, got={actual}"
            )


# ---------------------------------------------------------------------------
# 2. Boundary gate as runtime invariant
# ---------------------------------------------------------------------------


class TestBoundaryGateRuntimeInvariant:
    """ci/kernel_boundary_check.py must exit 0 from within a test assertion."""

    @pytest.mark.timeout(120)
    def test_boundary_gate_exits_zero(self):
        gate_script = _WORKSPACE_ROOT / "ci" / "kernel_boundary_check.py"
        assert gate_script.exists(), f"Boundary gate script not found: {gate_script}"

        proc = subprocess.run(
            [sys.executable, str(gate_script)],
            capture_output=True,
            text=True,
            cwd=str(_WORKSPACE_ROOT),
            timeout=90,
        )

        assert proc.returncode == 0, (
            f"Boundary gate FAILED (exit code {proc.returncode}).\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    @pytest.mark.timeout(180)
    def test_boundary_gate_is_reproducible(self):
        """Running the gate twice must produce the same exit code."""
        gate_script = _WORKSPACE_ROOT / "ci" / "kernel_boundary_check.py"
        results: list[int] = []
        for _ in range(2):
            proc = subprocess.run(
                [sys.executable, str(gate_script)],
                capture_output=True,
                text=True,
                cwd=str(_WORKSPACE_ROOT),
                timeout=90,
            )
            results.append(proc.returncode)
        assert results[0] == results[1] == 0, (
            f"Boundary gate non-deterministic or failing: runs returned {results}"
        )


# ---------------------------------------------------------------------------
# 3. Invariant stability across mixed-seed long run
# ---------------------------------------------------------------------------


class TestInvariantStabilityMixedSeed:
    """100-turn mixed-seed run — zero invariant violations allowed."""

    def test_zero_violations_across_100_mixed_seed_turns(self):
        runner = GraphRunner()
        checker = InvariantChecker()
        fuzzer = MutationFuzzer()

        # Cycle through all named seeds for the 100 turns
        violations: list[str] = []

        for turn in range(100):
            seed = _ALL_NAMED_SEEDS[turn % len(_ALL_NAMED_SEEDS)] + turn
            intents = fuzzer.generate_valid(seed=MUTATION_FUZZ + turn, count=3)
            registry = MockRegistry()
            graph = _build_canonical(registry)
            ctx = TurnFactory().build_turn(seed=seed, mutations=intents)
            result = runner.run(graph, ctx, registry)

            if not result.succeeded:
                violations.append(
                    f"turn={turn} seed={seed} execution-error={result.error}"
                )
                continue

            try:
                checker.validate(ctx, result, expect_save=True, expect_temporal=True)
            except InvariantViolation as exc:
                violations.append(f"turn={turn} seed={seed} invariant={exc}")

        assert not violations, (
            f"Mixed-seed run: {len(violations)} violations across 100 turns.\n"
            + "\n".join(violations)
        )

    def test_failure_accumulation_curve_is_flat(self):
        """Violation rate must not increase over time (no degradation under repeated load)."""
        runner = GraphRunner()
        checker = InvariantChecker()

        # Divide 200 turns into 4 windows of 50; count violations per window
        window_size = 50
        window_violations: list[int] = [0, 0, 0, 0]

        for turn in range(200):
            window_idx = turn // window_size
            seed = _ALL_NAMED_SEEDS[turn % len(_ALL_NAMED_SEEDS)] + turn + 10000
            registry = MockRegistry()
            graph = _build_canonical(registry)
            ctx = TurnFactory().build_turn(seed=seed)
            result = runner.run(graph, ctx, registry)

            if not result.succeeded:
                window_violations[window_idx] += 1
                continue

            try:
                checker.validate(ctx, result, expect_save=True, expect_temporal=True)
            except InvariantViolation:
                window_violations[window_idx] += 1

        # Any window with violations fails; also check no upward trend
        assert sum(window_violations) == 0, (
            f"Failure accumulation detected. Window violation counts: {window_violations}"
        )
