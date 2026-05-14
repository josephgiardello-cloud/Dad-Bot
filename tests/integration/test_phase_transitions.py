"""Integration — phase mapping per stage, no regressions, transition events."""

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
    TurnPhase,
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


class TestPhaseTransitionsAfterRun:
    def test_no_phase_regressions_in_history(self):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=BASELINE)
        GraphRunner().run(graph, ctx, registry)

        _PHASE_ORDER = [TurnPhase.PLAN, TurnPhase.ACT, TurnPhase.OBSERVE, TurnPhase.RESPOND]
        _PHASE_INDEX = {p: i for i, p in enumerate(_PHASE_ORDER)}

        prev_idx = -1
        for entry in ctx.phase_history:
            to_raw = entry.get("to")
            try:
                to_phase = TurnPhase(to_raw)
            except ValueError:
                continue
            idx = _PHASE_INDEX.get(to_phase, -1)
            assert idx >= prev_idx, f"Phase regression detected: history={ctx.phase_history}"
            prev_idx = idx

    def test_phase_sequence_plan_act_observe_respond(self):
        """Canonical pipeline transitions through all four phases."""
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=BASELINE)
        result = GraphRunner().run(graph, ctx, registry)
        result.assert_succeeded()

        seen_phases = [TurnPhase(e["to"]) for e in ctx.phase_history if e.get("to") in TurnPhase._value2member_map_]
        # PLAN is initial phase and may not appear in transition history.
        for phase in (TurnPhase.ACT, TurnPhase.OBSERVE, TurnPhase.RESPOND):
            assert phase in seen_phases, f"{phase.value!r} not found in phase history"

    @pytest.mark.parametrize("seed", [BASELINE, ADVERSARIAL, CHECKPOINT])
    def test_monotonic_transitions_across_seeds(self, seed):
        registry = MockRegistry()
        graph = _build_canonical(registry)
        ctx = TurnFactory().build_turn(seed=seed)
        GraphRunner().run(graph, ctx, registry)

        _PHASE_ORDER = [TurnPhase.PLAN, TurnPhase.ACT, TurnPhase.OBSERVE, TurnPhase.RESPOND]
        _PHASE_INDEX = {p: i for i, p in enumerate(_PHASE_ORDER)}

        prev_idx = -1
        for entry in ctx.phase_history:
            to_raw = entry.get("to")
            try:
                to_phase = TurnPhase(to_raw)
            except ValueError:
                continue
            idx = _PHASE_INDEX.get(to_phase, -1)
            assert idx >= prev_idx, f"Phase regression at seed={seed}: {entry}"
            prev_idx = idx
