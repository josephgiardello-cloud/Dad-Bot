"""Adversarial — TurnTemporalAxis freezing: temporal axis must remain
frozen even if VirtualClock ticks mid-execution.
"""

from __future__ import annotations

import pytest
from harness.deterministic_seeds import TEMPORAL_FREEZE
from harness.turn_factory import TurnFactory

from dadbot.core.graph import TurnTemporalAxis, VirtualClock


class TestTemporalFreeze:
    def test_temporal_axis_immutable_after_construction(self):
        axis = TurnTemporalAxis.from_lock_hash("deadbeefdeadbeef")
        original_wall_time = axis.wall_time
        original_epoch = axis.epoch_seconds

        # Attempt mutation — must raise
        with pytest.raises((TypeError, AttributeError)):
            axis.wall_time = "2099-01-01T00:00:00"

        # Fields unchanged
        assert axis.wall_time == original_wall_time
        assert axis.epoch_seconds == original_epoch

    def test_virtual_clock_ticks_do_not_alter_temporal_axis(self):
        ctx = TurnFactory().build_turn(seed=TEMPORAL_FREEZE)
        original_wall_time = ctx.temporal.wall_time
        original_epoch = ctx.temporal.epoch_seconds

        # Tick the virtual clock many times
        vc = ctx.virtual_clock
        if vc is not None:
            for _ in range(100):
                vc.tick()

        # ctx.temporal must be completely unchanged
        assert ctx.temporal.wall_time == original_wall_time
        assert ctx.temporal.epoch_seconds == original_epoch

    def test_context_temporal_field_is_frozen_dataclass(self):
        ctx = TurnFactory().build_turn(seed=TEMPORAL_FREEZE)
        axis = ctx.temporal
        with pytest.raises((TypeError, AttributeError)):
            axis.wall_date = "2099-12-31"

    def test_two_independent_clocks_do_not_share_epoch(self):
        vc_a = VirtualClock(base_epoch=1_000_000.0, step_size_seconds=10.0)
        vc_b = VirtualClock(base_epoch=2_000_000.0, step_size_seconds=10.0)
        for _ in range(50):
            vc_a.tick()
            vc_b.tick()
        # Clocks are independent
        assert vc_a.now() != vc_b.now()

    def test_temporal_state_in_context_unchanged_after_clock_ticks(self):
        """state['temporal'] snapshot (set by TemporalNode) must not update on clock tick."""
        import asyncio

        from harness.kernel_mock import MockRegistry

        from dadbot.core.graph import ContextBuilderNode, HealthNode, TemporalNode, TurnGraph

        registry = MockRegistry()
        # Build minimal graph: just temporal node
        g = TurnGraph(registry=registry)
        from dadbot.core.graph import InferenceNode, ReflectionNode, SafetyNode, SaveNode

        prev = None
        for name, node in [
            ("temporal", TemporalNode()),
            ("health", HealthNode()),
            ("context_builder", ContextBuilderNode()),
            ("inference", InferenceNode()),
            ("safety", SafetyNode()),
            ("reflection", ReflectionNode()),
            ("save", SaveNode()),
        ]:
            g.add_node(name, node)
            if prev:
                g.set_edge(prev, name)
            prev = name

        ctx = TurnFactory().build_turn(seed=TEMPORAL_FREEZE)
        asyncio.run(g.execute(ctx))

        saved_wall_time = ctx.state["temporal"]["wall_time"]

        # Tick clock after execution
        if ctx.virtual_clock:
            for _ in range(1000):
                ctx.virtual_clock.tick()

        # state['temporal'] must be unchanged
        assert ctx.state["temporal"]["wall_time"] == saved_wall_time

    def test_from_lock_hash_same_input_always_frozen(self):
        """Same lock hash → same frozen axis; state cannot diverge."""
        lock = "cafebabecafebabe"
        axes = [TurnTemporalAxis.from_lock_hash(lock) for _ in range(10)]
        first = axes[0]
        for a in axes[1:]:
            assert a.wall_time == first.wall_time
            assert a.epoch_seconds == first.epoch_seconds
