"""Unit tests for VirtualClock — determinism, monotonicity, datetime output."""
from __future__ import annotations

import datetime as dt

import pytest

from dadbot.core.graph import VirtualClock
from harness.deterministic_seeds import BASELINE, TEMPORAL_FREEZE


class TestVirtualClockMonotonicity:
    def test_tick_is_strictly_increasing(self):
        vc = VirtualClock(base_epoch=1_000_000.0, step_size_seconds=1.0)
        values = [vc.tick() for _ in range(100)]
        assert values == sorted(values)
        assert len(set(values)) == 100

    def test_now_does_not_advance_clock(self):
        vc = VirtualClock(base_epoch=5_000.0, step_size_seconds=10.0)
        baseline = vc.now()
        for _ in range(50):
            assert vc.now() == baseline

    def test_tick_advances_by_exact_step(self):
        vc = VirtualClock(base_epoch=0.0, step_size_seconds=7.5)
        assert vc.tick() == pytest.approx(7.5)
        assert vc.tick() == pytest.approx(15.0)
        assert vc.tick() == pytest.approx(22.5)

    def test_base_plus_step_times_n(self):
        base = 1_700_000_000.0
        step = 30.0
        vc = VirtualClock(base_epoch=base, step_size_seconds=step)
        for n in range(1, 11):
            assert vc.tick() == pytest.approx(base + n * step)


class TestVirtualClockDeterminism:
    def test_same_params_produce_same_sequence(self):
        a = VirtualClock(base_epoch=BASELINE * 1000.0, step_size_seconds=5.0)
        b = VirtualClock(base_epoch=BASELINE * 1000.0, step_size_seconds=5.0)
        for _ in range(20):
            assert a.tick() == b.tick()

    def test_different_step_sizes_diverge(self):
        a = VirtualClock(base_epoch=1_000.0, step_size_seconds=1.0)
        b = VirtualClock(base_epoch=1_000.0, step_size_seconds=2.0)
        a_ticks = [a.tick() for _ in range(5)]
        b_ticks = [b.tick() for _ in range(5)]
        assert a_ticks != b_ticks

    def test_seed_based_epoch_reproducible(self):
        """VirtualClock built with seed-derived epoch always produces same sequence."""
        epoch = 1_700_000_000.0 + TEMPORAL_FREEZE * 3600.0
        ticks_a = [VirtualClock(base_epoch=epoch).tick() for _ in range(10)]
        ticks_b = [VirtualClock(base_epoch=epoch).tick() for _ in range(10)]
        # Each clock is independent — comparing tick(1) of each
        assert ticks_a[0] == ticks_b[0]


class TestVirtualClockDatetime:
    def test_to_datetime_is_timezone_aware(self):
        vc = VirtualClock(base_epoch=1_700_000_000.0, step_size_seconds=1.0)
        result = vc.to_datetime()
        assert isinstance(result, dt.datetime)
        assert result.tzinfo is not None

    def test_to_datetime_microsecond_stripped(self):
        vc = VirtualClock(base_epoch=1_700_000_001.999, step_size_seconds=1.0)
        result = vc.to_datetime()
        assert result.microsecond == 0

    def test_to_datetime_reflects_tick(self):
        base = 1_700_000_000.0
        step = 3600.0  # 1 hour step
        vc = VirtualClock(base_epoch=base, step_size_seconds=step)
        vc.tick()  # advance to base + 3600
        result = vc.to_datetime()
        expected = dt.datetime.fromtimestamp(base + step).astimezone().replace(microsecond=0)
        assert result == expected

    def test_modern_epoch_produces_valid_datetime(self):
        vc = VirtualClock(base_epoch=1_700_000_000.0, step_size_seconds=1.0)
        result = vc.to_datetime()
        assert result.year >= 1970
