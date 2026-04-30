"""Unit tests for TurnTemporalAxis — from_now(), from_lock_hash(), to_dict()."""

from __future__ import annotations

import datetime as dt
import json

import pytest

pytestmark = pytest.mark.unit
from harness.deterministic_seeds import REPLAY_A, REPLAY_B, TEMPORAL_FREEZE

from dadbot.core.graph import TurnTemporalAxis


class TestFromNow:
    def test_from_now_produces_today(self):
        today = dt.date.today().isoformat()
        axis = TurnTemporalAxis.from_now()
        assert axis.wall_date == today

    def test_from_now_has_timezone(self):
        axis = TurnTemporalAxis.from_now()
        assert axis.timezone and axis.timezone != ""

    def test_from_now_wall_time_equals_turn_started_at(self):
        axis = TurnTemporalAxis.from_now()
        assert axis.wall_time == axis.turn_started_at

    def test_from_now_epoch_positive(self):
        axis = TurnTemporalAxis.from_now()
        assert axis.epoch_seconds > 0

    def test_successive_from_now_calls_advance_or_equal(self):
        a = TurnTemporalAxis.from_now()
        b = TurnTemporalAxis.from_now()
        assert b.epoch_seconds >= a.epoch_seconds


class TestFromLockHash:
    def test_same_hash_produces_identical_axis(self):
        a = TurnTemporalAxis.from_lock_hash("abc123def456789a")
        b = TurnTemporalAxis.from_lock_hash("abc123def456789a")
        assert a == b  # frozen dataclass equality

    def test_different_hashes_differ(self):
        a = TurnTemporalAxis.from_lock_hash("aaaaaaaaaaaaaaaa")
        b = TurnTemporalAxis.from_lock_hash("bbbbbbbbbbbbbbbb")
        assert a.epoch_seconds != b.epoch_seconds

    def test_empty_hash_falls_back_to_today(self):
        today = dt.date.today().isoformat()
        axis = TurnTemporalAxis.from_lock_hash("")
        assert axis.wall_date == today

    def test_non_hex_hash_falls_back_to_today(self):
        today = dt.date.today().isoformat()
        axis = TurnTemporalAxis.from_lock_hash("not-hex-at-all!!")
        assert axis.wall_date == today

    def test_seed_derived_hashes_reproducible(self):
        import hashlib

        lock = hashlib.sha256(f"temporal-lock-seed-{TEMPORAL_FREEZE}".encode()).hexdigest()[:16]
        a = TurnTemporalAxis.from_lock_hash(lock)
        b = TurnTemporalAxis.from_lock_hash(lock)
        assert a.epoch_seconds == b.epoch_seconds
        assert a.wall_time == b.wall_time

    def test_replay_pair_seeds_produce_different_axes(self):
        import hashlib

        lock_a = hashlib.sha256(f"temporal-lock-seed-{REPLAY_A}".encode()).hexdigest()[:16]
        lock_b = hashlib.sha256(f"temporal-lock-seed-{REPLAY_B}".encode()).hexdigest()[:16]
        a = TurnTemporalAxis.from_lock_hash(lock_a)
        b = TurnTemporalAxis.from_lock_hash(lock_b)
        assert a.epoch_seconds != b.epoch_seconds

    @pytest.mark.parametrize("short_hash", ["ab", "abcd", "abcdef0"])
    def test_short_hash_accepted_without_error(self, short_hash):
        # from_lock_hash uses first 16 chars; shorter input should not crash
        axis = TurnTemporalAxis.from_lock_hash(short_hash)
        assert axis.wall_date  # non-empty


class TestToDict:
    def test_to_dict_is_json_serialisable(self):
        axis = TurnTemporalAxis.from_lock_hash("deadbeefdeadbeef")
        serialised = json.dumps(axis.to_dict())
        assert axis.wall_time in serialised
        assert axis.wall_date in serialised

    def test_to_dict_contains_all_expected_keys(self):
        axis = TurnTemporalAxis.from_now()
        d = axis.to_dict()
        for key in ("turn_started_at", "wall_time", "wall_date", "timezone", "utc_offset_minutes", "epoch_seconds"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_epoch_is_float(self):
        axis = TurnTemporalAxis.from_now()
        assert isinstance(axis.to_dict()["epoch_seconds"], float)

    def test_to_dict_utc_offset_is_int(self):
        axis = TurnTemporalAxis.from_now()
        assert isinstance(axis.to_dict()["utc_offset_minutes"], int)

    def test_frozen_immutability(self):
        axis = TurnTemporalAxis.from_now()
        with pytest.raises((TypeError, AttributeError)):
            axis.wall_date = "2099-01-01"  # type: ignore
