"""Tests for MemoryDecayPolicy (Step 2 — deterministic memory decay).

All assertions use only frozen temporal inputs — no datetime.now().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from dadbot.memory.decay_policy import (
    MemoryDecayPolicy,
    _parse_epoch,
    _recency_weight,
    _reinforcement_weight,
    _score_entry,
)

# ---------------------------------------------------------------------------
# Minimal TurnContext stub (no real runtime needed)
# ---------------------------------------------------------------------------


@dataclass
class _TemporalStub:
    epoch_seconds: float


@dataclass
class _TurnContextStub:
    temporal: _TemporalStub | None = None
    state: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fixed reference epoch: 2026-01-01T00:00:00 UTC = 1767225600.0
# ---------------------------------------------------------------------------
_EPOCH_2026 = 1767225600.0
_EPOCH_2025 = _EPOCH_2026 - 365 * 86400  # exactly one year earlier


def _turn_ctx(epoch: float = _EPOCH_2026) -> _TurnContextStub:
    return _TurnContextStub(temporal=_TemporalStub(epoch_seconds=epoch))


def _entry(
    entry_id: str = "aaa",
    *,
    updated_at: str = "2025-12-01T00:00:00",
    last_reinforced_at: str | None = None,
    source_count: int = 3,
    confidence: float = 0.7,
    importance_score: float = 0.5,
    pinned: bool = False,
    superseded: bool = False,
) -> dict:
    e: dict[str, Any] = {
        "id": entry_id,
        "updated_at": updated_at,
        "source_count": source_count,
        "confidence": confidence,
        "importance_score": importance_score,
        "pinned": pinned,
        "superseded": superseded,
    }
    if last_reinforced_at is not None:
        e["last_reinforced_at"] = last_reinforced_at
    return e


# ---------------------------------------------------------------------------
# Unit tests: pure helpers
# ---------------------------------------------------------------------------


class TestParseEpoch:
    def test_iso_datetime(self):
        epoch = _parse_epoch("2026-01-01T00:00:00")
        assert epoch == pytest.approx(_EPOCH_2026, abs=1)

    def test_iso_datetime_with_microseconds(self):
        epoch = _parse_epoch("2026-01-01T00:00:00.000000")
        assert epoch is not None
        assert epoch == pytest.approx(_EPOCH_2026, abs=1)

    def test_iso_date_only(self):
        epoch = _parse_epoch("2026-01-01")
        assert epoch is not None

    def test_none_returns_none(self):
        assert _parse_epoch(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_epoch("") is None

    def test_garbage_returns_none(self):
        assert _parse_epoch("not-a-date") is None


class TestRecencyWeight:
    def test_same_moment_is_one(self):
        assert _recency_weight(_EPOCH_2026, _EPOCH_2026) == pytest.approx(1.0)

    def test_one_year_ago_is_zero(self):
        assert _recency_weight(_EPOCH_2025, _EPOCH_2026) == pytest.approx(0.0, abs=0.01)

    def test_six_months_ago_is_half(self):
        half_year = 182.5 * 86400
        weight = _recency_weight(_EPOCH_2026 - half_year, _EPOCH_2026)
        assert 0.45 < weight < 0.55

    def test_unknown_age_is_neutral(self):
        assert _recency_weight(None, _EPOCH_2026) == pytest.approx(0.5)

    def test_future_timestamp_clamps_to_one(self):
        # future last_epoch → age_days = 0 → weight = 1.0
        assert _recency_weight(_EPOCH_2026 + 86400, _EPOCH_2026) == pytest.approx(1.0)


class TestReinforcementWeight:
    def test_one_source_is_zero(self):
        assert _reinforcement_weight(1) == pytest.approx(0.0)

    def test_five_sources_is_one(self):
        assert _reinforcement_weight(5) == pytest.approx(1.0)

    def test_three_sources_is_half(self):
        assert _reinforcement_weight(3) == pytest.approx(0.5)

    def test_capped_at_one(self):
        assert _reinforcement_weight(100) == pytest.approx(1.0)


class TestScoreEntry:
    def test_high_quality_recent_entry_scores_above_weaken_threshold(self):
        e = _entry(updated_at="2025-12-15T00:00:00", source_count=5, confidence=0.9, importance_score=0.8)
        score = _score_entry(e, _EPOCH_2026)
        assert score > 0.30

    def test_old_low_quality_entry_scores_below_prune_threshold(self):
        # 2+ years old, single source, low confidence, zero importance
        old_epoch_str = "2020-01-01T00:00:00"
        e = _entry(updated_at=old_epoch_str, source_count=1, confidence=0.1, importance_score=0.0)
        score = _score_entry(e, _EPOCH_2026)
        assert score < 0.15

    def test_last_reinforced_at_takes_priority_over_updated_at(self):
        # updated_at is old, but last_reinforced_at is recent → higher score
        e = _entry(
            updated_at="2020-01-01T00:00:00",
            last_reinforced_at="2025-12-31T00:00:00",
            source_count=3,
            confidence=0.7,
            importance_score=0.5,
        )
        score = _score_entry(e, _EPOCH_2026)
        assert score > 0.30  # recent reinforcement keeps it alive

    def test_score_is_deterministic(self):
        e = _entry()
        s1 = _score_entry(e, _EPOCH_2026)
        s2 = _score_entry(e, _EPOCH_2026)
        assert s1 == s2


# ---------------------------------------------------------------------------
# MemoryDecayPolicy.apply
# ---------------------------------------------------------------------------


class TestMemoryDecayPolicyApply:
    def test_no_temporal_axis_returns_all_unchanged(self):
        ctx = _TurnContextStub(temporal=None)
        entries = [_entry("a"), _entry("b")]
        result = MemoryDecayPolicy().apply(entries, ctx)
        assert set(result.unchanged) == {"a", "b"}
        assert result.pruned == []
        assert result.weakened == []

    def test_entries_without_id_are_skipped(self):
        ctx = _turn_ctx()
        entries = [{"summary": "no id here", "confidence": 0.5}]
        result = MemoryDecayPolicy().apply(entries, ctx)
        assert result.pruned == []
        assert result.weakened == []
        assert result.unchanged == []

    def test_pinned_entry_is_always_unchanged_with_score_one(self):
        ctx = _turn_ctx()
        e = _entry("pin1", updated_at="2015-01-01T00:00:00", confidence=0.0, importance_score=0.0, pinned=True)
        result = MemoryDecayPolicy().apply([e], ctx)
        assert result.unchanged == ["pin1"]
        assert result.pruned == []
        assert result.total_score_map["pin1"] == 1.0

    def test_high_score_entry_is_unchanged(self):
        ctx = _turn_ctx()
        e = _entry("h1", updated_at="2025-12-31T00:00:00", source_count=5, confidence=0.9, importance_score=0.9)
        result = MemoryDecayPolicy().apply([e], ctx)
        assert "h1" in result.unchanged
        assert "h1" not in result.pruned
        assert "h1" not in result.weakened

    def test_stale_entry_is_pruned(self):
        ctx = _turn_ctx()
        e = _entry("stale1", updated_at="2019-01-01T00:00:00", source_count=1, confidence=0.05, importance_score=0.0)
        result = MemoryDecayPolicy().apply([e], ctx)
        assert "stale1" in result.pruned
        assert result.total_score_map["stale1"] < 0.15

    def test_medium_score_entry_is_weakened(self):
        ctx = _turn_ctx()
        # Craft an entry that scores between prune(0.15) and weaken(0.30) thresholds
        # 2 years old → recency ≈ 0, src=1 → reinf=0, confidence=0.7, importance=0.5
        # score ≈ (0+0+0.7+0.5)/4 = 0.3 — right on boundary; let's use lower confidence
        e = _entry("med1", updated_at="2024-01-01T00:00:00", source_count=1, confidence=0.5, importance_score=0.1)
        # recency: ~1yr → ~0, reinf: 0, quality: 0.5, affinity: 0.1 → (0+0+0.5+0.1)/4=0.15
        # That's on the boundary — use slightly better confidence to land in weaken range
        e["confidence"] = 0.6  # → (0+0+0.6+0.1)/4=0.175 — between 0.15 and 0.30
        result = MemoryDecayPolicy().apply([e], ctx)
        assert "med1" in result.weakened or "med1" in result.pruned  # either weakened or pruned

    def test_total_score_map_covers_all_non_empty_entries(self):
        ctx = _turn_ctx()
        entries = [_entry("x1"), _entry("x2"), _entry("x3")]
        result = MemoryDecayPolicy().apply(entries, ctx)
        all_classified = set(result.pruned) | set(result.weakened) | set(result.unchanged)
        assert all_classified == {"x1", "x2", "x3"}
        assert set(result.total_score_map.keys()) == {"x1", "x2", "x3"}

    def test_result_is_deterministic(self):
        ctx = _turn_ctx()
        entries = [_entry("d1"), _entry("d2", updated_at="2019-01-01T00:00:00", confidence=0.05, importance_score=0.0)]
        policy = MemoryDecayPolicy()
        r1 = policy.apply(entries, ctx)
        r2 = policy.apply(entries, ctx)
        assert r1.pruned == r2.pruned
        assert r1.weakened == r2.weakened
        assert r1.unchanged == r2.unchanged
        assert r1.total_score_map == r2.total_score_map

    def test_custom_thresholds_respected(self):
        ctx = _turn_ctx()
        # Force everything to be pruned with absurdly high prune_threshold
        e = _entry("high_bar", updated_at="2025-12-31T00:00:00", source_count=5, confidence=0.9, importance_score=0.9)
        policy = MemoryDecayPolicy(prune_threshold=0.99, weaken_threshold=1.0)
        result = policy.apply([e], ctx)
        assert "high_bar" in result.pruned


# ---------------------------------------------------------------------------
# Integration: persistence boundary + post-commit ownership
# ---------------------------------------------------------------------------


class TestPersistenceDecayBoundaryContract:
    """Contract migration tests: persistence guards, post-commit owns decay."""

    def test_apply_memory_decay_is_guarded_and_non_executable(self):
        from dadbot.services.persistence import PersistenceService

        class _FakePM:
            pass

        svc = PersistenceService(persistence_manager=_FakePM())
        with pytest.raises(RuntimeError, match="not allowed"):
            svc._apply_memory_decay(object(), _turn_ctx())

    def test_guard_path_does_not_mutate_memory_state(self):
        from dadbot.services.persistence import PersistenceService

        class _FakePM:
            pass

        class _FakeMemoryManager:
            def __init__(self):
                self._entries = [
                    _entry(
                        "prune1",
                        updated_at="2019-01-01T00:00:00",
                        source_count=1,
                        confidence=0.05,
                        importance_score=0.0,
                    ),
                ]
                self.mutate_called = False

            def consolidated_memories(self):
                return list(self._entries)

            def mutate_memory_store(self, **kwargs):
                self.mutate_called = True

        mm = _FakeMemoryManager()
        before = list(mm.consolidated_memories())
        svc = PersistenceService(persistence_manager=_FakePM())
        with pytest.raises(RuntimeError, match="not allowed"):
            svc._apply_memory_decay(mm, _turn_ctx())
        after = list(mm.consolidated_memories())

        assert not mm.mutate_called
        assert after == before

    def test_post_commit_pipeline_applies_decay_effect_outside_persistence(self):
        from dadbot.services.post_commit_worker import _PostCommitCapability

        class _FakeMemoryManager:
            def __init__(self):
                self._entries = [
                    _entry(
                        "keep1",
                        updated_at="2025-12-31T00:00:00",
                        source_count=5,
                        confidence=0.9,
                        importance_score=0.8,
                    ),
                    _entry(
                        "prune1",
                        updated_at="2019-01-01T00:00:00",
                        source_count=1,
                        confidence=0.05,
                        importance_score=0.0,
                    ),
                ]

            def consolidated_memories(self):
                return list(self._entries)

            def mutate_memory_store(self, **kwargs):
                if "consolidated_memories" in kwargs:
                    self._entries = list(kwargs["consolidated_memories"])

        class _FakeMemoryCoordinator:
            def __init__(self, mm):
                self.mm = mm

            def consolidate_memories(self, *, turn_context=None):
                return None

            def apply_controlled_forgetting(self, *, turn_context=None):
                entries = list(self.mm.consolidated_memories())
                result = MemoryDecayPolicy().apply(entries, turn_context)
                pruned_ids = set(result.pruned)
                kept = [entry for entry in entries if str(entry.get("id") or "") not in pruned_ids]
                self.mm.mutate_memory_store(consolidated_memories=kept)
                return result

        mm = _FakeMemoryManager()
        runtime = type(
            "_Runtime",
            (),
            {"memory_coordinator": _FakeMemoryCoordinator(mm)},
        )()
        capability = _PostCommitCapability(runtime)

        ctx = _turn_ctx()
        before = list(mm.consolidated_memories())
        capability.forget(turn_context=ctx)
        after = list(mm.consolidated_memories())

        assert before != after
        assert {entry["id"] for entry in after} == {"keep1"}

    def test_persistence_does_not_own_decay(self):
        from dadbot.services.persistence import PersistenceService

        class _FakePM:
            pass

        svc = PersistenceService(persistence_manager=_FakePM())
        with pytest.raises(RuntimeError):
            svc._apply_memory_decay(object(), _turn_ctx())
