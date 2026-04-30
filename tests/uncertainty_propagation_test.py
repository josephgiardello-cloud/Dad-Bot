"""Tests for Phase 4 — Uncertainty Propagation System (uncertainty_model.py)."""

from __future__ import annotations

import pytest

from dadbot.core.uncertainty_model import (
    ConfidenceFusion,
    ConfidenceVector,
    FusionStrategy,
    PlannerHint,
    PlannerWeightMode,
    UncertaintyPropagator,
)

# ---------------------------------------------------------------------------
# 4.1 ConfidenceVector
# ---------------------------------------------------------------------------


class TestConfidenceVector:
    def test_construction_valid(self):
        cv = ConfidenceVector(0.9, 0.8, 0.7, source_tool="tool_a", result_status="ok")
        assert cv.reliability_score == 0.9
        assert cv.source_tool == "tool_a"

    def test_aggregate_perfect_returns_one(self):
        cv = ConfidenceVector(1.0, 1.0, 1.0)
        assert cv.aggregate == 1.0

    def test_aggregate_zero_reliability_dominates(self):
        cv = ConfidenceVector(0.0, 1.0, 1.0)
        # Zero reliability collapses aggregate toward 0
        assert cv.aggregate < 0.2

    def test_aggregate_is_in_range(self):
        for r, f, c in [(0.5, 0.5, 0.5), (0.1, 0.9, 0.8), (0.99, 0.01, 0.5)]:
            cv = ConfidenceVector(r, f, c)
            assert 0.0 <= cv.aggregate <= 1.0

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            ConfidenceVector(1.5, 0.5, 0.5)
        with pytest.raises(ValueError):
            ConfidenceVector(0.5, -0.1, 0.5)

    def test_is_low_confidence_threshold(self):
        low = ConfidenceVector(0.1, 0.1, 0.1)
        high = ConfidenceVector(0.9, 0.9, 0.9)
        assert low.is_low_confidence(threshold=0.5)
        assert not high.is_low_confidence(threshold=0.5)

    def test_to_dict_includes_aggregate(self):
        cv = ConfidenceVector(0.8, 0.7, 0.6, source_tool="t")
        d = cv.to_dict()
        assert "aggregate" in d
        assert d["source_tool"] == "t"

    def test_from_tool_result_ok_status(self):
        cv = ConfidenceVector.from_tool_result("tool_a", "ok", partial_confidence=1.0, historical_reliability=0.95)
        assert cv.reliability_score > 0.85
        assert cv.completeness_score == 1.0

    def test_from_tool_result_error_status(self):
        cv = ConfidenceVector.from_tool_result("tool_a", "error", historical_reliability=0.95)
        assert cv.reliability_score < 0.5
        assert cv.completeness_score == 0.0

    def test_from_tool_result_timeout_status(self):
        cv = ConfidenceVector.from_tool_result("tool_a", "timeout", historical_reliability=0.95)
        assert cv.reliability_score < 0.5
        assert cv.completeness_score == 0.0

    def test_from_tool_result_partial_status(self):
        cv = ConfidenceVector.from_tool_result("tool_a", "partial", partial_confidence=0.6)
        assert cv.completeness_score == pytest.approx(0.6, abs=0.01)

    def test_freshness_decays_with_age(self):
        fresh = ConfidenceVector.from_tool_result("t", "ok", data_age_seconds=0)
        stale = ConfidenceVector.from_tool_result("t", "ok", data_age_seconds=3600)
        assert fresh.freshness_score > stale.freshness_score

    def test_freshness_half_life_respected(self):
        cv = ConfidenceVector.from_tool_result("t", "ok", data_age_seconds=3600, freshness_half_life_s=3600)
        assert cv.freshness_score == pytest.approx(0.5, abs=0.02)


# ---------------------------------------------------------------------------
# 4.2 UncertaintyPropagator
# ---------------------------------------------------------------------------


class TestUncertaintyPropagator:
    def setup_method(self):
        self.propagator = UncertaintyPropagator(
            low_confidence_threshold=0.5,
            critical_threshold=0.25,
            critic_penalty_floor=0.3,
        )

    def test_high_confidence_use_as_is(self):
        cv = ConfidenceVector(0.9, 0.9, 0.9, source_tool="t")
        hint = self.propagator.planner_hint(cv)
        assert hint.weight_mode == PlannerWeightMode.USE_AS_IS
        assert hint.planning_weight == 1.0

    def test_low_confidence_discount(self):
        cv = ConfidenceVector(0.3, 0.4, 0.3, source_tool="t")
        hint = self.propagator.planner_hint(cv)
        assert hint.weight_mode == PlannerWeightMode.DISCOUNT
        assert 0.0 < hint.planning_weight < 1.0

    def test_critical_confidence_exclude(self):
        cv = ConfidenceVector(0.1, 0.1, 0.1, source_tool="t")
        hint = self.propagator.planner_hint(cv)
        assert hint.weight_mode == PlannerWeightMode.EXCLUDE
        assert hint.planning_weight == 0.0

    def test_planner_hint_returns_plannerehint(self):
        cv = ConfidenceVector(0.7, 0.7, 0.7)
        hint = self.propagator.planner_hint(cv)
        assert isinstance(hint, PlannerHint)
        assert isinstance(hint.reason, str)

    def test_critic_penalty_high_confidence_minimal_penalty(self):
        cv = ConfidenceVector(1.0, 1.0, 1.0, source_tool="t")
        penalty = self.propagator.critic_penalty(cv)
        assert penalty.penalty_factor == pytest.approx(1.0, abs=0.01)

    def test_critic_penalty_low_confidence_applies_floor(self):
        cv = ConfidenceVector(0.0, 0.0, 0.0, source_tool="t")
        penalty = self.propagator.critic_penalty(cv)
        assert penalty.penalty_factor >= 0.3

    def test_critic_penalty_scales_with_confidence(self):
        mid_cv = ConfidenceVector(0.5, 0.5, 0.5)
        high_cv = ConfidenceVector(0.9, 0.9, 0.9)
        mid_penalty = self.propagator.critic_penalty(mid_cv)
        high_penalty = self.propagator.critic_penalty(high_cv)
        assert mid_penalty.penalty_factor < high_penalty.penalty_factor

    def test_propagate_chain_returns_hints_for_each(self):
        vectors = [
            ConfidenceVector(0.9, 0.9, 0.9),
            ConfidenceVector(0.3, 0.3, 0.3),
            ConfidenceVector(0.1, 0.1, 0.1),
        ]
        hints = self.propagator.propagate_chain(vectors)
        assert len(hints) == 3
        modes = {h.weight_mode for h in hints}
        assert PlannerWeightMode.USE_AS_IS in modes
        assert PlannerWeightMode.EXCLUDE in modes


# ---------------------------------------------------------------------------
# 4.3 ConfidenceFusion
# ---------------------------------------------------------------------------


class TestConfidenceFusion:
    def _cv(self, r: float, f: float, c: float, tool: str = "t") -> ConfidenceVector:
        return ConfidenceVector(r, f, c, source_tool=tool)

    def test_conservative_takes_min(self):
        vectors = [self._cv(0.9, 0.8, 0.7), self._cv(0.3, 0.6, 0.9)]
        fused = ConfidenceFusion.conservative(vectors)
        assert fused.reliability_score == pytest.approx(0.3, abs=0.01)
        assert fused.freshness_score == pytest.approx(0.6, abs=0.01)
        assert fused.completeness_score == pytest.approx(0.7, abs=0.01)

    def test_optimistic_takes_max(self):
        vectors = [self._cv(0.9, 0.8, 0.7), self._cv(0.3, 0.6, 0.9)]
        fused = ConfidenceFusion.optimistic(vectors)
        assert fused.reliability_score == pytest.approx(0.9, abs=0.01)
        assert fused.freshness_score == pytest.approx(0.8, abs=0.01)
        assert fused.completeness_score == pytest.approx(0.9, abs=0.01)

    def test_bayesian_between_min_and_max(self):
        vectors = [self._cv(0.9, 0.8, 0.7), self._cv(0.3, 0.6, 0.9)]
        conservative = ConfidenceFusion.conservative(vectors)
        optimistic = ConfidenceFusion.optimistic(vectors)
        bayesian = ConfidenceFusion.bayesian(vectors)
        assert conservative.reliability_score <= bayesian.reliability_score <= optimistic.reliability_score

    def test_single_vector_passthrough(self):
        cv = self._cv(0.8, 0.7, 0.6)
        fused = ConfidenceFusion.bayesian([cv])
        assert fused is not None
        assert fused.reliability_score == pytest.approx(0.8, abs=0.01)

    def test_empty_list_returns_none(self):
        assert ConfidenceFusion.fuse([]) is None

    def test_fused_aggregate_is_bounded(self):
        vectors = [self._cv(r, f, c) for r, f, c in [(0.1, 0.2, 0.3), (0.9, 0.8, 0.7), (0.5, 0.5, 0.5)]]
        for strategy in FusionStrategy:
            fused = ConfidenceFusion.fuse(vectors, strategy)
            assert 0.0 <= fused.aggregate <= 1.0

    def test_fused_to_dict_includes_strategy(self):
        vectors = [self._cv(0.8, 0.7, 0.6)]
        fused = ConfidenceFusion.conservative(vectors)
        d = fused.to_dict()
        assert d["strategy"] == "conservative"
        assert d["source_count"] == 1

    def test_source_tools_captured(self):
        vectors = [self._cv(0.8, 0.7, 0.6, "tool_a"), self._cv(0.5, 0.6, 0.7, "tool_b")]
        fused = ConfidenceFusion.bayesian(vectors)
        assert "tool_a" in fused.source_tools
        assert "tool_b" in fused.source_tools

    def test_conservative_dominated_by_weak_tool(self):
        strong = self._cv(0.95, 0.95, 0.95, "good_tool")
        weak = self._cv(0.1, 0.1, 0.1, "bad_tool")
        fused = ConfidenceFusion.conservative([strong, weak])
        assert fused.reliability_score < 0.2  # dominated by weak
