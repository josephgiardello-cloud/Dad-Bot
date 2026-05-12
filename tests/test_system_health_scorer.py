"""Unit tests for SystemHealthScorer.

Covers:
  - Zero-signal baseline (score=100, no failures)
  - Invariant violation penalty capping
  - Reconciliation lag detection
  - Persistence SLO miss penalties
  - Tool error rate threshold
  - Combined multi-source scoring
  - is_healthy threshold boundary
  - to_dict serialization
"""

from __future__ import annotations

import pytest

from dadbot.core.system_health_scorer import SystemHealthReport, SystemHealthScorer, score_from_gate

pytestmark = pytest.mark.unit


class _FakeGate:
    """Minimal InvariantGate stub."""
    def __init__(self, violations: int = 0):
        self.violations_observed = violations


class TestSystemHealthScorerBaseline:
    def test_no_signals_returns_100(self):
        scorer = SystemHealthScorer()
        report = scorer.score()
        assert report.overall_score == 100
        assert report.is_healthy is True
        assert report.failure_modes == []
        assert report.invariant_violations == 0

    def test_captured_at_is_utc_iso(self):
        scorer = SystemHealthScorer()
        report = scorer.score()
        # ISO 8601 basic check — has date+time separator
        assert "T" in report.captured_at
        # timezone offset present (+00:00 or Z)
        assert "+00:00" in report.captured_at or report.captured_at.endswith("Z")


class TestInvariantViolationPenalty:
    def test_single_violation_lowers_score(self):
        scorer = SystemHealthScorer()
        report = scorer.score(invariant_gate=_FakeGate(violations=1))
        assert report.overall_score < 100
        assert report.invariant_violations == 1
        assert any("invariant_violation" in f for f in report.failure_modes)

    def test_penalty_caps_at_60(self):
        scorer = SystemHealthScorer()
        report = scorer.score(invariant_gate=_FakeGate(violations=100))
        # 100 * 15 = 1500, capped at 60 → score = 40
        assert report.overall_score == 40
        assert report.invariant_violations == 100

    def test_zero_violations_no_penalty(self):
        scorer = SystemHealthScorer()
        report = scorer.score(invariant_gate=_FakeGate(violations=0))
        assert report.overall_score == 100
        assert report.invariant_violations == 0


class TestReconciliationPenalty:
    def test_partial_convergence_triggers_mode(self):
        scorer = SystemHealthScorer()
        report = scorer.score(
            reconcile_metrics={"attempted": 10, "converged": 8, "remaining": 2}
        )
        assert report.overall_score < 100
        assert any("reconciliation_lag" in f for f in report.failure_modes)
        assert report.signals["reconcile_convergence_rate"] == pytest.approx(0.8)

    def test_full_convergence_no_penalty(self):
        scorer = SystemHealthScorer()
        report = scorer.score(
            reconcile_metrics={"attempted": 50, "converged": 50, "remaining": 0}
        )
        assert report.overall_score == 100
        assert report.failure_modes == []

    def test_only_remaining_with_zero_attempted_triggers_backlog(self):
        scorer = SystemHealthScorer()
        report = scorer.score(reconcile_metrics={"attempted": 0, "converged": 0, "remaining": 5})
        assert any("reconciliation_backlog" in f for f in report.failure_modes)


class TestPersistenceSLOPenalty:
    def test_write_p95_miss_penalized(self):
        scorer = SystemHealthScorer()
        report = scorer.score(
            persistence_telemetry={"slo_ok": {"write_p95": False, "compaction_p95": True}}
        )
        assert report.overall_score < 100
        assert "persistence_write_p95_slo_miss" in report.failure_modes

    def test_compaction_p95_miss_penalized(self):
        scorer = SystemHealthScorer()
        report = scorer.score(
            persistence_telemetry={"slo_ok": {"write_p95": True, "compaction_p95": False}}
        )
        assert "persistence_compaction_p95_slo_miss" in report.failure_modes

    def test_all_slo_ok_no_penalty(self):
        scorer = SystemHealthScorer()
        report = scorer.score(
            persistence_telemetry={"slo_ok": {"write_p95": True, "compaction_p95": True}}
        )
        assert report.overall_score == 100


class TestToolErrorRatePenalty:
    def test_below_threshold_no_penalty(self):
        scorer = SystemHealthScorer()
        report = scorer.score(tool_execution_stats={"total": 100, "errors": 4})
        assert report.overall_score == 100
        assert not any("tool_error" in f for f in report.failure_modes)

    def test_above_threshold_penalized(self):
        scorer = SystemHealthScorer()
        report = scorer.score(tool_execution_stats={"total": 100, "errors": 10})
        assert any("tool_error_rate_elevated" in f for f in report.failure_modes)

    def test_zero_total_no_error_rate_signal(self):
        scorer = SystemHealthScorer()
        report = scorer.score(tool_execution_stats={"total": 0, "errors": 0})
        assert report.signals["tool_error_rate"] is None

    def test_critical_failure_classes_are_penalized(self):
        scorer = SystemHealthScorer()
        report = scorer.score(
            tool_execution_stats={
                "total": 10,
                "errors": 1,
                "failure_classes": {"timeout": 2, "unknown": 1},
            }
        )
        assert any("tool_failure_classes_elevated" in mode for mode in report.failure_modes)

    def test_tool_latency_spike_p95_is_penalized(self):
        scorer = SystemHealthScorer()
        report = scorer.score(
            tool_execution_stats={
                "total": 20,
                "errors": 0,
                "p95_latency_ms": 3000.0,
            }
        )
        assert any("tool_latency_spike" in mode for mode in report.failure_modes)


class TestMultiSourceScoring:
    def test_multiple_failures_stack(self):
        scorer = SystemHealthScorer()
        report = scorer.score(
            invariant_gate=_FakeGate(violations=2),
            reconcile_metrics={"attempted": 10, "converged": 5, "remaining": 5},
            persistence_telemetry={"slo_ok": {"write_p95": False}},
            tool_execution_stats={"total": 20, "errors": 2},
        )
        assert report.overall_score < 100
        assert len(report.failure_modes) >= 3

    def test_healthy_threshold_boundary(self):
        scorer = SystemHealthScorer(healthy_threshold=70)
        # 2 violations = -30, score = 70 → still healthy
        report = scorer.score(invariant_gate=_FakeGate(violations=2))
        assert report.overall_score == 70
        assert report.is_healthy is True

    def test_unhealthy_when_score_below_threshold(self):
        scorer = SystemHealthScorer(healthy_threshold=70)
        # 3 violations = -45, score = 55
        report = scorer.score(invariant_gate=_FakeGate(violations=3))
        assert report.overall_score == 55
        assert report.is_healthy is False


class TestScoreFromGateConvenience:
    def test_convenience_function(self):
        gate = _FakeGate(violations=1)
        report = score_from_gate(gate)
        assert isinstance(report, SystemHealthReport)
        assert report.invariant_violations == 1


class TestSystemHealthReportSerialization:
    def test_to_dict_contains_required_keys(self):
        scorer = SystemHealthScorer()
        report = scorer.score(invariant_gate=_FakeGate(violations=1))
        d = report.to_dict()
        assert set(d.keys()) >= {
            "overall_score", "invariant_violations", "failure_modes",
            "is_healthy", "signals", "captured_at",
        }
        assert isinstance(d["failure_modes"], list)
        assert isinstance(d["signals"], dict)
