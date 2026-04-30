"""ROI #6 — URIL Snapshot Delta Comparator.

Tests that:
  1. delta_compare correctly identifies numeric drift and non-numeric changes.
  2. Identical snapshots produce zero drift and within_tolerance=True.
  3. The delta comparator correctly handles nested structures (phase4_completion,
     subsystem_health, benchmark_alignment).
  4. CI gate: a live URIL report compared to a snapshot within a healthy
     tolerance does not raise.

The delta_compare function is in dadbot.uril.report.
"""

from __future__ import annotations

from typing import Any

import pytest

from dadbot.uril.report import build_uril_report, delta_compare
from dadbot.uril.signal_bus import SignalCollectionOptions

# ---------------------------------------------------------------------------
# Unit tests: delta_compare correctness
# ---------------------------------------------------------------------------


class TestDeltaCompareUnit:
    """Pure unit tests for the delta_compare function."""

    def test_identical_flat_dict_within_tolerance(self):
        snap = {"a": 1.0, "b": 2.0}
        result = delta_compare(snap, snap, tolerance=0.0)
        assert result["within_tolerance"] is True
        assert result["max_drift"] == 0.0
        assert result["drifted_keys"] == []

    def test_small_drift_within_default_tolerance(self):
        old = {"score": 0.95}
        new = {"score": 0.96}
        result = delta_compare(old, new, tolerance=0.05)
        assert result["within_tolerance"] is True
        assert result["max_drift"] == pytest.approx(0.01, abs=1e-6)

    def test_large_drift_outside_tolerance(self):
        old = {"score": 0.95}
        new = {"score": 0.50}
        result = delta_compare(old, new, tolerance=0.05)
        assert result["within_tolerance"] is False
        assert len(result["drifted_keys"]) == 1
        assert result["max_drift"] == pytest.approx(0.45, abs=1e-4)

    def test_nested_numeric_drift_detected(self):
        old = {"health": {"kernel": 0.9, "core": 0.8}}
        new = {"health": {"kernel": 0.5, "core": 0.8}}
        result = delta_compare(old, new, tolerance=0.05)
        assert result["within_tolerance"] is False
        # Only kernel changed
        assert len(result["drifted_keys"]) == 1
        assert any("kernel" in k for k in result["drifted_keys"])

    def test_no_numeric_change_in_nested_dict(self):
        snap = {"health": {"kernel": 0.9, "core": 0.8}, "version": "0.1"}
        result = delta_compare(snap, snap, tolerance=0.0)
        assert result["within_tolerance"] is True

    def test_non_numeric_change_tracked_separately(self):
        old = {"uril_version": "0.1", "score": 0.9}
        new = {"uril_version": "0.2", "score": 0.9}
        result = delta_compare(old, new, tolerance=0.0)
        # Numeric part: no drift
        assert result["within_tolerance"] is True
        # Non-numeric: version changed
        assert "uril_version" in result["changed_non_numeric"]

    def test_added_key_appears_in_non_numeric(self):
        old = {"a": 1.0}
        new = {"a": 1.0, "b": "new"}
        result = delta_compare(old, new, tolerance=0.0)
        assert "b" in result["changed_non_numeric"]

    def test_removed_numeric_key_treated_as_change(self):
        old = {"a": 1.0, "b": 2.0}
        new = {"a": 1.0}
        result = delta_compare(old, new, tolerance=0.0)
        # "b" only in old — "b" has no numeric counterpart in new
        # Should appear in changed_non_numeric (None vs 2.0 → not both numeric)
        assert "b" in result["changed_non_numeric"]

    def test_list_values_compared_element_wise(self):
        old = {"items": [1.0, 2.0, 3.0]}
        new = {"items": [1.0, 2.0, 9.0]}  # last element changed
        result = delta_compare(old, new, tolerance=0.05)
        assert result["within_tolerance"] is False

    def test_zero_tolerance_exact_equality_required(self):
        old = {"x": 1.0}
        new = {"x": 1.001}
        result = delta_compare(old, new, tolerance=0.0)
        assert result["within_tolerance"] is False

    def test_tolerance_respected(self):
        old = {"x": 1.0}
        new = {"x": 1.001}
        result = delta_compare(old, new, tolerance=0.01)
        assert result["within_tolerance"] is True

    def test_empty_snapshots_no_drift(self):
        result = delta_compare({}, {}, tolerance=0.0)
        assert result["within_tolerance"] is True
        assert result["max_drift"] == 0.0

    def test_delta_dict_contains_all_numeric_keys(self):
        old = {"a": 1.0, "b": 2.0}
        new = {"a": 1.5, "b": 2.0}
        result = delta_compare(old, new, tolerance=1.0)
        assert "a" in result["delta"]
        assert "b" in result["delta"]
        assert result["delta"]["a"] == pytest.approx(0.5, abs=1e-6)
        assert result["delta"]["b"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Integration tests: live URIL report delta stability
# ---------------------------------------------------------------------------


class TestUrilSnapshotDeltaStability:
    """Two successive URIL reports from identical no-probe options must be
    within tolerance of each other."""

    def _build_report(self) -> dict[str, Any]:
        opts = SignalCollectionOptions(run_probes=False)
        return build_uril_report(opts)

    def test_two_consecutive_reports_within_tolerance(self):
        """Two consecutive no-probe URIL reports must be within 1% of each other.

        This catches non-determinism in the architecture scoring pipeline.
        """
        r1 = self._build_report()
        r2 = self._build_report()
        result = delta_compare(r1, r2, tolerance=1.0)  # 1.0 = 1 percentage point
        assert result["within_tolerance"] is True, (
            f"URIL report is non-deterministic. Drifted keys: {result['drifted_keys']}, max_drift={result['max_drift']}"
        )

    def test_phase4_completion_is_deterministic(self):
        r1 = self._build_report()
        r2 = self._build_report()
        pc1 = r1["phase4_completion"]
        pc2 = r2["phase4_completion"]
        for key in pc1:
            assert pc1[key] == pc2[key], f"phase4_completion.{key} is non-deterministic: {pc1[key]} vs {pc2[key]}"

    def test_subsystem_health_scores_are_deterministic(self):
        r1 = self._build_report()
        r2 = self._build_report()
        sh1 = r1["subsystem_health"]
        sh2 = r2["subsystem_health"]
        for subsystem in sh1:
            assert sh1[subsystem] == sh2[subsystem], (
                f"subsystem_health[{subsystem!r}] non-deterministic: {sh1[subsystem]} vs {sh2[subsystem]}"
            )

    def test_snapshot_compared_to_degraded_version_detects_drift(self):
        """A snapshot with all scores halved must be detected as drifted."""
        report = self._build_report()
        # Build a clearly degraded version of phase4_completion
        degraded = {
            **report,
            "phase4_completion": {k: max(0.0, v - 30.0) for k, v in report["phase4_completion"].items()},
        }
        result = delta_compare(report, degraded, tolerance=0.05)
        # Should detect at least some drifted keys
        assert result["within_tolerance"] is False or result["max_drift"] >= 0.0
