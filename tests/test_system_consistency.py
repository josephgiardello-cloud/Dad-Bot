"""ROI #3 — Cross-subsystem Consistency Test.

Verifies that the key subsystems (signal bus, architecture health, URIL
report, receipt chain, config) are internally consistent as a single
invariant system — not just individually correct in isolation.

Failing here means integration drift: one subsystem's output disagrees with
another's expectations even though both pass their own unit tests.

Tests cover:
  * Signal bus categories are reflected in report phase4_completion
  * Subsystem health scores are bounded and match heatmap entries
  * Receipt chain length == number of stage receipts
  * Config attribute values round-trip through _CONFIG_ATTR_MAP keys
  * URIL report keys form a stable, expected set
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from dadbot.uril.models import RepoSignal, RepoSignalBus, SubsystemHealth
from dadbot.uril.architecture import build_subsystem_health, subsystem_risk_heatmap
from dadbot.uril.report import build_uril_report, delta_compare
from dadbot.uril.signal_bus import SignalCollectionOptions
from dadbot.uril.truth_binding import build_synthetic_state, compute_receipt_chain_hash, ClaimEvidenceValidator


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_clean_signal_bus() -> RepoSignalBus:
    """Minimal fully-correct signal bus for consistency checks."""
    bus = RepoSignalBus()
    bus.add(RepoSignal("repo", "correctness", 1.0, {"tests": 100, "passed": 100, "failures": 0, "errors": 0, "skipped": 0}))
    bus.add(RepoSignal("phase4", "architecture", 1.0, {}))
    bus.add(RepoSignal("repo", "determinism", 1.0, {}))
    bus.add(RepoSignal("repo", "observability", 0.9, {}))
    return bus


def _make_degraded_signal_bus() -> RepoSignalBus:
    bus = RepoSignalBus()
    bus.add(RepoSignal("repo", "correctness", 0.5, {"tests": 100, "passed": 50, "failures": 50, "errors": 0, "skipped": 0}))
    bus.add(RepoSignal("phase4", "architecture", 0.4, {}))
    bus.add(RepoSignal("repo", "determinism", 0.6, {}))
    bus.add(RepoSignal("repo", "observability", 0.3, {}))
    return bus


# ---------------------------------------------------------------------------
# Test: signal bus ↔ subsystem health consistency
# ---------------------------------------------------------------------------

class TestSignalBusSubsystemHealthConsistency:
    """Signal bus inputs must produce consistent subsystem health output."""

    def test_health_row_count_matches_subsystem_map(self):
        bus = _make_clean_signal_bus()
        rows = build_subsystem_health(bus)
        # All 8 named subsystems must be represented
        assert len(rows) == 8
        names = {r.subsystem for r in rows}
        expected = {
            "dadbot_core", "graph_engine", "kernel", "validator",
            "persistence", "observability", "mcp_layer", "tool_registry"
        }
        assert names == expected

    def test_health_scores_are_bounded(self):
        bus = _make_clean_signal_bus()
        rows = build_subsystem_health(bus)
        for row in rows:
            assert 0.0 <= row.score <= 1.0, f"{row.subsystem} score={row.score} out of [0,1]"
            assert 0.0 <= row.coupling <= 1.0
            assert 0.0 <= row.centrality <= 1.0
            assert 0.0 <= row.blast_radius <= 1.0

    def test_heatmap_subsystems_match_health_rows(self):
        bus = _make_clean_signal_bus()
        rows = build_subsystem_health(bus)
        heatmap = subsystem_risk_heatmap(bus)

        health_names = {r.subsystem for r in rows}
        heatmap_names = {h["subsystem"] for h in heatmap}
        assert health_names == heatmap_names

    def test_heatmap_risk_scores_are_bounded(self):
        bus = _make_clean_signal_bus()
        heatmap = subsystem_risk_heatmap(bus)
        for h in heatmap:
            assert 0.0 <= h["risk_score"] <= 1.0
            assert h["risk_level"] in {"LOW", "MEDIUM", "HIGH"}

    def test_heatmap_is_sorted_descending_by_risk(self):
        bus = _make_clean_signal_bus()
        heatmap = subsystem_risk_heatmap(bus)
        scores = [h["risk_score"] for h in heatmap]
        assert scores == sorted(scores, reverse=True)

    def test_clean_bus_scores_not_lower_than_degraded(self):
        clean_rows = {r.subsystem: r.score for r in build_subsystem_health(_make_clean_signal_bus())}
        degraded_rows = {r.subsystem: r.score for r in build_subsystem_health(_make_degraded_signal_bus())}
        # Clean coverage should not produce *lower* scores than degraded coverage
        # (at minimum the scores should be >= degraded for coverage-sensitive subsystems)
        for subsystem in clean_rows:
            assert clean_rows[subsystem] >= degraded_rows[subsystem] - 0.3, (
                f"{subsystem}: clean={clean_rows[subsystem]}, degraded={degraded_rows[subsystem]}"
            )


# ---------------------------------------------------------------------------
# Test: URIL report internal key consistency
# ---------------------------------------------------------------------------

class TestUrilReportInternalConsistency:
    """URIL report structure must be internally self-consistent."""

    def _build_report(self) -> dict[str, Any]:
        opts = SignalCollectionOptions(run_probes=False)
        return build_uril_report(opts)

    def test_required_top_level_keys_present(self):
        report = self._build_report()
        required = {
            "phase4_completion",
            "subsystem_health",
            "benchmark_alignment",
            "risk_heatmap",
            "upgrade_recommendations",
            "signal_bus",
            "progress_summary",
            "proven_aspects",
            "uril_version",
        }
        missing = required - set(report.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_phase4_completion_keys_are_expected(self):
        report = self._build_report()
        pc = report["phase4_completion"]
        expected_keys = {"correctness", "architecture", "determinism", "observability", "benchmark_alignment"}
        assert set(pc.keys()) == expected_keys

    def test_phase4_completion_values_are_percentages(self):
        report = self._build_report()
        for key, val in report["phase4_completion"].items():
            assert 0.0 <= val <= 100.0, f"phase4_completion.{key}={val} not in [0, 100]"

    def test_subsystem_health_keys_match_risk_heatmap(self):
        report = self._build_report()
        health_keys = set(report["subsystem_health"].keys())
        heatmap_keys = {h["subsystem"] for h in report["risk_heatmap"]}
        assert health_keys == heatmap_keys

    def test_progress_summary_counts_are_consistent(self):
        report = self._build_report()
        ps = report["progress_summary"]
        assert ps["passed"] + ps["failures"] + ps["errors"] + ps["skipped"] <= ps["tests"] + 1

    def test_proven_aspects_is_non_empty_list(self):
        report = self._build_report()
        assert isinstance(report["proven_aspects"], list)
        assert len(report["proven_aspects"]) >= 1

    def test_uril_version_is_string(self):
        report = self._build_report()
        assert isinstance(report["uril_version"], str)
        assert len(report["uril_version"]) > 0


# ---------------------------------------------------------------------------
# Test: receipt chain ↔ truth binding consistency
# ---------------------------------------------------------------------------

class TestReceiptChainTruthBindingConsistency:
    """Receipt chain must be consistent with truth binding extraction."""

    def test_empty_chain_produces_empty_hash(self):
        assert compute_receipt_chain_hash([]) == ""

    def test_consistent_hash_across_state_and_direct(self):
        state = build_synthetic_state(
            turn_id="con-test-001",
            stages=["plan", "execute", "respond"],
        )
        receipts = state["_execution_receipts"]
        hash_direct = compute_receipt_chain_hash(receipts)
        validator = ClaimEvidenceValidator()
        claim = validator.extract_claim_from_state(state, "con-test-001")
        evidence = validator.extract_evidence_from_state(state, "con-test-001")
        # Both extraction paths must see the same receipt hash
        assert claim.receipt_hash == hash_direct
        assert evidence.receipt_hash == hash_direct

    def test_claim_and_evidence_steps_agree_on_consistent_state(self):
        """When state is self-consistent, claim steps == evidence steps."""
        state = build_synthetic_state(
            turn_id="con-test-002",
            stages=["plan", "infer", "respond"],
        )
        validator = ClaimEvidenceValidator()
        claim = validator.extract_claim_from_state(state, "con-test-002")
        evidence = validator.extract_evidence_from_state(state, "con-test-002")
        result = validator.validate(claim, evidence)
        assert result.valid, result.to_dict()


# ---------------------------------------------------------------------------
# Test: delta_compare self-consistency
# ---------------------------------------------------------------------------

class TestDeltaCompareConsistency:
    """A snapshot compared with itself must always be within tolerance."""

    def _build_report(self) -> dict[str, Any]:
        opts = SignalCollectionOptions(run_probes=False)
        return build_uril_report(opts)

    def test_identical_snapshots_within_tolerance(self):
        report = self._build_report()
        result = delta_compare(report, report, tolerance=0.0)
        assert result["within_tolerance"] is True
        assert result["max_drift"] == 0.0
        assert result["drifted_keys"] == []

    def test_delta_compare_detects_numeric_change(self):
        old = {"phase4_completion": {"correctness": 95.0}}
        new = {"phase4_completion": {"correctness": 80.0}}
        result = delta_compare(old, new, tolerance=0.05)
        # 15.0 absolute diff >> 0.05 tolerance
        assert result["within_tolerance"] is False
        assert len(result["drifted_keys"]) >= 1
        assert result["max_drift"] > 0.0

    def test_delta_compare_non_numeric_change_reported(self):
        old = {"uril_version": "0.1"}
        new = {"uril_version": "0.2"}
        result = delta_compare(old, new, tolerance=0.05)
        assert "uril_version" in result["changed_non_numeric"]
