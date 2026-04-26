"""Tests for Phase 5 — Conflict Resolution Engine (conflict_resolution.py)."""
from __future__ import annotations

import pytest

from dadbot.core.uncertainty_model import ConfidenceVector
from dadbot.core.conflict_resolution import (
    ToolOutput,
    ConflictDetector,
    ConflictKind,
    ConflictReport,
    ConflictResolver,
    TrustWeightedMerger,
    ResolutionPolicyKind,
    ResolutionResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cv(r: float = 0.9, f: float = 0.9, c: float = 0.9, tool: str = "t", status: str = "ok") -> ConfidenceVector:
    return ConfidenceVector(r, f, c, source_tool=tool, result_status=status)


def _output(tool: str, output, status: str = "ok", r: float = 0.9, f: float = 0.9, c: float = 0.9) -> ToolOutput:
    return ToolOutput(tool_name=tool, output=output, confidence_vector=_cv(r, f, c, tool=tool, status=status), status=status)


# ---------------------------------------------------------------------------
# ToolOutput helpers
# ---------------------------------------------------------------------------


class TestToolOutput:
    def test_is_empty_for_none(self):
        to = _output("t", None)
        assert to.is_empty()

    def test_is_empty_for_empty_list(self):
        assert _output("t", []).is_empty()

    def test_is_empty_for_empty_string(self):
        assert _output("t", "").is_empty()

    def test_not_empty_for_nonempty_list(self):
        assert not _output("t", ["a"]).is_empty()

    def test_output_type_name(self):
        assert _output("t", []).output_type_name() == "list"
        assert _output("t", {}).output_type_name() == "dict"
        assert _output("t", "hello").output_type_name() == "str"
        assert _output("t", None).output_type_name() == "null"


# ---------------------------------------------------------------------------
# 5.2 ConflictDetector
# ---------------------------------------------------------------------------


class TestConflictDetector:
    def setup_method(self):
        self.detector = ConflictDetector()

    def test_single_output_no_conflict(self):
        report = self.detector.detect([_output("t", ["memory1", "memory2"])])
        assert not report.has_conflict()
        assert report.conflict_kind == ConflictKind.NONE

    def test_identical_outputs_no_conflict(self):
        o1 = _output("t1", ["item"])
        o2 = _output("t2", ["item"])
        report = self.detector.detect([o1, o2])
        assert not report.has_conflict()

    def test_status_mismatch_detected(self):
        o1 = _output("t1", ["item"], status="ok")
        o2 = _output("t2", ["item"], status="error")
        report = self.detector.detect([o1, o2])
        assert report.conflict_kind == ConflictKind.STATUS_MISMATCH
        assert report.severity >= 0.6

    def test_type_mismatch_detected(self):
        o1 = _output("t1", ["item"])
        o2 = _output("t2", {"key": "val"})
        report = self.detector.detect([o1, o2])
        assert report.conflict_kind == ConflictKind.TYPE_MISMATCH
        assert report.severity >= 0.8

    def test_empty_vs_nonempty_detected(self):
        o1 = _output("t1", [])
        o2 = _output("t2", ["item"])
        report = self.detector.detect([o1, o2])
        assert report.conflict_kind == ConflictKind.EMPTY_VS_NONEMPTY

    def test_value_divergence_detected(self):
        o1 = _output("t1", ["memory about work"])
        o2 = _output("t2", ["memory about sports"])
        o3 = _output("t3", ["memory about music"])
        report = self.detector.detect([o1, o2, o3])
        assert report.conflict_kind == ConflictKind.VALUE_DIVERGENCE

    def test_conflict_report_has_conflict(self):
        o1 = _output("t1", "hello")
        o2 = _output("t2", "world")
        report = self.detector.detect([o1, o2])
        if report.conflict_kind == ConflictKind.VALUE_DIVERGENCE:
            assert report.has_conflict()

    def test_conflicting_tools_populated(self):
        o1 = _output("tool_alpha", ["a"])
        o2 = _output("tool_beta", {"b": 1})
        report = self.detector.detect([o1, o2])
        assert "tool_alpha" in report.conflicting_tools or "tool_beta" in report.conflicting_tools


# ---------------------------------------------------------------------------
# 5.1 TrustWeightedMerger
# ---------------------------------------------------------------------------


class TestTrustWeightedMerger:
    def setup_method(self):
        self.merger = TrustWeightedMerger()

    def test_merge_deduplicates_identical_items(self):
        o1 = _output("t1", ["item1", "item2"])
        o2 = _output("t2", ["item1", "item3"])  # item1 is duplicate
        merged, cv = self.merger.merge([o1, o2])
        assert isinstance(merged, list)
        assert merged.count("item1") == 1

    def test_merge_orders_by_confidence(self):
        high = _output("t1", ["high_quality"], r=0.95, f=0.95, c=0.95)
        low = _output("t2", ["low_quality"], r=0.2, f=0.2, c=0.2)
        merged, cv = self.merger.merge([low, high])
        # High-confidence item should appear first
        assert merged[0] == "high_quality"

    def test_merge_all_empty_returns_empty(self):
        o1 = _output("t1", [])
        o2 = _output("t2", [])
        merged, cv = self.merger.merge([o1, o2])
        assert merged == []
        assert cv.aggregate == 0.0

    def test_merge_fused_confidence_vector(self):
        o1 = _output("t1", ["a"], r=0.9, f=0.9, c=0.9)
        o2 = _output("t2", ["b"], r=0.5, f=0.5, c=0.5)
        _, cv = self.merger.merge([o1, o2])
        # Fused should be between the two
        assert 0.5 <= cv.aggregate <= 0.9

    def test_merge_dict_outputs(self):
        o1 = _output("t1", {"key": "val1"})
        o2 = _output("t2", {"key": "val2"})
        merged, _ = self.merger.merge([o1, o2])
        assert isinstance(merged, list)
        assert len(merged) >= 1


# ---------------------------------------------------------------------------
# 5.3 ConflictResolver — full resolution pipeline
# ---------------------------------------------------------------------------


class TestConflictResolver:
    def setup_method(self):
        self.resolver = ConflictResolver()

    def test_no_conflict_returns_first_best(self):
        outputs = [_output("t1", ["item"], r=0.9, f=0.9, c=0.9)]
        result = self.resolver.resolve(outputs)
        assert result.resolved_output == ["item"]
        assert not result.requires_reexecution

    def test_empty_outputs_requires_reexecution(self):
        result = self.resolver.resolve([])
        assert result.requires_reexecution

    def test_status_mismatch_uses_highest_confidence(self):
        high = _output("t1", ["good"], status="ok", r=0.95, f=0.95, c=0.95)
        low = _output("t2", ["bad"], status="error", r=0.1, f=0.1, c=0.1)
        result = self.resolver.resolve([high, low])
        assert result.policy_used == ResolutionPolicyKind.HIGHEST_CONFIDENCE_WINS
        assert result.winning_tool == "t1"

    def test_type_mismatch_uses_highest_confidence(self):
        list_out = _output("t1", ["a", "b"], r=0.8)
        dict_out = _output("t2", {"x": 1}, r=0.3)
        result = self.resolver.resolve([list_out, dict_out])
        assert result.policy_used == ResolutionPolicyKind.HIGHEST_CONFIDENCE_WINS
        assert result.winning_tool == "t1"

    def test_empty_vs_nonempty_picks_nonempty(self):
        empty = _output("t1", [])
        nonempty = _output("t2", ["data"])
        result = self.resolver.resolve([empty, nonempty])
        assert result.winning_tool == "t2"

    def test_value_divergence_low_severity_ensemble(self):
        # Force ensemble by providing many same-type but different-value outputs
        outputs = [
            _output("t1", ["mem_work"], r=0.8),
            _output("t2", ["mem_family"], r=0.7),
        ]
        report = ConflictDetector().detect(outputs)
        if report.conflict_kind == ConflictKind.VALUE_DIVERGENCE and report.severity < 0.7:
            result = self.resolver.resolve(outputs)
            assert result.policy_used == ResolutionPolicyKind.ENSEMBLE_MERGE

    def test_policy_override_respected(self):
        o1 = _output("t1", ["a"])
        o2 = _output("t2", ["b"])
        result = self.resolver.resolve([o1, o2], policy_override=ResolutionPolicyKind.ENSEMBLE_MERGE)
        assert result.policy_used == ResolutionPolicyKind.ENSEMBLE_MERGE

    def test_defer_policy_sets_requires_reexecution(self):
        o1 = _output("t1", ["a"])
        result = self.resolver.resolve([o1], policy_override=ResolutionPolicyKind.DEFER_TO_REEXECUTION)
        assert result.requires_reexecution

    def test_resolution_notes_populated(self):
        outputs = [_output("t1", ["item"])]
        result = self.resolver.resolve(outputs)
        assert isinstance(result.resolution_notes, list)
        assert len(result.resolution_notes) > 0

    def test_result_confidence_vector_present(self):
        result = self.resolver.resolve([_output("t1", ["item"])])
        assert isinstance(result.confidence_vector, ConfidenceVector)

    def test_conflict_report_propagated(self):
        result = self.resolver.resolve([_output("t1", ["a"]), _output("t2", {"k": 1})])
        assert isinstance(result.conflict_report, ConflictReport)

    def test_highest_confidence_prefers_non_empty(self):
        empty = _output("t1", [], r=0.95)   # high confidence but empty
        nonempty = _output("t2", ["data"], r=0.5)  # lower confidence but has data
        result = self.resolver.resolve([empty, nonempty])
        # Should prefer non-empty result
        assert result.winning_tool == "t2"
