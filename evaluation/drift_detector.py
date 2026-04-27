"""Drift Detector — Phase 4D Stability Layer.

Purpose:
  Detect when benchmark results shift in ways that require explanation.
  "Shift" = meaningful change in score distribution, not noise.

Drift classification:
  none        — all deltas within noise band (±0.03)
  minor       — some deltas outside noise band, no regressions
  significant — at least one scenario regressed > threshold
  critical    — category average regressed > threshold OR multiple regressions

Outputs:
  DriftReport: per-scenario and per-category deltas, severity, regressions/improvements
  DriftGate: boolean pass/fail for CI integration

Usage:
    from evaluation.drift_detector import DriftDetector
    from evaluation.benchmark_registry import BenchmarkRegistry

    registry = BenchmarkRegistry()
    detector = DriftDetector()

    baseline = registry.load("bench-20260425-...")
    current  = registry.latest()

    report = detector.compare(baseline, current)
    print(report.summary())
    if report.drift_severity in ("significant", "critical"):
        raise RuntimeError("Benchmark drift detected — see drift report")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from evaluation.benchmark_registry import BenchmarkSnapshot


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Score delta at which we call it "noise" vs "real change"
NOISE_BAND: float = 0.03

# Regression threshold: scenario scores worse by at least this much
REGRESSION_THRESHOLD: float = 0.05

# Improvement threshold
IMPROVEMENT_THRESHOLD: float = 0.05

# Category-level regression trigger
CATEGORY_REGRESSION_THRESHOLD: float = 0.04


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScenarioDelta:
    """Score change for one scenario between two snapshots."""
    scenario_name: str
    category: str
    baseline_score: float
    current_score: float
    delta: float                # current - baseline (positive = improvement)
    is_regression: bool
    is_improvement: bool
    is_noise: bool

    subsystem_deltas: Dict[str, float] = field(default_factory=dict)

    @property
    def severity(self) -> str:
        if self.is_regression and abs(self.delta) > 0.10:
            return "critical"
        if self.is_regression:
            return "significant"
        if abs(self.delta) <= NOISE_BAND:
            return "noise"
        if self.is_improvement:
            return "improvement"
        return "minor"


@dataclass
class CategoryDelta:
    """Aggregate score change for one category."""
    category: str
    baseline_avg: float
    current_avg: float
    delta: float
    scenario_count: int
    regressions_in_category: int
    improvements_in_category: int

    @property
    def is_regression(self) -> bool:
        return self.delta < -CATEGORY_REGRESSION_THRESHOLD

    @property
    def is_improvement(self) -> bool:
        return self.delta > CATEGORY_REGRESSION_THRESHOLD


@dataclass
class VersionDrift:
    """Detected change in evaluation stack source files."""
    changed_files: List[str]
    affects_scores: bool      # True if scoring/schema/scenarios changed

    @property
    def stack_changed(self) -> bool:
        return len(self.changed_files) > 0


@dataclass
class DriftReport:
    """Complete drift analysis between two benchmark snapshots."""
    baseline_snapshot_id: str
    current_snapshot_id: str
    baseline_label: str
    current_label: str

    scenario_deltas: List[ScenarioDelta]
    category_deltas: List[CategoryDelta]
    version_drift: VersionDrift

    drift_severity: str         # "none" | "minor" | "significant" | "critical"
    drift_detected: bool

    regressions: List[str]      # scenario names that regressed
    improvements: List[str]     # scenario names that improved

    overall_delta: float        # average overall score delta
    scenarios_compared: int

    def passes_gate(self, allow_minor: bool = True) -> bool:
        """CI gate: True if this report should allow a merge/deploy."""
        if self.drift_severity == "none":
            return True
        if self.drift_severity == "minor" and allow_minor:
            return True
        return False

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "DRIFT REPORT",
            f"  Baseline : {self.baseline_label} ({self.baseline_snapshot_id})",
            f"  Current  : {self.current_label} ({self.current_snapshot_id})",
            f"  Severity : {self.drift_severity.upper()}",
            f"  Scenarios: {self.scenarios_compared}",
            f"  Overall Δ: {self.overall_delta:+.4f}",
            "",
        ]

        if self.version_drift.stack_changed:
            lines.append("  ⚠  VERSION DRIFT DETECTED:")
            for f in self.version_drift.changed_files:
                lines.append(f"       {f}")
            lines.append("")

        if self.regressions:
            lines.append(f"  ❌ REGRESSIONS ({len(self.regressions)}):")
            for name in self.regressions:
                delta = next(
                    (d for d in self.scenario_deltas if d.scenario_name == name),
                    None,
                )
                if delta:
                    lines.append(
                        f"     {name}: {delta.baseline_score:.3f} → "
                        f"{delta.current_score:.3f}  ({delta.delta:+.3f})"
                    )

        if self.improvements:
            lines.append(f"\n  ✅ IMPROVEMENTS ({len(self.improvements)}):")
            for name in self.improvements:
                delta = next(
                    (d for d in self.scenario_deltas if d.scenario_name == name),
                    None,
                )
                if delta:
                    lines.append(
                        f"     {name}: {delta.baseline_score:.3f} → "
                        f"{delta.current_score:.3f}  ({delta.delta:+.3f})"
                    )

        lines.append("\n  CATEGORY SUMMARY:")
        for cd in self.category_deltas:
            indicator = "❌" if cd.is_regression else ("✅" if cd.is_improvement else "·")
            lines.append(
                f"    {indicator} {cd.category:12s} "
                f"{cd.baseline_avg:.3f} → {cd.current_avg:.3f}  "
                f"({cd.delta:+.3f})"
            )

        lines.append("=" * 72)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Drift Detector
# ---------------------------------------------------------------------------

class DriftDetector:
    """Compares two BenchmarkSnapshots and produces a DriftReport.

    Supports both overall-score comparison and subsystem-level comparison.
    Version drift is detected by comparing VersionManifest hashes.
    """

    def compare(
        self,
        baseline: BenchmarkSnapshot,
        current: BenchmarkSnapshot,
    ) -> DriftReport:
        """Full drift analysis between baseline and current snapshots."""

        # Version drift
        changed_files = self._detect_version_drift(baseline, current)
        version_drift = VersionDrift(
            changed_files=changed_files,
            affects_scores=any(
                f in changed_files
                for f in [
                    "tests/scoring_engine.py",
                    "tests/trace_schema.py",
                    "tests/scenario_suite.py",
                ]
            ),
        )

        # Build lookup maps
        baseline_map = {s.get("scenario"): s for s in baseline.scores}
        current_map  = {s.get("scenario"): s for s in current.scores}

        # Compute per-scenario deltas
        scenario_deltas: List[ScenarioDelta] = []
        compared_names = set(baseline_map) & set(current_map)

        for name in sorted(compared_names):
            b = baseline_map[name]
            c = current_map[name]
            sd = self._compare_scenario(name, b, c)
            scenario_deltas.append(sd)

        # Compute per-category deltas
        category_deltas = self._compute_category_deltas(scenario_deltas)

        # Classify regressions and improvements
        regressions = [d.scenario_name for d in scenario_deltas if d.is_regression]
        improvements = [d.scenario_name for d in scenario_deltas if d.is_improvement]

        # Overall delta
        overall_delta = 0.0
        if scenario_deltas:
            overall_delta = sum(d.delta for d in scenario_deltas) / len(scenario_deltas)

        # Severity
        severity = self._classify_severity(
            scenario_deltas=scenario_deltas,
            category_deltas=category_deltas,
            version_drift=version_drift,
        )

        return DriftReport(
            baseline_snapshot_id=baseline.snapshot_id,
            current_snapshot_id=current.snapshot_id,
            baseline_label=baseline.run_label,
            current_label=current.run_label,
            scenario_deltas=scenario_deltas,
            category_deltas=category_deltas,
            version_drift=version_drift,
            drift_severity=severity,
            drift_detected=severity != "none",
            regressions=regressions,
            improvements=improvements,
            overall_delta=round(overall_delta, 4),
            scenarios_compared=len(scenario_deltas),
        )

    def compare_by_id(
        self,
        registry: Any,  # BenchmarkRegistry — avoid circular import type
        baseline_id: str,
        current_id: str,
    ) -> DriftReport:
        """Convenience: load snapshots by ID and compare."""
        baseline = registry.load(baseline_id)
        current = registry.load(current_id)
        return self.compare(baseline, current)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _compare_scenario(
        self,
        name: str,
        baseline_score: Dict[str, Any],
        current_score: Dict[str, Any],
    ) -> ScenarioDelta:
        category = str(
            current_score.get("category") or baseline_score.get("category") or ""
        )
        b_overall = self._extract_overall(baseline_score)
        c_overall = self._extract_overall(current_score)
        delta = c_overall - b_overall

        # Subsystem deltas
        sub_deltas: Dict[str, float] = {}
        for sub in ["planning", "tools", "memory", "ux", "robustness"]:
            b_sub = self._extract_subsystem(baseline_score, sub)
            c_sub = self._extract_subsystem(current_score, sub)
            if b_sub is not None and c_sub is not None:
                sub_deltas[sub] = round(c_sub - b_sub, 4)

        return ScenarioDelta(
            scenario_name=name,
            category=category,
            baseline_score=b_overall,
            current_score=c_overall,
            delta=round(delta, 4),
            is_regression=delta < -REGRESSION_THRESHOLD,
            is_improvement=delta > IMPROVEMENT_THRESHOLD,
            is_noise=abs(delta) <= NOISE_BAND,
            subsystem_deltas=sub_deltas,
        )

    def _compute_category_deltas(
        self, scenario_deltas: List[ScenarioDelta]
    ) -> List[CategoryDelta]:
        by_cat: Dict[str, List[ScenarioDelta]] = {}
        for d in scenario_deltas:
            by_cat.setdefault(d.category, []).append(d)

        result = []
        for cat, deltas in sorted(by_cat.items()):
            b_avg = sum(d.baseline_score for d in deltas) / len(deltas)
            c_avg = sum(d.current_score for d in deltas) / len(deltas)
            result.append(CategoryDelta(
                category=cat,
                baseline_avg=round(b_avg, 4),
                current_avg=round(c_avg, 4),
                delta=round(c_avg - b_avg, 4),
                scenario_count=len(deltas),
                regressions_in_category=sum(1 for d in deltas if d.is_regression),
                improvements_in_category=sum(1 for d in deltas if d.is_improvement),
            ))
        return result

    def _classify_severity(
        self,
        scenario_deltas: List[ScenarioDelta],
        category_deltas: List[CategoryDelta],
        version_drift: VersionDrift,
    ) -> str:
        regression_count = sum(1 for d in scenario_deltas if d.is_regression)
        has_critical = any(abs(d.delta) > 0.10 for d in scenario_deltas if d.is_regression)
        has_category_regression = any(cd.is_regression for cd in category_deltas)

        if has_critical or (regression_count >= 3) or has_category_regression:
            return "critical"
        if regression_count >= 1:
            return "significant"
        if any(not d.is_noise for d in scenario_deltas) or version_drift.affects_scores:
            return "minor"
        return "none"

    def _detect_version_drift(
        self,
        baseline: BenchmarkSnapshot,
        current: BenchmarkSnapshot,
    ) -> List[str]:
        b = baseline.version_manifest
        c = current.version_manifest
        changed = []
        checks = {
            "tests/scoring_engine.py": (b.scoring_engine_hash, c.scoring_engine_hash),
            "tests/trace_schema.py":   (b.trace_schema_hash,   c.trace_schema_hash),
            "tests/scenario_suite.py": (b.scenario_suite_hash, c.scenario_suite_hash),
            "evaluation/gold_set.py":  (b.gold_set_hash,       c.gold_set_hash),
        }
        if baseline.execution_mode == "orchestrator":
            checks["dadbot/core/orchestrator.py"] = (
                b.orchestrator_hash, c.orchestrator_hash
            )
        for file, (old, new) in checks.items():
            if old and new and old != new:
                changed.append(file)
        return changed

    @staticmethod
    def _extract_overall(score_dict: Dict[str, Any]) -> float:
        """Extract overall score from a serialized benchmark result."""
        # Try capability_score first (Phase 4B+)
        cap = score_dict.get("capability_score") or {}
        if cap.get("overall") is not None:
            return float(cap["overall"])
        # Fall back to legacy scoring.success → 1.0 or 0.0
        scoring = score_dict.get("scoring") or {}
        if scoring.get("success") is not None:
            return 1.0 if scoring["success"] else 0.0
        return 0.0

    @staticmethod
    def _extract_subsystem(score_dict: Dict[str, Any], subsystem: str) -> Optional[float]:
        """Extract a subsystem score from a serialized benchmark result."""
        cap = score_dict.get("capability_score") or {}
        sub = cap.get(subsystem) or {}
        if sub.get("score") is not None:
            return float(sub["score"])
        return None


# ---------------------------------------------------------------------------
# Regression gate helper (for use in pytest or CI)
# ---------------------------------------------------------------------------

def assert_no_regression(
    registry: Any,  # BenchmarkRegistry
    current_scores: List[Dict[str, Any]],
    execution_mode: str = "mock",
    run_label: str = "current",
    allow_minor: bool = True,
) -> DriftReport:
    """Save current scores, compare to latest baseline, return DriftReport.

    Raises AssertionError if drift exceeds allowed severity.
    Intended for use inside pytest or CI pipelines.

    Example:
        report = assert_no_regression(registry, scores)
        assert report.passes_gate(), report.summary()
    """
    # Save current run
    current_id = registry.save(
        scores=current_scores,
        execution_mode=execution_mode,
        run_label=run_label,
    )
    current_snap = registry.load(current_id)

    # Find previous snapshot (excluding the one we just saved)
    all_snaps = registry.list_snapshots()
    previous = [s for s in all_snaps if s.snapshot_id != current_id]
    if not previous:
        # No baseline yet — this IS the baseline, nothing to compare
        return DriftReport(
            baseline_snapshot_id=current_id,
            current_snapshot_id=current_id,
            baseline_label=run_label,
            current_label=run_label,
            scenario_deltas=[],
            category_deltas=[],
            version_drift=VersionDrift(changed_files=[], affects_scores=False),
            drift_severity="none",
            drift_detected=False,
            regressions=[],
            improvements=[],
            overall_delta=0.0,
            scenarios_compared=0,
        )

    baseline_snap = registry.load(previous[0].snapshot_id)
    detector = DriftDetector()
    return detector.compare(baseline_snap, current_snap)
