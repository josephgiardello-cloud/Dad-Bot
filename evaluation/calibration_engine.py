"""Calibration Engine — Phase 4C.

Purpose:
  Convert "scoring engine output + gold standard" into statistically
  meaningful, calibrated scores.

The calibration loop:
  1. Run scoring engine on all scenarios → raw scores
  2. Compare raw scores to gold pseudo-labels → compute deviations
  3. Derive per-subsystem normalization offsets
  4. Store offsets in CalibrationState
  5. Apply CalibrationState to future scores → calibrated scores

Key concepts:
  - CalibrationState: a persistent set of learned offsets
  - NormalizationOffset: per-subsystem additive correction
  - CalibrationResult: per-scenario deviation analysis
  - CalibrationReport: aggregate across all scenarios

Design decisions:
  - Offsets are ADDITIVE (not multiplicative) — simpler, more stable
  - Offsets are clamped so they cannot invert the score direction
  - Confidence grows as more scored runs accumulate (running average)
  - Category weights are re-calibrated based on observed variance

Usage:
    from evaluation.calibration_engine import CalibrationEngine, CalibrationState
    from evaluation.gold_set import GOLD_SET

    engine = CalibrationEngine(GOLD_SET)
    report = engine.calibrate(capability_scores)
    calibrated = engine.apply(some_score, report.state)
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tests.scoring_engine import CapabilityScore, SubsystemScore


# ---------------------------------------------------------------------------
# Core calibration data structures
# ---------------------------------------------------------------------------

@dataclass
class NormalizationOffset:
    """Additive correction for one subsystem derived from gold deviation."""
    subsystem: str
    offset: float           # additive: calibrated = raw + offset
    confidence: float       # 0.0–1.0; grows with more calibration runs
    sample_count: int       # how many observations contributed
    mean_deviation: float   # mean(gold_ideal - observed)
    std_deviation: float    # standard deviation of deviations


@dataclass
class CalibrationResult:
    """Per-scenario calibration analysis."""
    scenario_id: str
    category: str

    # Raw observed scores
    observed_overall: float
    observed_planning: Optional[float]
    observed_tools: Optional[float]
    observed_memory: Optional[float]
    observed_ux: Optional[float]
    observed_robustness: Optional[float]

    # Gold ideal scores
    gold_overall: float
    gold_planning: float
    gold_tools: float
    gold_memory: float
    gold_ux: float
    gold_robustness: float

    # Deviation analysis
    overall_deviation: float    # observed - gold_overall
    within_gold_bounds: bool    # |deviation| <= gold.acceptable_variance
    subsystem_deviations: Dict[str, float] = field(default_factory=dict)

    # Band violations (subsystems outside acceptable floor/ceiling)
    band_violations: List[str] = field(default_factory=list)

    @property
    def is_calibrated(self) -> bool:
        """True if overall score is within gold bounds and no critical violations."""
        return self.within_gold_bounds and len(self.band_violations) == 0


@dataclass
class CategoryWeightAdjustment:
    """Proposed adjustment to category weights based on observed variance."""
    subsystem: str
    current_weight: float
    proposed_weight: float
    reason: str


@dataclass
class CalibrationState:
    """Persistent calibration state derived from one or more calibration runs.

    Store and reload this between runs to maintain calibration continuity.
    """
    offsets: Dict[str, NormalizationOffset] = field(default_factory=dict)
    category_weight_adjustments: List[CategoryWeightAdjustment] = field(default_factory=list)
    calibration_run_count: int = 0
    last_calibrated_scenario_count: int = 0

    def get_offset(self, subsystem: str) -> float:
        if subsystem in self.offsets:
            return self.offsets[subsystem].offset
        return 0.0

    def is_mature(self) -> bool:
        """Calibration is considered mature after 3+ runs with 10+ scenarios."""
        return (
            self.calibration_run_count >= 3
            and self.last_calibrated_scenario_count >= 10
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "offsets": {k: asdict(v) for k, v in self.offsets.items()},
            "category_weight_adjustments": [asdict(a) for a in self.category_weight_adjustments],
            "calibration_run_count": self.calibration_run_count,
            "last_calibrated_scenario_count": self.last_calibrated_scenario_count,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CalibrationState":
        offsets = {
            k: NormalizationOffset(**v)
            for k, v in (data.get("offsets") or {}).items()
        }
        adjustments = [
            CategoryWeightAdjustment(**a)
            for a in (data.get("category_weight_adjustments") or [])
        ]
        return cls(
            offsets=offsets,
            category_weight_adjustments=adjustments,
            calibration_run_count=int(data.get("calibration_run_count") or 0),
            last_calibrated_scenario_count=int(data.get("last_calibrated_scenario_count") or 0),
        )


@dataclass
class CalibrationReport:
    """Full calibration report for one run over all scenarios."""
    results: List[CalibrationResult]
    state: CalibrationState

    # Aggregate health metrics
    scenarios_within_bounds: int
    scenarios_out_of_bounds: int
    mean_overall_deviation: float
    std_overall_deviation: float
    calibration_quality: str    # "excellent" | "good" | "fair" | "poor"

    @property
    def pass_rate(self) -> float:
        total = len(self.results)
        if total == 0:
            return 0.0
        return self.scenarios_within_bounds / total

    def summary(self) -> str:
        lines = [
            f"CALIBRATION REPORT",
            f"  Scenarios: {len(self.results)}",
            f"  Within gold bounds: {self.scenarios_within_bounds}/{len(self.results)} ({self.pass_rate:.0%})",
            f"  Mean deviation: {self.mean_overall_deviation:+.4f}",
            f"  Std deviation: {self.std_overall_deviation:.4f}",
            f"  Quality: {self.calibration_quality}",
            f"  Calibration runs: {self.state.calibration_run_count}",
            "",
            "  NORMALIZATION OFFSETS:",
        ]
        for sub, offset in self.state.offsets.items():
            lines.append(
                f"    {sub:12s} offset={offset.offset:+.4f}  "
                f"confidence={offset.confidence:.2f}  n={offset.sample_count}"
            )
        out_of_bounds = [r for r in self.results if not r.within_gold_bounds]
        if out_of_bounds:
            lines.append("\n  OUT-OF-BOUNDS SCENARIOS:")
            for r in out_of_bounds:
                lines.append(
                    f"    {r.scenario_id}: observed={r.observed_overall:.3f}  "
                    f"gold={r.gold_overall:.3f}  "
                    f"deviation={r.overall_deviation:+.3f}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Calibration Engine
# ---------------------------------------------------------------------------

_SUBSYSTEMS = ["planning", "tools", "memory", "ux", "robustness"]
_BASE_WEIGHTS = {"planning": 0.25, "tools": 0.25, "memory": 0.20, "ux": 0.15, "robustness": 0.15}


class CalibrationEngine:
    """Computes calibration state from scored runs vs gold standards.

    Typical usage:
        engine = CalibrationEngine(GOLD_SET)
        report = engine.calibrate(scores)         # compare to gold
        calibrated = engine.apply(score, report.state)  # apply offsets
    """

    def __init__(self, gold_set: Dict[str, Any], state_path: Optional[Path] = None):
        self._gold = gold_set
        self._state_path = state_path or Path("evaluation/calibration_state.json")

    # -----------------------------------------------------------------------
    # Primary API
    # -----------------------------------------------------------------------

    def calibrate(
        self,
        scores: List[CapabilityScore],
        previous_state: Optional[CalibrationState] = None,
    ) -> CalibrationReport:
        """Compare observed scores to gold, compute offsets, return report."""
        results: List[CalibrationResult] = []

        for cs in scores:
            gold = self._gold.get(cs.scenario_name)
            if gold is None:
                continue  # no gold for this scenario — skip
            result = self._analyze_scenario(cs, gold)
            results.append(result)

        if not results:
            state = previous_state or CalibrationState()
            return CalibrationReport(
                results=[],
                state=state,
                scenarios_within_bounds=0,
                scenarios_out_of_bounds=0,
                mean_overall_deviation=0.0,
                std_overall_deviation=0.0,
                calibration_quality="poor",
            )

        # Compute offsets from this run
        offsets = self._compute_offsets(results, previous_state)

        # Compute weight adjustments
        weight_adjustments = self._compute_weight_adjustments(results)

        # Update state
        run_count = (previous_state.calibration_run_count if previous_state else 0) + 1
        state = CalibrationState(
            offsets=offsets,
            category_weight_adjustments=weight_adjustments,
            calibration_run_count=run_count,
            last_calibrated_scenario_count=len(results),
        )

        # Aggregate stats
        deviations = [r.overall_deviation for r in results]
        mean_dev = sum(deviations) / len(deviations)
        std_dev = _std(deviations)
        within = sum(1 for r in results if r.within_gold_bounds)
        out_of = len(results) - within
        quality = self._quality_rating(within / len(results), std_dev)

        return CalibrationReport(
            results=results,
            state=state,
            scenarios_within_bounds=within,
            scenarios_out_of_bounds=out_of,
            mean_overall_deviation=mean_dev,
            std_overall_deviation=std_dev,
            calibration_quality=quality,
        )

    def apply(self, score: CapabilityScore, state: CalibrationState) -> CapabilityScore:
        """Apply calibration offsets to a raw CapabilityScore.

        Returns a new CapabilityScore with adjusted values.
        Offsets are additive and clamped to [0.0, 1.0].
        The original score object is NOT mutated.
        """
        if not state.offsets:
            return score  # no calibration data yet — return as-is

        def _adjust(sub: Optional[SubsystemScore], name: str) -> Optional[SubsystemScore]:
            if sub is None:
                return None
            offset = state.get_offset(name)
            new_score = max(0.0, min(1.0, sub.score + offset))
            # Return adjusted copy — SubsystemScore is a dataclass
            return SubsystemScore(
                subsystem=sub.subsystem,
                score=new_score,
                partial_success=0.0 < new_score < 1.0,
                signals=sub.signals,
                penalties=sub.penalties,
                confidence=sub.confidence,
                notes=sub.notes + [f"calibration_offset={offset:+.4f}"],
            )

        adj_planning   = _adjust(score.planning, "planning")
        adj_tools      = _adjust(score.tools, "tools")
        adj_memory     = _adjust(score.memory, "memory")
        adj_ux         = _adjust(score.ux, "ux")
        adj_robustness = _adjust(score.robustness, "robustness")

        # Recompute weighted overall
        subs = {
            "planning": adj_planning,
            "tools": adj_tools,
            "memory": adj_memory,
            "ux": adj_ux,
            "robustness": adj_robustness,
        }
        overall_offset = state.get_offset("overall")
        new_overall = max(0.0, min(1.0,
            sum(
                subs[cat].score * _BASE_WEIGHTS[cat]
                for cat in _BASE_WEIGHTS
                if subs[cat] is not None
            ) + overall_offset
        ))

        return CapabilityScore(
            scenario_name=score.scenario_name,
            category=score.category,
            overall=round(new_overall, 4),
            planning=adj_planning,
            tools=adj_tools,
            memory=adj_memory,
            ux=adj_ux,
            robustness=adj_robustness,
            execution_mode=score.execution_mode,
            is_mock_data=score.is_mock_data,
        )

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save_state(self, state: CalibrationState) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_path, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2)

    def load_state(self) -> Optional[CalibrationState]:
        if not self._state_path.exists():
            return None
        with open(self._state_path, encoding="utf-8") as f:
            return CalibrationState.from_dict(json.load(f))

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _analyze_scenario(self, cs: CapabilityScore, gold: Any) -> CalibrationResult:
        """Compare one CapabilityScore to its GoldScenario."""
        label = gold.pseudo_label

        # Extract observed subsystem scores
        obs = {
            "planning":   cs.planning.score if cs.planning else None,
            "tools":      cs.tools.score if cs.tools else None,
            "memory":     cs.memory.score if cs.memory else None,
            "ux":         cs.ux.score if cs.ux else None,
            "robustness": cs.robustness.score if cs.robustness else None,
        }

        # Extract gold ideals from pseudo-label
        gold_vals = {
            "planning":   label.planning_expected if label else 0.75,
            "tools":      label.tools_expected if label else 0.75,
            "memory":     label.memory_expected if label else 0.75,
            "ux":         label.ux_expected if label else 0.75,
            "robustness": label.robustness_expected if label else 0.75,
        }
        gold_overall = label.overall_expected if label else gold.ideal_score

        overall_deviation = cs.overall - gold_overall
        within_bounds = abs(overall_deviation) <= gold.acceptable_variance

        # Per-subsystem deviations
        sub_devs: Dict[str, float] = {}
        for sub in _SUBSYSTEMS:
            if obs[sub] is not None:
                sub_devs[sub] = obs[sub] - gold_vals[sub]

        # Band violations
        violations: List[str] = []
        for sub in _SUBSYSTEMS:
            if obs[sub] is not None:
                if not gold.within_band(sub, obs[sub]):
                    violations.append(sub)

        return CalibrationResult(
            scenario_id=cs.scenario_name,
            category=cs.category,
            observed_overall=cs.overall,
            observed_planning=obs["planning"],
            observed_tools=obs["tools"],
            observed_memory=obs["memory"],
            observed_ux=obs["ux"],
            observed_robustness=obs["robustness"],
            gold_overall=gold_overall,
            gold_planning=gold_vals["planning"],
            gold_tools=gold_vals["tools"],
            gold_memory=gold_vals["memory"],
            gold_ux=gold_vals["ux"],
            gold_robustness=gold_vals["robustness"],
            overall_deviation=overall_deviation,
            within_gold_bounds=within_bounds,
            subsystem_deviations=sub_devs,
            band_violations=violations,
        )

    def _compute_offsets(
        self,
        results: List[CalibrationResult],
        previous_state: Optional[CalibrationState],
    ) -> Dict[str, NormalizationOffset]:
        """Compute additive normalization offsets per subsystem.

        Strategy:
        - For each subsystem: offset = -mean_deviation
          (so calibrated = observed - mean_deviation → centers on gold)
        - Blend with previous offsets if available (exponential moving average, α=0.3)
        - Cap offsets at ±0.25 to prevent score inversion
        """
        offsets: Dict[str, NormalizationOffset] = {}
        alpha = 0.3  # weight for new observation vs previous

        for sub in _SUBSYSTEMS + ["overall"]:
            if sub == "overall":
                devs = [r.overall_deviation for r in results]
            else:
                devs = [
                    r.subsystem_deviations[sub]
                    for r in results
                    if sub in r.subsystem_deviations
                ]

            if not devs:
                continue

            mean_dev = sum(devs) / len(devs)
            std_dev = _std(devs)
            raw_offset = -mean_dev  # correct for systematic bias

            # Blend with previous
            prev_offset = 0.0
            prev_n = 0
            if previous_state and sub in previous_state.offsets:
                prev = previous_state.offsets[sub]
                prev_offset = prev.offset
                prev_n = prev.sample_count

            blended = (alpha * raw_offset + (1 - alpha) * prev_offset) if prev_n > 0 else raw_offset
            blended = max(-0.25, min(0.25, blended))  # hard cap

            total_n = len(devs) + prev_n
            confidence = min(1.0, total_n / 30.0)  # 30 observations = full confidence

            offsets[sub] = NormalizationOffset(
                subsystem=sub,
                offset=round(blended, 5),
                confidence=round(confidence, 4),
                sample_count=total_n,
                mean_deviation=round(mean_dev, 5),
                std_deviation=round(std_dev, 5),
            )

        return offsets

    def _compute_weight_adjustments(
        self,
        results: List[CalibrationResult],
    ) -> List[CategoryWeightAdjustment]:
        """Propose category weight adjustments based on variance.

        Logic: high-variance subsystems are less reliable → down-weight
        Low-variance subsystems with high confidence → up-weight
        Adjustments are capped at ±0.05 per cycle.
        """
        adjustments: List[CategoryWeightAdjustment] = []

        for sub in _SUBSYSTEMS:
            devs = [r.subsystem_deviations.get(sub, 0.0) for r in results]
            variance = _variance(devs)
            current = _BASE_WEIGHTS.get(sub, 0.20)

            if variance > 0.04:  # std > 0.2 = high variance
                delta = -min(0.05, variance * 0.1)
                reason = f"high_variance={variance:.3f}"
            elif variance < 0.01:  # std < 0.1 = stable
                delta = min(0.05, 0.005)
                reason = f"low_variance={variance:.3f}"
            else:
                continue  # no adjustment needed

            adjustments.append(CategoryWeightAdjustment(
                subsystem=sub,
                current_weight=current,
                proposed_weight=max(0.05, min(0.40, current + delta)),
                reason=reason,
            ))

        return adjustments

    def _quality_rating(self, pass_rate: float, std_dev: float) -> str:
        if pass_rate >= 0.90 and std_dev < 0.05:
            return "excellent"
        if pass_rate >= 0.75 and std_dev < 0.10:
            return "good"
        if pass_rate >= 0.60:
            return "fair"
        return "poor"


# ---------------------------------------------------------------------------
# Statistics helpers (no scipy dependency)
# ---------------------------------------------------------------------------

def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _variance(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return sum((v - m) ** 2 for v in values) / (len(values) - 1)


def _std(values: List[float]) -> float:
    return math.sqrt(_variance(values))
