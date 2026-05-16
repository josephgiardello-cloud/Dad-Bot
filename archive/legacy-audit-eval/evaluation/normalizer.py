"""Normalization Engine — Phase 4F Industry Comparison Layer.

Purpose:
  Convert raw scores (0.0-1.0 internal values) into:
    1. Percentile ranks vs registered baselines
    2. Z-scores relative to the baseline distribution
    3. Per-capability indices with rankings
    4. A human-readable Capability Profile report

The output answers: "How good is this system compared to known agent types?"

Normalization model:
  Given N baseline profiles, we have a distribution of N known points
  per subsystem. For a new observed score s:

    percentile = |{baseline_score < s}| / N * 100
    z_score    = (s - mean(baselines)) / std(baselines)

  With only 4 baselines, this is a small distribution.
  The system is honest about this: confidence is marked as "low" until
  more baselines are added or measured.

Usage:
    from evaluation.normalizer import NormalizationEngine, CapabilityProfile
    from evaluation.baselines import BASELINES
    from tests.scoring_engine import CapabilityScore

    engine = NormalizationEngine(BASELINES)
    profile = engine.normalize_score(capability_score)
    engine.print_profile(profile)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from evaluation.baselines import BASELINES, BaselineProfile

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SubsystemIndex:
    """Normalized capability index for one subsystem."""

    subsystem: str
    raw_score: float
    percentile: float  # 0–100
    z_score: float
    above_baselines: list[str]  # baseline names this score beats
    below_baselines: list[str]  # baseline names this score trails
    baseline_scores: dict[str, float] = field(default_factory=dict)

    @property
    def rank_label(self) -> str:
        """Human label for percentile range."""
        if self.percentile >= 90:
            return "Exceptional"
        if self.percentile >= 75:
            return "Strong"
        if self.percentile >= 50:
            return "Competitive"
        if self.percentile >= 25:
            return "Developing"
        return "Below Baseline"

    @property
    def confidence(self) -> str:
        """How reliable is this percentile given baseline count."""
        n = len(self.baseline_scores)
        if n >= 10:
            return "high"
        if n >= 5:
            return "medium"
        return "low"


@dataclass
class CapabilityProfile:
    """Full normalized capability profile for one benchmark run.

    This is the external-facing output of Phase 4F.
    """

    scenario_name: str
    category: str
    execution_mode: str

    indices: dict[str, SubsystemIndex]  # subsystem → index
    overall_percentile: float
    overall_z_score: float
    overall_raw: float

    @property
    def primary_index(self) -> SubsystemIndex | None:
        return self.indices.get(self.category)

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario_name,
            "category": self.category,
            "overall_percentile": round(self.overall_percentile, 1),
            "overall_z_score": round(self.overall_z_score, 3),
            "overall_raw": round(self.overall_raw, 4),
            "indices": {
                sub: {
                    "raw_score": round(idx.raw_score, 4),
                    "percentile": round(idx.percentile, 1),
                    "z_score": round(idx.z_score, 3),
                    "rank": idx.rank_label,
                    "confidence": idx.confidence,
                    "above": idx.above_baselines,
                    "below": idx.below_baselines,
                }
                for sub, idx in self.indices.items()
            },
        }


@dataclass
class AggregateCapabilityProfile:
    """Normalized profile aggregated across all scenarios in a benchmark run."""

    execution_mode: str
    scenario_count: int

    # Category-level normalized scores
    category_indices: dict[str, SubsystemIndex]  # category → index

    # Overall
    overall_percentile: float
    overall_z_score: float
    overall_raw: float

    def to_dict(self) -> dict:
        return {
            "scenario_count": self.scenario_count,
            "execution_mode": self.execution_mode,
            "overall_percentile": round(self.overall_percentile, 1),
            "overall_z_score": round(self.overall_z_score, 3),
            "overall_raw": round(self.overall_raw, 4),
            "categories": {
                cat: {
                    "raw_score": round(idx.raw_score, 4),
                    "percentile": round(idx.percentile, 1),
                    "z_score": round(idx.z_score, 3),
                    "rank": idx.rank_label,
                    "confidence": idx.confidence,
                }
                for cat, idx in self.category_indices.items()
            },
        }


# ---------------------------------------------------------------------------
# Normalization Engine
# ---------------------------------------------------------------------------

_SUBSYSTEMS = ["planning", "tools", "memory", "ux", "robustness"]
_WEIGHTS = {"planning": 0.25, "tools": 0.25, "memory": 0.20, "ux": 0.15, "robustness": 0.15}


class NormalizationEngine:
    """Converts raw CapabilityScores into normalized CapabilityProfiles.

    The engine is stateless — it reads from the baselines dict passed at init.
    """

    def __init__(self, baselines: dict[str, BaselineProfile] | None = None):
        self._baselines = baselines or BASELINES

    def normalize_score(
        self,
        raw: float,
        subsystem: str,
        scenario_name: str = "",
        category: str = "",
        execution_mode: str = "mock",
    ) -> SubsystemIndex:
        """Normalize a single raw score against the baseline distribution."""
        baseline_scores = {name: b.get(subsystem) for name, b in self._baselines.items()}
        return self._compute_index(subsystem, raw, baseline_scores)

    def normalize_capability_score(self, cap_score: Any) -> CapabilityProfile:
        """Normalize all subsystems of a CapabilityScore at once."""
        indices: dict[str, SubsystemIndex] = {}

        for sub in _SUBSYSTEMS:
            sub_obj = getattr(cap_score, sub, None)
            raw = sub_obj.score if sub_obj else 0.0
            baseline_scores = {name: b.get(sub) for name, b in self._baselines.items()}
            indices[sub] = self._compute_index(sub, raw, baseline_scores)

        # Overall
        baseline_overalls = {name: b.overall for name, b in self._baselines.items()}
        overall_percentile = _percentile(cap_score.overall, list(baseline_overalls.values()))
        overall_z = _z_score(cap_score.overall, list(baseline_overalls.values()))

        return CapabilityProfile(
            scenario_name=str(getattr(cap_score, "scenario_name", "")),
            category=str(getattr(cap_score, "category", "")),
            execution_mode=str(getattr(cap_score, "execution_mode", "mock")),
            indices=indices,
            overall_percentile=overall_percentile,
            overall_z_score=overall_z,
            overall_raw=cap_score.overall,
        )

    def normalize_all(
        self,
        cap_scores: list[Any],
    ) -> AggregateCapabilityProfile:
        """Aggregate normalization across all scenarios in a benchmark run."""
        if not cap_scores:
            empty_idx = SubsystemIndex(
                subsystem="",
                raw_score=0.0,
                percentile=0.0,
                z_score=0.0,
                above_baselines=[],
                below_baselines=[],
            )
            return AggregateCapabilityProfile(
                execution_mode="mock",
                scenario_count=0,
                category_indices={},
                overall_percentile=0.0,
                overall_z_score=0.0,
                overall_raw=0.0,
            )

        execution_mode = str(getattr(cap_scores[0], "execution_mode", "mock"))

        # Aggregate per-subsystem
        sub_raws: dict[str, list[float]] = {s: [] for s in _SUBSYSTEMS}
        for cs in cap_scores:
            for sub in _SUBSYSTEMS:
                sub_obj = getattr(cs, sub, None)
                if sub_obj:
                    sub_raws[sub].append(sub_obj.score)

        category_indices: dict[str, SubsystemIndex] = {}
        for sub in _SUBSYSTEMS:
            if not sub_raws[sub]:
                continue
            avg_raw = sum(sub_raws[sub]) / len(sub_raws[sub])
            baseline_scores = {name: b.get(sub) for name, b in self._baselines.items()}
            category_indices[sub] = self._compute_index(sub, avg_raw, baseline_scores)

        # Overall
        overall_raws = [cs.overall for cs in cap_scores]
        avg_overall = sum(overall_raws) / len(overall_raws)
        baseline_overalls = {name: b.overall for name, b in self._baselines.items()}
        overall_percentile = _percentile(avg_overall, list(baseline_overalls.values()))
        overall_z = _z_score(avg_overall, list(baseline_overalls.values()))

        return AggregateCapabilityProfile(
            execution_mode=execution_mode,
            scenario_count=len(cap_scores),
            category_indices=category_indices,
            overall_percentile=overall_percentile,
            overall_z_score=overall_z,
            overall_raw=round(avg_overall, 4),
        )

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _compute_index(
        self,
        subsystem: str,
        raw: float,
        baseline_scores: dict[str, float],
    ) -> SubsystemIndex:
        vals = list(baseline_scores.values())
        above = [name for name, score in baseline_scores.items() if raw > score]
        below = [name for name, score in baseline_scores.items() if raw <= score]

        return SubsystemIndex(
            subsystem=subsystem,
            raw_score=round(raw, 4),
            percentile=round(_percentile(raw, vals), 1),
            z_score=round(_z_score(raw, vals), 3),
            above_baselines=above,
            below_baselines=below,
            baseline_scores={name: round(s, 4) for name, s in baseline_scores.items()},
        )


# ---------------------------------------------------------------------------
# Capability Profile Report printer
# ---------------------------------------------------------------------------


def print_capability_profile(
    profile: AggregateCapabilityProfile,
    title: str = "CAPABILITY PROFILE",
    baselines: dict[str, BaselineProfile] | None = None,
) -> None:
    """Print a human-readable capability profile with baseline comparison."""
    baselines = baselines or BASELINES

    bar_width = 32

    print("\n" + "=" * 72)
    print(title)
    print(
        f"  Scenarios: {profile.scenario_count}  |  "
        f"Mode: {profile.execution_mode}  |  "
        f"Overall: {profile.overall_raw:.2%}  "
        f"(p{profile.overall_percentile:.0f}  z={profile.overall_z_score:+.2f})"
    )
    print("=" * 72)

    print("\n  CAPABILITY INDEX vs BASELINES:\n")
    print(f"  {'Subsystem':<12}  {'Score':>6}  {'%ile':>5}  {'Rank':<15}  Bar")
    print(f"  {'-' * 12}  {'-' * 6}  {'-' * 5}  {'-' * 15}  {'-' * bar_width}")

    for sub in _SUBSYSTEMS:
        idx = profile.category_indices.get(sub)
        if idx is None:
            continue
        filled = int(idx.percentile / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"  {sub:<12}  {idx.raw_score:>6.3f}  {idx.percentile:>5.1f}  {idx.rank_label:<15}  {bar}")

    print("\n  BASELINE COMPARISON:\n")
    print(f"  {'Subsystem':<12}", end="")
    for bname in baselines:
        print(f"  {bname[:12]:>12}", end="")
    print(f"  {'[YOU]':>8}")
    print(f"  {'-' * 12}", end="")
    for _ in baselines:
        print(f"  {'':>12}", end="")
    print(f"  {'':>8}")

    for sub in _SUBSYSTEMS:
        idx = profile.category_indices.get(sub)
        raw = idx.raw_score if idx else 0.0
        print(f"  {sub:<12}", end="")
        for bname in baselines:
            bscore = baselines[bname].get(sub)
            marker = ">" if raw > bscore else ("=" if abs(raw - bscore) < 0.01 else " ")
            print(f"  {bscore:>11.3f}{marker}", end="")
        print(f"  {raw:>8.3f}")

    print(f"\n  {'overall':<12}", end="")
    for bname in baselines:
        bscore = baselines[bname].overall
        print(f"  {bscore:>12.3f}", end="")
    print(f"  {profile.overall_raw:>8.3f}")

    print("\n  NOTE: Baselines are seeded estimates. Run update_baseline()")
    print("        to replace with measured values for higher confidence.")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _percentile(score: float, distribution: list[float]) -> float:
    """Percentile of score in distribution (0–100).

    Uses a simple empirical percentile: fraction of distribution < score.
    """
    if not distribution:
        return 50.0
    below = sum(1 for v in distribution if score > v)
    return (below / len(distribution)) * 100.0


def _z_score(score: float, distribution: list[float]) -> float:
    """Z-score of score relative to distribution."""
    if not distribution:
        return 0.0
    m = _mean(distribution)
    s = _std(distribution)
    if s == 0.0:
        return 0.0
    return (score - m) / s
