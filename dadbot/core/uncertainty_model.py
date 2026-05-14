"""Phase 4 — Uncertainty Propagation System.

Converts raw tool outputs into probabilistic inputs for the planner and critic.
Without this, the system treats all tool outputs as equally credible — which is
incorrect once the external tool runtime introduces partial results, degraded
modes, fallback chains, and variable-reliability tools.

4.1  ConfidenceVector
     A three-dimensional credibility score per tool result:
       - reliability_score  — historical reliability of the tool (0–1)
       - freshness_score    — how recent / time-sensitive the data is (0–1)
       - completeness_score — how complete the result is vs. expected (0–1)

     Derives a single ``aggregate`` score as a weighted harmonic mean.

4.2  UncertaintyPropagator
     Propagation hooks that let the planner and critic adjust their
     behavior based on confidence vectors:
       - planner: weights low-confidence outputs differently in goal planning
       - critic: penalizes overconfident reasoning on weak-confidence tools

4.3  ConfidenceFusion
     Aggregates confidence across a chain of tool results (e.g. fallback chain,
     multi-tool query) into a single fused confidence vector using:
       - Conservative fusion (min over dimensions)
       - Optimistic fusion (max over dimensions)
       - Bayesian fusion (weighted harmonic mean, default)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# 4.1 — ConfidenceVector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceVector:
    """Three-dimensional credibility score for a single tool result.

    Attributes:
        reliability_score:   How historically reliable this tool is (0–1).
        freshness_score:     How recent / time-appropriate the data is (0–1).
        completeness_score:  How complete the result is vs. expected (0–1).
        source_tool:         Which tool produced this result.
        result_status:       Terminal execution status (ok/partial/error/…).
        recorded_at:         Epoch timestamp when this vector was created.

    """

    reliability_score: float
    freshness_score: float
    completeness_score: float
    source_tool: str = ""
    result_status: str = "ok"
    recorded_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        # Validate bounds at construction time
        for attr in ("reliability_score", "freshness_score", "completeness_score"):
            v = getattr(self, attr)
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"ConfidenceVector.{attr} must be in [0, 1], got {v!r}",
                )

    @property
    def aggregate(self) -> float:
        """Weighted harmonic mean of the three dimensions.

        Weighting:
          reliability × 0.45  (dominates: a broken tool is useless)
          freshness   × 0.30  (significant: stale data misleads)
          completeness× 0.25  (relevant: partial results degrade quality)
        """
        w = (0.45, 0.30, 0.25)
        vals = (self.reliability_score, self.freshness_score, self.completeness_score)
        # Weighted harmonic mean: W / Σ(w_i / v_i)
        # Guard against zero-division with a small epsilon
        eps = 1e-9
        denom = sum(wi / max(vi, eps) for wi, vi in zip(w, vals))
        return round(sum(w) / denom, 4)

    def is_low_confidence(self, threshold: float = 0.5) -> bool:
        return self.aggregate < threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "reliability_score": round(self.reliability_score, 4),
            "freshness_score": round(self.freshness_score, 4),
            "completeness_score": round(self.completeness_score, 4),
            "aggregate": self.aggregate,
            "source_tool": self.source_tool,
            "result_status": self.result_status,
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_tool_result(
        cls,
        tool_name: str,
        status: str,
        partial_confidence: float = 1.0,
        historical_reliability: float = 0.9,
        data_age_seconds: float = 0.0,
        freshness_half_life_s: float = 3600.0,
    ) -> ConfidenceVector:
        """Construct a ConfidenceVector from execution metadata.

        Args:
            tool_name:              Name of the executing tool.
            status:                 Terminal execution status.
            partial_confidence:     Self-reported confidence from partial outputs (0–1).
            historical_reliability: Long-run reliability fraction for this tool.
            data_age_seconds:       How old the underlying data is in seconds.
            freshness_half_life_s:  Half-life for exponential freshness decay (default 1 hour).

        """
        # Reliability: use historical rate; penalize error/timeout/degraded
        status_penalty = {
            "ok": 1.0,
            "cached": 0.95,
            "partial": 0.75,
            "skipped": 0.6,
            "degraded": 0.55,
            "timeout": 0.3,
            "error": 0.2,
        }.get(str(status or "ok").lower(), 0.5)
        reliability = max(0.0, min(1.0, historical_reliability * status_penalty))

        # Freshness: exponential decay from data_age_seconds
        if data_age_seconds <= 0:
            freshness = 1.0
        else:
            hl = max(1.0, freshness_half_life_s)
            freshness = max(
                0.0,
                min(1.0, math.exp(-math.log(2) * data_age_seconds / hl)),
            )

        # Completeness: direct from partial_confidence, floored for errors
        if str(status or "ok").lower() in {"error", "timeout"}:
            completeness = 0.0
        else:
            completeness = max(0.0, min(1.0, float(partial_confidence or 1.0)))

        return cls(
            reliability_score=round(reliability, 4),
            freshness_score=round(freshness, 4),
            completeness_score=round(completeness, 4),
            source_tool=str(tool_name or ""),
            result_status=str(status or "ok"),
        )


# ---------------------------------------------------------------------------
# 4.2 — Uncertainty Propagator
# ---------------------------------------------------------------------------


class PlannerWeightMode(str, Enum):
    """How the planner should weight a low-confidence tool output."""

    USE_AS_IS = "use_as_is"  # treat output as fully credible
    DISCOUNT = "discount"  # reduce the output's weight in planning
    FLAG_FOR_REVIEW = "flag_for_review"  # mark for human/critic review
    EXCLUDE = "exclude"  # exclude from planning entirely


@dataclass(frozen=True)
class PlannerHint:
    """Hint produced for the planner about a tool result's credibility."""

    tool_name: str
    confidence_vector: ConfidenceVector
    weight_mode: PlannerWeightMode
    planning_weight: float  # 0–1; multiply against goal weights
    reason: str


@dataclass(frozen=True)
class CriticPenalty:
    """Penalty signal for the critic when reasoning on weak-confidence inputs."""

    tool_name: str
    confidence_vector: ConfidenceVector
    penalty_factor: float  # 0–1; applied to critic confidence scores
    reason: str


class UncertaintyPropagator:
    """Translates ConfidenceVectors into actionable hints for planner and critic.

    Thresholds (all configurable):
      low_confidence_threshold  — aggregate below this → DISCOUNT mode
      critical_threshold        — aggregate below this → EXCLUDE mode
      critic_penalty_floor      — minimum critic penalty for low-confidence inputs
    """

    def __init__(
        self,
        low_confidence_threshold: float = 0.5,
        critical_threshold: float = 0.25,
        critic_penalty_floor: float = 0.3,
    ) -> None:
        self._low = max(0.0, min(1.0, low_confidence_threshold))
        self._critical = max(0.0, min(self._low, critical_threshold))
        self._penalty_floor = max(0.0, min(1.0, critic_penalty_floor))

    def planner_hint(self, cv: ConfidenceVector) -> PlannerHint:
        """Produce a planning weight hint from a ConfidenceVector."""
        agg = cv.aggregate

        if agg >= self._low:
            mode = PlannerWeightMode.USE_AS_IS
            weight = 1.0
            reason = "Confidence above threshold; use output as-is."
        elif agg >= self._critical:
            mode = PlannerWeightMode.DISCOUNT
            # Linear interpolation between low and critical thresholds
            fraction = (agg - self._critical) / max(self._low - self._critical, 1e-9)
            weight = round(0.3 + 0.5 * fraction, 4)  # [0.3, 0.8]
            reason = f"Low confidence (agg={agg:.3f}); discounting planning weight to {weight:.3f}."
        else:
            mode = PlannerWeightMode.EXCLUDE
            weight = 0.0
            reason = f"Critical confidence (agg={agg:.3f}); excluding from planning."

        return PlannerHint(
            tool_name=cv.source_tool,
            confidence_vector=cv,
            weight_mode=mode,
            planning_weight=weight,
            reason=reason,
        )

    def critic_penalty(self, cv: ConfidenceVector) -> CriticPenalty:
        """Produce a critic penalty from a ConfidenceVector.

        A low-confidence tool result should make the critic less willing to
        endorse conclusions that heavily rely on that result.
        """
        agg = cv.aggregate
        # Penalty scales inversely with confidence:
        # agg=1.0 → penalty_factor=1.0 (no penalty)
        # agg=0.5 → penalty_factor ≈ 0.65
        # agg=0.0 → penalty_factor = penalty_floor
        penalty = max(
            self._penalty_floor,
            round(self._penalty_floor + (1.0 - self._penalty_floor) * agg, 4),
        )
        reason = (
            f"Tool '{cv.source_tool}' has aggregate confidence {agg:.3f}; critic penalty factor set to {penalty:.3f}."
        )
        return CriticPenalty(
            tool_name=cv.source_tool,
            confidence_vector=cv,
            penalty_factor=penalty,
            reason=reason,
        )

    def propagate_chain(self, vectors: list[ConfidenceVector]) -> list[PlannerHint]:
        """Propagate hints for an entire tool chain."""
        return [self.planner_hint(cv) for cv in vectors]


# ---------------------------------------------------------------------------
# 4.3 — Confidence Fusion
# ---------------------------------------------------------------------------


class FusionStrategy(str, Enum):
    CONSERVATIVE = "conservative"  # min per dimension (pessimistic)
    OPTIMISTIC = "optimistic"  # max per dimension
    BAYESIAN = "bayesian"  # weighted harmonic mean (default)


@dataclass(frozen=True)
class FusedConfidenceVector:
    """A single ConfidenceVector representing the fusion of multiple inputs."""

    reliability_score: float
    freshness_score: float
    completeness_score: float
    strategy: FusionStrategy
    source_count: int
    source_tools: list[str]

    @property
    def aggregate(self) -> float:
        w = (0.45, 0.30, 0.25)
        vals = (self.reliability_score, self.freshness_score, self.completeness_score)
        eps = 1e-9
        denom = sum(wi / max(vi, eps) for wi, vi in zip(w, vals))
        return round(sum(w) / denom, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reliability_score": round(self.reliability_score, 4),
            "freshness_score": round(self.freshness_score, 4),
            "completeness_score": round(self.completeness_score, 4),
            "aggregate": self.aggregate,
            "strategy": self.strategy.value,
            "source_count": self.source_count,
            "source_tools": self.source_tools,
        }


class ConfidenceFusion:
    """Fuses multiple ConfidenceVectors into a single aggregate.

    Use ``fuse()`` for a named strategy, or the shortcut methods
    ``conservative()``, ``optimistic()``, ``bayesian()``.
    """

    @staticmethod
    def fuse(
        vectors: list[ConfidenceVector],
        strategy: FusionStrategy = FusionStrategy.BAYESIAN,
    ) -> FusedConfidenceVector | None:
        if not vectors:
            return None

        tools = [cv.source_tool for cv in vectors if cv.source_tool]
        n = len(vectors)

        if strategy == FusionStrategy.CONSERVATIVE:
            r = min(cv.reliability_score for cv in vectors)
            f = min(cv.freshness_score for cv in vectors)
            c = min(cv.completeness_score for cv in vectors)

        elif strategy == FusionStrategy.OPTIMISTIC:
            r = max(cv.reliability_score for cv in vectors)
            f = max(cv.freshness_score for cv in vectors)
            c = max(cv.completeness_score for cv in vectors)

        else:  # BAYESIAN: weighted harmonic mean over equal weights
            eps = 1e-9

            def harmonic_mean(vals: list[float]) -> float:
                if not vals:
                    return 0.0
                denom = sum(1.0 / max(v, eps) for v in vals)
                return len(vals) / denom

            r = harmonic_mean([cv.reliability_score for cv in vectors])
            f = harmonic_mean([cv.freshness_score for cv in vectors])
            c = harmonic_mean([cv.completeness_score for cv in vectors])

        return FusedConfidenceVector(
            reliability_score=round(max(0.0, min(1.0, r)), 4),
            freshness_score=round(max(0.0, min(1.0, f)), 4),
            completeness_score=round(max(0.0, min(1.0, c)), 4),
            strategy=strategy,
            source_count=n,
            source_tools=tools,
        )

    @classmethod
    def conservative(
        cls,
        vectors: list[ConfidenceVector],
    ) -> FusedConfidenceVector | None:
        return cls.fuse(vectors, FusionStrategy.CONSERVATIVE)

    @classmethod
    def optimistic(
        cls,
        vectors: list[ConfidenceVector],
    ) -> FusedConfidenceVector | None:
        return cls.fuse(vectors, FusionStrategy.OPTIMISTIC)

    @classmethod
    def bayesian(
        cls,
        vectors: list[ConfidenceVector],
    ) -> FusedConfidenceVector | None:
        return cls.fuse(vectors, FusionStrategy.BAYESIAN)
