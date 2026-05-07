"""
Composite Friction Scoring: Multi-signal friction detection for goal re-synthesis triggers.

This module computes a unified friction score from multiple behavioral signals
without relying on simple consecutive halt counts.

Signals incorporated:
- Halt streak and recovery effectiveness
- Topic drift frequency and recurrence
- Session duration exhaustion
- Context switching rate
- Pattern persistence and confidence

Design Principle:
- Centralized evidence accumulation (no scattered heuristics)
- Statistical weighting, not magic numbers
- Support for adaptive thresholds based on user baseline
- Clear separation of friction detection from intervention logic
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class FrictionSignals:
    """Aggregated friction indicators from a session."""
    halt_streak: int = 0  # Current consecutive halt count
    recovery_success_rate: float = 1.0  # % of halts that led to successful realignment
    topic_drift_frequency: float = 0.0  # Proportion of recent turns with topic drift
    recurring_topic_patterns: int = 0  # Number of distinct repeated drift topics
    session_turn_count: int = 0  # Total turns in session
    average_turns_per_context: float = float('inf')  # Estimate of context switching frequency
    unresolved_objectives_count: int = 0  # Incomplete goals from prior attempts
    checkpoint_stability: float = 1.0  # Ratio of stable vs unstable checkpoints


@dataclass
class FrictionAnalysis:
    """
    Result of friction score computation.
    
    Fields:
        composite_score: 0.0-1.0 unified friction metric
        risk_level: "low", "moderate", "high", "critical"
        should_trigger_re_synthesis: bool indicating if score exceeds threshold
        individual_signals: Dict of normalized scores for each friction component
        confidence: How confident we are in this assessment (0.0-1.0)
        primary_friction_factor: The dominant friction signal
        recommended_intervention: Specific action (e.g., "scope_reduction", "pause_and_reset")
    """
    composite_score: float = 0.0
    risk_level: str = "low"
    should_trigger_re_synthesis: bool = False
    individual_signals: dict = field(default_factory=dict)
    confidence: float = 0.0
    primary_friction_factor: Optional[str] = None
    recommended_intervention: str = ""


class CompositeFrictionEngine:
    """
    Computes unified friction score from multiple behavioral signals.
    
    This engine:
    - Normalizes diverse signals to comparable scales
    - Weights signals based on statistical evidence
    - Incorporates user baseline for adaptive thresholds
    - Determines re-synthesis trigger with high confidence
    """

    # Signal weights: How much each component contributes to overall friction
    # These can be tuned based on empirical data
    SIGNAL_WEIGHTS = {
        "halt_streak": 0.25,  # Consecutive halts (25% of score)
        "recovery_failure": 0.20,  # Poor recovery effectiveness (20%)
        "topic_drift": 0.20,  # High drift frequency (20%)
        "context_exhaustion": 0.15,  # Session running long (15%)
        "pattern_recurrence": 0.10,  # Repeated drift patterns (10%)
        "checkpoint_instability": 0.10,  # Unstable state (10%)
    }

    # Thresholds for friction levels
    FRICTION_THRESHOLDS = {
        "low": 0.25,  # < 0.25: low friction
        "moderate": 0.50,  # 0.25-0.50: moderate
        "high": 0.75,  # 0.50-0.75: high
        "critical": 1.0,  # >= 0.75: critical
    }

    # Trigger threshold for goal re-synthesis
    # Only trigger if composite score AND confidence are both sufficiently high
    RESYNTHESIS_SCORE_THRESHOLD = 0.65
    RESYNTHESIS_CONFIDENCE_THRESHOLD = 0.60

    def __init__(self):
        """Initialize the friction engine."""
        self.baseline_drift_frequency = 0.15  # Expected baseline drift rate
        self.baseline_halt_streak = 1  # Normal halt occurrences
        self.session_exhaustion_turns = 50  # Turn count at which exhaustion kicks in

    def compute_friction(self, signals: FrictionSignals) -> FrictionAnalysis:
        """
        Compute composite friction score and analysis from multiple signals.
        
        Args:
            signals: FrictionSignals with raw metric values
            
        Returns:
            FrictionAnalysis with composite score and recommendations
        """
        individual_scores = {}
        evidence_count = 0

        # Signal 1: Halt Streak Score
        halt_score = self._score_halt_streak(signals.halt_streak)
        if halt_score is not None:
            individual_scores["halt_streak"] = halt_score
            evidence_count += 1

        # Signal 2: Recovery Effectiveness
        recovery_score = self._score_recovery_effectiveness(signals.recovery_success_rate)
        if recovery_score is not None:
            individual_scores["recovery_failure"] = recovery_score
            evidence_count += 1

        # Signal 3: Topic Drift Frequency
        drift_score = self._score_topic_drift_frequency(signals.topic_drift_frequency)
        if drift_score is not None:
            individual_scores["topic_drift"] = drift_score
            evidence_count += 1

        # Signal 4: Session Exhaustion
        exhaustion_score = self._score_session_exhaustion(signals.session_turn_count)
        if exhaustion_score is not None:
            individual_scores["context_exhaustion"] = exhaustion_score
            evidence_count += 1

        # Signal 5: Pattern Recurrence
        pattern_score = self._score_pattern_recurrence(signals.recurring_topic_patterns)
        if pattern_score is not None:
            individual_scores["pattern_recurrence"] = pattern_score
            evidence_count += 1

        # Signal 6: Checkpoint Stability
        checkpoint_score = self._score_checkpoint_stability(signals.checkpoint_stability)
        if checkpoint_score is not None:
            individual_scores["checkpoint_instability"] = checkpoint_score
            evidence_count += 1

        # Compute weighted composite score
        composite_score = self._compute_weighted_score(individual_scores)
        confidence = self._estimate_confidence(evidence_count, signals)

        # Determine risk level
        risk_level = self._classify_risk_level(composite_score)

        # Identify primary friction factor
        primary_factor = None
        if individual_scores:
            weighted_scores = {
                k: v * self.SIGNAL_WEIGHTS.get(k, 0.1)
                for k, v in individual_scores.items()
            }
            primary_factor = max(weighted_scores.items(), key=lambda x: x[1])[0] if weighted_scores else None

        # Determine if re-synthesis should be triggered
        should_trigger = (
            composite_score >= self.RESYNTHESIS_SCORE_THRESHOLD
            and confidence >= self.RESYNTHESIS_CONFIDENCE_THRESHOLD
        )

        # Generate recommendation
        recommendation = self._recommend_intervention(
            composite_score, primary_factor, signals
        )

        return FrictionAnalysis(
            composite_score=composite_score,
            risk_level=risk_level,
            should_trigger_re_synthesis=should_trigger,
            individual_signals=individual_scores,
            confidence=confidence,
            primary_friction_factor=primary_factor,
            recommended_intervention=recommendation,
        )

    # ============ Signal Scoring Methods ============

    def _score_halt_streak(self, streak: int) -> Optional[float]:
        """
        Score halt streak: 1-2 halts is normal, >3 is friction.
        
        Returns normalized score 0.0-1.0 (0=no friction, 1=maximum friction).
        """
        if streak < 1:
            return 0.0
        if streak <= 2:
            return 0.2  # 1-2 halts: minor friction
        if streak <= 4:
            return 0.6  # 3-4 halts: moderate-high friction
        return 1.0  # 5+ halts: critical friction

    def _score_recovery_effectiveness(self, success_rate: float) -> Optional[float]:
        """
        Score recovery effectiveness: high success rate is good.
        
        A user with 0% recovery rate (all halts remain unresolved) has high friction.
        A user with 100% recovery rate has low friction.
        """
        if success_rate >= 0.8:
            return 0.0  # 80%+ success: no friction
        if success_rate >= 0.6:
            return 0.3  # 60-80%: low friction
        if success_rate >= 0.4:
            return 0.7  # 40-60%: high friction
        return 1.0  # <40%: critical friction

    def _score_topic_drift_frequency(self, drift_freq: float) -> Optional[float]:
        """
        Score topic drift frequency against baseline.
        
        Args:
            drift_freq: Proportion of recent turns with topic drift (0.0-1.0)
        """
        # Compare to baseline expectation
        excess_drift = max(0.0, drift_freq - self.baseline_drift_frequency)
        if excess_drift < 0.05:
            return 0.0  # Close to baseline: no friction
        if excess_drift < 0.15:
            return 0.3  # Slightly elevated: low friction
        if excess_drift < 0.30:
            return 0.7  # Elevated: high friction
        return 1.0  # Very high: critical friction

    def _score_session_exhaustion(self, turn_count: int) -> Optional[float]:
        """
        Score session exhaustion: very long sessions accumulate fatigue.
        
        Assumes exhaustion increases with session length, but is soft
        (a 50-turn session is not automatically exhausted).
        """
        if turn_count < self.session_exhaustion_turns:
            return 0.0  # Not exhausted yet
        if turn_count < self.session_exhaustion_turns * 1.5:
            return 0.4  # Mild exhaustion
        if turn_count < self.session_exhaustion_turns * 2:
            return 0.7  # Significant exhaustion
        return 1.0  # Severe exhaustion

    def _score_pattern_recurrence(self, pattern_count: int) -> Optional[float]:
        """
        Score recurring drift patterns.
        
        Multiple distinct patterns suggest environmental triggers,
        single pattern suggests habitual response.
        """
        if pattern_count == 0:
            return 0.0  # No patterns: no friction
        if pattern_count == 1:
            return 0.3  # One pattern: somewhat predictable
        if pattern_count == 2:
            return 0.6  # Two patterns: fragmented friction
        return 1.0  # 3+ patterns: complex friction environment

    def _score_checkpoint_stability(self, stability: float) -> Optional[float]:
        """
        Score checkpoint stability: unstable state = higher friction.
        
        Args:
            stability: Ratio of stable to total checkpoints (0.0-1.0)
        """
        if stability >= 0.9:
            return 0.0  # Stable: no friction
        if stability >= 0.7:
            return 0.2  # Mostly stable: low friction
        if stability >= 0.5:
            return 0.6  # Often unstable: high friction
        return 1.0  # Rarely stable: critical friction

    # ============ Composition & Analysis ============

    def _compute_weighted_score(self, signals: dict[str, float]) -> float:
        """
        Compute weighted composite score from individual signals.
        
        Returns:
            Composite score 0.0-1.0
        """
        if not signals:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for signal_name, score in signals.items():
            weight = self.SIGNAL_WEIGHTS.get(signal_name, 0.1)
            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def _estimate_confidence(self, evidence_count: int, signals: FrictionSignals) -> float:
        """
        Estimate confidence in the friction assessment.
        
        Higher with more signals and when signals are consistent.
        """
        # Base confidence on number of evidence signals (max 6)
        base_confidence = min(evidence_count / 6.0, 1.0)

        # Boost confidence if we have strong signals like halt streak
        if signals.halt_streak >= 3:
            base_confidence = min(base_confidence + 0.15, 1.0)

        # Reduce confidence if signals are sparse
        if evidence_count < 3:
            base_confidence *= 0.7

        return base_confidence

    def _classify_risk_level(self, composite_score: float) -> str:
        """Classify risk level based on composite score."""
        if composite_score < 0.25:
            return "low"
        elif composite_score < 0.50:
            return "moderate"
        elif composite_score < 0.75:
            return "high"
        else:
            return "critical"

    def _recommend_intervention(
        self,
        composite_score: float,
        primary_factor: Optional[str],
        signals: FrictionSignals,
    ) -> str:
        """
        Generate specific intervention recommendation.
        
        Returns a concrete suggestion for goal re-synthesis.
        """
        if composite_score < 0.25:
            return "No intervention needed. Continue current trajectory."

        if composite_score < 0.50:
            return "Monitor for friction. Proactive goal review recommended if patterns persist."

        # Moderate to critical: recommend specific interventions
        recommendations = []

        if primary_factor == "halt_streak" and signals.halt_streak >= 4:
            recommendations.append("Consider breaking goal into smaller, time-boxed milestones.")

        if primary_factor == "recovery_failure" and signals.recovery_success_rate < 0.5:
            recommendations.append("Previous interventions ineffective. Explore alternative work environments or times.")

        if primary_factor == "topic_drift" and signals.topic_drift_frequency > 0.4:
            recommendations.append("Drift patterns indicate distraction triggers. Proactive breaks or context switches recommended.")

        if primary_factor == "context_exhaustion" and signals.session_turn_count > 60:
            recommendations.append("Session fatigue evident. Consider breaking goal into multi-session checkpoints.")

        if primary_factor == "pattern_recurrence" and signals.recurring_topic_patterns >= 2:
            recommendations.append("Multiple drift patterns detected. Restructure goal to address friction triggers directly.")

        if not recommendations:
            recommendations.append("Re-synthesis review recommended due to accumulated friction.")

        return " | ".join(recommendations)
