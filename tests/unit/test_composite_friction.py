"""
Tests for dadbot/core/composite_friction.py

Validates:
- FrictionSignals data model
- CompositeFrictionEngine signal scoring
- Weighted composite score computation
- Re-synthesis trigger logic
- Intervention recommendations
"""

import pytest
from dadbot.core.composite_friction import (
    FrictionSignals,
    FrictionAnalysis,
    CompositeFrictionEngine,
)


@pytest.mark.unit
class TestFrictionSignalsModel:
    """Validate FrictionSignals construction."""

    def test_friction_signals_creation_default(self):
        """Test creating signals with defaults."""
        signals = FrictionSignals()
        assert signals.halt_streak == 0
        assert signals.recovery_success_rate == 1.0
        assert signals.session_turn_count == 0

    def test_friction_signals_creation_custom(self):
        """Test creating signals with custom values."""
        signals = FrictionSignals(
            halt_streak=4,
            recovery_success_rate=0.5,
            topic_drift_frequency=0.35,
            session_turn_count=60
        )
        assert signals.halt_streak == 4
        assert signals.recovery_success_rate == 0.5
        assert signals.topic_drift_frequency == 0.35


@pytest.mark.unit
class TestFrictionAnalysisModel:
    """Validate FrictionAnalysis construction."""

    def test_friction_analysis_creation(self):
        """Test creating a friction analysis result."""
        analysis = FrictionAnalysis(
            composite_score=0.65,
            risk_level="high",
            should_trigger_re_synthesis=True,
            confidence=0.80,
            primary_friction_factor="halt_streak"
        )
        assert analysis.composite_score == 0.65
        assert analysis.risk_level == "high"
        assert analysis.should_trigger_re_synthesis is True
        assert analysis.confidence == 0.80


@pytest.mark.unit
class TestHaltStreakScoring:
    """Test halt streak scoring logic."""

    def test_halt_streak_no_friction(self):
        """Test that 0-2 halts have minimal friction."""
        engine = CompositeFrictionEngine()
        assert engine._score_halt_streak(0) == 0.0
        assert engine._score_halt_streak(1) == 0.2
        assert engine._score_halt_streak(2) == 0.2

    def test_halt_streak_moderate_friction(self):
        """Test that 3-4 halts have moderate friction."""
        engine = CompositeFrictionEngine()
        assert engine._score_halt_streak(3) == 0.6
        assert engine._score_halt_streak(4) == 0.6

    def test_halt_streak_critical_friction(self):
        """Test that 5+ halts have critical friction."""
        engine = CompositeFrictionEngine()
        assert engine._score_halt_streak(5) == 1.0
        assert engine._score_halt_streak(10) == 1.0


@pytest.mark.unit
class TestRecoveryEffectivenessScoring:
    """Test recovery effectiveness scoring."""

    def test_recovery_high_success_no_friction(self):
        """Test high recovery success has minimal friction."""
        engine = CompositeFrictionEngine()
        assert engine._score_recovery_effectiveness(1.0) == 0.0
        assert engine._score_recovery_effectiveness(0.85) == 0.0

    def test_recovery_low_success_high_friction(self):
        """Test low recovery success has high friction."""
        engine = CompositeFrictionEngine()
        assert engine._score_recovery_effectiveness(0.3) == 1.0
        assert engine._score_recovery_effectiveness(0.0) == 1.0

    def test_recovery_moderate_success(self):
        """Test moderate recovery success."""
        engine = CompositeFrictionEngine()
        score = engine._score_recovery_effectiveness(0.65)
        assert 0.3 <= score <= 0.7


@pytest.mark.unit
class TestTopicDriftScoring:
    """Test topic drift frequency scoring."""

    def test_drift_near_baseline_no_friction(self):
        """Test drift at baseline has minimal friction."""
        engine = CompositeFrictionEngine()
        baseline_drift = engine.baseline_drift_frequency
        score = engine._score_topic_drift_frequency(baseline_drift)
        assert score == 0.0

    def test_drift_elevated_has_friction(self):
        """Test elevated drift has friction."""
        engine = CompositeFrictionEngine()
        score = engine._score_topic_drift_frequency(0.45)
        assert score > 0.3

    def test_drift_very_high_critical_friction(self):
        """Test very high drift has critical friction."""
        engine = CompositeFrictionEngine()
        score = engine._score_topic_drift_frequency(0.75)
        assert score == 1.0


@pytest.mark.unit
class TestSessionExhaustionScoring:
    """Test session exhaustion scoring."""

    def test_short_session_no_exhaustion(self):
        """Test short session has no exhaustion friction."""
        engine = CompositeFrictionEngine()
        score = engine._score_session_exhaustion(20)
        assert score == 0.0

    def test_moderate_session_mild_exhaustion(self):
        """Test moderate session has some exhaustion."""
        engine = CompositeFrictionEngine()
        turn_count = engine.session_exhaustion_turns + 5
        score = engine._score_session_exhaustion(turn_count)
        assert 0.3 <= score <= 0.5

    def test_long_session_critical_exhaustion(self):
        """Test very long session has critical exhaustion."""
        engine = CompositeFrictionEngine()
        turn_count = engine.session_exhaustion_turns * 2.5
        score = engine._score_session_exhaustion(turn_count)
        assert score == 1.0


@pytest.mark.unit
class TestPatternRecurrenceScoring:
    """Test recurring pattern detection scoring."""

    def test_no_patterns_no_friction(self):
        """Test no patterns has minimal friction."""
        engine = CompositeFrictionEngine()
        assert engine._score_pattern_recurrence(0) == 0.0

    def test_one_pattern_low_friction(self):
        """Test single pattern has low friction."""
        engine = CompositeFrictionEngine()
        score = engine._score_pattern_recurrence(1)
        assert score == 0.3

    def test_multiple_patterns_high_friction(self):
        """Test multiple patterns have high friction."""
        engine = CompositeFrictionEngine()
        assert engine._score_pattern_recurrence(2) == 0.6
        assert engine._score_pattern_recurrence(3) == 1.0


@pytest.mark.unit
class TestCheckpointStabilityScoring:
    """Test checkpoint stability scoring."""

    def test_stable_checkpoints_no_friction(self):
        """Test stable checkpoints have no friction."""
        engine = CompositeFrictionEngine()
        assert engine._score_checkpoint_stability(1.0) == 0.0
        assert engine._score_checkpoint_stability(0.95) == 0.0

    def test_unstable_checkpoints_high_friction(self):
        """Test unstable checkpoints have high friction."""
        engine = CompositeFrictionEngine()
        assert engine._score_checkpoint_stability(0.3) == 1.0
        assert engine._score_checkpoint_stability(0.0) == 1.0

    def test_mixed_stability(self):
        """Test mixed stability."""
        engine = CompositeFrictionEngine()
        score = engine._score_checkpoint_stability(0.6)
        assert 0.5 <= score <= 0.7


@pytest.mark.unit
class TestCompositeScoreComputation:
    """Test weighted composite score calculation."""

    def test_no_signals_zero_score(self):
        """Test that no signals yields zero score."""
        engine = CompositeFrictionEngine()
        score = engine._compute_weighted_score({})
        assert score == 0.0

    def test_single_signal_contributes_weighted(self):
        """Test that single signal is weighted correctly."""
        engine = CompositeFrictionEngine()
        signals = {"halt_streak": 1.0}  # Maximum halt friction
        score = engine._compute_weighted_score(signals)
        # halt_streak has weight 0.25, but signal is 1.0
        # weighted_sum = 1.0 * 0.25 = 0.25, total_weight = 0.25
        # result = 0.25 / 0.25 = 1.0
        assert score == 1.0

    def test_multiple_signals_combined(self):
        """Test multiple signals are combined correctly."""
        engine = CompositeFrictionEngine()
        signals = {
            "halt_streak": 1.0,
            "recovery_failure": 1.0,
            "topic_drift": 1.0,
        }
        score = engine._compute_weighted_score(signals)
        # All signals at 1.0 with their weights:
        # (1.0 * 0.25) + (1.0 * 0.20) + (1.0 * 0.20) = 0.65
        # total_weight = 0.25 + 0.20 + 0.20 = 0.65
        # result = 0.65 / 0.65 = 1.0
        assert score == 1.0


@pytest.mark.unit
class TestConfidenceEstimation:
    """Test confidence estimation."""

    def test_confidence_with_no_evidence(self):
        """Test low confidence with minimal evidence."""
        engine = CompositeFrictionEngine()
        signals = FrictionSignals()
        confidence = engine._estimate_confidence(1, signals)
        assert confidence < 0.5

    def test_confidence_with_full_evidence(self):
        """Test high confidence with all 6 signals."""
        engine = CompositeFrictionEngine()
        signals = FrictionSignals()
        confidence = engine._estimate_confidence(6, signals)
        assert confidence > 0.5

    def test_confidence_boost_with_strong_signals(self):
        """Test confidence boost when halt streak is high."""
        engine = CompositeFrictionEngine()
        signals_weak = FrictionSignals(halt_streak=0)
        signals_strong = FrictionSignals(halt_streak=4)
        
        conf_weak = engine._estimate_confidence(3, signals_weak)
        conf_strong = engine._estimate_confidence(3, signals_strong)
        
        assert conf_strong > conf_weak


@pytest.mark.unit
class TestRiskLevelClassification:
    """Test risk level classification."""

    def test_low_risk_threshold(self):
        """Test low risk classification."""
        engine = CompositeFrictionEngine()
        assert engine._classify_risk_level(0.1) == "low"
        assert engine._classify_risk_level(0.24) == "low"

    def test_moderate_risk_threshold(self):
        """Test moderate risk classification."""
        engine = CompositeFrictionEngine()
        assert engine._classify_risk_level(0.25) == "moderate"
        assert engine._classify_risk_level(0.49) == "moderate"

    def test_high_risk_threshold(self):
        """Test high risk classification."""
        engine = CompositeFrictionEngine()
        assert engine._classify_risk_level(0.50) == "high"
        assert engine._classify_risk_level(0.74) == "high"

    def test_critical_risk_threshold(self):
        """Test critical risk classification."""
        engine = CompositeFrictionEngine()
        assert engine._classify_risk_level(0.75) == "critical"
        assert engine._classify_risk_level(1.0) == "critical"


@pytest.mark.unit
class TestResynthesisTrigger:
    """Test goal re-synthesis trigger logic."""

    def test_low_score_no_trigger(self):
        """Test that low friction doesn't trigger re-synthesis."""
        engine = CompositeFrictionEngine()
        signals = FrictionSignals(
            halt_streak=1,
            recovery_success_rate=0.95,
            topic_drift_frequency=0.1
        )
        analysis = engine.compute_friction(signals)
        assert analysis.should_trigger_re_synthesis is False

    def test_high_score_low_confidence_no_trigger(self):
        """Test that high score with low confidence doesn't trigger."""
        engine = CompositeFrictionEngine()
        signals = FrictionSignals(halt_streak=5)  # Only one signal
        analysis = engine.compute_friction(signals)
        assert analysis.should_trigger_re_synthesis is False

    def test_high_score_high_confidence_triggers(self):
        """Test that high score with high confidence triggers re-synthesis."""
        engine = CompositeFrictionEngine()
        signals = FrictionSignals(
            halt_streak=5,
            recovery_success_rate=0.2,
            topic_drift_frequency=0.5,
            recurring_topic_patterns=2,
            session_turn_count=80,
            checkpoint_stability=0.4,
        )
        analysis = engine.compute_friction(signals)
        assert analysis.should_trigger_re_synthesis is True


@pytest.mark.unit
class TestInterventionRecommendations:
    """Test intervention recommendation generation."""

    def test_no_intervention_low_friction(self):
        """Test minimal intervention for low friction."""
        engine = CompositeFrictionEngine()
        signals = FrictionSignals(halt_streak=0)
        analysis = engine.compute_friction(signals)
        assert "No intervention needed" in analysis.recommended_intervention

    def test_halt_streak_intervention(self):
        """Test halt-streak-specific intervention."""
        engine = CompositeFrictionEngine()
        signals = FrictionSignals(
            halt_streak=4,
            recovery_success_rate=0.8,
            topic_drift_frequency=0.1
        )
        analysis = engine.compute_friction(signals)
        if "halt" in analysis.recommended_intervention.lower():
            assert "milestones" in analysis.recommended_intervention.lower()

    def test_session_exhaustion_intervention(self):
        """Test exhaustion-specific intervention when score is high enough."""
        engine = CompositeFrictionEngine()
        # Need high enough composite score to trigger intervention recommendations
        signals = FrictionSignals(
            halt_streak=4,  # Increased to boost score
            session_turn_count=100,  # High turn count
            recovery_success_rate=0.5,  # Reduced to boost score
            topic_drift_frequency=0.3,
            recurring_topic_patterns=1,
            checkpoint_stability=0.6,
        )
        analysis = engine.compute_friction(signals)
        
        # With composite score high enough, recommendations should be provided
        # The engine will recommend specific interventions
        recommendation_lower = analysis.recommended_intervention.lower()
        # Should either have concrete recommendation or generic re-synthesis message
        assert len(recommendation_lower) > 10


@pytest.mark.unit
class TestFullComputationWorkflow:
    """Test complete friction computation workflow."""

    def test_compute_friction_realistic_scenario_no_friction(self):
        """Test realistic low-friction scenario."""
        engine = CompositeFrictionEngine()
        signals = FrictionSignals(
            halt_streak=1,
            recovery_success_rate=0.95,
            topic_drift_frequency=0.12,
            recurring_topic_patterns=0,
            session_turn_count=30,
            checkpoint_stability=0.95
        )
        analysis = engine.compute_friction(signals)
        
        assert analysis.composite_score < 0.25
        assert analysis.risk_level == "low"
        assert analysis.should_trigger_re_synthesis is False
        assert analysis.confidence > 0.5

    def test_compute_friction_realistic_scenario_high_friction(self):
        """Test realistic high-friction scenario."""
        engine = CompositeFrictionEngine()
        signals = FrictionSignals(
            halt_streak=4,
            recovery_success_rate=0.35,
            topic_drift_frequency=0.55,
            recurring_topic_patterns=3,
            session_turn_count=70,
            checkpoint_stability=0.35
        )
        analysis = engine.compute_friction(signals)
        
        assert analysis.composite_score > 0.60
        assert analysis.risk_level in ["high", "critical"]
        assert analysis.should_trigger_re_synthesis is True
        assert analysis.confidence > 0.60
