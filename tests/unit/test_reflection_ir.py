"""
Tests for dadbot/core/reflection_ir.py

Validates:
- DriftEpisode and BehavioralPattern data models
- DriftReflectionEngine pattern detection
- Temporal and topic clustering
- Recovery effectiveness tracking
- ReflectionSummary synthesis
"""

import pytest
import tempfile
import json
from datetime import datetime
from pathlib import Path

from dadbot.core.reflection_ir import (
    DriftEpisode,
    BehavioralPattern,
    ReflectionSummary,
    DriftReflectionEngine,
)


@pytest.mark.unit
class TestDriftEpisodeModel:
    """Validate DriftEpisode construction and properties."""

    def test_drift_episode_creation_resolved(self):
        """Test creating a resolved drift episode."""
        episode = DriftEpisode(
            start_turn=5,
            end_turn=8,
            duration=3,
            trigger_topic="fishing",
            trigger_time_bucket="11_PM",
            emotional_signature="fatigue",
            recovery_method="break",
            halt_count=2,
            resumed_successfully=True,
            recovery_time_minutes=15,
            burnout_probability=0.7
        )
        assert episode.start_turn == 5
        assert episode.end_turn == 8
        assert episode.resumed_successfully is True
        assert episode.burnout_probability == 0.7

    def test_drift_episode_creation_ongoing(self):
        """Test creating an ongoing (unresolved) drift episode."""
        episode = DriftEpisode(
            start_turn=20,
            end_turn=None,
            duration=5,
            trigger_topic="gaming",
            trigger_time_bucket=None,
            emotional_signature="distraction",
            recovery_method=None,
            halt_count=1,
            resumed_successfully=False,
        )
        assert episode.end_turn is None
        assert episode.resumed_successfully is False


@pytest.mark.unit
class TestBehavioralPatternModel:
    """Validate BehavioralPattern construction and properties."""

    def test_behavioral_pattern_creation(self):
        """Test creating a behavioral pattern."""
        pattern = BehavioralPattern(
            pattern_id="late_night_fatigue",
            pattern_name="Late-night burnout",
            frequency=3,
            confidence=0.75,
            correlated_times=["11_PM", "12_AM"],
            correlated_topics=["fishing", "doomscroll"],
            average_episode_duration=2.5,
            average_recovery_time=20.0,
            success_after_intervention=80.0,
            last_observed_turn=42,
            evidence_weight=0.6
        )
        assert pattern.pattern_id == "late_night_fatigue"
        assert pattern.frequency == 3
        assert pattern.confidence == 0.75
        assert "11_PM" in pattern.correlated_times


@pytest.mark.unit
class TestReflectionSummaryModel:
    """Validate ReflectionSummary construction."""

    def test_reflection_summary_high_risk(self):
        """Test high-risk reflection summary."""
        pattern = BehavioralPattern(
            pattern_id="recurring_avoidance",
            pattern_name="Post-task avoidance",
            frequency=5,
            confidence=0.85
        )
        summary = ReflectionSummary(
            current_risk_level="high",
            predicted_drift_probability=0.7,
            likely_trigger_category="avoidance",
            primary_pattern=pattern,
            confidence_score=0.85,
            observable_signals=["high_diversion_frequency"],
            recent_episode_count=3
        )
        assert summary.current_risk_level == "high"
        assert summary.predicted_drift_probability == 0.7
        assert summary.primary_pattern == pattern


@pytest.mark.unit
class TestDriftReflectionEngineNoLedger:
    """Test DriftReflectionEngine with no ledger file."""

    def test_engine_creation_without_ledger(self):
        """Test engine initialization without ledger."""
        engine = DriftReflectionEngine(ledger_path=None)
        summary = engine.analyze_ledger()
        
        assert summary.current_risk_level == "low"
        assert summary.predicted_drift_probability == 0.0
        assert summary.confidence_score == 0.0

    def test_engine_with_missing_ledger_file(self):
        """Test engine gracefully handles missing ledger file."""
        engine = DriftReflectionEngine(ledger_path="/nonexistent/path/ledger.jsonl")
        summary = engine.analyze_ledger()
        
        assert summary.current_risk_level == "low"
        assert summary.confidence_score == 0.1


@pytest.mark.unit
class TestDriftReflectionEngineWithLedger:
    """Test DriftReflectionEngine with sample ledger data."""

    @pytest.fixture
    def sample_ledger(self):
        """Create a temporary ledger file with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            base_time = datetime(2026, 1, 5, 23, 0, 0)

            # Turn 0-2: aligned
            for turn in range(3):
                entry = {
                    "turn_index": turn,
                    "recorded_at": base_time.replace(minute=turn).isoformat(),
                    "session_goals": ["code_feature"],
                    "relational_state": {"is_aligned": True},
                    "goal_alignment_diversion_streak": 0
                }
                f.write(json.dumps(entry) + "\n")

            # Turn 3-5: diverged (drift episode 1)
            for turn in range(3, 6):
                entry = {
                    "turn_index": turn,
                    "recorded_at": base_time.replace(minute=turn).isoformat(),
                    "session_goals": ["code_feature"],
                    "relational_state": {"is_aligned": False},
                    "goal_alignment_diversion_streak": turn - 2
                }
                f.write(json.dumps(entry) + "\n")

            # Turn 6-8: aligned again
            for turn in range(6, 9):
                entry = {
                    "turn_index": turn,
                    "recorded_at": base_time.replace(minute=turn).isoformat(),
                    "session_goals": ["code_feature"],
                    "relational_state": {"is_aligned": True},
                    "goal_alignment_diversion_streak": 0
                }
                f.write(json.dumps(entry) + "\n")

            # Turn 9-11: diverged again (drift episode 2 - same topic)
            for turn in range(9, 12):
                entry = {
                    "turn_index": turn,
                    "recorded_at": base_time.replace(minute=turn).isoformat(),
                    "session_goals": ["code_feature"],
                    "relational_state": {"is_aligned": False},
                    "goal_alignment_diversion_streak": turn - 8
                }
                f.write(json.dumps(entry) + "\n")

            ledger_path = f.name

        yield ledger_path

        # Cleanup
        Path(ledger_path).unlink(missing_ok=True)

    def test_engine_extracts_drift_episodes(self, sample_ledger):
        """Test that engine correctly identifies drift episodes."""
        engine = DriftReflectionEngine(ledger_path=sample_ledger)
        engine._load_and_parse_ledger()
        engine._extract_drift_episodes()

        assert len(engine.drift_episodes) >= 2
        # First episode: turns 3-5 (inclusive)
        assert engine.drift_episodes[0].start_turn == 3
        assert engine.drift_episodes[0].end_turn == 5
        assert engine.drift_episodes[0].duration == 3
        assert engine.drift_episodes[0].resumed_successfully is True

    def test_engine_detects_topic_patterns(self, sample_ledger):
        """Test that engine detects recurring topics."""
        engine = DriftReflectionEngine(ledger_path=sample_ledger)
        engine._load_and_parse_ledger()
        engine._extract_drift_episodes()
        engine._detect_topic_patterns()

        # Both episodes have same topic ("code_feature" divergence)
        # Should create pattern
        assert len(engine.behavioral_patterns) >= 1

    def test_engine_computes_recovery_effectiveness(self, sample_ledger):
        """Test that engine computes recovery success rates."""
        engine = DriftReflectionEngine(ledger_path=sample_ledger)
        engine._load_and_parse_ledger()
        engine._extract_drift_episodes()
        engine._detect_topic_patterns()
        engine._compute_recovery_effectiveness()

        for pattern in engine.behavioral_patterns.values():
            # All episodes in this test are recovered
            assert pattern.success_after_intervention >= 0.0

    def test_engine_synthesizes_reflection_summary(self, sample_ledger):
        """Test full analysis workflow and summary generation."""
        engine = DriftReflectionEngine(ledger_path=sample_ledger)
        summary = engine.analyze_ledger()

        assert isinstance(summary, ReflectionSummary)
        assert summary.current_risk_level in ["low", "moderate", "high"]
        assert 0.0 <= summary.predicted_drift_probability <= 1.0
        assert 0.0 <= summary.confidence_score <= 1.0

    def test_engine_builds_evidence_graph(self, sample_ledger):
        """Evidence graph should contain event/episode/pattern/outcome links."""
        engine = DriftReflectionEngine(ledger_path=sample_ledger)
        engine.analyze_ledger()
        summary = engine.get_pattern_summary()

        assert summary["evidence_edges"] > 0

    def test_engine_uses_timestamp_time_bucket(self, sample_ledger):
        """Temporal bucket should come from parsed recorded_at timestamp."""
        engine = DriftReflectionEngine(ledger_path=sample_ledger)
        engine._load_and_parse_ledger()
        bucket = engine._infer_time_bucket(3)

        assert bucket == "Mon_h23"

    def test_summary_with_no_drift_episodes(self):
        """Test summary when no drift episodes are detected."""
        # Create ledger with all aligned turns
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for turn in range(10):
                entry = {
                    "turn_index": turn,
                    "session_goals": ["task"],
                    "relational_state": {"is_aligned": True},
                    "goal_alignment_diversion_streak": 0
                }
                f.write(json.dumps(entry) + "\n")
            ledger_path = f.name

        try:
            engine = DriftReflectionEngine(ledger_path=ledger_path)
            summary = engine.analyze_ledger()

            assert summary.current_risk_level == "low"
            assert summary.predicted_drift_probability <= 0.15
            assert "no_drift_episodes_observed" in summary.observable_signals
        finally:
            Path(ledger_path).unlink(missing_ok=True)


@pytest.mark.unit
class TestConfidenceScoring:
    """Test confidence computation and pattern ranking."""

    def test_confidence_increases_with_frequency(self):
        """Test that confidence increases as frequency increases."""
        engine = DriftReflectionEngine()

        conf1 = engine._compute_confidence(1, 10)
        conf2 = engine._compute_confidence(2, 10)
        conf5 = engine._compute_confidence(5, 10)

        assert conf1 < conf2 < conf5

    def test_confidence_capped_at_one(self):
        """Test that confidence never exceeds 1.0."""
        engine = DriftReflectionEngine()
        conf = engine._compute_confidence(100, 10)
        assert conf <= 1.0

    def test_confidence_zero_for_no_episodes(self):
        """Test that confidence is zero when total is zero."""
        engine = DriftReflectionEngine()
        conf = engine._compute_confidence(5, 0)
        assert conf == 0.0


@pytest.mark.unit
class TestPatternMatchingLogic:
    """Test pattern matching and episode classification."""

    def test_episode_matches_topic_pattern(self):
        """Test episode matching against topic pattern."""
        engine = DriftReflectionEngine()
        episode = DriftEpisode(
            start_turn=5,
            end_turn=8,
            duration=3,
            trigger_topic="fishing",
            trigger_time_bucket=None,
            emotional_signature=None,
            recovery_method=None,
            halt_count=0,
            resumed_successfully=True
        )
        pattern = BehavioralPattern(
            pattern_id="fishing_distraction",
            pattern_name="Fishing rabbit hole",
            correlated_topics=["fishing", "gaming"]
        )

        assert engine._episode_matches_pattern(episode, pattern) is True

    def test_episode_does_not_match_mismatched_topic(self):
        """Test episode doesn't match different topic."""
        engine = DriftReflectionEngine()
        episode = DriftEpisode(
            start_turn=5,
            end_turn=8,
            duration=3,
            trigger_topic="coding",
            trigger_time_bucket=None,
            emotional_signature=None,
            recovery_method=None,
            halt_count=0,
            resumed_successfully=True
        )
        pattern = BehavioralPattern(
            pattern_id="gaming_distraction",
            pattern_name="Gaming rabbit hole",
            correlated_topics=["gaming", "streaming"]
        )

        assert engine._episode_matches_pattern(episode, pattern) is False


@pytest.mark.unit
class TestTriggerCategoryInference:
    """Test trigger category detection."""

    def test_infer_fatigue_category(self):
        """Test detection of fatigue-related patterns."""
        engine = DriftReflectionEngine()
        pattern = BehavioralPattern(
            pattern_id="late_night",
            pattern_name="Late-night drift after 11 PM"
        )
        category = engine._infer_trigger_category(pattern, [])
        assert category == "fatigue"

    def test_infer_avoidance_category(self):
        """Test detection of avoidance-related patterns."""
        engine = DriftReflectionEngine()
        pattern = BehavioralPattern(
            pattern_id="post_task",
            pattern_name="Avoidance after difficult task"
        )
        category = engine._infer_trigger_category(pattern, [])
        assert category == "avoidance"

    def test_infer_distraction_category(self):
        """Test detection of distraction-related patterns."""
        engine = DriftReflectionEngine()
        pattern = BehavioralPattern(
            pattern_id="rabbit_hole",
            pattern_name="Rabbit hole distraction"
        )
        category = engine._infer_trigger_category(pattern, [])
        assert category == "distraction"


@pytest.mark.unit
class TestPatternSummary:
    """Test pattern summary export."""

    def test_get_pattern_summary(self):
        """Test pattern summary generation."""
        engine = DriftReflectionEngine()
        
        # Add some patterns
        pattern1 = BehavioralPattern(
            pattern_id="pattern1",
            pattern_name="Pattern 1",
            frequency=3,
            confidence=0.8
        )
        engine.behavioral_patterns["pattern1"] = pattern1

        summary = engine.get_pattern_summary()
        assert isinstance(summary, dict)
        assert "total_episodes" in summary
        assert "patterns_detected" in summary
        assert "patterns" in summary
        assert len(summary["patterns"]) == 1
        assert summary["patterns"][0]["id"] == "pattern1"
