"""Tests for memory influence tracking and output coherence metrics."""

from __future__ import annotations

import pytest

from dadbot.core.coherence_metrics import OutputCoherenceTracker, ToneVector
from dadbot.core.memory_influence import MemoryInfluenceTracker


class TestToneVector:
    """Tests for ToneVector — tone fingerprinting."""

    def test_tone_vector_computes_scores(self):
        """ToneVector should compute voice, valence, and length tier."""
        reply = "Ha! That's great!"
        tone = ToneVector(reply)
        assert 0.0 <= tone.voice_score <= 1.0
        assert -1.0 <= tone.valence_score <= 1.0
        assert tone.length_tier in {0, 1, 2}

    def test_identical_replies_identical_scores(self):
        """Same reply should produce same scores."""
        reply = "This is a test reply."
        t1 = ToneVector(reply)
        t2 = ToneVector(reply)
        assert t1.voice_score == t2.voice_score
        assert t1.valence_score == t2.valence_score
        assert t1.length_tier == t2.length_tier

    def test_similarity_self_is_one(self):
        """Comparing a tone to itself should yield similarity 1.0."""
        reply = "Test reply with tone."
        t1 = ToneVector(reply)
        t2 = ToneVector(reply)
        assert t1.similarity_to(t2) == pytest.approx(1.0)

    def test_similarity_is_normalized(self):
        """Similarity should always be in [0.0, 1.0]."""
        replies = [
            "Ha! LOL that's funny!",
            "This is very formal.",
            "I'm sad and upset.",
        ]
        for r1_text in replies:
            for r2_text in replies:
                t1 = ToneVector(r1_text)
                t2 = ToneVector(r2_text)
                sim = t1.similarity_to(t2)
                assert 0.0 <= sim <= 1.0


class TestOutputCoherenceTracker:
    """Tests for OutputCoherenceTracker — multi-turn coherence."""

    def test_empty_tracker_full_coherence(self):
        """Empty tracker should report perfect coherence."""
        tracker = OutputCoherenceTracker()
        assert tracker.compute_window_coherence() == pytest.approx(1.0)

    def test_single_reply_full_coherence(self):
        """Single reply should report perfect coherence."""
        tracker = OutputCoherenceTracker()
        tracker.record_reply("Test reply.")
        assert tracker.compute_window_coherence() == pytest.approx(1.0)

    def test_coherence_in_valid_range(self):
        """Coherence should always be [0, 1]."""
        tracker = OutputCoherenceTracker(window_size=3)
        for i in range(5):
            tracker.record_reply(f"Reply number {i} with some tone.")
            assert 0.0 <= tracker.compute_window_coherence() <= 1.0

    def test_detect_drift_returns_valid_dict(self):
        """detect_personality_drift should return structured result."""
        tracker = OutputCoherenceTracker()
        tracker.record_reply("First reply.")
        tracker.record_reply("Second reply.")
        result = tracker.detect_personality_drift(threshold=0.5)
        assert "drifted" in result
        assert "coherence" in result
        assert "threshold" in result
        assert isinstance(result["drifted"], bool)
        assert 0.0 <= result["coherence"] <= 1.0

    def test_summarize_tone_profile_complete(self):
        """summarize_tone_profile should return all required keys."""
        tracker = OutputCoherenceTracker()
        tracker.record_reply("Test reply.")
        profile = tracker.summarize_tone_profile()
        required_keys = {
            "avg_voice_score",
            "avg_valence_score",
            "dominant_length_tier",
            "voice_profile",
            "emotional_profile",
        }
        assert required_keys <= set(profile.keys())


class TestMemoryInfluenceTracker:
    """Tests for MemoryInfluenceTracker — memory-to-response influence."""

    def test_extract_memories_returns_list(self):
        """extract_memories_from_context should return list."""
        context_state = {
            "memories": [{"id": "m1", "summary": "Test memory.", "category": "general"}]
        }
        tracker = MemoryInfluenceTracker()
        result = tracker.extract_memories_from_context(context_state)
        assert isinstance(result, list)

    def test_score_memory_influence_returns_dict(self):
        """score_memory_influence should return {id: score} dict."""
        reply = "Test reply with content."
        memories = [{"id": "m1", "summary": "Test memory."}]
        tracker = MemoryInfluenceTracker()
        scores = tracker.score_memory_influence(reply, memories)
        assert isinstance(scores, dict)
        assert "m1" in scores
        assert 0.0 <= scores["m1"] <= 1.0

    def test_verbatim_memory_scores_high(self):
        """Memories with verbatim phrases should score >= 0.8."""
        reply = "Tony likes pizza very much."
        memories = [{"id": "m1", "summary": "Tony likes pizza."}]
        tracker = MemoryInfluenceTracker()
        scores = tracker.score_memory_influence(reply, memories)
        # "Tony likes pizza" appears verbatim in reply
        assert scores["m1"] >= 0.8

    def test_unrelated_memory_scores_low(self):
        """Unrelated memories should score ~0.0."""
        reply = "The weather is sunny today."
        memories = [{"id": "m1", "summary": "Tony likes pizza and pasta."}]
        tracker = MemoryInfluenceTracker()
        scores = tracker.score_memory_influence(reply, memories)
        assert scores["m1"] == 0.0

    def test_log_influence_feedback_complete(self):
        """log_influence_feedback should return complete summary."""
        memories = [{"id": "m1", "summary": "Memory 1"}, {"id": "m2", "summary": "Memory 2"}]
        scores = {"m1": 0.8, "m2": 0.1}
        tracker = MemoryInfluenceTracker()
        feedback = tracker.log_influence_feedback(
            turn_id="t1",
            memory_entries=memories,
            influence_scores=scores,
            reply_text="Test reply",
        )
        assert feedback["turn_id"] == "t1"
        assert "high_influence_count" in feedback
        assert "unused_count" in feedback
        assert "avg_influence_score" in feedback
        assert 0.0 <= feedback["avg_influence_score"] <= 1.0
