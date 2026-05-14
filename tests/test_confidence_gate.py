"""
Test suite for the confidence/robustness gate in tool execution.
Verifies that invalid/low-confidence tool intents are rejected before execution.
"""
from dadbot.services.turn_service import TurnService


class TestConfidenceGateValidation:
    """Test the _validate_tool_intent_robustness() method (unit-level, no fixtures)."""

    def test_semantic_keywords_extraction(self):
        """Test keyword extraction filters stopwords correctly."""
        # Create a minimal TurnService instance
        # (We only need the helper methods, not full initialization)
        ts = TurnService.__new__(TurnService)
        
        keywords = ts._extract_intent_keywords(
            "Why did the dad go to the bank to get money out?"
        )
        # Should include content words, exclude stopwords
        assert "dad" in keywords
        assert "bank" in keywords
        assert "money" in keywords
        # Should exclude stopwords
        assert "the" not in keywords
        assert "did" not in keywords
        assert "to" not in keywords

    def test_semantic_overlap_computation(self):
        """Test Jaccard similarity computation."""
        ts = TurnService.__new__(TurnService)
        
        set_a = {"reminder", "call", "mom"}
        set_b = {"reminder", "phone", "parent"}
        
        overlap = ts._compute_semantic_overlap(set_a, set_b)
        # Intersection: {"reminder"}, Union: {"reminder", "call", "mom", "phone", "parent"}
        # Jaccard = 1/5 = 0.2
        assert 0.15 < overlap < 0.25

    def test_semantic_mismatch_detected(self):
        """Verify semantic mismatch detection works (low overlap = rejection)."""
        ts = TurnService.__new__(TurnService)
        
        # "joke" vs "reminder" - very low semantic overlap
        joke_keywords = {"dad", "bank"}  # From the joke
        reminder_keywords = {"reminder", "mom"}  # From reminder intent
        
        overlap = ts._compute_semantic_overlap(joke_keywords, reminder_keywords)
        # Should be close to 0 (no shared keywords)
        assert overlap < 0.35  # Threshold in the gate

    def test_reminder_param_validation_negative_time(self):
        """Test rejection of negative reminder time."""
        ts = TurnService.__new__(TurnService)
        
        is_valid, reason = ts._validate_reminder_params(
            {"minutes_from_now": -5},
            "Remind me"
        )
        assert is_valid is False
        assert "past" in reason.lower() or "invalid" in reason.lower()

    def test_reminder_param_validation_valid_time(self):
        """Test acceptance of valid reminder time."""
        ts = TurnService.__new__(TurnService)
        
        is_valid, reason = ts._validate_reminder_params(
            {"minutes_from_now": 30},
            "Remind me in 30 minutes"
        )
        assert is_valid is True
        assert "valid" in reason.lower()

    def test_search_param_validation_short_query(self):
        """Test rejection of short search query."""
        ts = TurnService.__new__(TurnService)
        
        is_valid, reason = ts._validate_search_params(
            {"query": "ab"},  # Too short
            "Search"
        )
        assert is_valid is False
        assert "short" in reason.lower()

    def test_search_param_validation_valid_query(self):
        """Test acceptance of valid search query."""
        ts = TurnService.__new__(TurnService)
        
        is_valid, reason = ts._validate_search_params(
            {"query": "Python documentation"},
            "Search for Python"
        )
        assert is_valid is True
        assert "valid" in reason.lower()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
