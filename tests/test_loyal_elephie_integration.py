"""Quick validation tests for Loyal-Elephie integrations."""

import pytest
from dadbot.memory.hybrid_retriever import HybridRetriever
from dadbot.streamlit_helpers import save_important_memory, load_important_memories, get_relationship_health_stats
from dadbot.core.memory_ranking import rank_memory_hybrid


class TestHybridRetriever:
    """Test hybrid memory retrieval (semantic + BM25)."""

    def test_hybrid_retriever_initialization(self):
        """Test HybridRetriever can initialize."""
        retriever = HybridRetriever()
        assert retriever is not None
        assert retriever.bm25_enabled is not None

    def test_tokenize(self):
        """Test text tokenization."""
        retriever = HybridRetriever()
        tokens = retriever._tokenize("Hello world this is a test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_rrf_fusion(self):
        """Test Reciprocal Rank Fusion merging."""
        retriever = HybridRetriever()
        vector_ranked = [
            (0.9, {"id": "1", "text": "family memory"}),
            (0.8, {"id": "2", "text": "goal memory"}),
        ]
        bm25_ranked = [
            (0.95, {"id": "2", "text": "goal memory"}),
            (0.7, {"id": "1", "text": "family memory"}),
        ]
        merged = retriever._reciprocal_rank_fusion(vector_ranked, bm25_ranked, k=60)
        assert len(merged) > 0
        # Item 2 should rank higher due to high scores in both lists
        assert merged[0].get("id") == "2"

    def test_hybrid_rank_with_fallback(self):
        """Test hybrid ranking with BM25 disabled (fallback to semantic)."""
        retriever = HybridRetriever()
        items = [
            {"id": "1", "text": "Dad advice", "created_at": 1000, "score": 0.8},
            {"id": "2", "text": "Family story", "created_at": 2000, "score": 0.6},
        ]
        semantic_scores = {"1": 0.8, "2": 0.6}
        
        ranked = retriever.rank_with_hybrid(
            items=items,
            user_input="family",
            semantic_scores=semantic_scores,
            limit=2,
        )
        assert len(ranked) > 0


class TestStreamlitHelpers:
    """Test Streamlit helper functions for memory and relationship visualization."""

    def test_save_important_memory_validation(self):
        """Test that save_important_memory validates input."""
        # Empty content should return False
        result = save_important_memory(content="", context="test", tags=["family"])
        assert result is False

    def test_relationship_health_stats_default(self):
        """Test relationship health stats with empty world model."""
        stats = get_relationship_health_stats(None)
        assert "trust_level" in stats
        assert "openness" in stats
        assert "recent_mood" in stats
        assert "relationship_trend" in stats
        assert stats["trust_level"] == 50  # Default
        assert stats["openness"] == 50  # Default

    def test_relationship_health_stats_from_model(self):
        """Test relationship health stats extraction."""
        world_model = {
            "trust_metric": 75,
            "openness_level": 80,
            "recent_mood": "happy",
            "mood_history": [
                {"valence": 0.5},  # Older
                {"valence": 0.7},  # Newer (improving)
            ],
        }
        stats = get_relationship_health_stats(world_model)
        assert stats["trust_level"] == 75
        assert stats["openness"] == 80
        assert stats["recent_mood"] == "happy"
        assert stats["relationship_trend"] == "improving"

    def test_format_relationship_card(self):
        """Test relationship card formatting."""
        health_stats = {
            "trust_level": 80,
            "openness": 75,
            "recent_mood": "happy",
            "relationship_trend": "improving",
        }
        card = __import__("dadbot.streamlit_helpers", fromlist=["format_relationship_card"]).format_relationship_card(health_stats)
        assert "80%" in card or "80" in card
        assert "Relationship Health" in card


class TestMemoryRankingHybrid:
    """Test hybrid ranking integration with memory ranking module."""

    def test_rank_memory_hybrid_exists(self):
        """Test that rank_memory_hybrid function exists and is callable."""
        from dadbot.core.memory_ranking import rank_memory_hybrid
        assert callable(rank_memory_hybrid)

    def test_rank_memory_hybrid_with_items(self):
        """Test hybrid ranking of memory items."""
        items = [
            {"id": "1", "text": "Remember when you helped me", "created_at": 1000, "score": 0.7},
            {"id": "2", "text": "Dad gave me advice about work", "created_at": 2000, "score": 0.6},
            {"id": "3", "text": "Family dinner story", "created_at": 1500, "score": 0.8},
        ]
        
        ranked = rank_memory_hybrid(
            items=items,
            user_input="family advice",
            limit=2,
        )
        assert len(ranked) <= 2
        assert all(isinstance(item, dict) for item in ranked)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
