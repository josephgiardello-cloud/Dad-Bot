"""Hybrid memory retrieval: semantic vector search + BM25 keyword ranking.

Combines ChromaDB vector embeddings with BM25Okapi keyword scoring for balanced recall.
Uses Reciprocal Rank Fusion (RRF) to merge ranked result sets.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None  # type: ignore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid memory retriever: vector search + keyword-based BM25 ranking.

    Maintains a dual-path retrieval strategy:
    1. Vector embeddings (semantic similarity) via ChromaDB
    2. BM25 keyword-based scoring for term overlap
    3. Reciprocal Rank Fusion combines both ranked lists

    Result: Better recall than pure semantic, better precision than pure keyword.
    """

    def __init__(self) -> None:
        """Initialize hybrid retriever. BM25 module is optional."""
        self.bm25_enabled = BM25Okapi is not None
        if not self.bm25_enabled:
            logger.warning(
                "rank_bm25 not available; hybrid retriever will fall back to semantic-only. "
                "Install with: pip install rank-bm25"
            )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text for BM25: lowercase, split on whitespace, min length 3."""
        tokens = (
            str(text or "")
            .lower()
            .replace("\n", " ")
            .replace("\t", " ")
            .split()
        )
        return [t.strip() for t in tokens if len(t.strip()) >= 3]

    @staticmethod
    def _reciprocal_rank_fusion(
        vector_ranked: list[tuple[float, dict[str, Any]]],
        bm25_ranked: list[tuple[float, dict[str, Any]]],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """Merge two ranked lists using Weighted Reciprocal Rank Fusion.

        Weighted RRF formula: score = sum(original_score * 1 / (k + rank))
        Incorporates both original relevance scores and rank position.
        """
        if not vector_ranked and not bm25_ranked:
            return []

        # Build weighted RRF scores from both ranked lists
        rrf_scores: dict[str, float] = {}
        id_to_item: dict[str, dict[str, Any]] = {}

        # Add vector search results (weight by relevance score)
        for rank, (score, item) in enumerate(vector_ranked, start=1):
            item_id = item.get("id") or str(item.get("doc_id") or "")
            if item_id:
                weighted_score = (score * 1.0) / (k + rank)  # Use original score
                rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + weighted_score
                id_to_item[item_id] = item

        # Add BM25 results (weight by relevance score)
        for rank, (score, item) in enumerate(bm25_ranked, start=1):
            item_id = item.get("id") or str(item.get("doc_id") or "")
            if item_id:
                weighted_score = (score * 1.0) / (k + rank)  # Use original score
                rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + weighted_score
                id_to_item[item_id] = item

        # Sort by weighted RRF score and return merged list
        merged = sorted(
            [
                (score, id_to_item[item_id])
                for item_id, score in rrf_scores.items()
            ],
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [item for _score, item in merged]

    def rank_with_hybrid(
        self,
        *,
        items: list[dict[str, Any]],
        user_input: str,
        semantic_scores: dict[str, float],
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Rank items using hybrid semantic + BM25 scoring.

        Args:
            items: List of memory items to rank
            user_input: Current query/context
            semantic_scores: Pre-computed semantic similarity scores (dict: item_id -> score)
            limit: Max results to return

        Returns:
            Ranked list of items, merged by RRF
        """
        if not items:
            return []

        if not self.bm25_enabled:
            # Fallback: pure semantic ranking
            vector_ranked = [
                (semantic_scores.get(item.get("id") or "", 0.0), item)
                for item in items
            ]
            vector_ranked.sort(key=lambda pair: pair[0], reverse=True)
            return [item for _score, item in vector_ranked[:limit]]

        # Prepare documents for BM25
        texts = [str(item.get("text") or item.get("content") or "") for item in items]
        tokenized = [self._tokenize(text) for text in texts]

        # Skip if no valid documents
        if not any(tokenized):
            vector_ranked = [
                (semantic_scores.get(item.get("id") or "", 0.0), item)
                for item in items
            ]
            vector_ranked.sort(key=lambda pair: pair[0], reverse=True)
            return [item for _score, item in vector_ranked[:limit]]

        # BM25 ranking
        bm25 = BM25Okapi(tokenized)
        query_tokens = self._tokenize(user_input)
        bm25_scores = bm25.get_scores(query_tokens)

        # Build ranked pairs for RRF
        vector_ranked = [
            (semantic_scores.get(item.get("id") or "", 0.0), item)
            for item in items
        ]
        bm25_ranked = [(float(score), item) for score, item in zip(bm25_scores, items)]

        # Sort for RRF
        vector_ranked.sort(key=lambda pair: pair[0], reverse=True)
        bm25_ranked.sort(key=lambda pair: pair[0], reverse=True)

        # Merge via RRF
        merged = self._reciprocal_rank_fusion(
            vector_ranked,
            bm25_ranked,
            k=60,
        )

        return merged[:max(0, int(limit))]
