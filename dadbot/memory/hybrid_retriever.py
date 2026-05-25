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
    ) -> list[tuple[float, dict[str, Any]]]:
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

        # Sort by weighted RRF score and return merged scored list
        return sorted(
            [(score, id_to_item[item_id]) for item_id, score in rrf_scores.items()],
            key=lambda pair: pair[0],
            reverse=True,
        )

    @staticmethod
    def _stable_dedupe(items: list[dict[str, Any]], *, key_field: str) -> list[dict[str, Any]]:
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for item in items:
            key = str(item.get(key_field) or "").strip().lower()
            if not key:
                deduped.append(item)
                continue
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @staticmethod
    def _diversity_rerank(
        scored_items: list[tuple[float, dict[str, Any]]],
        *,
        limit: int,
        max_per_bucket: int = 2,
    ) -> list[dict[str, Any]]:
        """Greedy diversity selection.

        Buckets on (category, type/source) when available to avoid near-duplicate
        context blocks from the same theme dominating the top-N.
        """
        selected: list[dict[str, Any]] = []
        bucket_counts: dict[tuple[str, str], int] = {}

        def bucket_for(item: dict[str, Any]) -> tuple[str, str]:
            category = str(item.get("category") or "").strip().lower() or "general"
            source = str(item.get("source_type") or item.get("type") or item.get("kind") or "").strip().lower()
            if not source:
                source = "memory"
            return category, source

        for _score, item in scored_items:
            bucket = bucket_for(item)
            count = int(bucket_counts.get(bucket, 0) or 0)
            if count >= max_per_bucket:
                continue
            bucket_counts[bucket] = count + 1
            selected.append(item)
            if len(selected) >= limit:
                return selected

        # If diversity gate was too strict (missing metadata, etc.), backfill.
        if len(selected) < limit:
            selected_ids = {str(item.get("id") or item.get("doc_id") or "") for item in selected}
            for _score, item in scored_items:
                item_id = str(item.get("id") or item.get("doc_id") or "")
                if item_id and item_id in selected_ids:
                    continue
                selected.append(item)
                if len(selected) >= limit:
                    break
        return selected

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
        merged_scored = self._reciprocal_rank_fusion(
            vector_ranked,
            bm25_ranked,
            k=60,
        )

        limit = max(0, int(limit))
        merged_items = [item for _score, item in merged_scored]

        # Post-process: content dedupe + diversity
        merged_items = self._stable_dedupe(merged_items, key_field="id")
        merged_items = self._stable_dedupe(merged_items, key_field="doc_id")
        merged_items = self._stable_dedupe(merged_items, key_field="text")
        merged_items = self._stable_dedupe(merged_items, key_field="content")

        # Re-score list for diversity selection: keep original RRF order/score.
        scored_for_diversity = [(score, item) for score, item in merged_scored if item in merged_items]
        return self._diversity_rerank(scored_for_diversity, limit=limit)
