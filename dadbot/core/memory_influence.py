"""Track which memories actually influence response generation.

Purpose: Build a feedback loop where high-influence memories get higher
decay scores, unused memories decay faster, and the retrieval system
learns which memory types drive quality responses.

This module runs post-commit (after SaveNode) and logs influence data
without blocking the execution path.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MemoryInfluenceTracker:
    """Detect and score memory-to-response influence."""

    def __init__(self):
        self._influence_cache: dict[str, float] = {}

    def extract_memories_from_context(self, context_state: dict) -> list[dict]:
        """Extract memory entries that were included in turn context."""
        memories: list[dict] = []

        # Recent buffer (highest priority)
        recent_buffer = context_state.get("rich_context", {}).get("recent_buffer")
        if isinstance(recent_buffer, list):
            memories.extend(
                {
                    "source": "recent_buffer",
                    "priority": 1,
                    "content": str(msg.get("content", ""))[:200],
                }
                for msg in recent_buffer
            )

        # Semantic memories from retrieval
        semantic_mems = list(context_state.get("memories") or [])
        memories.extend(
            {
                "source": "semantic",
                "priority": 3,
                "id": str(mem.get("id", "")),
                "summary": str(mem.get("summary", ""))[:150],
                "category": str(mem.get("category", "general")),
            }
            for mem in semantic_mems
        )

        # Graph results (long-term patterns)
        graph_result = context_state.get("graph_result")
        if isinstance(graph_result, dict):
            memories.append(
                {
                    "source": "graph",
                    "priority": 2,
                    "content": graph_result.get("compressed_summary", "")[:200],
                }
            )

        # Archive notes (prior sessions)
        archive_entries = list(context_state.get("archive_entries") or [])
        memories.extend(
            {
                "source": "archive",
                "priority": 2,
                "created_at": str(entry.get("created_at", "")),
                "summary": str(entry.get("summary", ""))[:150],
            }
            for entry in archive_entries
        )

        return memories

    def score_memory_influence(
        self,
        reply_text: str,
        memory_entries: list[dict],
        *,
        tokenize_fn: Any = None,
    ) -> dict[str, float]:
        """Score how much each memory influenced the reply.

        Returns: {memory_id_or_source: influence_score in [0.0, 1.0]}

        Heuristics:
        - Memory content appearing verbatim in reply: 0.9+
        - Topic keywords from memory appearing in reply: 0.5-0.8
        - Similar mood/category to reply context: ignored (too weak)
        - No detectable contribution: 0.0
        """
        scores: dict[str, float] = {}
        reply_lower = str(reply_text or "").lower()

        for memory in memory_entries:
            memory_id = str(memory.get("id") or memory.get("source", "unknown"))
            content = str(memory.get("content") or memory.get("summary") or "").lower()

            if not content:
                scores[memory_id] = 0.0
                continue

            # 1. Verbatim phrase matching (strong signal)
            phrases = [
                phrase.strip()
                for phrase in content.split(".")
                if len(phrase.strip()) > 5
            ]
            verbatim_matches = sum(
                1 for phrase in phrases if phrase in reply_lower
            )
            if verbatim_matches > 0:
                scores[memory_id] = min(0.95, 0.85 + 0.1 * verbatim_matches)
                continue

            # 2. Significant token overlap (moderate signal)
            if tokenize_fn is not None:
                memory_tokens = set(tokenize_fn(content))
                reply_tokens = set(tokenize_fn(reply_lower))
                overlap = memory_tokens & reply_tokens
                if len(memory_tokens) > 0:
                    overlap_ratio = len(overlap) / len(memory_tokens)
                    if overlap_ratio > 0.4:
                        scores[memory_id] = min(0.80, 0.5 + 0.3 * overlap_ratio)
                        continue

            # 3. No other signals (avoid weak heuristics)
            scores[memory_id] = 0.0

        return scores

    def log_influence_feedback(
        self,
        *,
        turn_id: str,
        memory_entries: list[dict],
        influence_scores: dict[str, float],
        reply_text: str,
    ) -> dict[str, Any]:
        """Log influence data for post-turn analysis.

        Returns: {summary, high_influence_count, unused_count, avg_influence}
        """
        high_influence = [
            (mid, score)
            for mid, score in influence_scores.items()
            if score >= 0.5
        ]
        unused = [
            (mid, score)
            for mid, score in influence_scores.items()
            if score < 0.1
        ]

        avg_influence = (
            sum(influence_scores.values()) / len(influence_scores)
            if influence_scores
            else 0.0
        )

        logger.info(
            "Memory influence feedback (turn=%s): high=%d unused=%d avg_influence=%.3f",
            turn_id,
            len(high_influence),
            len(unused),
            avg_influence,
        )

        return {
            "turn_id": turn_id,
            "high_influence_count": len(high_influence),
            "unused_count": len(unused),
            "avg_influence_score": round(avg_influence, 3),
            "high_influence_memories": high_influence[:3],  # top 3 for tracing
        }
