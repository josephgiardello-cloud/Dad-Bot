"""Goal-aware memory ranker used by the core orchestrator pipeline.

This module is intentionally in ``dadbot.core`` so Phase 4 entrypoints do not
depend on ignored/side-band package paths.
"""

from __future__ import annotations

import re
from typing import Any

_MAX_GOAL_BOOST: float = 0.40
_TOKEN_BOOST_STEP: float = 0.08
_STOPWORDS: frozenset[str] = frozenset(
    {
        "i",
        "a",
        "an",
        "the",
        "is",
        "to",
        "of",
        "in",
        "it",
        "my",
        "me",
        "we",
        "be",
        "do",
        "go",
        "so",
        "at",
        "by",
        "he",
        "she",
        "you",
        "for",
        "and",
        "or",
        "not",
        "but",
        "was",
        "are",
        "has",
        "had",
        "have",
        "that",
        "this",
        "with",
        "from",
        "they",
        "them",
        "their",
        "what",
        "when",
        "where",
    },
)


def _tokenise(text: str) -> frozenset[str]:
    raw = re.split(r"\W+", str(text or "").lower())
    return frozenset(t for t in raw if len(t) >= 3 and t not in _STOPWORDS)


def _memory_text(mem: Any) -> str:
    if isinstance(mem, dict):
        return " ".join(
            str(v)
            for v in [
                mem.get("content"),
                mem.get("summary"),
                mem.get("text"),
                mem.get("note"),
            ]
            if v
        )
    return str(mem or "")


class GoalAwareRanker:
    """Re-rank memory entries by relevance to active goals."""

    def __init__(
        self,
        max_boost: float = _MAX_GOAL_BOOST,
        token_boost_step: float = _TOKEN_BOOST_STEP,
    ) -> None:
        self.max_boost = max_boost
        self.token_boost_step = token_boost_step

    def rerank(
        self,
        memories: list[Any],
        active_goals: list[dict[str, Any]],
    ) -> list[Any]:
        if not memories or not active_goals:
            return list(memories)

        goal_tokens: frozenset[str] = frozenset()
        for goal in active_goals:
            desc = str(
                goal.get("description", goal) if isinstance(goal, dict) else goal,
            )
            goal_tokens = goal_tokens | _tokenise(desc)

        if not goal_tokens:
            return list(memories)

        scored: list[tuple[float, int, Any]] = []
        for idx, mem in enumerate(memories):
            mem_tokens = _tokenise(_memory_text(mem))
            overlap = len(goal_tokens & mem_tokens)
            boost = min(self.max_boost, overlap * self.token_boost_step)
            scored.append((boost, idx, mem))

        scored.sort(key=lambda t: (-t[0], t[1]))
        return [mem for _, _, mem in scored]

    def goal_relevance_scores(
        self,
        memories: list[Any],
        active_goals: list[dict[str, Any]],
    ) -> list[float]:
        if not active_goals:
            return [0.0] * len(memories)

        goal_tokens: frozenset[str] = frozenset()
        for goal in active_goals:
            desc = str(
                goal.get("description", goal) if isinstance(goal, dict) else goal,
            )
            goal_tokens = goal_tokens | _tokenise(desc)

        scores: list[float] = []
        for mem in memories:
            mem_tokens = _tokenise(_memory_text(mem))
            overlap = len(goal_tokens & mem_tokens)
            scores.append(min(self.max_boost, overlap * self.token_boost_step))
        return scores
