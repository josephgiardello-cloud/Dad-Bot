"""Goal-aware memory ranker: re-ranks retrieved memories by active-goal relevance.

The base MemoryScorer in ``memory/scoring.py`` ranks entries by emotional
intensity, recency, and relationship impact.  This ranker adds a fourth
dimension: *goal relevance*.

When the user has active goals, memories that relate to those goals are
surfaced higher so InferenceNode can connect past context to current objectives.
The boost is additive and capped to avoid completely overriding semantic
similarity scores.

Usage (from ContextBuilderNode)::

    ranker = GoalAwareRanker()
    context.state["memories"] = ranker.rerank(memories, active_goals)
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum fractional boost applied to any one memory entry (keeps ranker
#: from completely overriding the base semantic ranking).
_MAX_GOAL_BOOST: float = 0.40

#: Each overlapping content token adds this much to the boost score.
_TOKEN_BOOST_STEP: float = 0.08

#: Stopwords excluded from overlap computation.
_STOPWORDS: frozenset[str] = frozenset({
    "i", "a", "an", "the", "is", "to", "of", "in", "it", "my", "me", "we",
    "be", "do", "go", "so", "at", "by", "he", "she", "you", "for", "and",
    "or", "not", "but", "was", "are", "has", "had", "have", "that", "this",
    "with", "from", "they", "them", "their", "what", "when", "where",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> frozenset[str]:
    """Lower-cased word tokens, filtered to length ≥ 3 and not stopwords."""
    raw = re.split(r"\W+", str(text or "").lower())
    return frozenset(t for t in raw if len(t) >= 3 and t not in _STOPWORDS)


def _memory_text(mem: Any) -> str:
    """Extract text content from a memory entry (dict or string)."""
    if isinstance(mem, dict):
        return " ".join(str(v) for v in [
            mem.get("content"),
            mem.get("summary"),
            mem.get("text"),
            mem.get("note"),
        ] if v)
    return str(mem or "")


# ---------------------------------------------------------------------------
# GoalAwareRanker
# ---------------------------------------------------------------------------

class GoalAwareRanker:
    """Re-rank a list of memory entries by relevance to active goals.

    The ranker computes a *goal boost* for each memory based on token overlap
    between the memory's text and the combined description of all active goals.
    Memories with higher overlap float to the top of the list.

    This runs in O(n * g * tokens) time — entirely in-process with no I/O.
    """

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
        """Return a re-ranked copy of ``memories``.

        Memories are NOT modified; the list order is the only thing that changes.

        Args:
            memories: List of memory entries (dicts or strings).
            active_goals: List of GoalRecord-like dicts with a ``description`` key.

        Returns:
            Re-ranked list (new list, original objects unchanged).
        """
        if not memories or not active_goals:
            return list(memories)

        # Build union of goal tokens (all active goals contribute equally).
        goal_tokens: frozenset[str] = frozenset()
        for goal in active_goals:
            desc = str(goal.get("description", goal) if isinstance(goal, dict) else goal)
            goal_tokens = goal_tokens | _tokenise(desc)

        if not goal_tokens:
            return list(memories)

        # Compute boost per memory.
        scored: list[tuple[float, int, Any]] = []
        for idx, mem in enumerate(memories):
            mem_tokens = _tokenise(_memory_text(mem))
            overlap = len(goal_tokens & mem_tokens)
            boost = min(self.max_boost, overlap * self.token_boost_step)
            # Preserve original index as tiebreaker so equal-boost items retain order.
            scored.append((boost, idx, mem))

        # Stable sort: higher boost first, original index as tiebreaker.
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [mem for _, _, mem in scored]

    def goal_relevance_scores(
        self,
        memories: list[Any],
        active_goals: list[dict[str, Any]],
    ) -> list[float]:
        """Return a parallel list of [0, max_boost] relevance scores for each memory."""
        if not active_goals:
            return [0.0] * len(memories)

        goal_tokens: frozenset[str] = frozenset()
        for goal in active_goals:
            desc = str(goal.get("description", goal) if isinstance(goal, dict) else goal)
            goal_tokens = goal_tokens | _tokenise(desc)

        scores: list[float] = []
        for mem in memories:
            mem_tokens = _tokenise(_memory_text(mem))
            overlap = len(goal_tokens & mem_tokens)
            scores.append(min(self.max_boost, overlap * self.token_boost_step))
        return scores
