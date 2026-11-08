from __future__ import annotations

from dadbot.managers.memory_query import MEMORY_DENSE_WEIGHT
from dadbot.memory.reranker import (
    CANDIDATE_LIMIT,
    MARGIN,
    MODEL_ID,
    MODEL_REVISION,
    LocalQwenReranker,
    reranked_candidates,
    select_candidates,
)

__all__ = [
    "CANDIDATE_LIMIT",
    "MARGIN",
    "MODEL_ID",
    "MODEL_REVISION",
    "LocalQwenReranker",
    "gated_candidates",
    "reranked_candidates",
]


def gated_candidates(bot, rows):
    ranked = sorted(
        (
            (round(MEMORY_DENSE_WEIGHT * dense + (1 - MEMORY_DENSE_WEIGHT) * sparse, 6), memory)
            for dense, sparse, memory, _ in rows
        ),
        key=lambda item: (item[0], item[1].get("updated_at", ""), item[1].get("summary", "")),
        reverse=True,
    )
    return select_candidates(ranked, bot.memory_query.memory_diversity_key)
