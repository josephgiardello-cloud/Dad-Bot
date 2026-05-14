from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


class SemanticMemoryRanker:
    """Ranks semantic memory items against the current query."""

    def __init__(self, now_fn: Callable[[], float] | None = None) -> None:
        self._now = now_fn or time.time

    @staticmethod
    def tokenize_memory_text(value: str) -> set[str]:
        raw = str(value or "").strip().lower()
        if not raw:
            return set()
        return {token for token in raw.replace("\n", " ").split(" ") if len(token.strip()) >= 3}

    def rank_semantic_memory_items(
        self,
        *,
        items: list[dict[str, Any]],
        user_input: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        query_tokens = self.tokenize_memory_text(user_input)
        now = float(self._now())
        ranked: list[tuple[float, dict[str, Any]]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "")
            if not text:
                continue
            item_tokens = self.tokenize_memory_text(text)
            lexical_overlap = 0.0
            if query_tokens and item_tokens:
                lexical_overlap = float(len(query_tokens.intersection(item_tokens))) / float(max(1, len(query_tokens)))
            base_score = float(item.get("score") or 0.0)
            created_at = float(item.get("created_at") or now)
            recency = max(0.0, 1.0 - min(1.0, (now - created_at) / 86_400.0))
            final_score = (0.55 * base_score) + (0.35 * lexical_overlap) + (0.10 * recency)
            candidate = dict(item)
            candidate["retrieval_score"] = round(float(final_score), 6)
            ranked.append((final_score, candidate))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [candidate for _score, candidate in ranked[:max(0, int(limit))]]


_default_ranker = SemanticMemoryRanker()


def tokenize_memory_text(value: str) -> set[str]:
    return _default_ranker.tokenize_memory_text(value)


def rank_semantic_memory_items(
    *,
    items: list[dict[str, Any]],
    user_input: str,
    limit: int,
) -> list[dict[str, Any]]:
    return _default_ranker.rank_semantic_memory_items(items=items, user_input=user_input, limit=limit)
