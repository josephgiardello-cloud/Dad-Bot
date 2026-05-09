from __future__ import annotations

import logging
from typing import Any, Protocol

from dadbot.core.execution_memory_view import merge_memory_retrieval_sets

logger = logging.getLogger(__name__)


class SupportsMemoryRetrieval(Protocol):
    def graph_retrieval_for_input(self, query: str, limit: int = 3) -> Any: ...

    def relevant_memories_for_input(
        self,
        query: str,
        limit: int = 5,
        graph_result: Any | None = None,
    ) -> list[dict[str, Any]]: ...


class MemoryService:
    """Unified retrieval facade for runtime memory reads.

    This is intentionally thin: it consolidates call paths without changing
    retrieval semantics.
    """

    def __init__(self, runtime: SupportsMemoryRetrieval):
        self.runtime = runtime

    def retrieve_for_query(
        self,
        *,
        query: str,
        graph_limit: int = 3,
        memory_limit: int = 5,
    ) -> list[dict[str, Any]]:
        graph_result = None
        try:
            graph_result = self.runtime.graph_retrieval_for_input(query, limit=graph_limit)
        except Exception:
            graph_result = None

        try:
            return list(
                self.runtime.relevant_memories_for_input(
                    query,
                    limit=memory_limit,
                    graph_result=graph_result,
                )
                or []
            )
        except Exception as exc:
            logger.debug("MemoryService retrieval failed (non-fatal): %s", exc)
            return []

    @staticmethod
    def merge_retrieval_sets(
        existing: list[dict[str, Any]],
        incoming: list[dict[str, Any]],
        *,
        source_labels: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return merge_memory_retrieval_sets(
            existing,
            incoming,
            source_labels=source_labels,
        )
