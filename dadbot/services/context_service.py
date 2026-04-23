from __future__ import annotations

import logging
from typing import Any

from dadbot.context import ContextBuilder
from dadbot.utils import significant_tokens as _significant_tokens

logger = logging.getLogger(__name__)


class ContextService:
    """Service wrapper for memory/profile/relationship context composition.

    When a ``semantic_index`` (``SQLiteSemanticIndex`` or ``PGVectorSemanticIndex``)
    is wired at construction time, ``build_context`` augments the standard context
    sections with targeted semantic search results — RAG over long-term memory.
    This lets the model surface relevant history beyond the most-recent window
    without bloating the prompt with the entire memory store.
    """

    def __init__(
        self,
        context_builder: ContextBuilder,
        memory_manager: Any,
        semantic_index: Any = None,
    ):
        self.context_builder = context_builder
        self.memory_manager = memory_manager
        self.semantic_index = semantic_index  # Optional; wired by ServiceRegistry.boot()

    def build_context(self, turn_context: Any) -> dict[str, Any]:
        user_input = str(getattr(turn_context, "user_input", "") or "")
        ctx: dict[str, Any] = {
            "core_persona": self.context_builder.build_core_persona_prompt(),
            "dynamic_profile": self.context_builder.build_dynamic_profile_context(),
            "relationship": self.context_builder.build_relationship_context(),
            "session_summary": self.context_builder.build_session_summary_context(),
            "memory": self.context_builder.build_memory_context(user_input),
            "relevant": self.context_builder.build_relevant_context(user_input),
            "cross_session": self.context_builder.build_cross_session_context(user_input),
        }

        # Semantic RAG: query long-term index for targeted memory hits
        if self.semantic_index is not None and user_input.strip():
            try:
                tokens = _significant_tokens(user_input)
                query_embedding = None
                embed_text = getattr(self.memory_manager, "embed_text", None)
                if callable(embed_text):
                    query_embedding = embed_text(user_input)
                hits = self.semantic_index.fetch_candidates(
                    query_embedding=query_embedding,
                    query_tokens=tokens,
                    query_category="general",
                    query_mood="neutral",
                    limit=5,
                )
                if hits:
                    # Deduplicate against already-included base memory summaries
                    base_summaries = {
                        str(m.get("summary", ""))
                        for m in (ctx.get("memory") or [])
                        if isinstance(m, dict)
                    }
                    unique_hits = [h for h in hits if str(h.get("summary", "")) not in base_summaries]
                    if unique_hits:
                        ctx["semantic"] = unique_hits
            except Exception as exc:
                logger.debug("ContextService: semantic index query failed (non-fatal): %s", exc)

        return ctx
