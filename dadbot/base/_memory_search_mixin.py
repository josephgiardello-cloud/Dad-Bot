from __future__ import annotations

import logging

from dadbot_system.semantic_index import PGVectorSemanticIndex

logger = logging.getLogger(__name__)


class MemorySearchMixin:
    """Semantic/search delegation mixin focused on deterministic retrieval flow."""

    def embed_texts(self, texts, purpose="semantic retrieval"):
        return self._semantic_manager.embed_texts(texts, purpose=purpose)

    def embed_text(self, text):
        return self._semantic_manager.embed_text(text)

    def embedding_cache_db_path(self):
        return self._semantic_manager.embedding_cache_db_path()

    def with_embedding_cache_db(self, operation, write=False):
        return self._semantic_manager.with_embedding_cache_db(operation, write=write)

    def ensure_embedding_cache_storage(self):
        return self._semantic_manager.ensure_embedding_cache_storage()

    @staticmethod
    def embedding_cache_key(text, model_name):
        from dadbot.memory.semantic_manager import SemanticIndexManager

        return SemanticIndexManager.embedding_cache_key(text, model_name)

    def cached_embeddings_for_texts(self, model_name, texts):
        return self._semantic_manager.cached_embeddings_for_texts(model_name, texts)

    def store_cached_embeddings(self, model_name, text_to_embedding):
        return self._semantic_manager.store_cached_embeddings(
            model_name,
            text_to_embedding,
        )

    @staticmethod
    def snapshot_memory_entries(memories):
        from dadbot.memory.semantic_manager import SemanticIndexManager

        return SemanticIndexManager.snapshot_memory_entries(memories)

    def queue_semantic_memory_index(self, memories):
        return self._semantic_manager.queue_semantic_memory_index(memories)

    def _drain_semantic_index_queue(self):
        return self._semantic_manager._drain_semantic_index_queue()

    def wait_for_semantic_index_idle(self, timeout=None):
        return self._semantic_manager.wait_for_semantic_index_idle(timeout=timeout)

    def with_semantic_db(self, operation, write=False):
        return self._semantic_manager.with_semantic_db(operation, write=write)

    @staticmethod
    def _extract_embeddings_from_response(response):
        from dadbot.memory.semantic_manager import SemanticIndexManager

        return SemanticIndexManager._extract_embeddings_from_response(response)

    def ensure_semantic_memory_db(self):
        return self._semantic_manager.ensure_semantic_memory_db()

    def clear_semantic_memory_index(self):
        return self._semantic_manager.clear_semantic_memory_index()

    def semantic_memory_key(self, memory):
        return self._semantic_manager.semantic_memory_key(memory)

    def memory_embedding_text(self, memory):
        return self._semantic_manager.memory_embedding_text(memory)

    def semantic_index_row_count(self):
        return self._semantic_manager.semantic_index_row_count()

    def semantic_memory_status(self):
        return self._semantic_manager.semantic_memory_status()

    def sync_semantic_memory_index(self, memories):
        return self._semantic_manager.sync_semantic_memory_index(memories)

    def semantic_memory_lookup(self, memories):
        return self._semantic_manager.semantic_memory_lookup(memories)

    def semantic_query_context(self, query, limit):
        return self._semantic_manager.semantic_query_context(query, limit)

    @staticmethod
    def semantic_memory_filters(query_tokens, query_category, query_mood):
        from dadbot.memory.semantic_manager import SemanticIndexManager

        return SemanticIndexManager.semantic_memory_filters(
            query_tokens,
            query_category,
            query_mood,
        )

    def recent_semantic_rows(self, candidate_limit):
        return self._semantic_manager.recent_semantic_rows(candidate_limit)

    def filtered_semantic_rows(self, where_clauses, params, candidate_limit):
        return self._semantic_manager.filtered_semantic_rows(
            where_clauses,
            params,
            candidate_limit,
        )

    def score_semantic_rows(self, rows, current_memories, query_embedding):
        return self._semantic_manager.score_semantic_rows(
            rows,
            current_memories,
            query_embedding,
        )

    def semantic_memory_matches(self, query, memories, limit=3):
        # Search path must remain read-only: retrieval only, no catalog/index mutation.
        if not memories:
            return []

        status = {"index_pending": False, "degraded": False}
        self._semantic_manager._last_retrieval_status = status
        with self.bot._semantic_index_lock:
            future = self.bot._semantic_index_future
            status["index_pending"] = bool(future and not future.done())
        query_context = self.semantic_query_context(query, limit)
        if query_context is None:
            status.update(degraded=True, reason="embedding_unavailable")
            return []

        current_memories = self.semantic_memory_lookup(memories)
        backend = self._semantic_manager._semantic_index_backend
        candidate_limit = self.bot.runtime_config.memory_candidate_limit
        try:
            if candidate_limit > 0 and len(current_memories) > candidate_limit and isinstance(
                backend, PGVectorSemanticIndex,
            ):
                rows = backend.fetch_nearest_by_keys(
                    current_memories.keys(), query_context["query_embedding"], candidate_limit,
                )
                status["bounded"] = True
            else:
                rows = self.catalog_semantic_rows(current_memories.keys())
                status["bounded"] = False
        except Exception as exc:
            logger.warning("Semantic memory lookup failed: %s", exc)
            status.update(degraded=True, reason="semantic_lookup_failed")
            return []
        status["candidate_count"] = len(rows)
        if not status["bounded"] and len(rows) < len(current_memories):
            status.update(degraded=True, reason="incomplete_index")

        recent_topics = self.bot.recent_memory_topics(limit=4)
        mood_trend = self.bot.current_memory_mood_trend()
        scored = []
        for similarity, memory in self.score_semantic_rows(
            rows,
            current_memories,
            query_context["query_embedding"],
        ):
            freshness = self.bot.semantic_memory_freshness_weight(memory)
            alignment = self.bot.memory_alignment_weight(
                memory,
                query_tokens=query_context["query_tokens"],
                query_category=query_context["query_category"],
                query_mood=query_context["query_mood"],
                recent_topics=recent_topics,
                mood_trend=mood_trend,
            )
            if alignment <= 0:
                continue
            score = similarity * 5.0 * freshness * alignment
            impact_bonus = min(
                1.5,
                max(0.0, self.bot.memory_impact_score(memory)) * 0.35,
            )
            if score > 0:
                score += impact_bonus
            if score > 0.1:
                scored.append((round(score, 4), memory))

        ranked = sorted(
            scored,
            key=lambda item: (
                item[0],
                item[1].get("updated_at", ""),
                item[1].get("summary", ""),
            ),
            reverse=True,
        )
        return self.bot.select_diverse_ranked_memories(ranked, limit)
