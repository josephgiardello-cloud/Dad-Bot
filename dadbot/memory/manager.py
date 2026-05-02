from __future__ import annotations

import logging
from datetime import date

from dadbot.contracts import DadBotContext, SupportsDadBotAccess
from dadbot.core.execution_boundary import MemoryWriteOwnerScope
from dadbot.core.execution_context import ensure_execution_trace_root
from dadbot.memory.graph_manager import MemoryGraphManager
from dadbot.memory.lifecycle import MemoryLifecycleManager
from dadbot.memory.migration import CURRENT_SCHEMA_VERSION
from dadbot.memory.normalizers import MemoryNormalizer
from dadbot.memory.semantic_manager import SemanticIndexManager
from dadbot.memory.storage import MemoryStorageBackend

logger = logging.getLogger(__name__)


class MemoryManager:
    """Owns memory normalization, catalog access, and semantic-memory persistence."""

    def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot
        self._memory_store = {}
        self._graph_manager = MemoryGraphManager(self.bot, self)
        self._semantic_manager = SemanticIndexManager(self.bot)
        self._normalizer = MemoryNormalizer(self.bot)  # data-shape policy
        self._storage = MemoryStorageBackend(self.bot, self)  # persistence I/O
        self._lifecycle = MemoryLifecycleManager(self.bot, self)  # catalog access

    @property
    def memory_store(self):
        return self._memory_store

    @memory_store.setter
    def memory_store(self, value):
        self._memory_store = value if isinstance(value, dict) else {}

    def initialize_memory_store_defaults(self):
        if "schema_version" not in self._memory_store:
            self._memory_store["schema_version"] = CURRENT_SCHEMA_VERSION
        if "last_scheduled_proactive_at" not in self._memory_store:
            self._memory_store["last_scheduled_proactive_at"] = None
        if "last_memory_compaction_at" not in self._memory_store:
            self._memory_store["last_memory_compaction_at"] = None
        if "last_memory_compaction_summary" not in self._memory_store:
            self._memory_store["last_memory_compaction_summary"] = ""
        if "mcp_local_store" not in self._memory_store:
            self._memory_store["mcp_local_store"] = {}
        if "narrative_memories" not in self._memory_store:
            self._memory_store["narrative_memories"] = []
        if "heritage_cross_links" not in self._memory_store:
            self._memory_store["heritage_cross_links"] = []
        if "advice_audits" not in self._memory_store:
            self._memory_store["advice_audits"] = []
        if "environmental_cues_history" not in self._memory_store:
            self._memory_store["environmental_cues_history"] = []
        if "longitudinal_insights" not in self._memory_store:
            self._memory_store["longitudinal_insights"] = []
        return self._memory_store

    def load_memory_store(self):
        self.memory_store = self._storage.load_memory_store()
        self.initialize_memory_store_defaults()
        return self.memory_store

    def should_do_daily_checkin(self) -> bool:
        last_date = str(self.memory_store.get("last_mood_updated_at") or "").strip()
        return last_date != date.today().isoformat()

    def save_mood_state(self, mood):
        normalized_mood = self.bot.normalize_mood(mood)
        today_stamp = date.today().isoformat()
        recent = self.recent_mood_history()
        recent.append({"mood": normalized_mood, "date": today_stamp})
        self.mutate_memory_store(
            last_mood=normalized_mood,
            last_mood_updated_at=today_stamp,
            recent_moods=recent[-12:],
        )

    def consolidate_turn_outcome(self, turn_context=None, result=None):
        """Compatibility hook for SaveNode-owned post-turn memory consolidation."""
        consolidate = getattr(self.bot, "consolidate_memories", None)
        if not callable(consolidate):
            return {"consolidated_count": 0}
        try:
            consolidated = consolidate() or []
            return {"consolidated_count": len(consolidated)}
        except Exception as exc:
            logger.warning(
                "MemoryManager.consolidate_turn_outcome failed (non-fatal): %s",
                exc,
            )
            return {"consolidated_count": 0, "error": str(exc)}

    # ------------------------------------------------------------------ Sub-managers

    @property
    def graph_manager(self) -> MemoryGraphManager:
        return self._graph_manager

    @property
    def _graph_store_backend(self):
        # Backward-compatible facade for tests/integration code.
        return self._graph_manager._graph_store_backend

    @property
    def _graph_prompt_compressor(self):
        # Backward-compatible facade for tests/integration code.
        return self._graph_manager._graph_prompt_compressor

    @property
    def semantic(self) -> SemanticIndexManager:
        return self._semantic_manager

    def close(self):
        # Best-effort shutdown hook used by DadBot.shutdown.
        self.wait_for_semantic_index_idle(timeout=5)

    # ------------------------------------------------------------------ Semantic delegation

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

    def semantic_candidate_rows(
        self,
        where_clauses,
        params,
        candidate_limit,
        query_embedding=None,
        query_tokens=None,
        query_category="general",
        query_mood="neutral",
    ):
        return self._semantic_manager.semantic_candidate_rows(
            where_clauses,
            params,
            candidate_limit,
            query_embedding=query_embedding,
            query_tokens=query_tokens,
            query_category=query_category,
            query_mood=query_mood,
        )

    def score_semantic_rows(self, rows, current_memories, query_embedding):
        return self._semantic_manager.score_semantic_rows(
            rows,
            current_memories,
            query_embedding,
        )

    def semantic_memory_matches(self, query, memories, limit=3):
        # Keep orchestration at facade level for backward-compatibility with
        # monkeypatching in tests/integration code that patches MemoryManager methods.
        if not memories:
            return []

        self.queue_semantic_memory_index(memories)

        query_context = self.semantic_query_context(query, limit)
        if query_context is None:
            return []

        current_memories = self.semantic_memory_lookup(memories)
        where_clauses, params = self.semantic_memory_filters(
            query_context["query_tokens"],
            query_context["query_category"],
            query_context["query_mood"],
        )

        try:
            rows = self.semantic_candidate_rows(
                where_clauses,
                params,
                query_context["candidate_limit"],
                query_embedding=query_context["query_embedding"],
                query_tokens=query_context["query_tokens"],
                query_category=query_context["query_category"],
                query_mood=query_context["query_mood"],
            )
        except Exception as exc:
            logger.warning("Semantic memory lookup failed for query %r: %s", query, exc)
            return []

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

    # ------------------------------------------------------------------ Graph delegation

    def ensure_graph_store(self):
        return self._graph_manager.ensure_graph_store()

    def clear_graph_store(self):
        return self._graph_manager.clear_graph_store()

    @staticmethod
    def graph_node_key(node_type, identifier):
        return MemoryGraphManager.graph_node_key(node_type, identifier)

    @staticmethod
    def graph_edge_key(source_key, target_key, relation_type):
        return MemoryGraphManager.graph_edge_key(source_key, target_key, relation_type)

    def graph_source_id(self, source_type, entry):
        return self._graph_manager.graph_source_id(source_type, entry)

    def graph_keyword_tokens(self, text, limit=4):
        return self._graph_manager.graph_keyword_tokens(text, limit)

    @staticmethod
    def graph_slug(text):
        return MemoryGraphManager.graph_slug(text)

    def canonical_graph_entity(self, label, semantic_type=None):
        return self._graph_manager.canonical_graph_entity(
            label,
            semantic_type=semantic_type,
        )

    def normalize_relation_type(self, relation_type):
        return self._graph_manager.normalize_relation_type(relation_type)

    def relation_type_for_entity(
        self,
        source_type,
        entity_type,
        text,
        fallback="mentions",
    ):
        return self._graph_manager.relation_type_for_entity(
            source_type,
            entity_type,
            text,
            fallback=fallback,
        )

    def extract_typed_graph_facts(
        self,
        text,
        source_type,
        *,
        default_category="general",
        default_mood="neutral",
        default_day="",
    ):
        return self._graph_manager.extract_typed_graph_facts(
            text,
            source_type,
            default_category=default_category,
            default_mood=default_mood,
            default_day=default_day,
        )

    def llm_typed_graph_facts(self, text, source_type):
        return self._graph_manager.llm_typed_graph_facts(text, source_type)

    def graph_source_confidence(self, source_type, entry):
        return self._graph_manager.graph_source_confidence(source_type, entry)

    def graph_source_weight(self, source_type, entry):
        return self._graph_manager.graph_source_weight(source_type, entry)

    @staticmethod
    def upsert_graph_node(node_map, **node):
        return MemoryGraphManager.upsert_graph_node(node_map, **node)

    @staticmethod
    def upsert_graph_edge(edge_map, **edge):
        return MemoryGraphManager.upsert_graph_edge(edge_map, **edge)

    def build_graph_projection(self):
        return self._graph_manager.build_graph_projection()

    def preview_memory_graph(self, snapshot=None):
        return self._graph_manager.preview_memory_graph(snapshot=snapshot)

    def sync_graph_store(self):
        return self._graph_manager.sync_graph_store()

    def graph_snapshot(self):
        return self._graph_manager.graph_snapshot()

    def graph_source_summary(self, node):
        return self._graph_manager.graph_source_summary(node)

    def graph_source_lines(self, selected_items):
        return self._graph_manager.graph_source_lines(selected_items)

    def graph_retrieval_for_input(self, query, limit=3):
        return self._graph_manager.graph_retrieval_for_input(query, limit)

    def build_graph_summary_context(self, limit=3):
        return self._graph_manager.build_graph_summary_context(limit)

    def normalize_reminder_entry(self, reminder):
        return self._normalizer.normalize_reminder_entry(reminder)

    def normalize_session_archive_entry(self, entry):
        return self._normalizer.normalize_session_archive_entry(entry)

    def normalize_consolidated_memory_entry(self, entry):
        return self._normalizer.normalize_consolidated_memory_entry(entry)

    def normalize_persona_evolution_entry(self, entry):
        return self._normalizer.normalize_persona_evolution_entry(entry)

    def normalize_wisdom_entry(self, entry):
        return self._normalizer.normalize_wisdom_entry(entry)

    def normalize_life_pattern_entry(self, entry):
        return self._normalizer.normalize_life_pattern_entry(entry)

    def normalize_proactive_message_entry(self, entry):
        return self._normalizer.normalize_proactive_message_entry(entry)

    def normalize_memory_graph(self, graph):
        return self._normalizer.normalize_memory_graph(graph)

    def normalize_relationship_state(self, state):
        return self._normalizer.normalize_relationship_state(state)

    @staticmethod
    def memory_sort_key(memory):
        return MemoryNormalizer.memory_sort_key(memory)

    @staticmethod
    def infer_memory_category(summary):
        return MemoryNormalizer.infer_memory_category(summary)

    @staticmethod
    def _coerce_memory_confidence(value, *, default=0.5):
        return MemoryNormalizer._coerce_memory_confidence(value, default=default)

    @staticmethod
    def _coerce_impact_score(value, *, default=1.0):
        return MemoryNormalizer._coerce_impact_score(value, default=default)

    @staticmethod
    def _coerce_unit_float(value, *, default):
        return MemoryNormalizer._coerce_unit_float(value, default=default)

    @staticmethod
    def _normalize_memory_contradictions(value):
        return MemoryNormalizer._normalize_memory_contradictions(value)

    @staticmethod
    def _normalize_optional_timestamp(value):
        return MemoryNormalizer._normalize_optional_timestamp(value)

    @staticmethod
    def _normalize_memory_timestamp(value, *, fallback):
        return MemoryNormalizer._normalize_memory_timestamp(value, fallback=fallback)

    @staticmethod
    def _coerce_persona_strength(value, *, default=1.0):
        return MemoryNormalizer._coerce_persona_strength(value, default=default)

    @staticmethod
    def _coerce_persona_impact(value, *, default=0.0):
        return MemoryNormalizer._coerce_persona_impact(value, default=default)

    def normalize_memory_entry(self, memory):
        return self._normalizer.normalize_memory_entry(memory)

    def normalize_persisted_memory_entry(self, memory):
        return self._normalizer.normalize_persisted_memory_entry(memory)

    def memory_quality_score(self, memory):
        return self._normalizer.memory_quality_score(memory)

    def is_high_quality_memory(self, memory):
        return self._normalizer.is_high_quality_memory(memory)

    def memory_dedup_key(self, memory):
        return self._normalizer.memory_dedup_key(memory)

    def clean_memory_entries(self, memories):
        return self._normalizer.clean_memory_entries(memories)

    def save_memory_catalog(self, memories):
        cleaned = self.clean_memory_entries(memories[-50:])
        self.mutate_memory_store(memories=cleaned)
        self.queue_semantic_memory_index(cleaned)
        self.bot.mark_memory_graph_dirty()
        return cleaned

    def normalize_memory_store(self, store):
        return self._normalizer.normalize_memory_store(store)

    # ------------------------------------------------------------------ Storage I/O delegation shims
    # All persistence is owned by MemoryStorageBackend.

    def _write_json_memory_store_unlocked(self):
        return self._storage._write_json_memory_store_unlocked()

    def _write_memory_store_unlocked(self):
        return self._storage._write_memory_store_unlocked()

    def mutate_memory_store(self, mutator=None, normalize=True, save=True, **changes):
        with (
            ensure_execution_trace_root(
                operation="memory_write",
                prompt="[memory-manager-mutate]",
                metadata={"source": "MemoryManager.mutate_memory_store"},
                required=True,
            ),
            MemoryWriteOwnerScope.bind("MemoryManager"),
        ):
            return self._storage.mutate_memory_store(
                mutator=mutator,
                normalize=normalize,
                save=save,
                owner="MemoryManager",
                **changes,
            )

    def _load_memory_store(self):
        return self._storage.load_memory_store()

    def prepare_memory_store_for_save(self):
        return self._storage.prepare_memory_store_for_save()

    def save_memory_store(self):
        return self._storage.save_memory_store()

    def export_memory_store(self, export_path):
        return self._storage.export_memory_store(export_path)

    def clear_memory_store(self):
        return self._storage.clear_memory_store()

    def reminder_catalog(self, include_done=False):
        return self._lifecycle.reminder_catalog(include_done=include_done)

    def session_archive(self):
        return self._lifecycle.session_archive()

    def narrative_memories(self):
        return self._lifecycle.narrative_memories()

    def relationship_timeline(self):
        return self._lifecycle.relationship_timeline()

    def relationship_history(self, limit=60):
        return self._lifecycle.relationship_history(limit=limit)

    def persona_evolution_history(self):
        return self._lifecycle.persona_evolution_history()

    def wisdom_catalog(self):
        return self._lifecycle.wisdom_catalog()

    def life_patterns(self):
        return self._lifecycle.life_patterns()

    def pending_proactive_messages(self):
        return self._lifecycle.pending_proactive_messages()

    def queue_proactive_message(self, message, source="general"):
        return self._lifecycle.queue_proactive_message(message, source=source)

    def consume_proactive_message(self):
        return self._lifecycle.consume_proactive_message()

    def consolidated_memories(self):
        return self._lifecycle.consolidated_memories()

    def memory_graph_snapshot(self):
        return self._lifecycle.memory_graph_snapshot()

    def memory_catalog(self):
        return self._lifecycle.memory_catalog()

    def last_saved_mood(self):
        return self._lifecycle.last_saved_mood()

    def recent_mood_history(self):
        return self._lifecycle.recent_mood_history()

    def relationship_state(self):
        return self._lifecycle.relationship_state()


__all__ = ["MemoryManager"]
