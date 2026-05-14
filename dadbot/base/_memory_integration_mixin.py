from __future__ import annotations


class MemoryIntegrationMixin:
    """Graph + normalizer delegation mixin for integration/data-quality surfaces."""

    # ------------------------------------------------------------------ Graph delegation

    def ensure_graph_store(self):
        return self._graph_manager.ensure_graph_store()

    def clear_graph_store(self):
        return self._graph_manager.clear_graph_store()

    @staticmethod
    def graph_node_key(node_type, identifier):
        from dadbot.memory.graph_manager import MemoryGraphManager

        return MemoryGraphManager.graph_node_key(node_type, identifier)

    @staticmethod
    def graph_edge_key(source_key, target_key, relation_type):
        from dadbot.memory.graph_manager import MemoryGraphManager

        return MemoryGraphManager.graph_edge_key(source_key, target_key, relation_type)

    def graph_source_id(self, source_type, entry):
        return self._graph_manager.graph_source_id(source_type, entry)

    def graph_keyword_tokens(self, text, limit=4):
        return self._graph_manager.graph_keyword_tokens(text, limit)

    @staticmethod
    def graph_slug(text):
        from dadbot.memory.graph_manager import MemoryGraphManager

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
        from dadbot.memory.graph_manager import MemoryGraphManager

        return MemoryGraphManager.upsert_graph_node(node_map, **node)

    @staticmethod
    def upsert_graph_edge(edge_map, **edge):
        from dadbot.memory.graph_manager import MemoryGraphManager

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

    # ------------------------------------------------------------------ Normalizer delegation

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
        from dadbot.memory.normalizers import MemoryNormalizer

        return MemoryNormalizer.memory_sort_key(memory)

    @staticmethod
    def infer_memory_category(summary):
        from dadbot.memory.normalizers import MemoryNormalizer

        return MemoryNormalizer.infer_memory_category(summary)

    @staticmethod
    def _coerce_memory_confidence(value, *, default=0.5):
        from dadbot.memory.normalizers import MemoryNormalizer

        return MemoryNormalizer._coerce_memory_confidence(value, default=default)

    @staticmethod
    def _coerce_impact_score(value, *, default=1.0):
        from dadbot.memory.normalizers import MemoryNormalizer

        return MemoryNormalizer._coerce_impact_score(value, default=default)

    @staticmethod
    def _coerce_unit_float(value, *, default):
        from dadbot.memory.normalizers import MemoryNormalizer

        return MemoryNormalizer._coerce_unit_float(value, default=default)

    @staticmethod
    def _normalize_memory_contradictions(value):
        from dadbot.memory.normalizers import MemoryNormalizer

        return MemoryNormalizer._normalize_memory_contradictions(value)

    @staticmethod
    def _normalize_optional_timestamp(value):
        from dadbot.memory.normalizers import MemoryNormalizer

        return MemoryNormalizer._normalize_optional_timestamp(value)

    @staticmethod
    def _normalize_memory_timestamp(value, *, fallback):
        from dadbot.memory.normalizers import MemoryNormalizer

        return MemoryNormalizer._normalize_memory_timestamp(value, fallback=fallback)

    @staticmethod
    def _coerce_persona_strength(value, *, default=1.0):
        from dadbot.memory.normalizers import MemoryNormalizer

        return MemoryNormalizer._coerce_persona_strength(value, default=default)

    @staticmethod
    def _coerce_persona_impact(value, *, default=0.0):
        from dadbot.memory.normalizers import MemoryNormalizer

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

    def normalize_memory_store(self, store):
        return self._normalizer.normalize_memory_store(store)