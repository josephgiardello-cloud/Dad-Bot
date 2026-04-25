from __future__ import annotations

from datetime import datetime

from dadbot.agentic import AgenticHandler as PackagedAgenticHandler, ToolRegistry as PackagedToolRegistry
from dadbot.constants import PERSONA_PRESETS
from dadbot.defaults import (
    default_memory_store as _default_memory_store,
    default_relationship_state as _default_relationship_state,
    relationship_hypothesis_profiles as _relationship_hypothesis_profiles,
)
from dadbot.managers.internal_state import InternalStateManager as PackagedInternalStateManager
from dadbot.managers.memory_commands import MemoryCommandManager as PackagedMemoryCommandManager
from dadbot.managers.memory_coordination import MemoryCoordinator as PackagedMemoryCoordinator
from dadbot.managers.memory_query import MemoryQueryManager as PackagedMemoryQueryManager
from dadbot.managers.multimodal import MultimodalManager as PackagedMultimodalManager
from dadbot.managers.runtime_storage import RuntimeStorageManager as PackagedRuntimeStorageManager
from dadbot.memory.manager import MemoryManager as PackagedMemoryManager
from dadbot.profile import ProfileContextManager as PackagedProfileContextManager
from dadbot.relationship import RelationshipManager as PackagedRelationshipManager
from dadbot.utils import (
    normalize_memory_text as cached_normalize_memory_text,
    significant_tokens as cached_significant_tokens,
    tokenize_text,
)
from dadbot.utils.graph import (
    accumulate_memory_graph_edge as _util_accumulate_memory_graph_edge,
    accumulate_memory_graph_node as _util_accumulate_memory_graph_node,
    build_memory_graph_edges as _util_build_memory_graph_edges,
    build_memory_graph_nodes as _util_build_memory_graph_nodes,
    pattern_identity as _util_pattern_identity,
    persona_announcement as _util_persona_announcement,
)
from dadbot.utils.llm import (
    extract_ollama_message_content as _util_extract_ollama_message_content,
    extract_ollama_message_payload as _util_extract_ollama_message_payload,
    parse_model_json_content as _util_parse_model_json_content,
    transcript_from_messages as _util_transcript_from_messages,
)
from dadbot.utils.memory import (
    memory_sort_key as _util_memory_sort_key,
    naturalize_memory_summary as _util_naturalize_memory_summary,
)
from dadbot.utils.mood import (
    build_style_examples as _util_build_style_examples,
    command_help_text as _util_command_help_text,
    normalize_mood as _util_normalize_mood,
    normalize_mood_detection_key as _util_normalize_mood_detection_key,
)
from dadbot.utils.relationship import (
    clamp_score as _util_clamp_score,
    confidence_label as _util_confidence_label,
    days_since_iso_date as _util_days_since_iso_date,
    decay_relationship_level as _util_decay_relationship_level,
    normalize_confidence as _util_normalize_confidence,
    parse_iso_date as _util_parse_iso_date,
    recency_weight as _util_recency_weight,
)


class DadBotCompatMixin:
    """Compatibility helpers that do not own orchestration or mutable runtime state."""

    def clear_memory_store(self):
        return self.memory.clear_memory_store()

    def export_memory_store(self, export_path):
        return self.memory.export_memory_store(export_path)

    @staticmethod
    def json_backup_path(destination):
        return PackagedRuntimeStorageManager.json_backup_path(destination)

    @staticmethod
    def corrupt_json_snapshot_path(destination):
        return PackagedRuntimeStorageManager.corrupt_json_snapshot_path(destination)

    @staticmethod
    def extract_ollama_message_payload(response):
        return _util_extract_ollama_message_payload(response)

    @staticmethod
    def extract_ollama_message_content(response):
        return _util_extract_ollama_message_content(response)

    @staticmethod
    def normalize_mood_detection_key(user_input, recent_history=None):
        return _util_normalize_mood_detection_key(user_input, recent_history)

    @staticmethod
    def clamp_score(value, minimum=0, maximum=100):
        return _util_clamp_score(value, minimum, maximum)

    @staticmethod
    def parse_iso_date(value):
        return _util_parse_iso_date(value)

    def days_since_iso_date(self, value):
        return _util_days_since_iso_date(value)

    def decay_relationship_level(self, score, last_updated, midpoint=50, daily_decay=0.985):
        return _util_decay_relationship_level(score, last_updated, midpoint, daily_decay)

    def recency_weight(self, value, freshest=4):
        return _util_recency_weight(value, freshest)

    def default_relationship_state(self):
        return _default_relationship_state()

    @staticmethod
    def relationship_hypothesis_profiles():
        return _relationship_hypothesis_profiles()

    def default_relationship_hypotheses(self):
        from dadbot.defaults import default_relationship_hypotheses
        return default_relationship_hypotheses()

    def default_memory_store(self):
        return _default_memory_store()

    @staticmethod
    def default_internal_state():
        return PackagedInternalStateManager.default_state()

    @staticmethod
    def default_memory_graph():
        return {
            "nodes": [],
            "edges": [],
            "updated_at": None,
        }

    @staticmethod
    def normalize_confidence(value, source_count=1, contradiction_count=0, updated_at=None):
        return _util_normalize_confidence(value, source_count, contradiction_count, updated_at)

    @staticmethod
    def confidence_label(confidence):
        return _util_confidence_label(confidence)

    @staticmethod
    def transcript_from_messages(messages):
        return _util_transcript_from_messages(messages)

    @staticmethod
    def parse_model_json_content(content):
        return _util_parse_model_json_content(content)

    @staticmethod
    def persona_preset_catalog():
        return PERSONA_PRESETS

    def runtime_timestamp(self):
        temporal = getattr(self, "_current_turn_time_base", None)
        turn_started_at = getattr(temporal, "turn_started_at", None)
        if turn_started_at:
            return str(turn_started_at)
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def snapshot_memory_entries(memories):
        return PackagedMemoryManager.snapshot_memory_entries(memories)

    @staticmethod
    def extract_embeddings_from_response(response):
        if isinstance(response, dict):
            if response.get("embeddings") is not None:
                return response["embeddings"]
            if response.get("embedding") is not None:
                return [response["embedding"]]

        embeddings = getattr(response, "embeddings", None)
        if embeddings is not None:
            return embeddings

        embedding = getattr(response, "embedding", None)
        if embedding is not None:
            return [embedding]

        return None

    @classmethod
    def extract_embedding(cls, response):
        embeddings = cls.extract_embeddings_from_response(response)
        if embeddings is None:
            return None

        if embeddings and isinstance(embeddings[0], (int, float)):
            return embeddings

        if isinstance(embeddings, list) and embeddings:
            return embeddings[0]

        return None

    @staticmethod
    def cosine_similarity(left, right):
        if not left or not right or len(left) != len(right):
            return 0.0

        dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
        left_norm = sum(value * value for value in left) ** 0.5
        right_norm = sum(value * value for value in right) ** 0.5
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot_product / (left_norm * right_norm)

    @staticmethod
    def normalize_chat_attachment(attachment):
        return PackagedMultimodalManager.normalize_chat_attachment(attachment)

    @staticmethod
    def history_attachment_metadata(attachment):
        return PackagedMultimodalManager.history_attachment_metadata(attachment)

    @staticmethod
    def semantic_memory_filters(query_tokens, query_category, query_mood):
        return PackagedMemoryManager.semantic_memory_filters(query_tokens, query_category, query_mood)

    @staticmethod
    def command_help_text():
        return _util_command_help_text()

    @staticmethod
    def reminder_has_date_signal(detail):
        return PackagedToolRegistry.reminder_has_date_signal(detail)

    @staticmethod
    def normalize_relative_reminder_phrase(detail, reference):
        return PackagedToolRegistry.normalize_relative_reminder_phrase(detail, reference)

    @classmethod
    def split_reminder_details(cls, detail):
        return PackagedToolRegistry.split_reminder_details(detail)

    @staticmethod
    def extract_related_topic_results(related_topics):
        return PackagedAgenticHandler.extract_related_topic_results(related_topics)

    @staticmethod
    def normalize_lookup_query(user_input):
        return PackagedToolRegistry.normalize_lookup_query(user_input)

    @staticmethod
    def persona_announcement(trait, reason):
        return _util_persona_announcement(trait, reason)

    @staticmethod
    def pattern_identity(pattern):
        return _util_pattern_identity(pattern)

    @staticmethod
    def accumulate_memory_graph_node(node_weights, node_types, label, node_type, weight=1):
        return _util_accumulate_memory_graph_node(node_weights, node_types, label, node_type, weight=weight)

    @staticmethod
    def accumulate_memory_graph_edge(edge_weights, left, right, weight=1):
        return _util_accumulate_memory_graph_edge(edge_weights, left, right, weight=weight)

    @staticmethod
    def build_memory_graph_nodes(node_weights, node_types):
        return _util_build_memory_graph_nodes(node_weights, node_types)

    @staticmethod
    def build_memory_graph_edges(edge_weights):
        return _util_build_memory_graph_edges(edge_weights)

    @staticmethod
    def ordinal(day):
        return PackagedProfileContextManager.ordinal(day)

    @staticmethod
    def natural_list(items):
        return PackagedProfileContextManager.natural_list(items)

    @staticmethod
    def normalize_memory_text(text):
        return cached_normalize_memory_text(text)

    @staticmethod
    def tokenize(text):
        return tokenize_text(text)

    @staticmethod
    def normalize_mood(mood):
        return _util_normalize_mood(mood)

    @staticmethod
    def naturalize_memory_summary(summary):
        return _util_naturalize_memory_summary(summary)

    @staticmethod
    def memory_sort_key(memory):
        return _util_memory_sort_key(memory)

    @staticmethod
    def infer_memory_category(summary):
        return PackagedMemoryManager.infer_memory_category(summary)

    @staticmethod
    def relationship_level_label(score):
        return PackagedRelationshipManager.level_label(score)

    @staticmethod
    def build_style_examples():
        return _util_build_style_examples()

    @staticmethod
    def memory_impact_score(memory):
        return PackagedMemoryQueryManager.memory_impact_score(memory)

    @staticmethod
    def format_memories_for_reply(memories):
        return PackagedMemoryQueryManager.format_memories_for_reply(memories)

    @staticmethod
    def parse_memory_command(user_input):
        return PackagedMemoryCommandManager.parse_memory_command(user_input)

    @staticmethod
    def memory_extraction_prompt():
        return PackagedMemoryCoordinator.memory_extraction_prompt()

    @staticmethod
    def significant_tokens(text):
        return cached_significant_tokens(text)

    @staticmethod
    def is_session_exit_command(stripped_input: str) -> bool:
        return stripped_input.lower() in ["bye", "goodnight", "exit", "quit"]
