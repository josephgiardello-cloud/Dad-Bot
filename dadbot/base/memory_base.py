"""Neutral memory base contracts and mixin shells.

Step A only: this module declares interfaces and empty mixins for
MemoryManager decomposition. It intentionally contains no behavioral logic.
"""
from __future__ import annotations

from typing import Any, Protocol

from dadbot.base._memory_integration_mixin import MemoryIntegrationMixin
from dadbot.base._memory_lifecycle_mixin import MemoryLifecycleMixin
from dadbot.base._memory_search_mixin import MemorySearchMixin


class GraphManagerProtocol(Protocol):
    """Contract for memory graph projection and retrieval services."""

    def ensure_graph_store(self) -> None: ...

    def clear_graph_store(self) -> None: ...

    @staticmethod
    def graph_node_key(node_type: Any, identifier: Any) -> str: ...

    @staticmethod
    def graph_edge_key(source_key: Any, target_key: Any, relation_type: Any) -> str: ...

    @staticmethod
    def graph_slug(text: Any) -> str: ...

    def graph_source_id(self, source_type: Any, entry: dict[str, Any]) -> str: ...

    def graph_keyword_tokens(self, text: Any, limit: int = 4) -> list[str]: ...

    def canonical_graph_entity(
        self,
        label: Any,
        semantic_type: Any | None = None,
    ) -> tuple[str, str]: ...

    def normalize_relation_type(self, relation_type: Any) -> str: ...

    def relation_type_for_entity(
        self,
        source_type: Any,
        entity_type: Any,
        text: Any,
        fallback: str = "mentions",
    ) -> str: ...

    def extract_typed_graph_facts(
        self,
        text: Any,
        source_type: Any,
        *,
        default_category: str = "general",
        default_mood: str = "neutral",
        default_day: str = "",
    ) -> list[dict[str, Any]]: ...

    def llm_typed_graph_facts(self, text: Any, source_type: Any) -> list[dict[str, Any]]: ...

    def graph_source_confidence(self, source_type: Any, entry: dict[str, Any]) -> float: ...

    def graph_source_weight(self, source_type: Any, entry: dict[str, Any]) -> float: ...

    @staticmethod
    def upsert_graph_node(node_map: dict[str, Any], **node: Any) -> Any: ...

    @staticmethod
    def upsert_graph_edge(edge_map: dict[str, Any], **edge: Any) -> Any: ...

    def build_graph_projection(self) -> Any: ...

    def preview_memory_graph(self, snapshot: Any | None = None) -> Any: ...

    def sync_graph_store(self) -> Any: ...

    def graph_snapshot(self) -> Any: ...

    def graph_source_summary(self, node: Any) -> Any: ...

    def graph_source_lines(self, selected_items: Any) -> Any: ...

    def graph_retrieval_for_input(self, query: Any, limit: int = 3) -> Any: ...

    def build_graph_summary_context(self, limit: int = 3) -> Any: ...


class MemoryLifecycleProtocol(Protocol):
    """Lifecycle/storage contract marker for facade thinning."""


class MemorySearchProtocol(Protocol):
    """Semantic/search contract marker for facade thinning."""


class MemoryIntegrationProtocol(Protocol):
    """Normalizer/graph contract marker for facade thinning."""


__all__ = [
    "GraphManagerProtocol",
    "MemoryIntegrationMixin",
    "MemoryIntegrationProtocol",
    "MemoryLifecycleMixin",
    "MemoryLifecycleProtocol",
    "MemorySearchMixin",
    "MemorySearchProtocol",
]
