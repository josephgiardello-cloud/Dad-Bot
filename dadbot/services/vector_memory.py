"""
SovereignMemory - Phase 5 Long-Term Cognitive Baseline

This module provides a local, offline-first vector memory layer for "Gideon" companion.
Instead of losing context at the 128k token window boundary, the bot can retrieve
relevant historical events from months ago using semantic search.

Architecture:
  - Vector Store: ChromaDB with persistent disk backend (./memory/vector_store)
  - Embeddings: ChromaDB's local all-MiniLM-L6-v2 default embedding function
  - Collection: sovereign_cognitive_baseline (automatically created)
  - Metadata: Indexed by turn_id, event_type, timestamp for filtered search

Key Properties:
  - Offline-first: No cloud provider dependency
  - Deterministic: Filtered by Safety Policy IR before context injection
  - Optional encryption: Can layer AES-GCM over vector store
  - Retroactive indexing: One-time sweep of relational_ledger.jsonl

Usage:
    from dadbot.services.vector_memory import SovereignMemory

    # Initialize
    memory = SovereignMemory(persist_directory="./memory/vector_store")

    # Index a new event
    memory.commit_to_long_term(
        turn_id="turn_12345",
        event_payload={"event_type": "TOOL_EXECUTION", "tool_name": "read_file", ...}
    )

    # Retrieve relevant context
    fragments = memory.retrieve_context(
        query="What did we discuss about HVAC safety?",
        limit=5,
        time_window_days=90
    )

    # Shutdown gracefully
    memory.shutdown()
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from dadbot.runtime.env_loader import is_production_environment

try:
    import chromadb
    from chromadb.config import Settings
except ModuleNotFoundError as exc:
    if exc.name != "chromadb":
        raise
    chromadb = None
    Settings = None

logger = logging.getLogger(__name__)

_QUERY_TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9]+\b")
_QUERY_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "can", "did", "do", "does",
        "earlier", "for", "from", "i", "in", "is", "it", "me", "my", "of",
        "on", "please", "remind", "the", "this", "to", "was", "what", "when",
        "where", "which", "who", "why", "you",
    },
)


def _query_term_weights(query: str) -> dict[str, float]:
    """Weight topic words above capitalized entity names during reranking."""
    weights: dict[str, float] = {}
    for match in _QUERY_TOKEN_PATTERN.finditer(query):
        token = match.group(0).lower()
        if token in _QUERY_STOP_WORDS or len(token) < 2:
            continue
        weight = 0.25 if match.group(0)[0].isupper() else 1.0
        weights[token] = max(weights.get(token, 0.0), weight)
    return weights


@dataclass
class MemoryFragment:
    """A retrieved memory fragment from the vector store."""

    event_id: str
    event_type: str
    timestamp: str
    content: str
    similarity_score: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "content": self.content,
            "similarity_score": self.similarity_score,
            "metadata": self.metadata,
        }


def _rank_memory_fragments(
    results: dict[str, Any],
    query: str,
    limit: int,
    distance_threshold: float,
) -> list[MemoryFragment]:
    query_terms = _query_term_weights(query)
    query_weight_total = sum(query_terms.values())
    ranked_fragments: list[tuple[float, float, int, MemoryFragment]] = []

    if results and results["documents"]:
        for idx, (doc, metadata, distance) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                strict=True,
            ),
        ):
            similarity_score = 1.0 / (1.0 + distance) if distance >= 0 else 0.0
            if distance > distance_threshold:
                continue

            try:
                payload = json.loads(doc)
            except (TypeError, ValueError):
                payload = {}
            summary = payload.get("summary") if isinstance(payload, dict) else None
            content_terms = {
                match.group(0).lower()
                for match in _QUERY_TOKEN_PATTERN.finditer(str(summary or doc))
            }
            lexical_coverage = (
                sum(weight for token, weight in query_terms.items() if token in content_terms)
                / query_weight_total
                if query_weight_total
                else 0.0
            )
            relevance_score = similarity_score + (0.15 * lexical_coverage)
            fragment = MemoryFragment(
                event_id=metadata.get("event_id", "?"),
                event_type=metadata.get("event_type", "UNKNOWN"),
                timestamp=metadata.get("timestamp", ""),
                content=doc,
                similarity_score=similarity_score,
                metadata={
                    key: value
                    for key, value in metadata.items()
                    if key not in {"event_id", "event_type", "timestamp"}
                },
            )
            ranked_fragments.append(
                (relevance_score, similarity_score, idx, fragment),
            )

    ranked_fragments.sort(
        key=lambda item: (item[0], item[1], -item[2]),
        reverse=True,
    )
    return [item[3] for item in ranked_fragments[:limit]]


class SovereignMemory:
    """
    Long-term cognitive memory layer for Gideon companion.

    This service manages a local vector store of indexed sovereign events,
    allowing semantic retrieval of relevant context from the bot's history
    without cloud provider dependency.
    """

    def __init__(
        self,
        persist_directory: str | Path = "./memory/vector_store",
        collection_name: str = "sovereign_cognitive_baseline",
        enable_telemetry: bool = False,
    ) -> None:
        """
        Initialize SovereignMemory service.

        Args:
            persist_directory: Path to persistent vector store directory
            collection_name: ChromaDB collection name for sovereign events
            enable_telemetry: Whether to enable ChromaDB telemetry (default: False for privacy)
        """
        if chromadb is None:
            raise ImportError(
                "chromadb is required for SovereignMemory. "
                "Install with: pip install chromadb"
            )

        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._initialized = False

        self._initialize_client(enable_telemetry)

    def _initialize_client(self, enable_telemetry: bool = False) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            # Create persistence directory if needed
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            # Initialize ChromaDB with privacy settings
            settings = Settings(
                allow_reset=True,
                anonymized_telemetry=enable_telemetry,
                is_persistent=True,
            )

            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=settings,
            )

            # Get or create collection
            # ChromaDB uses its local default embedding function and L2 distance.
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Sovereign Event Cognitive Baseline"},
            )
            self._backfill_timestamp_metadata()

            self._initialized = True
            logger.info(
                f"SovereignMemory initialized: {self.persist_directory} "
                f"[collection={self.collection_name}]"
            )

        except Exception as e:
            logger.error(f"Failed to initialize SovereignMemory: {e}")
            raise

    def commit_to_long_term(
        self,
        turn_id: str,
        event_payload: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Index a sovereign event for long-term retrieval.

        This method commits an event to the vector store, making it available
        for semantic search and context retrieval.

        Args:
            turn_id: Unique identifier for the turn
            event_payload: The event data to index (will be stringified for embedding)
            metadata: Optional metadata for indexing and filtering

        Returns:
            Document ID in the vector store
        """
        if not self._initialized or self._collection is None:
            logger.warning("SovereignMemory not initialized, skipping commit")
            return ""

        try:
            # Construct the document to be embedded
            event_type = event_payload.get("type", "UNKNOWN")
            event_id = event_payload.get("event_id", f"evt_{turn_id}")

            # Stringify the entire payload for embedding
            document = json.dumps(event_payload, default=str)

            # Generate deterministic document ID
            doc_id = self._generate_doc_id(event_id, document)

            # Prepare metadata for indexed filtering
            now = datetime.now(UTC).isoformat()
            indexed_metadata = {
                "turn_id": str(turn_id),
                "event_type": str(event_type),
                "event_id": str(event_id),
                "timestamp": now,
                "doc_id": doc_id,
            }

            # Merge with any provided metadata
            if metadata:
                indexed_metadata.update(metadata)

            # Add to collection
            # ChromaDB embeds documents with the collection's configured function.
            self._collection.add(
                documents=[document],
                metadatas=[indexed_metadata],
                ids=[doc_id],
            )

            logger.debug(f"Indexed event: {doc_id} (turn={turn_id}, type={event_type})")
            return doc_id

        except Exception as e:
            logger.error(f"Failed to commit event to long-term memory: {e}")
            raise

    def _backfill_timestamp_metadata(self, *, batch_size: int = 500) -> None:
        """Add numeric timestamps to existing records for Chroma range filters."""
        if self._collection is None:
            return

        count = int(self._collection.count())
        offset = 0
        while offset < count:
            result = self._collection.get(
                limit=min(batch_size, count - offset),
                offset=offset,
                include=["metadatas"],
            )
            ids = list(result.get("ids") or [])
            metadatas = list(result.get("metadatas") or [])
            if not ids:
                break

            update_ids: list[str] = []
            update_metadatas: list[dict[str, Any]] = []
            for record_id, raw_metadata in zip(ids, metadatas):
                metadata = dict(raw_metadata or {})
                if "timestamp_epoch" in metadata:
                    continue
                timestamp = str(metadata.get("timestamp") or "").strip()
                if not timestamp:
                    continue
                try:
                    parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError:
                    logger.warning("Skipping invalid memory timestamp during metadata migration: %s", record_id)
                    continue
                if parsed_timestamp.tzinfo is None:
                    parsed_timestamp = parsed_timestamp.replace(tzinfo=UTC)
                metadata["timestamp_epoch"] = parsed_timestamp.timestamp()
                update_ids.append(str(record_id))
                update_metadatas.append(metadata)

            if update_ids:
                self._collection.update(ids=update_ids, metadatas=update_metadatas)
            offset += len(ids)

    def retrieve_context(
        self,
        query: str,
        limit: int = 5,
        time_window_days: Optional[int] = None,
        event_type_filter: Optional[str] = None,
        distance_threshold: float = 2.0,  # ChromaDB uses L2 distance by default
    ) -> list[MemoryFragment]:
        """
        Retrieve relevant memory fragments via semantic search.

        This method performs a semantic similarity search to find events related
        to the query, optionally filtering by time window and event type.

        Args:
            query: Search query (natural language)
            limit: Maximum number of fragments to retrieve
            time_window_days: Only return events from last N days (None = no limit)
            event_type_filter: Only return events of this type (None = all types)
            distance_threshold: Maximum distance for results (lower = more similar)

        Returns:
            List of MemoryFragment objects ranked by similarity
        """
        if not self._initialized or self._collection is None:
            if is_production_environment():
                raise RuntimeError("Production vector memory is not initialized.")
            logger.warning("SovereignMemory not initialized, returning empty results")
            return []

        result_limit = int(limit)
        if result_limit < 1:
            raise ValueError("limit must be greater than zero")
        if distance_threshold < 0:
            raise ValueError("distance_threshold must not be negative")
        window_days = None
        if time_window_days is not None:
            window_days = int(time_window_days)
            if window_days < 0:
                raise ValueError("time_window_days must not be negative")
        if not str(query or "").strip():
            raise ValueError("query must not be empty")

        try:
            filters: list[dict[str, Any]] = []
            if window_days is not None:
                cutoff_timestamp = (datetime.now(UTC) - timedelta(days=window_days)).timestamp()
                filters.append({"timestamp_epoch": {"$gte": cutoff_timestamp}})

            if event_type_filter is not None:
                filters.append({"event_type": {"$eq": event_type_filter}})

            where_filter = None
            if len(filters) == 1:
                where_filter = filters[0]
            elif filters:
                where_filter = {"$and": filters}

            # Fetch a wider candidate set so lexical reranking can correct close semantic misses.
            candidate_limit = max(result_limit, min(result_limit * 5, 25))
            results = self._collection.query(
                query_texts=[query],
                n_results=candidate_limit,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            fragments = _rank_memory_fragments(
                results,
                query,
                result_limit,
                distance_threshold,
            )

            logger.debug(
                f"Retrieved {len(fragments)} fragments for query: {query[:50]}... "
                f"(window={time_window_days}d, type={event_type_filter})"
            )
            return fragments

        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            if is_production_environment():
                raise
            return []

    def delete_old_events(self, before_days: int = 365) -> int:
        """
        Delete events older than N days (maintenance operation).

        Args:
            before_days: Delete events older than this many days

        Returns:
            Number of events deleted
        """
        if not self._initialized or self._collection is None:
            if is_production_environment():
                raise RuntimeError("Production vector memory is not initialized.")
            return 0

        retention_days = int(before_days)
        if retention_days < 0:
            raise ValueError("before_days must not be negative")

        try:
            cutoff_timestamp = (datetime.now(UTC) - timedelta(days=retention_days)).timestamp()
            result = self._collection.get(where={"timestamp_epoch": {"$lt": cutoff_timestamp}})
            ids = list(result.get("ids") or [])
            if ids:
                self._collection.delete(ids=ids)
            return len(ids)

        except Exception as e:
            logger.error(f"Failed to delete old events: {e}")
            if is_production_environment():
                raise
            return 0

    def get_collection_stats(self) -> dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection metadata and event count
        """
        if not self._initialized or self._collection is None:
            return {"error": "Not initialized"}

        try:
            count = self._collection.count()
            return {
                "collection_name": self.collection_name,
                "event_count": count,
                "persist_directory": str(self.persist_directory),
                "initialized": self._initialized,
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}

    def reset_collection(self, confirm: bool = False) -> bool:
        """
        Reset (delete) the entire collection.

        WARNING: This is destructive and cannot be undone.

        Args:
            confirm: Must be True to actually perform the reset

        Returns:
            True if reset, False otherwise
        """
        if not confirm:
            logger.warning("reset_collection called without confirm=True, ignoring")
            return False

        if not self._initialized or self._client is None:
            return False

        try:
            self._client.delete_collection(name=self.collection_name)
            # Recreate the collection
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Sovereign Event Cognitive Baseline"},
            )
            logger.warning(f"Collection {self.collection_name} has been reset")
            return True
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False

    def shutdown(self) -> None:
        """Gracefully shutdown the service."""
        try:
            if self._client is not None:
                # ChromaDB doesn't require explicit shutdown, but we can log it
                logger.info("SovereignMemory shutdown")
                self._initialized = False
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    @staticmethod
    def _generate_doc_id(event_id: str, document: str) -> str:
        """Generate deterministic document ID from event and content."""
        seed = f"{event_id}:{document}"
        return hashlib.sha256(seed.encode()).hexdigest()[:24]


class NullSovereignMemory:
    """
    No-op fallback for SovereignMemory when chromadb is not installed.

    All public methods return safe empty/default values so the system
    boots and runs normally without vector-memory capability.
    """

    def commit_to_long_term(
        self,
        turn_id: str,
        event_payload: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        return ""

    def retrieve_context(
        self,
        query: str,
        limit: int = 5,
        time_window_days: Optional[int] = None,
        event_type_filter: Optional[str] = None,
        distance_threshold: float = 2.0,
    ) -> list[MemoryFragment]:
        return []

    def delete_old_events(self, before_days: int = 365) -> int:
        return 0

    def get_collection_stats(self) -> dict[str, Any]:
        return {"error": "chromadb not installed", "initialized": False}

    def reset_collection(self, confirm: bool = False) -> bool:
        return False

    def shutdown(self) -> None:
        pass


def build_sovereign_memory(
    persist_directory: str | Path = "./memory/vector_store",
    collection_name: str = "sovereign_cognitive_baseline",
    enable_telemetry: bool = False,
) -> "SovereignMemory | NullSovereignMemory":
    """
    Feature-gated factory.  Returns a fully-initialised SovereignMemory
    when chromadb is installed, or a NullSovereignMemory no-op when it is
    not.  The system will boot and operate normally in either case.
    """
    if chromadb is None:
        if is_production_environment():
            raise RuntimeError(
                "Production requires ChromaDB-backed memory; install the chromadb dependency.",
            )
        logger.warning(
            "chromadb is not installed – SovereignMemory running in no-op mode. "
            "Install with: pip install chromadb to enable vector memory."
        )
        return NullSovereignMemory()
    return SovereignMemory(
        persist_directory=persist_directory,
        collection_name=collection_name,
        enable_telemetry=enable_telemetry,
    )


# Convenience functions for singleton-like usage

_global_memory: Optional[SovereignMemory] = None


def initialize_global_memory(
    persist_directory: str | Path = "./memory/vector_store",
    collection_name: str = "sovereign_cognitive_baseline",
) -> SovereignMemory:
    """Initialize and return global SovereignMemory instance."""
    global _global_memory
    _global_memory = SovereignMemory(persist_directory, collection_name)
    return _global_memory


def get_global_memory() -> Optional[SovereignMemory]:
    """Get the current global SovereignMemory instance."""
    return _global_memory
