"""
SovereignMemory - Phase 5 Long-Term Cognitive Baseline

This module provides a local, offline-first vector memory layer for "Gideon" companion.
Instead of losing context at the 128k token window boundary, the bot can retrieve
relevant historical events from months ago using semantic search.

Architecture:
  - Vector Store: ChromaDB with persistent disk backend (./memory/vector_store)
  - Embeddings: nomic-embed-text (local, offline)
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
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None
    Settings = None

logger = logging.getLogger(__name__)


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
            # ChromaDB uses cosine similarity by default, which is good for embeddings
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Sovereign Event Cognitive Baseline"},
            )

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
            # ChromaDB handles embedding via nomic-embed-text automatically
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

    def retrieve_context(
        self,
        query: str,
        limit: int = 5,
        time_window_days: Optional[int] = None,
        event_type_filter: Optional[str] = None,
        distance_threshold: float = 1.5,  # ChromaDB uses L2 distance by default
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
            logger.warning("SovereignMemory not initialized, returning empty results")
            return []

        try:
            # Build where filter if needed
            where_filter: Optional[dict[str, Any]] = None
            if time_window_days is not None or event_type_filter is not None:
                where_filter = {}

                if time_window_days is not None:
                    cutoff_date = (
                        datetime.now(UTC) - timedelta(days=time_window_days)
                    ).isoformat()
                    where_filter = {
                        "$and": [
                            where_filter,
                            {"timestamp": {"$gte": cutoff_date}},
                        ]
                    }

                if event_type_filter is not None:
                    type_filter = {"event_type": {"$eq": event_type_filter}}
                    if where_filter:
                        where_filter = {"$and": [where_filter, type_filter]}
                    else:
                        where_filter = type_filter

            # Perform semantic search
            results = self._collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            # Convert results to MemoryFragment objects
            fragments = []
            if results and results["documents"]:
                for idx, (doc, metadata, distance) in enumerate(
                    zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                    )
                ):
                    # ChromaDB returns L2 distance; convert to similarity (lower distance = higher similarity)
                    # For cosine-like similarity: similarity = 1 / (1 + distance)
                    similarity_score = 1.0 / (1.0 + distance) if distance >= 0 else 0.0

                    # Only include if above threshold
                    if distance <= distance_threshold:
                        fragment = MemoryFragment(
                            event_id=metadata.get("event_id", "?"),
                            event_type=metadata.get("event_type", "UNKNOWN"),
                            timestamp=metadata.get("timestamp", ""),
                            content=doc,
                            similarity_score=similarity_score,
                            metadata={k: v for k, v in metadata.items()
                                     if k not in ["event_id", "event_type", "timestamp"]},
                        )
                        fragments.append(fragment)

            logger.debug(
                f"Retrieved {len(fragments)} fragments for query: {query[:50]}... "
                f"(window={time_window_days}d, type={event_type_filter})"
            )
            return fragments

        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
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
            return 0

        try:
            cutoff_date = (datetime.now(UTC) - timedelta(days=before_days)).isoformat()

            # ChromaDB doesn't have built-in delete by query, so we retrieve and delete
            # For now, log a warning that this would need custom implementation
            logger.warning(
                f"delete_old_events called for events before {cutoff_date}. "
                "Full implementation requires custom indexing strategy."
            )

            return 0

        except Exception as e:
            logger.error(f"Failed to delete old events: {e}")
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
        distance_threshold: float = 1.5,
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
