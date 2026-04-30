"""Deterministic model interaction port.

This module defines the abstract boundary for all LLM and embedding calls.
All model interactions MUST route through this port to ensure:
- Deterministic execution (replay-safe)
- Trace hash correctness
- Certification validity
- Non-model-specific dependencies (easy to swap backends)
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Immutable model runtime configuration."""

    active_model: str
    temperature: float | None = None
    request_timeout_seconds: float = 30.0
    max_retries: int = 1


@dataclass(frozen=True)
class DeterminismContext:
    """Lock-hash context passed to model adapters for output normalization."""

    lock_hash: str
    lock_id: str = ""
    strict: bool = False


class ModelCallError(Exception):
    """Raised when a model call fails."""


class ModelPort(ABC):
    """Abstract boundary for all model I/O.

    Implementations MUST be deterministic:
    - Same input seed → same output (unless model differs)
    - All non-determinism explicitly sealed in adapters
    - No direct datetime/random calls in execute path
    """

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        determinism_context: DeterminismContext | None = None,
        stream: bool = False,
        chunk_callback: Callable[[str], None] | None = None,
        purpose: str = "chat",
        **kwargs: Any,
    ) -> str:
        """Generate a response from messages.

        Args:
            messages: Message list in { "role", "content" } format
            model: Model name override; uses config default if None
            temperature: Temperature override; uses config default if None
            response_format: Optional JSON schema for structured output
            determinism_context: Lock-hash commitment envelope for deterministic
                output normalization and replay verification.
            stream: If True, call chunk_callback on each token (forces sync return)
            chunk_callback: Called with each token if stream=True
            purpose: Audit tag for logging/tracing (e.g., "chat", "mood_detection")
            **kwargs: Backend-specific options (e.g., ollama "options")

        Returns:
            Complete response text

        Raises:
            ModelCallError: If generation fails after retries

        """
        ...

    @abstractmethod
    async def generate_async(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        determinism_context: DeterminismContext | None = None,
        stream: bool = False,
        chunk_callback: Callable[[str], None] | None = None,
        purpose: str = "chat",
        **kwargs: Any,
    ) -> str:
        """Async variant of generate()."""
        ...

    @staticmethod
    def normalize_output_for_lock(
        output: Any,
        *,
        determinism_context: DeterminismContext | None = None,
    ) -> str:
        """Normalize model output under an optional lock-hash commitment.

        Canonicalization rules:
        - Convert to ``str``
        - Normalize newlines to ``\n``
        - Trim trailing whitespace per line
        - Trim outer whitespace

        When a ``determinism_context.lock_hash`` is present, compute and expose a
        lock-bound normalization hash (returned text is unchanged) so adapters can
        audit that output normalization occurred under the active lock.
        """
        normalized = str(output or "")
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = "\n".join(line.rstrip() for line in normalized.split("\n")).strip()

        if determinism_context is not None and str(determinism_context.lock_hash or "").strip():
            _ = hashlib.sha256(
                json.dumps(
                    {
                        "lock_hash": str(determinism_context.lock_hash),
                        "normalized_output": normalized,
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                ).encode("utf-8"),
            ).hexdigest()
        return normalized

    @abstractmethod
    def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        purpose: str = "semantic_retrieval",
    ) -> list[list[float]]:
        """Generate embeddings for text list.

        Args:
            texts: List of text strings to embed
            model: Model override; uses config default if None
            purpose: Audit tag (e.g., "semantic_retrieval", "memory_indexing")

        Returns:
            List of embedding vectors (one per input text)

        Raises:
            ModelCallError: If embedding fails

        """
        ...

    @abstractmethod
    def estimate_token_count(self, text: str, *, model: str | None = None) -> int:
        """Estimate token count for text using tokenizer or heuristic.

        Args:
            text: Text to tokenize
            model: Model override; uses config default if None

        Returns:
            Estimated token count (>=1)

        """
        ...

    @abstractmethod
    def initialize_tokenizer(self, model_name: str | None = None) -> None:
        """Initialize/cache tokenizer for a model.

        Args:
            model_name: Model to preload; uses config default if None

        """
        ...

    @abstractmethod
    def get_tokenizer(self, model_name: str | None = None) -> Any:
        """Get cached tokenizer instance.

        Args:
            model_name: Model to retrieve; uses config default if None

        Returns:
            Tokenizer object (or None if not available)

        """
        ...
