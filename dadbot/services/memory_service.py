from __future__ import annotations

import logging
import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from dadbot.core.execution_memory_view import merge_memory_retrieval_sets

logger = logging.getLogger(__name__)
_RECENT_RETRIEVAL_STATUSES: deque[str] = deque(maxlen=256)
_RETRIEVAL_STATUS_LOCK = threading.Lock()

MEMORY_FACT_EXTRACTION_PROMPT = """
Extract only durable, explicit personal facts stated by the user in the supplied conversation.
Return only a JSON array with at most five objects. Each object must contain exactly:
- "fact": one concise, atomic fact copied verbatim from a user message
- "category": a short topic category
- "mood": the user's expressed mood, or "neutral" when none is stated

Do not infer facts, repeat assistant claims as user facts, or include transient greetings,
unsupported conclusions, or sensitive identifiers. Do not paraphrase the user's statement.
Do not return trust scores or provenance; the application sets those from the verified
source. Return [] when there are no qualifying facts.
""".strip()
USER_MESSAGE_TRUST_LEVEL = 0.9


class ExtractedMemoryFact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    fact: str = Field(min_length=1, max_length=500)
    category: str = Field(min_length=1, max_length=40)
    mood: str = Field(min_length=1, max_length=32)

    @field_validator("fact", "category", "mood")
    @classmethod
    def require_non_whitespace(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field must contain non-whitespace text")
        return normalized


_EXTRACTED_MEMORY_FACTS = TypeAdapter(list[ExtractedMemoryFact])


class MemoryConsolidationProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    summary: str = Field(min_length=1, max_length=500)
    category: str = Field(min_length=1, max_length=40)
    source_refs: list[str] = Field(min_length=1, max_length=18)

    @field_validator("summary", "category")
    @classmethod
    def require_normalized_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or normalized != value:
            raise ValueError("Field must contain normalized non-whitespace text")
        return normalized

    @field_validator("summary")
    @classmethod
    def limit_summary_words(cls, value: str) -> str:
        if len(value.split()) > 22:
            raise ValueError("Consolidated summary must not exceed 22 words")
        return value

    @field_validator("source_refs")
    @classmethod
    def require_unique_source_refs(cls, value: list[str]) -> list[str]:
        if any(not item or item != item.strip() or len(item) > 128 for item in value):
            raise ValueError("Source references must be normalized nonempty strings")
        if len(set(value)) != len(value):
            raise ValueError("Source references must be unique")
        return value


_MEMORY_CONSOLIDATION_PROPOSALS = TypeAdapter(list[MemoryConsolidationProposal])


def validate_memory_consolidation_proposals(payload: Any) -> list[MemoryConsolidationProposal]:
    proposals = _MEMORY_CONSOLIDATION_PROPOSALS.validate_python(payload, strict=True)
    if len(proposals) > 6:
        raise ValueError("Memory consolidation exceeded the six-proposal limit")
    return proposals


class SupportsMemoryFactExtraction(Protocol):
    def call_ollama_chat(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: str,
        purpose: str,
    ) -> Any: ...

    def extract_ollama_message_content(self, response: Any) -> str: ...

    def parse_model_json_content(self, content: str) -> Any: ...


def extract_facts(
    runtime: SupportsMemoryFactExtraction,
    transcript: str,
) -> list[ExtractedMemoryFact]:
    """Extract typed facts and retain only those anchored to an exact user quote."""
    if not transcript.strip():
        return []

    response = runtime.call_ollama_chat(
        messages=[
            {"role": "system", "content": MEMORY_FACT_EXTRACTION_PROMPT},
            {"role": "user", "content": transcript},
        ],
        response_format="json",
        purpose="memory fact extraction",
    )
    content = runtime.extract_ollama_message_content(response)
    parsed = runtime.parse_model_json_content(content)
    facts = _EXTRACTED_MEMORY_FACTS.validate_python(parsed, strict=True)
    if len(facts) > 5:
        raise ValueError("Memory fact extraction exceeded the five-fact limit")

    supported = [fact for fact in facts if fact.fact in transcript]
    unsupported_count = len(facts) - len(supported)
    if unsupported_count:
        logger.warning(
            "Memory extraction rejected facts without exact source evidence (count=%d)",
            unsupported_count,
        )
    return supported


@dataclass(slots=True)
class MemoryRetrievalOutcome:
    memories: list[dict[str, Any]]
    status: str = "ok"
    errors: list[dict[str, str]] = field(default_factory=list)


class MemoryRetrievalError(RuntimeError):
    """Raised when a result-only retrieval API cannot represent degradation."""

    def __init__(self, outcome: MemoryRetrievalOutcome) -> None:
        self.status = outcome.status
        self.errors = list(outcome.errors)
        super().__init__(f"Memory retrieval did not complete successfully (status={outcome.status})")


class SupportsMemoryRetrieval(Protocol):
    def graph_retrieval_for_input(self, query: str, limit: int = 3) -> Any: ...

    def relevant_memories_for_input(
        self,
        query: str,
        limit: int = 5,
        graph_result: Any | None = None,
        *,
        from_date: str | None = None,
        to_date: str | None = None,
        record_access: bool = True,
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
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict[str, Any]]:
        outcome = self.retrieve_for_query_with_status(
            query=query,
            graph_limit=graph_limit,
            memory_limit=memory_limit,
            from_date=from_date,
            to_date=to_date,
        )
        if outcome.status != "ok":
            raise MemoryRetrievalError(outcome)
        return outcome.memories

    def retrieve_for_query_with_status(
        self,
        *,
        query: str,
        graph_limit: int = 3,
        memory_limit: int = 5,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> MemoryRetrievalOutcome:
        graph_result = None
        errors: list[dict[str, str]] = []
        try:
            graph_result = self.runtime.graph_retrieval_for_input(query, limit=graph_limit)
        except Exception as exc:
            logger.warning(
                "Memory graph retrieval failed (error_type=%s)",
                type(exc).__name__,
            )
            errors.append({"source": "graph", "error_type": type(exc).__name__})
            graph_result = None

        try:
            options: dict[str, Any] = {"limit": memory_limit, "graph_result": graph_result}
            if from_date is not None:
                options["from_date"] = from_date
            if to_date is not None:
                options["to_date"] = to_date
            memories = list(self.runtime.relevant_memories_for_input(query, **options) or [])
        except ValueError:
            raise
        except Exception as exc:
            logger.warning(
                "Memory retrieval failed (error_type=%s)",
                type(exc).__name__,
            )
            errors.append({"source": "memory_catalog", "error_type": type(exc).__name__})
            memories = []

        status = "degraded" if any(error["source"] == "memory_catalog" for error in errors) else (
            "partial" if errors else "ok"
        )
        return MemoryRetrievalOutcome(memories=memories, status=status, errors=errors)

    @staticmethod
    def record_retrieval_status(
        state: Any,
        *,
        status: str,
        errors: list[dict[str, str]],
        source: str,
    ) -> None:
        if not isinstance(state, dict):
            return
        current = dict(state.get("memory_retrieval_status") or {})
        severity = {"ok": 0, "partial": 1, "degraded": 2}
        current_status = str(current.get("status") or "ok")
        requested_status = str(status or "ok")
        if severity.get(requested_status, 2) > severity.get(current_status, 0):
            current_status = requested_status
        recorded_errors = list(current.get("errors") or [])
        for error in errors:
            recorded_errors.append(
                {
                    "source": f"{source}:{str(error.get('source') or 'unknown')}",
                    "error_type": str(error.get("error_type") or "UnknownError"),
                }
            )
        state["memory_retrieval_status"] = {
            "status": current_status,
            "errors": recorded_errors,
            "sources": list(
                dict.fromkeys(
                    [
                        *list(current.get("sources") or []),
                        source,
                    ]
                )
            ),
        }
        with _RETRIEVAL_STATUS_LOCK:
            _RECENT_RETRIEVAL_STATUSES.append(current_status)

    @staticmethod
    def recent_retrieval_status_counts(*, limit: int = 100) -> dict[str, int]:
        safe_limit = max(1, min(int(limit), _RECENT_RETRIEVAL_STATUSES.maxlen or 256))
        with _RETRIEVAL_STATUS_LOCK:
            statuses = list(_RECENT_RETRIEVAL_STATUSES)[-safe_limit:]
        counts = Counter(statuses)
        return {
            "window_size": len(statuses),
            "ok": counts.get("ok", 0),
            "partial": counts.get("partial", 0),
            "degraded": counts.get("degraded", 0),
        }

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

    @staticmethod
    def fetch_semantic_candidates(
        index: Any,
        *,
        query_embedding: Any,
        query_tokens: Any,
        query_category: str,
        query_mood: str,
        limit: int,
        state: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch candidates from a semantic index, isolated from caller error handling."""
        try:
            return list(
                index.fetch_candidates(
                    query_embedding=query_embedding,
                    query_tokens=query_tokens,
                    query_category=query_category,
                    query_mood=query_mood,
                    limit=limit,
                )
                or []
            )
        except Exception as exc:
            logger.warning(
                "Semantic memory index retrieval failed (error_type=%s)",
                type(exc).__name__,
            )
            MemoryService.record_retrieval_status(
                state,
                status="degraded",
                errors=[{"source": "semantic_index", "error_type": type(exc).__name__}],
                source="semantic_rag",
            )
            return []
