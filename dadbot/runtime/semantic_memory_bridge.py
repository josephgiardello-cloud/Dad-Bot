from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from dadbot.runtime.context_pruner import MemoryRecall


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _dict_rows(value: object, *, source: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise TypeError(f"{source} must be a list of dictionaries")
    return value


@dataclass(slots=True)
class MemoryIndexer:
    """Read scoped ledger events and explicitly repair the committed catalog index."""

    kernel: Any
    session_id: str = "default"
    max_events: int = 500
    tenant_id: str = field(init=False)
    _authority_required: bool = field(init=False)

    def __post_init__(self) -> None:
        from dadbot_system.delegation import live_authority_audit_context
        from dadbot_system.execution_identity import require_scoped_tenant_id

        self.tenant_id = require_scoped_tenant_id(
            _safe_str(getattr(getattr(self.kernel, "config", None), "tenant_id", "")),
        )
        self._authority_required = live_authority_audit_context()["authority_mode"] != "trusted_local"

    def _authorize(self) -> None:
        from dadbot_system.delegation import require_tenant_authority_if_active
        from dadbot_system.security import AuthorizationError

        configured_tenant = _safe_str(getattr(getattr(self.kernel, "config", None), "tenant_id", ""))
        if configured_tenant != self.tenant_id:
            raise AuthorizationError("Memory bridge runtime tenant changed")
        require_tenant_authority_if_active(self.tenant_id, required=self._authority_required)

    def _ledger_events(self) -> list[dict[str, Any]]:
        self._authorize()
        candidates = [
            getattr(self.kernel, "execution_ledger", None),
            getattr(self.kernel, "ledger", None),
            getattr(self.kernel, "_ledger", None),
        ]
        for ledger in candidates:
            if ledger is None:
                continue
            reader = getattr(ledger, "read", None)
            if callable(reader):
                rows = _dict_rows(reader(), source="Memory bridge ledger events")
                return [
                    dict(item) for item in rows
                    if _safe_str(item.get("tenant_id")) == self.tenant_id
                    and _safe_str(item.get("session_id")) == self.session_id
                ]
        return []

    def _memory_from_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(event.get("payload") or {})
        event_type = _safe_str(event.get("type") or payload.get("event_type"))
        session = _safe_str(event.get("session_id") or payload.get("session_id") or "default")
        if session != self.session_id or _safe_str(event.get("tenant_id")) != self.tenant_id:
            return None

        candidates = [
            payload.get("summary"),
            payload.get("user_input"),
            payload.get("reply"),
            payload.get("message"),
            payload.get("content"),
        ]
        summary = ""
        for value in candidates:
            text = _safe_str(value)
            if text:
                summary = text
                break
        if not summary:
            return None

        return {
            "summary": summary,
            "category": _safe_str(payload.get("category") or "general") or "general",
            "mood": _safe_str(payload.get("mood") or "neutral") or "neutral",
            "updated_at": _safe_str(
                event.get("timestamp")
                or payload.get("updated_at")
                or payload.get("occurred_at")
                or _iso_utc_now(),
            ),
            "source_event_type": event_type,
            "source_event_id": _safe_str(event.get("event_id")),
        }

    def materialize_memories(self) -> list[dict[str, Any]]:
        events = self._ledger_events()
        if not events:
            return []
        selected = events[-int(self.max_events) :]
        memories: list[dict[str, Any]] = []
        for event in selected:
            mapped = self._memory_from_event(event)
            if mapped is not None:
                memories.append(mapped)
        return memories

    def sync(self) -> int:
        self._authorize()
        catalog = getattr(self.kernel, "memory_catalog", None)
        if not callable(catalog):
            raise RuntimeError("Semantic index repair requires a committed memory catalog")
        memories = _dict_rows(catalog(), source="Memory catalog")
        sync_fn = getattr(self.kernel, "sync_semantic_memory_index", None)
        if not callable(sync_fn):
            raise RuntimeError("Semantic index repair requires sync_semantic_memory_index")
        self._authorize()
        sync_fn(memories)
        return len(memories)


class SemanticRetrievalHook:
    """Fetch top semantic memories for a query and return snippets for prompt injection."""

    def __init__(
        self,
        kernel: Any,
        *,
        top_k: int = 3,
        session_id: str = "default",
        sync_index: bool = False,
    ) -> None:
        self.kernel = kernel
        self.top_k = max(1, int(top_k))
        self.sync_index = bool(sync_index)
        self.indexer = MemoryIndexer(kernel, session_id=str(session_id or "default"))

    def __call__(self, query: str) -> list[str]:
        self.indexer._authorize()
        text = _safe_str(query)
        if not text:
            return []

        if self.sync_index:
            self.indexer.sync()

        memory_catalog = getattr(self.kernel, "memory_catalog", None)
        semantic_matches = getattr(self.kernel, "semantic_memory_matches", None)
        if callable(memory_catalog) and callable(semantic_matches):
            memories = _dict_rows(memory_catalog(), source="Memory catalog")
            if not memories:
                return []
            matches = semantic_matches(text, memories, limit=self.top_k)
            if not isinstance(matches, list):
                raise TypeError("Semantic memory matches must be a list")
            rows = [
                match[1] if isinstance(match, (tuple, list)) and len(match) == 2 else match
                for match in matches
            ]
            snippets = []
            for memory in _dict_rows(rows, source="Semantic memory matches")[: self.top_k]:
                summary = _safe_str(memory.get("summary"))
                if not summary:
                    continue
                category = _safe_str(memory.get("category") or "general")
                snippets.append(f"[{category}] {summary}")
            return snippets

        # Fallback: use freshly materialized ledger memories and lexical overlap.
        query_terms = {token for token in text.lower().split() if token}
        scored: list[tuple[int, dict[str, Any]]] = []
        for memory in self.indexer.materialize_memories():
            summary = _safe_str(memory.get("summary"))
            if not summary:
                continue
            overlap = len(query_terms.intersection(set(summary.lower().split())))
            if overlap > 0:
                scored.append((overlap, memory))
        ranked = sorted(scored, key=lambda item: item[0], reverse=True)
        return [f"[{_safe_str(m.get('category') or 'general')}] {_safe_str(m.get('summary'))}" for _s, m in ranked[: self.top_k]]


def build_semantic_snippet_provider(
    kernel: Any,
    *,
    top_k: int = 3,
    session_id: str = "default",
    sync_index: bool = False,
) -> Any:
    return SemanticRetrievalHook(
        kernel,
        top_k=top_k,
        session_id=session_id,
        sync_index=sync_index,
    )


def build_hybrid_memory_snippet_provider(
    kernel: Any,
    *,
    top_k: int = 3,
    session_id: str = "default",
    sync_index: bool = False,
) -> Any:
    """Expose the same evidence path as the structured observation provider."""
    recall = build_hybrid_memory_recall_provider(
        kernel, top_k=top_k, session_id=session_id, sync_index=sync_index,
    )

    return lambda query: recall(query).snippets


def build_hybrid_memory_recall_provider(
    kernel: Any,
    *,
    top_k: int = 3,
    session_id: str = "default",
    sync_index: bool = False,
) -> Any:
    """Return hybrid memories with bounded conflict and timestamp signals."""
    query_manager = getattr(kernel, "memory_query", None)
    retrieve = getattr(query_manager, "relevant_memories_for_input", None)
    retrieve_context = getattr(query_manager, "retrieve_context", None)
    if not callable(retrieve) and not callable(retrieve_context):
        snippets = build_semantic_snippet_provider(
            kernel,
            top_k=top_k,
            session_id=session_id,
            sync_index=sync_index,
        )

        def provide_semantic_recall(query: str) -> MemoryRecall:
            return MemoryRecall(
                snippets=list(snippets(query) or []),
                signals={
                    "conflicts": [], "temporal": [],
                    "retrieval": {"status": "degraded", "reason": "hybrid_query_unavailable"},
                },
            )

        return provide_semantic_recall

    limit = max(1, int(top_k))
    indexer = MemoryIndexer(kernel, session_id=str(session_id or "default"))

    def provide_hybrid_recall(query: str) -> MemoryRecall:
        indexer._authorize()
        if sync_index:
            indexer.sync()
        if callable(retrieve_context):
            context = retrieve_context(_safe_str(query), strategy="hybrid", limit=limit, record_access=False)
            return _recall_from_context(kernel, indexer, context, limit)

        memories = retrieve(_safe_str(query), limit=limit, record_access=False)
        memories = _dict_rows([] if memories is None else memories, source="Hybrid memory results")[:limit]

        snippets: list[str] = []
        for memory in memories:
            summary = _safe_str(memory.get("summary"))
            if summary:
                category = _safe_str(memory.get("category") or "general")[:80]
                snippets.append(f"[{category}] {summary}")
        signals = _memory_signals(kernel, memories)
        signals["retrieval"] = {"status": "degraded", "reason": "context_bundle_unavailable"}
        return MemoryRecall(snippets=snippets, signals=signals)

    return provide_hybrid_recall


def _recall_from_context(kernel: Any, indexer: MemoryIndexer, context: object, limit: int) -> MemoryRecall:
    from dadbot_system.security import AuthorizationError

    if not isinstance(context, dict) or not isinstance(context.get("retrieval_status"), dict):
        raise TypeError("Read-only memory context must expose its retrieval status")
    bundle = _dict_rows(context.get("bundle"), source="Memory context bundle")[:limit]
    memories = _dict_rows([item.get("payload") for item in bundle], source="Memory context evidence")
    indexer._authorize()
    evidence = []
    snippets = []
    for item, memory in zip(bundle, memories, strict=True):
        if memory.get("tenant_id", indexer.tenant_id) != indexer.tenant_id:
            raise AuthorizationError("Memory context evidence belongs to another tenant")
        source_id = item.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("Memory context evidence requires a source ID")
        metadata = {
            "source_id": source_id, "source_type": memory.get("source_type", item["type"]),
            "tenant_id": indexer.tenant_id,
            "authority_revision": context["retrieval_status"].get("authority_revision"),
            "provenance": memory.get("provenance"),
            "temporal": {
                name: memory[name] for name in ("event_time", "created_at", "updated_at")
                if memory.get(name)
            },
            "conflicts": list(memory.get("contradictions") or []),
            "source_metadata": memory.get("source_metadata", {}),
            "validity": memory.get("validity", []),
        }
        evidence.append(metadata)
        snippets.append(
            f"[{_safe_str(memory.get('category') or 'general')}] {_safe_str(memory.get('summary'))}\n"
            f"Evidence: {json.dumps(metadata, sort_keys=True, ensure_ascii=True)}"
        )
    signals = _memory_signals(kernel, memories, insights=context.get("consolidated_memories", []))
    signals.update(evidence=evidence, retrieval=dict(context["retrieval_status"]))
    indexer._authorize()
    return MemoryRecall(snippets=snippets, signals=signals)


def _memory_signals(
    kernel: Any, memories: list[dict[str, Any]], *, insights: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    temporal: list[dict[str, str]] = []
    timestamp_fields = ("event_time", "created_at", "updated_at", "last_reinforced_at")
    for memory in memories[:4]:
        summary = _safe_str(memory.get("summary"))
        timestamps: dict[str, str] = {}
        for name in timestamp_fields:
            value = _safe_str(memory.get(name))
            if value:
                timestamps[name] = value[:64]
        if summary and timestamps:
            temporal.append(
                {
                    "summary": summary[:240],
                    "category": _safe_str(memory.get("category") or "general")[:80],
                    **timestamps,
                },
            )

    conflicts: list[dict[str, Any]] = []
    coordinator = getattr(kernel, "memory_coordinator", None)
    detect = getattr(coordinator, "detect_memory_contradictions", None)
    if callable(detect) and memories:
        insight_reader = getattr(kernel, "consolidated_memory_catalog", None)
        insights = _dict_rows(
            insights if insights is not None else (insight_reader() if callable(insight_reader) else []),
            source="Consolidated memory catalog",
        )[-8:]
        findings = _dict_rows(detect(memories, insights), source="Memory contradictions")
        for finding in findings[:3]:
            left = _safe_str(finding.get("left"))
            right = _safe_str(finding.get("right"))
            reason = _safe_str(finding.get("reason"))
            if left and right and reason:
                conflicts.append(
                    {
                        "left": left[:240],
                        "right": right[:240],
                        "reason": reason[:160],
                    },
                )
    return {"conflicts": conflicts, "temporal": temporal}


class MemoryConsolidationJob:
    """Background consolidation job that emits MEMORY_CONSOLIDATED ledger events."""

    def __init__(self, kernel: Any, *, session_id: str = "default", window_size: int = 10) -> None:
        self.kernel = kernel
        self.session_id = str(session_id or "default")
        self.window_size = max(3, int(window_size))

    def _writer(self) -> Any:
        for name in ("ledger_writer", "_ledger_writer"):
            candidate = getattr(self.kernel, name, None)
            if callable(getattr(candidate, "write_event", None)):
                return candidate
        return None

    def _events(self) -> list[dict[str, Any]]:
        indexer = MemoryIndexer(self.kernel, session_id=self.session_id, max_events=1000)
        return indexer._ledger_events()

    def _summary_from_events(self, events: list[dict[str, Any]]) -> tuple[str, list[str]]:
        lines: list[str] = []
        ids: list[str] = []
        for event in events:
            payload = dict(event.get("payload") or {})
            ids.append(_safe_str(event.get("event_id")))
            candidate = _safe_str(payload.get("summary") or payload.get("user_input") or payload.get("reply"))
            if candidate:
                lines.append(candidate)
        unique_lines: list[str] = []
        for line in lines:
            if line not in unique_lines:
                unique_lines.append(line)
        if not unique_lines:
            return "", ids
        headline = "; ".join(unique_lines[:3])
        return f"Session discussion summary: {headline}", ids

    def run_once(self, *, force: bool = False) -> dict[str, Any]:
        writer = self._writer()
        if writer is None:
            return {"status": "unavailable", "reason": "ledger_writer_missing"}

        events = [
            event
            for event in self._events()
            if _safe_str(event.get("session_id") or "default") == self.session_id
        ]
        if len(events) < self.window_size and not force:
            return {"status": "skipped", "reason": "insufficient_events", "event_count": len(events)}

        window = events[-self.window_size :]
        summary, source_event_ids = self._summary_from_events(window)
        if not summary:
            return {"status": "skipped", "reason": "no_summary_text"}

        payload = {
            "event_type": "MemoryConsolidatedEvent",
            "schema_version": "memory-consolidation.v1",
            "summary": summary,
            "source_event_count": len(window),
            "source_event_ids": [event_id for event_id in source_event_ids if event_id],
            "occurred_at": _iso_utc_now(),
        }
        event = writer.write_event(
            "MEMORY_CONSOLIDATED",
            session_id=self.session_id,
            step_key="memory.consolidation.background",
            trace_token=f"memory-consolidation:{self.session_id}",
            payload=payload,
            committed=True,
        )
        return {
            "status": "written",
            "event_id": _safe_str(event.get("event_id")),
            "source_event_count": len(window),
            "summary": summary,
        }

    def schedule(self, *, force: bool = False) -> dict[str, Any]:
        submit = getattr(self.kernel, "submit_background_task", None)
        if not callable(submit):
            return self.run_once(force=force)
        task = submit(
            self.run_once,
            force=force,
            task_kind="memory-consolidation",
            metadata={"session_id": self.session_id, "window_size": self.window_size},
        )
        return {"status": "queued", "task_id": _safe_str(getattr(task, "dadbot_task_id", ""))}


__all__ = [
    "MemoryConsolidationJob",
    "MemoryIndexer",
    "SemanticRetrievalHook",
    "build_hybrid_memory_recall_provider",
    "build_hybrid_memory_snippet_provider",
    "build_semantic_snippet_provider",
]
