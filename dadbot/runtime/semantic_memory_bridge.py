from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


@dataclass(slots=True)
class MemoryIndexer:
    """Bridge execution ledger events into the existing semantic index backend."""

    kernel: Any
    session_id: str = "default"
    max_events: int = 500

    def _ledger_events(self) -> list[dict[str, Any]]:
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
                rows = reader()
                if isinstance(rows, list):
                    return [dict(item) for item in rows if isinstance(item, dict)]
        return []

    def _memory_from_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(event.get("payload") or {})
        event_type = _safe_str(event.get("type") or payload.get("event_type"))
        session = _safe_str(event.get("session_id") or payload.get("session_id") or "default")
        if session != _safe_str(self.session_id or "default"):
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
        memories = self.materialize_memories()
        if not memories:
            return 0
        sync_fn = getattr(self.kernel, "sync_semantic_memory_index", None)
        if callable(sync_fn):
            sync_fn(memories)
            return len(memories)
        return 0


class SemanticRetrievalHook:
    """Fetch top semantic memories for a query and return snippets for prompt injection."""

    def __init__(
        self,
        kernel: Any,
        *,
        top_k: int = 3,
        session_id: str = "default",
        sync_index: bool = True,
    ) -> None:
        self.kernel = kernel
        self.top_k = max(1, int(top_k))
        self.sync_index = bool(sync_index)
        self.indexer = MemoryIndexer(kernel, session_id=str(session_id or "default"))

    def __call__(self, query: str) -> list[str]:
        text = _safe_str(query)
        if not text:
            return []

        if self.sync_index:
            try:
                self.indexer.sync()
            except Exception:
                pass

        memory_catalog = getattr(self.kernel, "memory_catalog", None)
        semantic_matches = getattr(self.kernel, "semantic_memory_matches", None)
        if callable(memory_catalog) and callable(semantic_matches):
            memories = list(memory_catalog() or [])
            if not memories:
                return []
            matches = list(semantic_matches(text, memories, limit=self.top_k) or [])
            snippets = []
            for memory in matches[: self.top_k]:
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
    sync_index: bool = True,
) -> Any:
    return SemanticRetrievalHook(
        kernel,
        top_k=top_k,
        session_id=session_id,
        sync_index=sync_index,
    )


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
    "MemoryIndexer",
    "SemanticRetrievalHook",
    "build_semantic_snippet_provider",
    "MemoryConsolidationJob",
]
