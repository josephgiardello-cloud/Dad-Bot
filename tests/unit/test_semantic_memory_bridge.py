from __future__ import annotations

from types import SimpleNamespace

import pytest

from dadbot.runtime.semantic_memory_bridge import (
    MemoryConsolidationJob,
    MemoryIndexer,
    SemanticRetrievalHook,
    build_hybrid_memory_recall_provider,
)

pytestmark = pytest.mark.unit


class _FakeLedger:
    def __init__(self, events):
        self._events = list(events)

    def read(self):
        return list(self._events)


class _FakeWriter:
    def __init__(self) -> None:
        self.calls = []

    def write_event(self, event_type: str, **kwargs):
        self.calls.append((event_type, kwargs))
        return {"event_id": "evt-1", "type": event_type}


class _Kernel:
    def __init__(self, events):
        self.execution_ledger = _FakeLedger(events)
        self.synced = None
        self.ledger_writer = _FakeWriter()

    def sync_semantic_memory_index(self, memories):
        self.synced = list(memories)


def _event(event_id: str, session_id: str, payload: dict, event_type: str = "JOB_COMPLETED"):
    return {
        "event_id": event_id,
        "session_id": session_id,
        "type": event_type,
        "payload": payload,
        "timestamp": "2026-05-12T00:00:00Z",
    }


def test_memory_indexer_materializes_and_syncs():
    kernel = _Kernel(
        [
            _event("1", "s1", {"summary": "User allergic to peanuts", "category": "health", "mood": "calm"}),
            _event("2", "s2", {"summary": "ignore me"}),
            _event("3", "s1", {"reply": "Discussed tomatoes"}),
        ]
    )

    indexer = MemoryIndexer(kernel, session_id="s1")
    memories = indexer.materialize_memories()

    assert len(memories) == 2
    assert memories[0]["summary"] == "User allergic to peanuts"
    count = indexer.sync()
    assert count == 2
    assert isinstance(kernel.synced, list)
    assert len(kernel.synced) == 2


def test_semantic_retrieval_hook_uses_kernel_search():
    kernel = _Kernel([])
    kernel.memory_catalog = lambda: [
        {"summary": "Allergic to peanuts", "category": "health"},
        {"summary": "Plant tomatoes in spring", "category": "garden"},
    ]
    kernel.semantic_memory_matches = lambda q, memories, limit=3: memories[:limit]

    hook = SemanticRetrievalHook(kernel, top_k=2, sync_index=False)
    snippets = hook("What should I eat?")

    assert len(snippets) == 2
    assert snippets[0].startswith("[health]")


def test_memory_consolidation_job_writes_event():
    events = [
        _event(str(i), "s1", {"summary": f"topic {i}"}, event_type="JOB_COMPLETED")
        for i in range(1, 12)
    ]
    kernel = _Kernel(events)
    job = MemoryConsolidationJob(kernel, session_id="s1", window_size=10)

    result = job.run_once(force=False)

    assert result["status"] == "written"
    assert kernel.ledger_writer.calls
    event_type, kwargs = kernel.ledger_writer.calls[0]
    assert event_type == "MEMORY_CONSOLIDATED"
    assert kwargs["session_id"] == "s1"
    assert kwargs["payload"]["event_type"] == "MemoryConsolidatedEvent"


def test_memory_consolidation_job_skips_when_not_enough_events():
    events = [_event("1", "s1", {"summary": "small"})]
    kernel = _Kernel(events)
    job = MemoryConsolidationJob(kernel, session_id="s1", window_size=10)

    result = job.run_once(force=False)

    assert result["status"] == "skipped"
    assert result["reason"] == "insufficient_events"


def test_hybrid_recall_includes_conflict_and_temporal_signals():
    memories = [
        {
            "summary": "Lives in Seattle",
            "category": "x" * 100,
            "event_time": "2023-01-01",
            "updated_at": "2023-01-02T00:00:00Z",
        },
        {
            "summary": "Lives in Boston",
            "category": "x" * 100,
            "event_time": "t" * 100,
            "updated_at": "2025-01-02T00:00:00Z",
        },
    ]
    kernel = _Kernel([])
    kernel.memory_query = SimpleNamespace(
        relevant_memories_for_input=lambda _query, limit=3: memories[:limit],
    )
    kernel.memory_coordinator = SimpleNamespace(
        detect_memory_contradictions=lambda rows, _insights: [
            {
                "left": rows[0]["summary"],
                "right": rows[1]["summary"],
                "reason": "opposing locations",
            },
        ],
    )
    kernel.consolidated_memories = lambda: []

    provider = build_hybrid_memory_recall_provider(kernel, sync_index=False)
    recall = provider("where does this person live")

    assert recall.snippets == [
        f"[{'x' * 80}] Lives in Seattle",
        f"[{'x' * 80}] Lives in Boston",
    ]
    assert recall.signals["conflicts"] == [
        {
            "left": "Lives in Seattle",
            "right": "Lives in Boston",
            "reason": "opposing locations",
        }
    ]
    assert recall.signals["temporal"][1]["event_time"] == "t" * 64


def test_hybrid_recall_falls_back_to_semantic_snippets():
    kernel = _Kernel([])
    kernel.memory_catalog = lambda: [{"summary": "Likes tea", "category": "food"}]
    kernel.semantic_memory_matches = lambda _query, memories, limit=3: memories[:limit]

    recall = build_hybrid_memory_recall_provider(kernel, sync_index=False)("tea")

    assert recall.snippets == ["[food] Likes tea"]
    assert recall.signals == {"conflicts": [], "temporal": []}
