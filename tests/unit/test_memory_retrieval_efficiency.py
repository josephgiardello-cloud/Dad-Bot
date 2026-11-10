from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import date
from types import SimpleNamespace

import pytest

from dadbot.memory.storage import _only_memory_access_changed
from dadbot.services.memory_service import MemoryRetrievalError, MemoryService
from dadbot_system.semantic_index import PGVectorSemanticIndex

pytestmark = pytest.mark.unit


def test_access_only_comparison_does_not_hide_content_changes():
    before = {"memories": [{"summary": "A", "access_count": 1}]}
    after = {"memories": [{"summary": "A", "access_count": 2}]}
    assert _only_memory_access_changed(before, after, {"memories"})
    assert not _only_memory_access_changed(before, {"memories": [{"summary": "B"}]}, {"memories"})
    assert not _only_memory_access_changed(before, after, {"memories", "session_archive"})
    assert not _only_memory_access_changed(before, {"memories": []}, {"memories"})


def test_access_updates_remain_durable_without_index_or_graph_rebuild(bot, monkeypatch):
    bot.mutate_memory_store(memories=[{
        "summary": "My preferred fruit is an apple.",
        "category": "preferences",
        "updated_at": date.today().isoformat(),
    }])
    assert bot.wait_for_semantic_index_idle(timeout=30)
    memories = bot.memory_catalog()
    assert bot.wait_for_semantic_index_idle(timeout=30)
    calls = []
    monkeypatch.setattr(bot.memory_manager, "sync_semantic_memory_index", lambda *_: calls.append("semantic"))
    monkeypatch.setattr(bot.memory_manager.graph_manager, "sync_graph_store", lambda **_: calls.append("graph"))
    bot.memory_query._record_memory_access_signals([(0.9, memories[0])], memories)
    assert calls == []
    saved = json.loads(bot.MEMORY_PATH.read_text(encoding="utf-8"))
    assert saved["memories"][0]["access_count"] == 1
    assert saved["memories"][0]["confidence_history"]["high"] == 1
    assert saved["memories"][0]["last_accessed_at"] == date.today().isoformat()
    reloaded = bot.memory_manager.normalize_memory_store(saved)["memories"][0]
    assert reloaded["access_count"] == 1
    assert reloaded["last_accessed_at"] == date.today().isoformat()


def test_supplied_empty_graph_is_not_looked_up_again_and_diagnostics_are_safe(bot, monkeypatch):
    bot.mutate_memory_store(memories=[{
        "summary": "My preferred fruit is an apple.",
        "category": "preferences",
        "updated_at": date.today().isoformat(),
    }])
    monkeypatch.setattr(bot, "graph_retrieval_for_input", lambda *_args, **_kwargs: pytest.fail("duplicate graph"))
    monkeypatch.setattr(bot.memory_query, "_record_memory_access_signals", lambda *_args: None)
    bot.memory_query.relevant_memories_for_input("apple", graph_result={})
    diagnostics = bot.memory_query.get_retrieval_diagnostics()
    assert diagnostics["catalog_count"] == 1
    assert diagnostics["candidate_count"] == 1
    assert set(diagnostics["stage_seconds"]) == {
        "catalog", "graph", "semantic", "lexical", "ranking", "reranking", "access_update",
    }
    assert diagnostics["total_seconds"] >= 0
    assert "apple" not in repr(diagnostics)


def test_pg_nearest_candidates_are_catalog_scoped_and_bounded(monkeypatch):
    backend = PGVectorSemanticIndex("unused", vector_dimensions=3, ann_index="hnsw")
    calls = []
    cursor = SimpleNamespace(
        execute=lambda sql, params: calls.append((sql, params)),
        fetchall=lambda: [("a", "Synthetic", "general", "neutral", "", "[1,0,0]")],
    )
    connection = SimpleNamespace(cursor=lambda: nullcontext(cursor))
    monkeypatch.setattr(backend, "ensure_storage", lambda: None)
    monkeypatch.setattr(backend, "with_connection", lambda operation: operation(connection))
    assert [row["summary_key"] for row in backend.fetch_nearest_by_keys({"b", "a"}, [1, 0, 0], 1)] == ["a"]
    sql, params = calls[0]
    assert "summary_key = ANY(%s)" in sql
    assert "LIMIT %s" in sql
    assert params[0] == ["a", "b"]
    assert params[-1] == 1
    with pytest.raises(ValueError):
        backend.fetch_nearest_by_keys({"a"}, [1, 0, 0], 0)


def test_bounded_pg_ranking_keeps_semantic_and_lexical_candidates(bot, monkeypatch):
    entries = [
        {"summary": "My preferred fruit is an apple.", "category": "preferences"},
        {"summary": "My favorite pet is a rabbit.", "category": "pets"},
        {"summary": "My weekend hobby is pottery.", "category": "hobbies"},
        {"summary": "My favorite drink is tea.", "category": "preferences"},
    ]
    bot.mutate_memory_store(memories=entries)
    assert bot.wait_for_semantic_index_idle(timeout=30)
    manager = bot.memory_manager._semantic_manager
    backend = PGVectorSemanticIndex("unused", vector_dimensions=12)
    monkeypatch.setattr(manager, "_semantic_index_backend", backend)
    monkeypatch.setattr(manager, "queue_semantic_memory_index", lambda *_args: None)
    monkeypatch.setattr(manager, "semantic_query_context", lambda *_args: {
        "query_embedding": [1.0] * 12,
        "query_tokens": [],
        "query_category": "general",
        "query_mood": "neutral",
    })
    requested = []

    def nearest(keys, embedding, limit):
        requested.append((set(keys), limit))
        target = next(key for key in keys if "rabbit" in key)
        return [{"summary_key": target, "embedding_json": json.dumps(embedding)}]

    monkeypatch.setattr(backend, "fetch_nearest_by_keys", nearest)
    monkeypatch.setattr(bot.memory_query, "_record_memory_access_signals", lambda *_args: None)
    bot.runtime_config.memory_candidate_limit = 2
    result = bot.memory_query.relevant_memories_for_input(
        "apple", graph_result={"supporting_evidence": [{"summary": "pottery", "matched_nodes": []}]},
    )
    summaries = " ".join(memory["summary"].lower() for memory in result)
    assert "apple" in summaries
    assert "rabbit" in summaries
    assert "pottery" in summaries
    assert requested[0][1] == 2
    assert len(requested[0][0]) == 4
    diagnostics = bot.memory_query.get_retrieval_diagnostics()
    assert diagnostics["semantic_status"]["bounded"] is True
    assert diagnostics["candidate_count"] == 3
    assert diagnostics["graph_candidate_count"] == 1


def test_missing_embeddings_are_reported_without_disclosing_the_query(bot, monkeypatch):
    bot.mutate_memory_store(memories=[{"summary": "My preferred fruit is an apple.", "category": "preferences"}])
    monkeypatch.setattr(bot.memory_manager, "semantic_query_context", lambda *_args: None)
    monkeypatch.setattr(bot.memory_query, "_record_memory_access_signals", lambda *_args: None)
    bot.memory_query.relevant_memories_for_input("private unrelated question", graph_result={})
    status = bot.memory_query.get_retrieval_diagnostics()["semantic_status"]
    assert status["degraded"] is True
    assert status["reason"] == "embedding_unavailable"
    assert "private" not in repr(status)


def test_time_scoped_retrieval_preserves_out_of_scope_memories(bot, monkeypatch):
    entries = [
        {"summary": "My preferred fruit is an apple.", "category": "preferences", "created_at": "2025-01-05"},
        {"summary": "My preferred fruit is a pear.", "category": "preferences", "created_at": "2025-02-05"},
        {"summary": "My preferred fruit is a peach.", "category": "preferences", "created_at": "2025-03-05"},
    ]
    bot.mutate_memory_store(memories=entries)
    monkeypatch.setattr(bot, "semantic_memory_matches", lambda *_args, **_kwargs: [])
    result = bot.memory_query.relevant_memories_for_input(
        "preferred fruit",
        graph_result={},
        from_date="2025-02-01",
        to_date="2025-02-28",
    )
    assert len(result) == 1
    assert "pear" in result[0]["summary"]
    saved = bot.memory_catalog()
    assert len(saved) == 3
    assert next(item for item in saved if "pear" in item["summary"])["access_count"] == 1
    assert all(not item.get("access_count") for item in saved if "pear" not in item["summary"])


def test_time_scoped_retrieval_rejects_invalid_ranges(bot):
    with pytest.raises(ValueError, match="from_date"):
        bot.memory_query.relevant_memories_for_input("fruit", from_date="not-a-date")
    with pytest.raises(ValueError, match="from_date.*to_date"):
        bot.memory_query.relevant_memories_for_input("fruit", from_date="2025-03-01", to_date="2025-02-01")


def test_time_scoped_service_across_simulated_months(bot, monkeypatch):
    entries = [
        {"summary": "The project milestone is planning.", "created_at": "2025-01-15", "category": "work"},
        {"summary": "The project milestone is building.", "created_at": "2025-03-15", "category": "work"},
        {"summary": "The project milestone is launch.", "created_at": "2025-06-15", "category": "work"},
    ]
    bot.mutate_memory_store(memories=entries)
    monkeypatch.setattr(bot, "semantic_memory_matches", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(bot, "graph_retrieval_for_input", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(bot.memory_query, "_record_memory_access_signals", lambda *_args: None)
    service = MemoryService(bot)

    for start, end, expected in (
        ("2025-01-01", "2025-02-28", "planning"),
        ("2025-03-01", "2025-05-31", "building"),
        ("2025-06-01", "2025-06-30", "launch"),
    ):
        found = service.retrieve_for_query(query="project milestone", from_date=start, to_date=end)
        assert len(found) == 1
        assert expected in found[0]["summary"]
    assert service.retrieve_for_query(query="project milestone", from_date="2025-07-01") == []
    with pytest.raises(ValueError, match="from_date"):
        service.retrieve_for_query(query="project milestone", from_date="invalid")


def test_memory_retrieval_status_distinguishes_empty_results_from_backend_failure():
    class EmptyRuntime:
        def graph_retrieval_for_input(self, _query, limit=3):
            return {}

        def relevant_memories_for_input(self, _query, **_kwargs):
            return []

    class FailedRuntime(EmptyRuntime):
        def relevant_memories_for_input(self, _query, **_kwargs):
            raise OSError("backend unavailable")

    empty_service = MemoryService(EmptyRuntime())
    empty = empty_service.retrieve_for_query_with_status(query="an empty index")
    assert empty.memories == []
    assert empty.status == "ok"
    assert empty.errors == []
    assert empty_service.retrieve_for_query(query="an empty index") == []
    empty_turn_state = {}
    MemoryService.record_retrieval_status(
        empty_turn_state,
        status=empty.status,
        errors=empty.errors,
        source="post_tool",
    )
    assert empty_turn_state["memory_retrieval_status"] == {
        "status": "ok",
        "errors": [],
        "sources": ["post_tool"],
    }

    failed_service = MemoryService(FailedRuntime())
    failed = failed_service.retrieve_for_query_with_status(query="a failed index")
    assert failed.memories == []
    assert failed.status == "degraded"
    assert failed.errors == [{"source": "memory_catalog", "error_type": "OSError"}]
    with pytest.raises(MemoryRetrievalError) as error:
        failed_service.retrieve_for_query(query="a failed index")
    assert error.value.status == "degraded"
    assert error.value.errors == [{"source": "memory_catalog", "error_type": "OSError"}]
    failed_turn_state = {}
    MemoryService.record_retrieval_status(
        failed_turn_state,
        status=failed.status,
        errors=failed.errors,
        source="post_tool",
    )
    assert failed_turn_state["memory_retrieval_status"] == {
        "status": "degraded",
        "errors": [{"source": "post_tool:memory_catalog", "error_type": "OSError"}],
        "sources": ["post_tool"],
    }
    assert failed_turn_state["memory_retrieval_status"] != empty_turn_state["memory_retrieval_status"]


def test_semantic_index_failure_is_recorded_in_turn_retrieval_status():
    class FailedIndex:
        def fetch_candidates(self, **_kwargs):
            raise OSError("backend unavailable")

    state = {}
    MemoryService.record_retrieval_status(
        state,
        status="ok",
        errors=[],
        source="context_snapshot",
    )
    candidates = MemoryService.fetch_semantic_candidates(
        FailedIndex(),
        query_embedding=None,
        query_tokens=[],
        query_category="general",
        query_mood="neutral",
        limit=5,
        state=state,
    )

    assert candidates == []
    assert state["memory_retrieval_status"] == {
        "status": "degraded",
        "errors": [{"source": "semantic_rag:semantic_index", "error_type": "OSError"}],
        "sources": ["context_snapshot", "semantic_rag"],
    }


def test_time_scope_uses_recorded_occurrence_not_write_date(bot, monkeypatch):
    bot.mutate_memory_store(memories=[
        {
            "summary": "The project milestone is planning.",
            "category": "work",
            "event_time": "2025-01-15T10:00:00+00:00",
            "created_at": "2025-06-15",
        },
        {
            "summary": "The project milestone is launch.",
            "category": "work",
            "created_at": "2025-06-15",
        },
    ])
    monkeypatch.setattr(bot, "semantic_memory_matches", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(bot.memory_query, "_record_memory_access_signals", lambda *_args: None)
    assert any(item.get("event_time") for item in bot.memory_catalog())
    january = bot.memory_query.relevant_memories_for_input(
        "project milestone", graph_result={}, from_date="2025-01-01", to_date="2025-01-31",
    )
    assert len(january) == 1
    assert "planning" in january[0]["summary"]
    assert january[0]["event_time"].startswith("2025-01-15")
    saved = json.loads(bot.MEMORY_PATH.read_text(encoding="utf-8"))
    assert next(item for item in saved["memories"] if "planning" in item["summary"])["event_time"]


def test_invalid_event_time_fails_without_losing_existing_memories(bot):
    bot.mutate_memory_store(memories=[{"summary": "The project milestone is planning.", "category": "work"}])
    before = bot.memory_catalog()
    with pytest.raises(ValueError, match="event_time"):
        bot.mutate_memory_store(memories=[*before, {
            "summary": "The project milestone is launch.",
            "category": "work",
            "event_time": "invalid",
        }])
    assert bot.memory_catalog() == before
