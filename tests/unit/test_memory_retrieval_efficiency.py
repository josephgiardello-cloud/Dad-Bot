from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import date
from types import SimpleNamespace

import pytest

from dadbot.memory.storage import _only_memory_access_changed
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
