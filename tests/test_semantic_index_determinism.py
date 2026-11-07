from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from dadbot_system.semantic_index import PGVectorSemanticIndex, SQLiteSemanticIndex

pytestmark = pytest.mark.unit


def test_semantic_query_tokens_are_sorted_before_candidate_window(bot, monkeypatch):
    tokens = ["work", "tea", "project", "preference", "learning", "goal", "earlier", "about"]
    manager = bot.memory_manager._semantic_manager
    monkeypatch.setattr(manager, "embed_texts", lambda *_args, **_kwargs: [[1.0] * 12])
    monkeypatch.setattr(bot, "significant_tokens", lambda _query: tokens)
    first = manager.semantic_query_context("query", 3)
    tokens.reverse()
    second = manager.semantic_query_context("query", 3)

    assert first["query_tokens"] == second["query_tokens"] == sorted(tokens)[:6]


def _index(tmp_path: Path) -> SQLiteSemanticIndex:
    bot = SimpleNamespace(_io_lock=None)
    return SQLiteSemanticIndex(bot=bot, db_path=tmp_path / "semantic_test.db")


def test_fetch_candidates_stable_order_and_score(tmp_path: Path):
    index = _index(tmp_path)
    index.upsert_rows(
        [
            {
                "summary_key": "b",
                "summary": "dad memory beta",
                "category": "general",
                "mood": "neutral",
                "updated_at": "2026-01-01T00:00:00Z",
                "content_hash": "h-b",
                "embedding_json": json.dumps([0.5, 0.5]),
            },
            {
                "summary_key": "a",
                "summary": "dad memory alpha",
                "category": "general",
                "mood": "neutral",
                "updated_at": "2026-01-01T00:00:00Z",
                "content_hash": "h-a",
                "embedding_json": json.dumps([0.5, 0.5]),
            },
        ]
    )

    q = [0.5, 0.5]
    r1 = index.fetch_candidates(q, ["dad"], "general", "neutral", 5)
    r2 = index.fetch_candidates(q, ["dad"], "general", "neutral", 5)

    assert r1 == r2
    assert [item["summary_key"] for item in r1] == ["a", "b"]
    assert all("retrieval_score" in item for item in r1)


def test_fetch_candidates_cache_invalidation_after_upsert(tmp_path: Path):
    index = _index(tmp_path)
    index.upsert_rows(
        [
            {
                "summary_key": "k1",
                "summary": "old summary",
                "category": "general",
                "mood": "neutral",
                "updated_at": "2026-01-01T00:00:00Z",
                "content_hash": "h1",
                "embedding_json": json.dumps([1.0, 0.0]),
            }
        ]
    )

    first = index.fetch_candidates([1.0, 0.0], ["old"], "general", "neutral", 5)
    assert first and first[0]["summary"] == "old summary"

    index.upsert_rows(
        [
            {
                "summary_key": "k1",
                "summary": "new summary",
                "category": "general",
                "mood": "neutral",
                "updated_at": "2026-01-02T00:00:00Z",
                "content_hash": "h2",
                "embedding_json": json.dumps([1.0, 0.0]),
            }
        ]
    )

    second = index.fetch_candidates([1.0, 0.0], ["new"], "general", "neutral", 5)
    assert second and second[0]["summary"] == "new summary"


def test_filters_use_and_semantics(tmp_path: Path):
    index = _index(tmp_path)
    index.upsert_rows(
        [
            {
                "summary_key": "match",
                "summary": "dad school update",
                "category": "family",
                "mood": "neutral",
                "updated_at": "2026-01-01T00:00:00Z",
                "content_hash": "h-match",
                "embedding_json": json.dumps([0.1, 0.2]),
            },
            {
                "summary_key": "wrong-category",
                "summary": "dad school update",
                "category": "work",
                "mood": "neutral",
                "updated_at": "2026-01-01T00:00:00Z",
                "content_hash": "h-wc",
                "embedding_json": json.dumps([0.1, 0.2]),
            },
        ]
    )

    rows = index.fetch_candidates(None, ["school"], "family", "neutral", 10)
    assert [row["summary_key"] for row in rows] == ["match"]


def test_catalog_fetch_has_no_relevance_or_candidate_limit(tmp_path):
    index = _index(tmp_path)
    keys = [f"key-{number:04d}" for number in range(1100)]
    index.upsert_rows(
        [
            {
                "summary_key": key,
                "summary": "No lexical match",
                "category": "routines",
                "mood": "calm",
                "updated_at": "2020-01-01",
                "content_hash": key,
                "embedding_json": "[1.0, 0.0]",
            }
            for key in [*keys, "other-tenant"]
        ]
    )
    rows = index.fetch_by_keys(reversed(keys))
    assert [row["summary_key"] for row in rows] == keys
    assert index.fetch_by_keys([]) == []


@pytest.mark.parametrize(
    "query",
    [
        "Which shade would I pick?",
        "Which time of year do I enjoy most?",
        "When do I get out of bed on workdays?",
        "Where am I planning to go on holiday?",
    ],
)
def test_facade_scores_entire_current_catalog_before_limit(bot, monkeypatch, query):
    manager = bot.memory_manager._semantic_manager
    memories = [{"summary": f"Routine number {number}", "category": "routines", "mood": "calm"} for number in range(40)]
    current = bot.semantic_memory_lookup(memories)
    manager._semantic_index_backend.upsert_rows(
        [
            {
                "summary_key": key,
                "summary": memory["summary"],
                "category": memory["category"],
                "mood": memory["mood"],
                "updated_at": "2020-01-01",
                "content_hash": key,
                "embedding_json": "[1.0, 0.0]",
            }
            for key, memory in current.items()
        ]
    )
    monkeypatch.setattr(manager, "embed_texts", lambda *_args, **_kwargs: [[1.0, 0.0]])
    scored_keys = []
    original = manager.score_semantic_rows

    def capture(rows, catalog, embedding):
        scored_keys.extend(row["summary_key"] for row in rows)
        return original(rows, catalog, embedding)

    monkeypatch.setattr(manager, "score_semantic_rows", capture)
    bot.semantic_memory_matches(query, memories, limit=1)
    assert set(scored_keys) == set(current)
    assert len(scored_keys) == 40


def test_missing_index_entries_are_reported(bot, caplog):
    manager = bot.memory_manager._semantic_manager
    rows = manager.catalog_semantic_rows({"missing"})
    assert rows == []
    assert "1 of 1 current catalog entries have no indexed embedding" in caplog.text


def test_fusion_does_not_request_intermediate_dense_top_k(bot, monkeypatch):
    bot.MEMORY_STORE = {
        **bot.default_memory_store(),
        "memories": [
            {"summary": f"My synthetic project number {number} needs paint.", "category": "home"}
            for number in range(40)
        ],
    }
    requests = []

    def matches(_query, memories, limit):
        requests.append((limit, len(memories)))
        return [(0.5, memory) for memory in memories]

    monkeypatch.setattr(bot, "semantic_memory_matches", matches)
    monkeypatch.setattr(bot.memory_query, "_record_memory_access_signals", lambda *_args: None)
    bot.memory_query.relevant_memories_for_input("synthetic project", limit=3)
    assert len(requests) == 1
    assert requests[0][0] == requests[0][1]
    assert requests[0][0] > 9


@pytest.mark.parametrize("ann", [None, "hnsw", "ivfflat"])
def test_pgvector_catalog_fetch_does_not_use_ann_or_relevance_filters(monkeypatch, ann):
    backend = PGVectorSemanticIndex("unused", ann_index=ann, vector_dimensions=768)
    calls = []
    cursor = SimpleNamespace(
        execute=lambda sql, params: calls.append((sql, params)),
        fetchall=lambda: [("a", "Synthetic memory", "routines", "calm", "2020-01-01", "[1.0, 0.0]")],
    )
    connection = SimpleNamespace(cursor=lambda: nullcontext(cursor))
    monkeypatch.setattr(backend, "ensure_storage", lambda: None)
    monkeypatch.setattr(backend, "with_connection", lambda operation: operation(connection))
    assert [row["summary_key"] for row in backend.fetch_by_keys(["a"])] == ["a"]
    sql, params = calls[0]
    assert "WHERE summary_key = ANY(%s)" in sql
    assert params == (["a"],)
    assert "LIMIT" not in sql
    assert "<=>" not in sql
    assert "vector_dims" not in sql
    assert "LIKE" not in sql
