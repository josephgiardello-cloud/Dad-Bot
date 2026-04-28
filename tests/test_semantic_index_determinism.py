from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from dadbot_system.semantic_index import SQLiteSemanticIndex


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
