import os
import uuid
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from Dad import DadBot


def _fake_embed_texts(texts, purpose="semantic retrieval"):
    items = [texts] if isinstance(texts, str) else list(texts)
    vectors = []
    for item in items:
        lowered = str(item).lower()
        vector = [0.0, 0.0, 0.0]
        vector[0] = 1.0 if any(token in lowered for token in ("save", "saving", "budget", "money")) else 0.0
        vector[1] = 1.0 if any(token in lowered for token in ("work", "career", "boss")) else 0.0
        vector[2] = 1.0 if any(token in lowered for token in ("stress", "anxious", "overwhelmed")) else 0.0
        vectors.append(vector)
    return vectors


def test_semantic_memory_can_index_and_query_through_pgvector(monkeypatch):
    postgres_dsn = str(os.environ.get("DADBOT_TEST_POSTGRES_DSN") or "").strip()
    if not postgres_dsn:
        pytest.skip("Set DADBOT_TEST_POSTGRES_DSN to run the live PGVector integration test.")

    psycopg = pytest.importorskip("psycopg")
    table_name = f"semantic_memories_test_{uuid.uuid4().hex[:10]}"
    today_stamp = date.today().isoformat()
    ann_index_name = f"idx_{table_name}_embedding_hnsw"

    monkeypatch.setenv("DADBOT_POSTGRES_DSN", postgres_dsn)
    monkeypatch.setenv("DADBOT_SEMANTIC_INDEX_TABLE", table_name)
    monkeypatch.setenv("DADBOT_SEMANTIC_VECTOR_DIM", "3")
    monkeypatch.setenv("DADBOT_SEMANTIC_ANN_INDEX", "hnsw")
    monkeypatch.setenv("DADBOT_SEMANTIC_DISTANCE_METRIC", "cosine")

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        bot = None
        try:
            bot = DadBot()
            bot.MEMORY_PATH = temp_path / "dad_memory.json"
            bot.SEMANTIC_MEMORY_DB_PATH = temp_path / "dad_memory_semantic.sqlite3"
            bot.SESSION_LOG_DIR = temp_path / "session_logs"
            bot.SESSION_LOG_DIR.mkdir(parents=True, exist_ok=True)
            bot.MEMORY_STORE = bot.default_memory_store()
            bot.save_memory_store()
            bot.embed_texts = _fake_embed_texts

            memories = [
                {
                    "summary": "Tony has been saving money for a trip.",
                    "category": "finance",
                    "mood": "positive",
                    "created_at": today_stamp,
                    "updated_at": today_stamp,
                }
            ]

            saved_memories = bot.save_memory_catalog(memories)
            assert len(saved_memories) == 1
            assert bot.wait_for_semantic_index_idle(timeout=5)
            assert bot.semantic_memory_status()["backend"] == "pgvector"
            assert bot.semantic_memory_status()["ann_index"] == "hnsw"
            assert bot.semantic_memory_status()["vector_dimensions"] == 3
            assert bot.semantic_index_row_count() == 1
            assert not bot.SEMANTIC_MEMORY_DB_PATH.exists()

            with psycopg.connect(postgres_dsn, autocommit=True, connect_timeout=5) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row = cursor.fetchone()
                    assert row is not None
                    assert int(row[0]) == 1
                    cursor.execute(
                        "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public' AND tablename = %s",
                        (table_name,),
                    )
                    indexes = {index_name: index_def for index_name, index_def in cursor.fetchall()}
                    assert ann_index_name in indexes
                    assert "USING hnsw" in indexes[ann_index_name]

            matches = bot.semantic_memory_matches("saving money", saved_memories, limit=1)
            assert len(matches) == 1
            assert matches[0][1]["summary"] == saved_memories[0]["summary"]
        finally:
            if bot is not None:
                try:
                    bot.shutdown()
                except Exception:
                    pass
            with psycopg.connect(postgres_dsn, autocommit=True, connect_timeout=5) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
