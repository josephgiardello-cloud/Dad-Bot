import uuid
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from Dad import DadBot


@pytest.mark.integration
@pytest.mark.postgres
def test_graph_store_can_sync_and_query_through_postgres(monkeypatch, postgres_test_dsn):
    psycopg = pytest.importorskip("psycopg")
    table_prefix = f"dadbot_graph_test_{uuid.uuid4().hex[:10]}"
    today = date.today().isoformat()

    monkeypatch.setenv("DADBOT_POSTGRES_DSN", postgres_test_dsn)
    monkeypatch.setenv("DADBOT_GRAPH_TABLE_PREFIX", table_prefix)

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        bot = None
        try:
            bot = DadBot()
            bot.MEMORY_PATH = temp_path / "dad_memory.json"
            bot.SEMANTIC_MEMORY_DB_PATH = temp_path / "dad_memory_semantic.sqlite3"
            bot.GRAPH_STORE_DB_PATH = temp_path / "dad_memory_graph.sqlite3"
            bot.SESSION_LOG_DIR = temp_path / "session_logs"
            bot.SESSION_LOG_DIR.mkdir(parents=True, exist_ok=True)
            bot.MEMORY_STORE = bot.default_memory_store()
            bot.MEMORY_STORE["consolidated_memories"] = [
                {
                    "summary": "Tony has been saving money for an emergency fund.",
                    "category": "finance",
                    "source_count": 3,
                    "confidence": 0.84,
                    "supporting_summaries": ["Tony has been sticking to a budget."],
                    "contradictions": ["Spent impulsively after payday"],
                    "updated_at": today,
                }
            ]
            bot.MEMORY_STORE["session_archive"] = [
                {
                    "summary": "Work deadlines felt especially heavy this week.",
                    "topics": ["work"],
                    "dominant_mood": "stressed",
                    "turn_count": 5,
                    "created_at": today + "T09:00:00",
                    "id": "archive-1",
                }
            ]
            bot.save_memory_store()

            snapshot = bot.sync_graph_store()
            assert snapshot["nodes"]
            assert snapshot["edges"]
            assert bot.memory_manager._graph_store_backend.name == "postgres"
            assert not bot.GRAPH_STORE_DB_PATH.exists()

            graph_result = bot.graph_retrieval_for_input("budget and emergency fund", limit=2)
            assert graph_result is not None
            assert graph_result["supporting_evidence"]
            assert graph_result["compressed_summary"]

            with psycopg.connect(postgres_test_dsn, autocommit=True, connect_timeout=5) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_prefix}_nodes")
                    node_row = cursor.fetchone()
                    cursor.execute(f"SELECT COUNT(*) FROM {table_prefix}_edges")
                    edge_row = cursor.fetchone()
                    assert node_row is not None and int(node_row[0]) > 0
                    assert edge_row is not None and int(edge_row[0]) > 0
        finally:
            if bot is not None:
                try:
                    bot.shutdown()
                except Exception:
                    pass
            with psycopg.connect(postgres_test_dsn, autocommit=True, connect_timeout=5) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(f"DROP TABLE IF EXISTS {table_prefix}_edges")
                    cursor.execute(f"DROP TABLE IF EXISTS {table_prefix}_nodes")
