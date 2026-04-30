import os
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

pytest_plugins = ("tests.phase4_helpers",)

try:
    from Dad import DadBot as _DadBot
except ModuleNotFoundError:
    _DadBot = None  # type: ignore


@pytest.fixture
def bot():
    if _DadBot is None:
        pytest.skip("Dad module not importable in this environment")
    DadBot = _DadBot
    temp_dir = TemporaryDirectory()
    try:
        bot = DadBot()
        temp_path = Path(temp_dir.name)
        bot.MEMORY_PATH = temp_path / "dad_memory.json"
        bot.SEMANTIC_MEMORY_DB_PATH = temp_path / "dad_memory_semantic.sqlite3"
        bot.GRAPH_STORE_DB_PATH = temp_path / "dad_memory_graph.sqlite3"
        bot.SESSION_LOG_DIR = temp_path / "session_logs"
        bot.MEMORY_STORE = bot.default_memory_store()
        bot.save_memory_store()

        def fake_embed_texts(texts, purpose="semantic retrieval"):
            items = [texts] if isinstance(texts, str) else list(texts)
            vectors = []
            for item in items:
                lowered = str(item).lower()
                vector = [0.0] * 12
                vector[0] = 1.0 if any(token in lowered for token in ("save", "saving", "budget", "money")) else 0.0
                vector[1] = 1.0 if any(token in lowered for token in ("work", "career", "boss")) else 0.0
                vector[2] = 1.0 if "stress" in lowered or "anxious" in lowered else 0.0
                vectors.append(vector)
            return vectors

        bot.embed_texts = fake_embed_texts
        yield bot
    finally:
        try:
            bot.shutdown()
        except Exception:
            pass
        try:
            bot.wait_for_semantic_index_idle(5)
        except Exception:
            pass
        temp_dir.cleanup()


@pytest.fixture
def today_iso():
    return date.today().isoformat()


@pytest.fixture
def postgres_test_dsn():
    dsn = str(os.environ.get("DADBOT_TEST_POSTGRES_DSN") or "").strip()
    if not dsn:
        pytest.skip("Set DADBOT_TEST_POSTGRES_DSN to run Postgres integration tests.")
    return dsn
