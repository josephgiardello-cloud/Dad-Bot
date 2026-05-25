# Global monkeypatch: ensure every DadBot instance has PROFILE_PATH
import tempfile
from pathlib import Path
from dadbot.core.dadbot import DadBot as _RealDadBot
_orig_dadbot_init = _RealDadBot.__init__
def _patched_dadbot_init(self, *args, **kwargs):
    tmp_profile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    self.PROFILE_PATH = Path(tmp_profile.name)
    # Write an empty JSON object so the file is always valid
    with open(self.PROFILE_PATH, "w", encoding="utf-8") as f:
        f.write("{}\n")
    tmp_profile.close()
    _orig_dadbot_init(self, *args, **kwargs)
_RealDadBot.__init__ = _patched_dadbot_init
from dadbot.memory import semantic_manager as sm_mod
from dadbot_system.semantic_index import SQLiteSemanticIndex

# Global monkeypatch: force all SemanticIndexManager to use SQLiteSemanticIndex for tests
def _test_build_semantic_index_backend(self):
    import tempfile
    print("[TEST PATCH] Using SQLiteSemanticIndex for all tests (patch active)")
    db_path = getattr(self._bot, "SEMANTIC_INDEX_DB_PATH", tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False).name)
    return SQLiteSemanticIndex(self._bot, db_path)
sm_mod.SemanticIndexManager._build_semantic_index_backend = _test_build_semantic_index_backend
import tempfile
from dadbot.memory import graph_manager as gm_mod
from dadbot_system.graph_store import SQLiteGraphStore

# Global monkeypatch: force all MemoryGraphManager to use SQLiteGraphStore for tests
def _test_build_graph_store_backend(self):
    print("[TEST PATCH] Using SQLiteGraphStore for all tests (patch active)")
    if hasattr(self._bot, "GRAPH_STORE_DB_PATH"):
        db_path = self._bot.GRAPH_STORE_DB_PATH
    else:
        db_path = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False).name
    return SQLiteGraphStore(self._bot, db_path)
gm_mod.MemoryGraphManager._build_graph_store_backend = _test_build_graph_store_backend

import types
import os
from unittest.mock import MagicMock
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest

@pytest.fixture
def make_test_dadbot():
    """Factory for DadBot with all required dependencies mocked."""
    from dadbot.core.dadbot import DadBot
    memory_manager = MagicMock(name="memory_manager")
    relationship_manager = MagicMock(name="relationship_manager")
    mood_manager = MagicMock(name="mood_manager")
    profile_runtime = MagicMock(name="profile_runtime")
    event_bus = MagicMock(name="event_bus")
    from dadbot_system.state import InMemoryStateStore
    def factory(**overrides):
        # Patch MemoryGraphManager to always use SQLiteGraphStore with a temp file
        import tempfile
        from dadbot.memory import graph_manager as gm_mod
        from dadbot_system.graph_store import SQLiteGraphStore
        orig_build_backend = gm_mod.MemoryGraphManager._build_graph_store_backend
        def _test_build_graph_store_backend(self):
            # Always use a temp SQLite DB for tests
            if hasattr(self._bot, "GRAPH_STORE_DB_PATH"):
                db_path = self._bot.GRAPH_STORE_DB_PATH
            else:
                db_path = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False).name
            return SQLiteGraphStore(self._bot, db_path)
        gm_mod.MemoryGraphManager._build_graph_store_backend = _test_build_graph_store_backend

        args = dict(
            memory_manager=memory_manager,
            relationship_manager=relationship_manager,
            mood_manager=mood_manager,
            profile_runtime=profile_runtime,
            event_bus=event_bus,
            document_store=InMemoryStateStore(),
        )
        bot = DadBot(
            memory_manager=memory_manager,
            relationship_manager=relationship_manager,
            mood_manager=mood_manager,
            profile_runtime=profile_runtime,
            event_bus=event_bus,
        )
        # Patch PROFILE_PATH for test isolation
        import tempfile
        from pathlib import Path
        tmp_profile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        bot.PROFILE_PATH = Path(tmp_profile.name)
        tmp_profile.close()
        # Patch config._env_path to always return the override path if provided
        if hasattr(bot, "config"):
            orig_env_path = bot.config._env_path
            def _patched_env_path(env_var, fallback):
                # Use test override if present in overrides
                if env_var == "DADBOT_GRAPH_DB_PATH" and "graph_store_db_path" in overrides:
                    return Path(overrides["graph_store_db_path"])
                if env_var == "DADBOT_MEMORY_PATH" and "memory_path" in overrides:
                    return Path(overrides["memory_path"])
                if env_var == "DADBOT_SEMANTIC_DB_PATH" and "semantic_memory_db_path" in overrides:
                    return Path(overrides["semantic_memory_db_path"])
                if env_var == "DADBOT_SESSION_LOG_DIR" and "session_log_dir" in overrides:
                    return Path(overrides["session_log_dir"])
                return orig_env_path(env_var, fallback)
            bot.config._env_path = _patched_env_path
        return bot
    return factory

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
