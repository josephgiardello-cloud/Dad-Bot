import os
import json
from datetime import date

import ollama
import pytest

from dadbot.config import DadBotConfig
from dadbot.core.dadbot import DadBot
from dadbot.managers.runtime_client import RuntimeClientManager

pytest_plugins = ("tests.phase4_helpers",)


@pytest.fixture
def offline_model_transport(monkeypatch):
    """Reject accidental live inference in mock-backed bot fixtures."""
    monkeypatch.setenv("DADBOT_GRAPH_ENABLE_LLM_EXTRACTION", "0")

    def unexpected_sync_model_call(**_kwargs):
        raise AssertionError("Unexpected live Ollama call; mock the model port or transport")

    async def unexpected_async_model_call(*_args, **_kwargs):
        raise AssertionError("Unexpected live Ollama call; mock the model port or transport")

    monkeypatch.setattr(RuntimeClientManager, "_call_ollama_sync", staticmethod(unexpected_sync_model_call))
    monkeypatch.setattr(ollama.AsyncClient, "chat", unexpected_async_model_call)


@pytest.fixture
def make_test_dadbot(offline_model_transport, tmp_path, monkeypatch):
    """Build isolated real managers rather than replacing public APIs with stubs."""
    instances = []

    def factory(**overrides):
        root = tmp_path / f"bot-{len(instances)}"
        root.mkdir()
        profile_path = root / "profile.json"
        profile_path.write_text(json.dumps(DadBot.default_profile()), encoding="utf-8")
        for key, path in {
            "DADBOT_PROFILE_PATH": profile_path,
            "DADBOT_MEMORY_PATH": root / "memory.json",
            "DADBOT_SEMANTIC_DB_PATH": root / "semantic.sqlite3",
            "DADBOT_GRAPH_DB_PATH": root / "graph.sqlite3",
            "DADBOT_SESSION_LOG_DIR": root / "sessions",
        }.items():
            monkeypatch.setenv(key, str(path))
        config = overrides.pop("config", None) or DadBotConfig()
        for name in (
            "profile_path", "memory_path", "semantic_memory_db_path",
            "graph_store_db_path", "session_log_dir", "tenant_id",
            "model_name", "append_signoff", "light_mode",
        ):
            if name in overrides:
                setattr(config, name, overrides.pop(name))
        bot = DadBot(config=config, **overrides)
        instances.append(bot)

        def fake_embed_texts(texts, purpose="semantic retrieval"):
            items = [texts] if isinstance(texts, str) else list(texts)
            vectors = []
            for item in items:
                lowered = str(item).lower()
                vector = [0.0] * 12
                vector[0] = float(any(token in lowered for token in ("save", "saving", "budget", "money")))
                vector[1] = float(any(token in lowered for token in ("work", "career", "boss")))
                vector[2] = float("stress" in lowered or "anxious" in lowered)
                vectors.append(vector)
            return vectors

        bot.embed_texts = fake_embed_texts
        return bot

    yield factory
    for bot in reversed(instances):
        bot.shutdown()


@pytest.fixture
def bot(make_test_dadbot):
    return make_test_dadbot()


@pytest.fixture
def today_iso():
    return date.today().isoformat()


@pytest.fixture
def postgres_test_dsn():
    dsn = str(os.environ.get("DADBOT_TEST_POSTGRES_DSN") or "").strip()
    if not dsn:
        pytest.skip("Set DADBOT_TEST_POSTGRES_DSN to run Postgres integration tests.")
    return dsn
