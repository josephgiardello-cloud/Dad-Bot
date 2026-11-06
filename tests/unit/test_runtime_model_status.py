from types import SimpleNamespace

from dadbot.infrastructure.llm import OllamaModelAdapter
from dadbot.managers import runtime_model
from dadbot.managers.runtime_model import RuntimeModelManager


def _retryable_errors():
    return (OSError,)


def _summarize_error(exc):
    return f"connection failed: {exc}"


def test_ollama_status_normalizes_typed_list_response(monkeypatch):
    manager = RuntimeModelManager.__new__(RuntimeModelManager)
    manager.bot = SimpleNamespace(
        ACTIVE_MODEL="qwen3.5:9b",
        ollama_retryable_errors=_retryable_errors,
        ollama_error_summary=_summarize_error,
    )

    def list_models():
        return SimpleNamespace(
            models=[
                SimpleNamespace(model="qwen3.5:9b"),
                SimpleNamespace(name="llama3.2"),
            ],
        )

    monkeypatch.setattr(runtime_model.ollama, "list", list_models)

    status = manager.ollama_status()

    assert status == {
        "model": "qwen3.5:9b",
        "connected": True,
        "models": ["qwen3.5:9b", "llama3.2"],
        "models_count": 2,
        "error": "",
    }


def test_ollama_status_reports_connection_failure(monkeypatch):
    manager = RuntimeModelManager.__new__(RuntimeModelManager)
    manager.bot = SimpleNamespace(
        ACTIVE_MODEL="qwen3.5:9b",
        ollama_retryable_errors=_retryable_errors,
        ollama_error_summary=_summarize_error,
    )

    def fail_list():
        raise OSError("server unavailable")

    monkeypatch.setattr(runtime_model.ollama, "list", fail_list)

    status = manager.ollama_status()

    assert status["connected"] is False
    assert status["models_count"] == 0
    assert status["error"] == "connection failed: server unavailable"


def test_model_availability_accepts_typed_model_records():
    assert RuntimeModelManager.model_is_available(
        [SimpleNamespace(model="qwen3.5:9b")],
        "qwen3.5:9b",
    )


def test_ollama_model_adapter_normalizes_typed_list_response(monkeypatch):
    monkeypatch.setattr(
        "dadbot.infrastructure.llm.ollama.list",
        lambda: SimpleNamespace(models=[SimpleNamespace(model="qwen3.5:9b")]),
    )

    assert OllamaModelAdapter().list_models() == [{"model": "qwen3.5:9b"}]
    assert OllamaModelAdapter.model_is_available(
        [SimpleNamespace(model="qwen3.5:9b")],
        "qwen3.5:9b",
    )
