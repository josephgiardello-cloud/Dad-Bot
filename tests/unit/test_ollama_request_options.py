import asyncio
import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import ollama
import pytest

from dadbot.config import DadRuntimeConfig
from dadbot.core.execution_boundary import ModelGatewayScope
from dadbot.managers.runtime_client import RuntimeClientManager

pytestmark = pytest.mark.unit


def _manager(*, budget=6000, model_limit=32768, keep_alive=-1):
    guarded_calls = []

    def guard(messages, **kwargs):
        guarded_calls.append(kwargs)
        return messages

    manager = RuntimeClientManager.__new__(RuntimeClientManager)
    manager.bot = SimpleNamespace(
        ACTIVE_MODEL="test-model",
        RESERVED_RESPONSE_TOKENS=900,
        runtime_config=SimpleNamespace(ollama_keep_alive_seconds=keep_alive),
        model_runtime=SimpleNamespace(
            effective_context_token_budget=lambda _model: budget,
            model_context_length=lambda _model: model_limit,
            ollama_show_payload=lambda _model: {"capabilities": []},
        ),
        guard_chat_request_messages=guard,
        model_candidates=lambda: [],
        ollama_retryable_errors=lambda: (OSError,),
        extract_ollama_message_content=lambda response: response["message"]["content"],
    )
    return manager, guarded_calls


@pytest.mark.parametrize(
    ("budget", "model_limit", "expected"),
    [
        (256, 32768, 256),
        (512, 32768, 512),
        (513, 32768, 513),
        (1157, 32768, 1157),
        (1286, 32768, 1286),
        (1308, 32768, 1308),
        (6000, 32768, 6000),
        (9000, 32768, 9000),
        (6000, 4096, 4096),
        (6000, 7000, 6000),
        (6000, None, 6000),
    ],
)
def test_context_budget_preserves_prompt_guard_budget(budget, model_limit, expected):
    manager, guarded_calls = _manager(model_limit=model_limit)
    options = {"num_ctx": budget, "num_keep": 42, "temperature": 0.2}
    guarded, prepared = manager._prepare_ollama_request(
        "test-model", [{"role": "user", "content": "Hello"}], options, "chat",
    )

    assert prepared["num_ctx"] == expected
    assert guarded_calls[0]["context_budget"] == min(budget, model_limit or budget)
    assert prepared["num_keep"] == 42
    assert prepared["temperature"] == 0.2
    assert options["num_ctx"] == budget
    assert guarded[0]["content"] == "Hello"


def test_default_context_budget_does_not_depend_on_prompt_length():
    manager, guarded_calls = _manager()
    for text in ("short", "long " * 500):
        _, options = manager._prepare_ollama_request(
            "test-model", [{"role": "user", "content": text}], None, "chat",
        )
        assert options["num_ctx"] == 6000
        assert "num_keep" not in options
    assert all(call["context_budget"] == 6000 for call in guarded_calls)


@pytest.mark.parametrize("budget", [-1, 0, 255])
def test_context_bucket_rejects_invalid_budget(budget):
    manager, _ = _manager()
    with pytest.raises(ValueError, match="at least 256"):
        manager._prepare_ollama_request("test-model", [], {"num_ctx": budget}, "chat")


@pytest.mark.parametrize("method", ["sync", "explicit", "async", "stream", "async_stream"])
@pytest.mark.parametrize("keep_alive", [-1, 0, 300])
def test_all_chat_paths_send_top_level_residency_setting(monkeypatch, method, keep_alive):
    manager, _ = _manager(keep_alive=keep_alive)
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        if kwargs.get("stream"):
            return iter_stream()
        return {"message": {"content": "ok"}}

    def iter_stream():
        yield {"message": {"content": "ok"}}

    async def async_chat(**kwargs):
        return chat(**kwargs)

    monkeypatch.setattr(manager, "_call_ollama_sync", chat)
    monkeypatch.setattr(manager, "ollama_async_client", lambda: SimpleNamespace(chat=async_chat))
    monkeypatch.setattr(manager, "_collect_stream_chunks", lambda *_args, **_kwargs: (["ok"], False))

    async def collect_async(*_args, **_kwargs):
        return ["ok"], False

    monkeypatch.setattr(manager, "_collect_stream_chunks_async", collect_async)
    messages = [{"role": "user", "content": "Hello"}]
    with ModelGatewayScope.bind("ModelPort"):
        if method == "sync":
            manager.call_ollama_chat(messages, caller="ModelPort")
        elif method == "explicit":
            manager.call_ollama_chat_with_model("test-model", messages, caller="ModelPort")
        elif method == "async":
            asyncio.run(manager.call_ollama_chat_async(messages, caller="ModelPort"))
        elif method == "stream":
            manager.call_ollama_chat_stream(messages, caller="ModelPort")
        else:
            asyncio.run(manager.call_ollama_chat_stream_async(messages, caller="ModelPort"))

    assert len(calls) == 1
    assert calls[0]["keep_alive"] == keep_alive
    assert "keep_alive" not in calls[0]["options"]
    assert calls[0]["options"]["num_ctx"] == 6000


def test_residency_config_reads_environment_and_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("DADBOT_OLLAMA_KEEP_ALIVE_SECONDS", "300")
    assert DadRuntimeConfig().ollama_keep_alive_seconds == 300
    monkeypatch.setenv("DADBOT_OLLAMA_KEEP_ALIVE_SECONDS", "invalid")
    with pytest.raises(ValueError):
        DadRuntimeConfig()


@pytest.mark.parametrize(("value", "expected"), [("auto", None), ("true", True), ("false", False)])
def test_chat_thinking_configuration_is_validated(monkeypatch, value, expected):
    monkeypatch.setenv("DADBOT_OLLAMA_CHAT_THINK", value)
    assert DadRuntimeConfig().ollama_chat_think is expected
    monkeypatch.setenv("DADBOT_OLLAMA_CHAT_THINK", "invalid")
    with pytest.raises(ValueError, match="DADBOT_OLLAMA_CHAT_THINK"):
        DadRuntimeConfig()


@pytest.mark.parametrize("method", [
    "sync", "explicit", "async", "stream", "async_stream", "async_native", "async_stream_native",
])
@pytest.mark.parametrize(("configured", "explicit", "expected"), [
    (None, None, None), (False, None, False), (True, None, True),
    (False, True, True), (True, False, False),
])
def test_chat_thinking_setting_reaches_every_transport(monkeypatch, method, configured, explicit, expected):
    manager, guarded_calls = _manager()
    manager.bot.runtime_config.ollama_chat_think = configured
    manager.bot.model_runtime.ollama_show_payload = lambda _model: {"capabilities": ["thinking"]}
    manager.bot.STREAM_TIMEOUT_SECONDS = 30
    manager.bot.STREAM_MAX_CHARS = 100
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        payload = {"message": {"content": "ok"}, "done_reason": "stop"}
        return (item for item in [payload]) if kwargs.get("stream") else payload

    monkeypatch.setattr(manager, "_call_ollama_sync", chat)
    async def async_chat(**kwargs):
        if kwargs.get("stream"):
            async def chunks():
                for item in chat(**kwargs):
                    yield item
            return chunks()
        return chat(**kwargs)

    native = SimpleNamespace(chat=async_chat) if method.endswith("_native") else None
    monkeypatch.setattr(manager, "ollama_async_client", lambda: native)
    messages = [{"role": "user", "content": "hello"}]
    with ModelGatewayScope.bind("ModelPort"):
        target = {
            "sync": manager.call_ollama_chat,
            "explicit": manager.call_ollama_chat_with_model,
            "async": manager.call_ollama_chat_async,
            "stream": manager.call_ollama_chat_stream,
            "async_stream": manager.call_ollama_chat_stream_async,
        }[method.removesuffix("_native")]
        args = ("test-model", messages) if method == "explicit" else (messages,)
        result = target(*args, caller="ModelPort", think=explicit)
        if method.startswith("async"):
            asyncio.run(result)
    assert len(calls) == 1
    if expected is None:
        assert "think" not in calls[0]
    else:
        assert calls[0]["think"] is expected
    assert guarded_calls[0]["reserved_response_tokens"] == (900 if expected is False else 3000)


def test_chat_thinking_default_does_not_change_planning_or_reflection():
    manager, _ = _manager()
    manager.bot.runtime_config.ollama_chat_think = False
    assert manager._resolve_ollama_think(None, "tool planning") is None
    assert manager._resolve_ollama_think(None, "reflection") is None
    assert manager._resolve_ollama_think(False, "tool planning") is False


@pytest.mark.parametrize("method", ["sync", "explicit", "async"])
def test_empty_completed_response_fails_at_transport_boundary(monkeypatch, method, caplog):
    manager, _ = _manager()
    payload = {
        "message": {"content": "", "thinking": "private reasoning"},
        "done_reason": "stop", "prompt_eval_count": 100, "eval_count": 200,
    }
    monkeypatch.setattr(manager, "_call_ollama_sync", lambda **_kwargs: payload)
    monkeypatch.setattr(manager, "ollama_async_client", lambda: None)
    with caplog.at_level("INFO"), ModelGatewayScope.bind("ModelPort"):
        with pytest.raises(RuntimeError, match="completed without a visible answer"):
            if method == "sync":
                manager.call_ollama_chat([], caller="ModelPort")
            elif method == "explicit":
                manager.call_ollama_chat_with_model("test-model", [], caller="ModelPort")
            else:
                asyncio.run(manager.call_ollama_chat_async([], caller="ModelPort"))
    assert "private reasoning" not in caplog.text
    assert "prompt_tokens=100" in caplog.text


@pytest.mark.skipif(
    os.environ.get("DADBOT_RUN_REAL_OLLAMA_OPTIONS") != "1",
    reason="requires local Ollama and a downloaded chat model",
)
def test_real_ollama_context_bucket_and_residency():
    model = os.environ.get("DADBOT_OPTIONS_TEST_MODEL", "llama3.2")
    manager, _ = _manager()
    for budget in (1157, 1308):
        with ModelGatewayScope.bind("ModelPort"):
            response = manager.call_ollama_chat_with_model(
                model,
                [{"role": "user", "content": "Reply with the word READY."}],
                caller="ModelPort",
                options={"num_ctx": budget, "num_predict": 8, "temperature": 0.0},
                purpose="request-options probe",
            )
        assert response.message.content.strip()
        print(
            f"Live probe requested_ctx={budget} load_ms={response.load_duration / 1e6:.1f} "
            f"prompt_tokens={response.prompt_eval_count} generated_tokens={response.eval_count}"
        )
        loaded = next(
            item for item in ollama.ps().models
            if item.name in {model, f"{model}:latest"}
        )
        assert loaded.context_length == 2048
        assert loaded.expires_at > datetime.now(UTC) + timedelta(days=1)
