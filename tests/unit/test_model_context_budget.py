from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from dadbot.core.execution_boundary import ModelGatewayScope
from dadbot.managers.prompt_assembly import PromptAssemblyManager
from dadbot.managers.runtime_client import RuntimeClientManager
from dadbot.managers.runtime_model import RuntimeModelManager
from dadbot.services._tool_pipeline_mixin import ToolPipelineMixin


def _client():
    def cost(message):
        return len(str(message.get("content") or "").split()) + 8

    def trim(message, budget):
        return {**message, "content": " ".join(str(message["content"]).split()[:max(0, budget - 8)])}

    bot = SimpleNamespace(
        ACTIVE_MODEL="thinking-model", CONTEXT_TOKEN_BUDGET=6000, RESERVED_RESPONSE_TOKENS=512,
        runtime_config=SimpleNamespace(ollama_keep_alive_seconds=-1),
        effective_context_token_budget=lambda _model=None: 6000,
        adaptive_prompt_pressure_factor=lambda: 1.0,
        message_token_cost=cost, trim_message_to_token_budget=trim,
        prompt_guard_stats=lambda: {}, model_candidates=lambda: [],
        ollama_retryable_errors=lambda: (ConnectionError,), ollama_error_summary=str,
        extract_ollama_message_content=lambda response: response["message"].get("content", ""),
        model_runtime=SimpleNamespace(
            effective_context_token_budget=lambda _model: 6000,
            model_context_length=lambda _model: 32768,
            ollama_show_payload=lambda _model: {"capabilities": ["completion", "thinking"]},
        ),
    )
    bot.guard_chat_request_messages = PromptAssemblyManager(bot).guard_chat_request_messages
    return RuntimeClientManager(bot)


def test_modelfile_runtime_num_ctx_is_not_the_architectural_context_limit():
    bot = SimpleNamespace(
        ACTIVE_MODEL="thinking-model", MODEL_NAME="thinking-model", CONTEXT_TOKEN_BUDGET=6000,
        _model_metadata_cache={"thinking-model": {
            "model_info": {"qwen.context_length": 32768}, "parameters": "num_ctx 4096",
        }},
    )
    runtime = RuntimeModelManager(bot)
    assert runtime.model_context_length() == 32768
    assert runtime.effective_context_token_budget("thinking-model") == 6000
    bot._model_metadata_cache["thinking-model"]["model_info"]["qwen.context_length"] = 2048
    assert runtime.effective_context_token_budget("thinking-model") == 2048


def test_guard_reserves_thinking_space_trims_history_and_preserves_latest_request():
    client = _client()
    latest = {"role": "user", "content": "The current task and required output schema stay intact."}
    system = {"role": "system", "content": " ".join(["persona"] * 4000)}
    messages = [system, {"role": "assistant", "content": " ".join(["old"] * 2000)}, latest]
    guarded, options = client._prepare_ollama_request("thinking-model", messages, {}, "chat")
    assert options["num_ctx"] == 6000
    assert guarded[-1] == latest
    assert not any(message["role"] == "assistant" for message in guarded)
    assert sum(client.bot.message_token_cost(message) for message in guarded) <= 3000
    assert messages[0] == system
    assert len(system["content"].split()) == 4000


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_every_ollama_path_sets_context_and_applies_the_same_guard(monkeypatch, asynchronous, stream):
    client = _client()
    requests = []

    def respond(**kwargs):
        requests.append(kwargs)
        payload = {"message": {"content": "4"}, "done_reason": "stop"}
        return (item for item in [payload]) if stream else payload

    monkeypatch.setattr(client, "_call_ollama_sync", respond)
    monkeypatch.setattr(client, "ollama_async_client", lambda: None)
    client.bot.STREAM_TIMEOUT_SECONDS = 30
    client.bot.STREAM_MAX_CHARS = 100
    messages = [{"role": "user", "content": " ".join(["question"] * 4000)}]
    with ModelGatewayScope.bind("ModelPort"):
        if stream:
            method = client.call_ollama_chat_stream_async if asynchronous else client.call_ollama_chat_stream
        else:
            method = client.call_ollama_chat_async if asynchronous else client.call_ollama_chat
        result = method(messages, caller="ModelPort", options={"temperature": 0.4})
        if asynchronous:
            result = asyncio.run(result)
    assert result == ("4" if stream else {"message": {"content": "4"}, "done_reason": "stop"})
    assert requests[0]["options"] == {"temperature": 0.4, "num_ctx": 6000}
    assert sum(client.bot.message_token_cost(m) for m in requests[0]["messages"]) <= 3000
    assert "think" not in requests[0]


def test_explicit_context_override_is_used_by_the_guard_and_clamped_to_model_capacity():
    client = _client()
    client.bot.model_runtime.model_context_length = lambda _model: 8192
    messages = [{"role": "user", "content": "short question"}]
    _, options = client._prepare_ollama_request("thinking-model", messages, {"num_ctx": 16384}, "chat")
    assert options["num_ctx"] == 8192
    _, options = client._prepare_ollama_request("thinking-model", messages, {"num_ctx": 4096}, "chat")
    assert options["num_ctx"] == 4096


@pytest.mark.parametrize("num_ctx", [0, -1, 128])
def test_invalid_context_override_is_rejected(num_ctx):
    with pytest.raises(ValueError, match="at least 256"):
        _client()._prepare_ollama_request("thinking-model", [], {"num_ctx": num_ctx}, "chat")


def test_reflection_trimming_terminates_when_prompt_overhead_exceeds_budget():
    client = _client()
    client.bot.prompt_assembly = SimpleNamespace(_compute_prompt_budget=lambda: 128)
    calls = []

    def oversized(message):
        calls.append(message)
        assert len(calls) <= 20, "Reflection trimmer failed to make progress"
        return 1000

    client.bot.message_token_cost = oversized
    result = ToolPipelineMixin._reflection_prompt(
        SimpleNamespace(bot=client.bot), "question", "query", "observation " * 100,
    )
    observation = result.split("Observed result:\n", 1)[1].split("\n\nReturn ONLY", 1)[0]
    assert len(observation) <= 96
    assert observation.endswith("...")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("visible_content", ["", "4"])
def test_streaming_exhaustion_checks_accumulated_content(asynchronous, visible_content):
    client = _client()
    client.bot.STREAM_TIMEOUT_SECONDS = 30
    client.bot.STREAM_MAX_CHARS = 100
    chunks = [
        {"message": {"content": visible_content}},
        {"message": {"content": ""}, "done_reason": "length"},
    ]

    def collect():
        if not asynchronous:
            return client._collect_stream_chunks(iter(chunks), purpose="chat", chunk_callback=None)

        async def stream():
            for chunk in chunks:
                yield chunk

        return asyncio.run(client._collect_stream_chunks_async(stream(), purpose="chat", chunk_callback=None))

    if visible_content:
        assert collect()[0] == ["4"]
    else:
        with pytest.raises(RuntimeError, match="exhausted.*without a visible answer"):
            collect()


@pytest.mark.parametrize("stream", [False, True])
def test_native_async_transport_uses_the_same_context_guard(monkeypatch, stream):
    client = _client()
    requests = []
    client.bot.STREAM_TIMEOUT_SECONDS = 30
    client.bot.STREAM_MAX_CHARS = 100

    async def respond(**kwargs):
        requests.append(kwargs)
        payload = {"message": {"content": "4"}, "done_reason": "stop"}
        if not stream:
            return payload

        async def chunks():
            yield payload

        return chunks()

    monkeypatch.setattr(client, "ollama_async_client", lambda: SimpleNamespace(chat=respond))
    with ModelGatewayScope.bind("ModelPort"):
        method = client.call_ollama_chat_stream_async if stream else client.call_ollama_chat_async
        response = asyncio.run(method(
            [{"role": "user", "content": " ".join(["question"] * 4000)}], caller="ModelPort",
        ))
    assert response == ("4" if stream else {"message": {"content": "4"}, "done_reason": "stop"})
    assert requests[0]["options"]["num_ctx"] == 6000
    assert sum(client.bot.message_token_cost(message) for message in requests[0]["messages"]) <= 3000


def test_unknown_model_token_estimate_uses_the_live_config_projection():
    bot = SimpleNamespace(
        ACTIVE_MODEL="custom", MODEL_NAME="custom", approx_chars_per_token=3.5,
        _model_metadata_cache={"custom": {}},
    )
    assert RuntimeModelManager(bot).model_chars_per_token_estimate() == 3.5


@pytest.mark.parametrize("asynchronous", [False, True])
def test_reasoning_only_exhaustion_is_an_explicit_failure_not_a_success(monkeypatch, asynchronous):
    client = _client()
    monkeypatch.setattr(client, "ollama_async_client", lambda: None)
    monkeypatch.setattr(client, "_call_ollama_sync", lambda **_kwargs: {
        "message": {"content": "", "thinking": "private reasoning"},
        "done_reason": "length", "prompt_eval_count": 1820, "eval_count": 2276,
    })
    with ModelGatewayScope.bind("ModelPort"), pytest.raises(RuntimeError, match="exhausted.*without a visible answer"):
        if asynchronous:
            asyncio.run(client.call_ollama_chat_async([{"role": "user", "content": "hi"}], caller="ModelPort"))
        else:
            client.call_ollama_chat_with_model(
                "thinking-model", [{"role": "user", "content": "hi"}], caller="ModelPort",
            )
