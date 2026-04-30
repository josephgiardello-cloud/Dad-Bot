import asyncio

import dadbot.managers.runtime_client as runtime_client_module


class FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk


def test_call_ollama_chat_stream_async_uses_litellm_stream_once_and_awaits_callback(bot, monkeypatch):
    bot.LLM_PROVIDER = "openai"
    bot.LLM_MODEL = "gpt-4o-mini"

    guard_calls = []
    delivered_chunks = []
    runtime_calls = []

    def fake_guard(messages, purpose="chat"):
        guard_calls.append({"messages": messages, "purpose": purpose})
        return [{"role": "user", "content": "guarded hi"}]

    async def unexpected_runtime_fallback(messages, options=None, purpose="chat", chunk_callback=None):
        runtime_calls.append(
            {
                "messages": messages,
                "options": options,
                "purpose": purpose,
                "chunk_callback": chunk_callback,
            }
        )
        return "unexpected fallback"

    class FakeLiteLLM:
        async def acompletion(self, **kwargs):
            assert kwargs["model"] == "openai/gpt-4o-mini"
            assert kwargs["messages"] == [{"role": "user", "content": "guarded hi"}]
            assert kwargs["temperature"] == 0.25
            assert kwargs["stream"] is True
            return FakeAsyncStream(
                [
                    {"choices": [{"delta": {"content": "Hello"}}]},
                    {"choices": [{"delta": {"content": " there"}}]},
                ]
            )

    async def collect_chunk(chunk):
        delivered_chunks.append(chunk)

    monkeypatch.setattr(bot, "guard_chat_request_messages", fake_guard)
    monkeypatch.setattr(bot.runtime_client, "call_ollama_chat_stream_async", unexpected_runtime_fallback)
    monkeypatch.setattr(runtime_client_module, "litellm", FakeLiteLLM())

    reply = asyncio.run(
        bot.call_ollama_chat_stream_async(
            [{"role": "user", "content": "Hi"}],
            options={"temperature": 0.25},
            purpose="reply",
            chunk_callback=collect_chunk,
        )
    )

    assert reply == "Hello there"
    assert delivered_chunks == ["Hello", " there"]
    assert len(guard_calls) == 1
    assert guard_calls[0]["purpose"] == "reply"
    assert runtime_calls == []


def test_call_ollama_chat_stream_async_falls_back_to_runtime_client_when_litellm_fails(bot, monkeypatch):
    bot.LLM_PROVIDER = "openai"
    bot.LLM_MODEL = "gpt-4o-mini"

    guard_calls = []
    delivered_chunks = []
    runtime_calls = []

    def fake_guard(messages, purpose="chat"):
        guard_calls.append({"messages": messages, "purpose": purpose})
        return [{"role": "user", "content": "guarded fallback"}]

    class FakeLiteLLM:
        async def acompletion(self, **kwargs):
            raise RuntimeError(f"provider boom for {kwargs['model']}")

    async def fake_runtime_fallback(messages, options=None, purpose="chat", chunk_callback=None):
        runtime_calls.append(
            {
                "messages": messages,
                "options": options,
                "purpose": purpose,
                "chunk_callback": chunk_callback,
            }
        )
        if chunk_callback is not None:
            await chunk_callback("fallback")
            await chunk_callback(" reply")
        return "fallback reply"

    async def collect_chunk(chunk):
        delivered_chunks.append(chunk)

    monkeypatch.setattr(bot, "guard_chat_request_messages", fake_guard)
    monkeypatch.setattr(bot.runtime_client, "call_ollama_chat_stream_async", fake_runtime_fallback)
    monkeypatch.setattr(runtime_client_module, "litellm", FakeLiteLLM())

    reply = asyncio.run(
        bot.call_ollama_chat_stream_async(
            [{"role": "user", "content": "Hi"}],
            options={"temperature": 0.4},
            purpose="reply",
            chunk_callback=collect_chunk,
        )
    )

    assert reply == "fallback reply"
    assert delivered_chunks == ["fallback", " reply"]
    assert len(guard_calls) == 1
    assert runtime_calls == [
        {
            "messages": [{"role": "user", "content": "guarded fallback"}],
            "options": {"temperature": 0.4},
            "purpose": "reply",
            "chunk_callback": collect_chunk,
        }
    ]
