import ast
import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from dadbot.config import DadBotConfig
from dadbot.config_schema import ConfigSchema
from dadbot.context import ContextBuilder
from dadbot.core.execution_boundary import ModelGatewayScope
from dadbot.core.turn_mixin import DadBotTurnMixin
from dadbot.managers import runtime_client as runtime_client_module
from dadbot.managers.runtime_client import RuntimeClientManager
from dadbot.registry import ServiceRegistry
from dadbot.runtime.mcp import approval_service as mcp_approval_service_module
from dadbot.runtime.mcp.approval_service import MCPApprovalService
from dadbot.runtime.mcp.tool_client import (
    MCPApprovalRequest,
    run_mcp_tool_loop,
)
from dadbot.ui.chat_stream import draft_chunk_callback, run_with_chat_drafts
from dadbot_system.contracts import DEFAULT_MODEL, ServiceConfig


def test_model_defaults_and_explicit_overrides(monkeypatch):
    monkeypatch.delenv("DADBOT_LLM_MODEL", raising=False)
    assert DadBotConfig().active_model == DEFAULT_MODEL
    assert ConfigSchema().model_candidates == [DEFAULT_MODEL]
    assert ServiceConfig().default_model == DEFAULT_MODEL
    assert ServiceConfig.from_dict({}).default_model == DEFAULT_MODEL
    assert DadBotConfig(model_name="qwen3:14b").active_model == "qwen3:14b"
    monkeypatch.setenv("DADBOT_LLM_MODEL", "custom:latest")
    assert DadBotConfig().llm_model == "custom:latest"


@pytest.mark.parametrize("reuse_registry", [False, True])
def test_registry_boot_returns_the_requested_registry(tmp_path, reuse_registry):
    existing = ServiceRegistry()
    sentinel = object()
    existing.register("existing_service", sentinel)
    registry = ServiceRegistry.boot(
        config_path=str(tmp_path / "config.yaml"),
        registry=existing if reuse_registry else None,
    )
    assert (registry is existing) is reuse_registry
    assert registry.get("existing_service", optional=True) is (sentinel if reuse_registry else None)
    assert registry.get("turn_graph").registry is registry


def test_backend_container_boots_orchestrator_into_its_manager_registry(bot):
    bot.config.strict_graph_mode = False
    registry = bot.services.registry
    manager = registry.get("memory_manager")
    orchestrator = bot._get_turn_orchestrator()
    assert bot.lifecycle_state == "ready"
    assert bot.model_port is not None
    assert orchestrator is bot.services.turn_orchestrator
    assert orchestrator.registry is bot.service_registry is registry
    assert orchestrator.graph.registry is registry
    assert orchestrator.control_plane.graph is orchestrator.graph
    assert orchestrator.control_plane.kernel_executor.__self__ is orchestrator
    assert orchestrator.control_plane.kernel_executor.__func__ is type(orchestrator)._execute_job
    assert registry.get("turn_graph").registry is registry
    assert registry.get("persistence_service") is registry.get("storage")
    assert registry.get("memory_manager") is manager


def test_backend_container_orchestrator_uses_registered_inference_service(bot):
    bot.config.strict_graph_mode = False
    registry = bot.services.registry
    ServiceRegistry.boot(
        config_path=bot.config.turn_graph_config_path,
        bot=bot,
        registry=registry,
    )
    inference_service = object()
    registry.register("llm", inference_service)

    orchestrator = bot._get_turn_orchestrator()

    assert orchestrator.registry is registry
    assert orchestrator.graph.registry is registry
    assert orchestrator.registry.get("llm") is inference_service


def test_draft_rendering_precedes_completion_and_stays_on_caller_thread():
    rendered = threading.Event()
    caller_thread = threading.get_ident()
    updates = []

    def run_turn():
        assert threading.get_ident() != caller_thread
        draft_chunk_callback.get()("Provisional")
        assert rendered.wait(5), "Draft was buffered until turn completion"
        draft_chunk_callback.get()(" answer")
        return "Final validated answer", False

    def render(text):
        assert threading.get_ident() == caller_thread
        updates.append(text)
        rendered.set()

    assert run_with_chat_drafts(run_turn, render) == ("Final validated answer", False)
    assert updates == ["Provisional", "Provisional answer"]
    assert draft_chunk_callback.get() is None


def test_draft_turn_failure_propagates_and_cleans_up():
    def run_turn():
        draft_chunk_callback.get()("Incomplete draft")
        raise RuntimeError("final validation failed")

    with pytest.raises(RuntimeError, match="final validation failed"):
        run_with_chat_drafts(run_turn, lambda _text: None)
    assert draft_chunk_callback.get() is None


def test_coroutine_thread_bridge_preserves_draft_context():
    updates = []
    callback = updates.append
    token = draft_chunk_callback.set(callback)

    async def run_turn():
        assert draft_chunk_callback.get() is callback
        draft_chunk_callback.get()("A streamed sentence.")
        return "Final sentence."

    try:
        assert DadBotTurnMixin._run_coro_in_thread(run_turn()) == "Final sentence."
        assert draft_chunk_callback.get() is callback
        assert updates == ["A streamed sentence."]
    finally:
        draft_chunk_callback.reset(token)


def test_three_chunks_render_live_across_coroutine_and_to_thread_bridges():
    caller_thread = threading.get_ident()
    rendered = [threading.Event() for _ in range(3)]
    updates = []
    completed = threading.Event()

    def generate():
        assert threading.get_ident() != caller_thread
        callback = draft_chunk_callback.get()
        assert callback is not None
        for index, chunk in enumerate(["One ", "two ", "three."]):
            callback(chunk)
            assert rendered[index].wait(5), "render_draft did not fire before model completion"
        return "One two three."

    async def run_agent():
        return await asyncio.to_thread(generate)

    def run_turn():
        reply = DadBotTurnMixin._run_coro_in_thread(run_agent())
        completed.set()
        return reply, False

    def render_draft(text):
        assert threading.get_ident() == caller_thread
        assert not completed.is_set()
        updates.append(text)
        rendered[len(updates) - 1].set()

    assert run_with_chat_drafts(run_turn, render_draft) == ("One two three.", False)
    assert updates == ["One ", "One two ", "One two three."]
    assert completed.is_set()
    assert draft_chunk_callback.get() is None


def test_mcp_approval_is_stored_for_the_active_streamlit_thread(monkeypatch):
    monkeypatch.setenv("DADBOT_HA_MCP_URL", "http://home-assistant.example/mcp")
    monkeypatch.setenv("DADBOT_HA_MCP_ALLOWED_TOOLS", "light_control")
    proposal = json.dumps(
        {
            "should_continue": True,
            "action_input": "I can turn the kitchen light off.",
            "reasoning": "A device action was requested.",
            "tool_calls": [{"name": "light_control", "arguments": {"entity_id": "light.kitchen"}}],
            "confidence": 0.9,
        }
    )
    request = asyncio.run(
        run_mcp_tool_loop(
            [{"role": "user", "content": "Turn off light.kitchen"}],
            [{"name": "light_control", "input_schema": {"type": "object"}}],
            model_call=lambda _messages: proposal,
        )
    )
    assert isinstance(request, MCPApprovalRequest)

    bot = SimpleNamespace(
        active_thread_id="backend-thread",
        _streamlit_active_thread_id="streamlit-thread",
        ACTIVE_MODEL=DEFAULT_MODEL,
        prompt_assembly=SimpleNamespace(build_request_system_prompt=lambda *_args: "Dad persona"),
        config=SimpleNamespace(active_model=DEFAULT_MODEL),
        model_port=SimpleNamespace(generate=lambda *_args, **_kwargs: ""),
    )
    service = MCPApprovalService(bot)

    async def return_pending(*_args, **_kwargs):
        return request

    monkeypatch.setattr(mcp_approval_service_module, "run_mcp_with_fallback", return_pending)
    reply, should_end = asyncio.run(
        service.generate_candidate(
            SimpleNamespace(user_input="Turn off light.kitchen", attachments=None, state={}),
            {"messages": [{"role": "user", "content": "Turn off light.kitchen"}]},
        ),
    )

    assert should_end is False
    assert "review" in reply.lower()
    assert service.get_pending_mcp_approval("streamlit-thread") is request
    assert service.get_pending_mcp_approval("backend-thread") is None


def test_persona_preserves_profile_rules_and_tool_honesty():
    bot = SimpleNamespace(profile_runtime=SimpleNamespace(
        effective_behavior_rules=lambda: ["Keep established facts."],
        style={"name": "Dad"},
    ))
    prompt = ContextBuilder(bot).build_core_persona_prompt()
    assert "Keep established facts." in prompt
    assert "Don't repeat pet names" in prompt
    assert "never invent shared experiences" in prompt
    assert "confirming tool result" in prompt


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_runtime_preserves_selected_model_and_reasoning_defaults(monkeypatch, asynchronous, stream):
    requests = []
    chunks = []

    def chat(**kwargs):
        requests.append(kwargs)
        response = {"message": {"content": "Hello"}}
        return (item for item in [response]) if kwargs.get("stream") else response

    monkeypatch.setattr(RuntimeClientManager, "_call_ollama_sync", staticmethod(chat))
    bot = SimpleNamespace(
        ACTIVE_MODEL="old-model",
        model_runtime=SimpleNamespace(
            normalized_llm_provider=lambda: "ollama",
            normalized_llm_model=lambda model: model or DEFAULT_MODEL,
            resolve_temperature=lambda _options: 0.7,
        ),
        model_candidates=lambda: [],
        guard_chat_request_messages=lambda messages, **_kwargs: messages,
        ollama_retryable_errors=lambda: (ConnectionError,),
        ollama_error_summary=str,
        extract_ollama_message_content=lambda response: response["message"]["content"],
        STREAM_TIMEOUT_SECONDS=30,
        STREAM_MAX_CHARS=1000,
    )
    client = RuntimeClientManager(bot)
    monkeypatch.setattr(client, "ollama_async_client", lambda: None)
    with ModelGatewayScope.bind("ModelPort"):
        kwargs = {
            "caller": "ModelPort",
            "model": DEFAULT_MODEL,
            "stream": stream,
            "chunk_callback": chunks.append,
            "options": {"num_ctx": 4096},
        }
        if asynchronous:
            result = asyncio.run(client.call_llm_async([], **kwargs))
        else:
            result = client.call_llm([], **kwargs)
    assert result == ("Hello" if stream else {"message": {"content": "Hello"}})
    assert requests[0]["model"] == DEFAULT_MODEL
    assert "think" not in requests[0]
    assert requests[0]["options"] == {"num_ctx": 4096, "temperature": 0.7}
    assert chunks == (["Hello"] if stream else [])


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("stalled", [False, True])
def test_ollama_transport_timeout_and_cleanup(monkeypatch, asynchronous, stream, stalled):
    clients = []
    requests = []
    client_type = runtime_client_module.ollama.AsyncClient if asynchronous else runtime_client_module.ollama.Client

    def respond(request):
        requests.append(request)
        assert request.extensions["timeout"] == {
            key: 120.0 for key in ("connect", "read", "write", "pool")
        }
        if stalled:
            raise httpx.ReadTimeout("Ollama stopped responding", request=request)
        payload = {"message": {"role": "assistant", "content": "4"}, "done": True}
        return httpx.Response(200, content=json.dumps(payload) + "\n")

    def create_client(**kwargs):
        client = client_type(transport=httpx.MockTransport(respond), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(runtime_client_module.ollama, "AsyncClient" if asynchronous else "Client", create_client)
    manager = RuntimeClientManager(SimpleNamespace())
    kwargs = {"model": DEFAULT_MODEL, "messages": [{"role": "user", "content": "2 + 2?"}], "stream": stream}

    async def run_async():
        client = manager.ollama_async_client()
        try:
            result = await client.chat(**kwargs)
            return [chunk async for chunk in result] if stream else result
        finally:
            await manager._close_ollama_async_client_instance(client)

    def run():
        if asynchronous:
            return asyncio.run(run_async())
        result = manager._call_ollama_sync(**kwargs)
        return list(result) if stream else result

    if stalled:
        with pytest.raises(httpx.ReadTimeout, match="Ollama stopped responding"):
            run()
    else:
        result = run()
        assert (result[0] if stream else result).message.content == "4"
    assert len(requests) == 1
    assert clients[0]._client.is_closed


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("typed_response", [False, True])
def test_stream_collectors_preserve_whitespace_and_hide_thinking(asynchronous, typed_response):
    responses = [
        {"message": {"role": "assistant", "content": "", "thinking": "Private reasoning"}},
        *[
            {"message": {"role": "assistant", "content": text}}
            for text in ["Hello", " ", "world", ".\n", "Next line."]
        ],
        {"message": {"role": "assistant", "content": ""}, "done": True},
    ]
    stream_responses = (
        [runtime_client_module.ollama.ChatResponse(**item) for item in responses]
        if typed_response else responses
    )
    client = RuntimeClientManager(SimpleNamespace(
        STREAM_TIMEOUT_SECONDS=30,
        STREAM_MAX_CHARS=1000,
        extract_ollama_message_content=lambda response: (
            response.get("message", {}).get("content", "")
            if isinstance(response, dict)
            else response.message.content
        ),
    ))
    updates = []
    if asynchronous:
        async def stream():
            for response in stream_responses:
                yield response

        chunks, truncated = asyncio.run(client._collect_stream_chunks_async(
            stream(), purpose="chat", chunk_callback=updates.append,
        ))
    else:
        chunks, truncated = client._collect_stream_chunks(
            iter(stream_responses), purpose="chat", chunk_callback=updates.append,
        )
    assert not truncated
    assert updates == ["Hello", " ", "world", ".\n", "Next line."]
    assert "".join(chunks) == "Hello world.\nNext line."


@pytest.mark.parametrize("asynchronous", [False, True])
def test_cloud_stream_fallback_preserves_local_model_selection(monkeypatch, asynchronous):
    class FailedProvider:
        def completion(self, **_kwargs):
            raise ConnectionError("cloud unavailable")

        async def acompletion(self, **_kwargs):
            raise ConnectionError("cloud unavailable")

    bot = SimpleNamespace(
        model_runtime=SimpleNamespace(
            normalized_llm_provider=lambda: "openai",
            normalized_llm_model=lambda _model: "cloud-model",
            resolve_temperature=lambda _options: 0.7,
            litellm_model_identifier=lambda model: f"openai/{model}",
        ),
        guard_chat_request_messages=lambda messages, **_kwargs: messages,
    )
    client = RuntimeClientManager(bot)
    monkeypatch.setattr(runtime_client_module, "litellm", FailedProvider())

    def fallback(messages, caller=None, options=None, purpose="chat", chunk_callback=None):
        return "Local fallback"

    async def async_fallback(messages, caller=None, options=None, purpose="chat", chunk_callback=None):
        return "Local fallback"

    monkeypatch.setattr(client, "call_ollama_chat_stream", fallback)
    monkeypatch.setattr(client, "call_ollama_chat_stream_async", async_fallback)
    with ModelGatewayScope.bind("ModelPort"):
        if asynchronous:
            result = asyncio.run(client.call_llm_async([], caller="ModelPort", stream=True))
        else:
            result = client.call_llm([], caller="ModelPort", stream=True)
    assert result == "Local fallback"


@pytest.mark.parametrize("failed_turn", [False, True])
def test_chat_ui_only_saves_and_speaks_canonical_reply(monkeypatch, failed_turn):
    from streamlit.testing.v1 import AppTest

    path = Path(__file__).resolve().parents[2] / "dad_streamlit.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    build = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_STREAMLIT_RUNTIME_BUILD" for target in node.targets)
    )
    spoken = []

    def get_audio(_manager, text):
        spoken.append(text)
        return None, "No audio in test", "audio/mpeg"

    def process_message(_text, attachments=None):
        if failed_turn:
            raise RuntimeError("Test validation failure")
        return "Final validated response.", False

    monkeypatch.delenv("DADBOT_HA_MCP_URL", raising=False)
    monkeypatch.delenv("DADBOT_SPATIAL_AVATAR_URL", raising=False)
    monkeypatch.setattr("dadbot.managers.tts.TTSManager.get_tts_audio", get_audio)
    app = AppTest.from_file(str(path))
    test_bot = SimpleNamespace(PROFILE={}, process_user_message=process_message)
    from dadbot.managers.reply_finalization import ReplyFinalizationManager

    test_bot.reply_finalization = ReplyFinalizationManager(test_bot)
    app.session_state["bot"] = test_bot
    app.session_state["_streamlit_runtime_build"] = build
    app.run(timeout=20)
    assert not app.exception
    app.chat_input[0].set_value("Help me organize my desk.").run(timeout=20)
    assert not app.exception
    history = app.session_state["chat_threads"]["default"]
    expected = "Sorry, I hit a snag: Test validation failure" if failed_turn else "Final validated response."
    assert [message["text"] for message in history] == ["Help me organize my desk.", expected]
    assert spoken == [expected]
    next(button for button in app.button if button.label == "Duplicate").click().run(timeout=20)
    assert not app.exception
    fork_id = app.session_state["active_thread_id"]
    assert fork_id.startswith("fork-")
    assert app.session_state["thread-picker"] == fork_id
    assert app.session_state["chat_threads"][fork_id] == history
    next(button for button in app.button if button.label == "New chat").click().run(timeout=20)
    assert not app.exception
    new_id = app.session_state["active_thread_id"]
    assert new_id.startswith("thread-")
    assert app.session_state["thread-picker"] == new_id
    assert app.session_state["chat_threads"][new_id] == []
    assert app.session_state["chat_threads"]["default"] == history
    assert test_bot._streamlit_active_thread_id == new_id


@pytest.mark.parametrize("approved", [False, True])
def test_chat_approval_blocks_input_and_validates_result(monkeypatch, approved):
    from streamlit.testing.v1 import AppTest

    from dadbot.managers.reply_finalization import ReplyFinalizationManager

    path = Path(__file__).resolve().parents[2] / "dad_streamlit.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    build = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_STREAMLIT_RUNTIME_BUILD" for target in node.targets)
    )
    pending = SimpleNamespace(
        tool_name="light_control", arguments={"entity_id": "light.kitchen"}, request_id="test-approval",
    )
    resolved = []
    normalized = []

    async def resolve(thread_id, *, approved):
        nonlocal pending
        resolved.append((thread_id, approved))
        pending = None
        return "  Action completed.  " if approved else "  Action rejected.  "

    def normalize(reply):
        normalized.append(reply)
        return ReplyFinalizationManager.normalize_reply(reply)

    registry = ServiceRegistry()
    registry.register("mcp_tool_service", SimpleNamespace(
        get_pending_mcp_approval=lambda _thread_id: pending,
        resolve_mcp_approval=resolve,
    ))
    test_bot = SimpleNamespace(
        PROFILE={}, services=SimpleNamespace(registry=registry),
        reply_finalization=SimpleNamespace(normalize_reply=normalize),
    )
    monkeypatch.delenv("DADBOT_SPATIAL_AVATAR_URL", raising=False)
    app = AppTest.from_file(str(path))
    app.session_state["bot"] = test_bot
    app.session_state["_streamlit_runtime_build"] = build
    app.run(timeout=20)
    assert not app.exception
    assert not app.chat_input
    label = "Approve this action" if approved else "Reject this action"
    next(button for button in app.button if button.label == label).click().run(timeout=20)
    assert not app.exception
    assert resolved == [("default", approved)]
    assert len(normalized) == 1
    expected = "Action completed." if approved else "Action rejected."
    assert app.session_state["chat_threads"]["default"][-1]["text"] == expected
    assert app.chat_input
