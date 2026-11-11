import ast
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from dadbot.runtime_core.streamlit_runtime import StreamlitRuntime
from dadbot.ui.streamlit_context import StreamlitBotPool, set_runtime_thread_id, streamlit_turn_lock

pytestmark = pytest.mark.unit


def test_pool_reuses_tenant_runtime_and_closes_old_build(make_test_dadbot):
    pool = StreamlitBotPool()
    calls = []

    def build(tenant_id):
        calls.append(tenant_id)
        return make_test_dadbot(tenant_id=tenant_id)

    first = pool.get("tenant-a", "one", build)
    assert pool.get("tenant-a", "one", build) is first
    other = pool.get("tenant-b", "one", build)
    assert other is not first
    replacement = pool.get("tenant-a", "two", build)
    assert first._shutdown_completed
    assert replacement is not first
    assert calls == ["tenant-a", "tenant-b", "tenant-a"]
    pool.close()
    assert other._shutdown_completed
    assert replacement._shutdown_completed


def test_pool_does_not_return_closed_runtime_after_failed_upgrade(make_test_dadbot):
    pool = StreamlitBotPool()
    first = pool.get("tenant-a", "one", lambda tenant_id: make_test_dadbot(tenant_id=tenant_id))

    def unavailable(_tenant_id):
        raise OSError("Synthetic startup failure")

    with pytest.raises(OSError, match="startup failure"):
        pool.get("tenant-a", "two", unavailable)
    replacement = pool.get("tenant-a", "one", lambda tenant_id: make_test_dadbot(tenant_id=tenant_id))
    assert replacement is not first
    assert first._shutdown_completed
    pool.close()


def test_shared_turn_context_and_selection_are_serialized(bot, monkeypatch):
    tree = ast.parse((Path(__file__).resolve().parents[2] / "dad_streamlit.py").read_text(encoding="utf-8"))
    node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_process_user_message_with_thread"
    )
    local = threading.local()
    observed = []
    running = 0
    maximum_running = 0

    def process(user_input, **_kwargs):
        nonlocal running, maximum_running
        running += 1
        maximum_running = max(maximum_running, running)
        time.sleep(0.03)
        observed.append((user_input, bot.active_thread_id, list(bot._streamlit_context_messages)))
        running -= 1
        return "Synthetic response", False

    monkeypatch.setattr(bot, "process_user_message", process)
    namespace = {
        "bot": bot,
        "Any": Any,
        "streamlit_turn_lock": streamlit_turn_lock,
        "_active_thread_id": lambda: local.thread_id,
        "_thread_messages": lambda: [{"role": "user", "text": local.thread_id}],
        "serialize_thread_messages": lambda messages, _input: messages,
        "set_runtime_thread_id": set_runtime_thread_id,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<shared-ui-test>", "exec"), namespace)

    def send(thread_id):
        local.thread_id = thread_id
        return namespace["_process_user_message_with_thread"](thread_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(send, ["thread-a", "thread-b"]))
    assert results == [("Synthetic response", False)] * 2
    assert maximum_running == 1
    assert {item[0] for item in observed} == {"thread-a", "thread-b"}
    for prompt, thread_id, messages in observed:
        assert thread_id == prompt
        assert messages == [{"role": "user", "text": thread_id}]
    assert bot._streamlit_context_messages == []


def test_media_adapter_uses_bound_thread_and_preserves_canonical_strict_mode(bot, monkeypatch):
    from dadbot.ui import interaction_controller

    bot.config.strict_graph_mode = True
    monkeypatch.setattr(
        interaction_controller.st, "session_state",
        {"bot": bot, "active_thread_id": "voice-thread"},
    )

    def forbidden_standalone():
        raise AssertionError("Media built a second runtime")

    monkeypatch.setattr(interaction_controller, "_standalone_runtime", forbidden_standalone)
    runtime = interaction_controller.get_runtime()
    assert runtime.bot is bot
    assert runtime.canonical_ui
    assert runtime.api.active_thread_id == "voice-thread"
    assert bot.config.strict_graph_mode
    set_runtime_thread_id(bot, "other-thread")
    assert runtime.api.active_thread_id == "voice-thread"
    assert runtime.api.active_chat_thread()["thread_id"] == "voice-thread"
    assert interaction_controller.get_chat_event_api()._bot is bot


def test_canonical_media_disables_legacy_fast_path_and_scopes_turn(bot, monkeypatch):
    runtime = StreamlitRuntime(bot, canonical_ui=True, thread_id="voice-thread")
    calls = []

    def send(**kwargs):
        calls.append((kwargs, bot.active_thread_id, bot._streamlit_context_messages))
        return {"reply": "Synthetic media response"}

    monkeypatch.setattr(runtime.api, "send_user_message", send)
    bot._streamlit_context_messages = [{"role": "user", "content": "Other thread context"}]
    result = runtime.send_user_message(thread_id="voice-thread", content="hello")
    assert result == {"reply": "Synthetic media response"}
    assert calls == [(
        {"thread_id": "voice-thread", "content": "hello", "attachments": None, "metadata": {"disable_fast_path": True}},
        "voice-thread", [],
    )]
