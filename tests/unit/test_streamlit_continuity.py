import ast
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dadbot.ui.streamlit_context import SQLiteStreamlitThreadStore

pytestmark = pytest.mark.unit


@pytest.fixture
def open_app(make_test_dadbot, monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[2] / "dad_streamlit.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    build = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_STREAMLIT_RUNTIME_BUILD" for target in node.targets)
    )
    monkeypatch.delenv("DADBOT_SPATIAL_AVATAR_URL", raising=False)

    def create(query=None, tenant_id="tenant-a"):
        bot = make_test_dadbot(tenant_id=tenant_id)
        bot._streamlit_thread_store = SQLiteStreamlitThreadStore(
            tmp_path / "threads.sqlite3", tenant_id=tenant_id,
        )
        monkeypatch.setattr(bot, "process_user_message", lambda *_args, **_kwargs: ("Synthetic test reply.", False))
        app = AppTest.from_file(str(path))
        app.session_state["bot"] = bot
        app.session_state["_streamlit_runtime_build"] = build
        app.session_state["auto_speak_replies"] = False
        if query is not None:
            app.query_params["thread"] = query
        app.run(timeout=20)
        assert not app.exception, list(app.exception)
        return app, bot

    return create


def _button(app, label):
    return next(button for button in app.button if button.label == label)


def test_created_empty_thread_and_fork_survive_new_browser_session(open_app):
    app, _ = open_app()
    _button(app, "New chat").click().run(timeout=20)
    thread_id = app.session_state["active_thread_id"]
    assert app.query_params["thread"] == [thread_id]
    restored, _ = open_app(thread_id)
    assert restored.session_state["active_thread_id"] == thread_id
    assert restored.session_state["thread-picker"] == thread_id
    restored.chat_input[0].set_value("Synthetic continuity check.").run(timeout=20)
    assert not restored.exception
    history = restored.session_state["chat_history"]
    _button(restored, "Duplicate").click().run(timeout=20)
    fork_id = restored.session_state["active_thread_id"]
    forked, _ = open_app(fork_id)
    assert forked.session_state["chat_history"] == history
    assert forked.query_params["thread"] == [fork_id]
    assert forked.session_state["thread-picker"] == fork_id


def test_tab_selection_is_independent_and_existing_thread_picker_updates_url(open_app):
    first, _ = open_app()
    _button(first, "New chat").click().run(timeout=20)
    first_id = first.session_state["active_thread_id"]
    second, _ = open_app()
    _button(second, "New chat").click().run(timeout=20)
    second_id = second.session_state["active_thread_id"]
    assert first_id != second_id
    first_reload, _ = open_app(first_id)
    second_reload, _ = open_app(second_id)
    assert first_reload.session_state["active_thread_id"] == first_id
    assert second_reload.session_state["active_thread_id"] == second_id
    next(item for item in first_reload.selectbox if item.label == "Active conversation").set_value(second_id).run()
    assert first_reload.session_state["active_thread_id"] == second_id
    assert first_reload.query_params["thread"] == [second_id]


def test_unknown_or_other_tenant_thread_url_does_not_create_or_expose_history(open_app):
    owner, _ = open_app()
    _button(owner, "New chat").click().run(timeout=20)
    thread_id = owner.session_state["active_thread_id"]
    owner.chat_input[0].set_value("Owner-only synthetic message.").run(timeout=20)
    other, _ = open_app(thread_id, tenant_id="tenant-b")
    assert other.session_state["active_thread_id"] == "default"
    assert thread_id not in other.session_state["chat_threads"]
    assert other.query_params["thread"] == ["default"]
    assert other.warning
    assert not other.session_state["chat_history"]


def test_failed_turn_history_and_active_selection_survive_reload(open_app, monkeypatch):
    app, bot = open_app()
    _button(app, "New chat").click().run(timeout=20)
    thread_id = app.session_state["active_thread_id"]

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("Synthetic turn failure")

    monkeypatch.setattr(bot, "process_user_message", unavailable)
    app.chat_input[0].set_value("Synthetic failed turn.").run(timeout=20)
    assert not app.exception
    assert len(app.session_state["message_outbox"]) == 1
    restored, _ = open_app(thread_id)
    assert restored.session_state["active_thread_id"] == thread_id
    assert restored.session_state["chat_history"][-1]["text"] == "Sorry, I hit a snag: Synthetic turn failure"


def test_thread_save_failure_does_not_publish_a_successful_empty_thread(open_app, monkeypatch):
    app, bot = open_app()
    before = dict(app.session_state["chat_threads"])

    def unavailable(*_args, **_kwargs):
        raise OSError("Synthetic display-store outage")

    monkeypatch.setattr(bot._streamlit_thread_store, "save_thread", unavailable)
    _button(app, "New chat").click().run(timeout=20)
    assert app.exception
    assert "Synthetic display-store outage" in app.exception[0].message
    assert app.session_state["chat_threads"] == before
    assert app.session_state["active_thread_id"] == "default"


def test_tenant_change_clears_previous_browser_state(open_app):
    app, _ = open_app()
    _button(app, "New chat").click().run(timeout=20)
    thread_id = app.session_state["active_thread_id"]
    app.chat_input[0].set_value("Owner-only synthetic message.").run(timeout=20)
    app.session_state["message_outbox"] = [{"prompt": "Owner-only pending item"}]
    app.session_state["runtime_guardrails"] = [{"detail": "Owner-only runtime detail"}]
    _, other_bot = open_app(tenant_id="tenant-b")
    app.session_state["bot"] = other_bot
    app.run(timeout=20)
    assert not app.exception
    assert thread_id not in app.session_state["chat_threads"]
    assert app.session_state["active_thread_id"] == "default"
    assert app.query_params["thread"] == ["default"]
    assert not app.session_state["chat_history"]
    assert not app.session_state["message_outbox"]
    assert not app.session_state["runtime_guardrails"]
