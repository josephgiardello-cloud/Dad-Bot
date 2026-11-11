import sqlite3
from types import SimpleNamespace

import pytest

from dadbot.services.context_service import ContextService
from dadbot.ui.streamlit_context import (
    SQLiteStreamlitThreadStore,
    StreamlitContextService,
    StreamlitReflectionService,
    classify_profile_question,
    serialize_thread_messages,
    set_runtime_thread_id,
)

pytestmark = pytest.mark.unit


def test_thread_serializer_keeps_displayed_history_and_appends_current_input():
    messages = serialize_thread_messages(
        [
            {"role": "user", "text": "Turn off the kitchen light."},
            {"role": "assistant", "text": "I can help with that."},
        ],
        "Then set the thermostat to 68.",
    )

    assert messages == [
        {"role": "user", "content": "Turn off the kitchen light."},
        {"role": "assistant", "content": "I can help with that."},
        {"role": "user", "content": "Then set the thermostat to 68."},
    ]


def test_thread_serializer_does_not_duplicate_current_message_or_non_chat_roles():
    messages = serialize_thread_messages(
        [
            {"role": "user", "text": "Hello"},
            {"role": "tool", "text": "ignored"},
        ],
        "Hello",
    )

    assert messages == [{"role": "user", "content": "Hello"}]


def test_profile_question_classifier_ignores_technical_who_are_you_questions():
    assert classify_profile_question("Who are you in this code?") == ""
    assert classify_profile_question("Who are you?") == "identity"
    assert classify_profile_question("How old are you in this project?") == ""
    assert classify_profile_question("How old are you?") == "age"


def test_streamlit_context_service_returns_thread_and_temporal_context():
    bot = SimpleNamespace(_streamlit_context_messages=[{"role": "user", "content": "Earlier turn"}])
    service = StreamlitContextService(bot)

    context = service.build_context(SimpleNamespace(state={"temporal": {"mood": "neutral"}}))

    assert context["messages"] == [{"role": "user", "content": "Earlier turn"}]
    assert context["temporal"] == {"mood": "neutral"}


def test_five_turn_dependent_conversation_retains_prior_user_and_assistant_context():
    bot = SimpleNamespace(_streamlit_context_messages=[])
    service = StreamlitContextService(bot)
    thread = []

    for turn_number in range(1, 6):
        user_text = "My friend's name is Maria." if turn_number == 1 else f"Turn {turn_number}, continue the plan from before."
        bot._streamlit_context_messages = serialize_thread_messages(thread, user_text)
        rich_context = service.build_context(SimpleNamespace(state={"temporal": {}}))
        received = rich_context["messages"]

        assert received[-1] == {"role": "user", "content": user_text}
        assert "Maria" in received[0]["content"]
        if turn_number > 1:
            assert received[-2] == {"role": "assistant", "content": thread[-1]["text"]}

        remembered_name = received[0]["content"].split("is ", 1)[-1].rstrip(".")
        assistant_text = f"Turn {turn_number} remembers {remembered_name}."
        thread.extend(
            [
                {"role": "user", "text": user_text},
                {"role": "assistant", "text": assistant_text},
            ]
        )

    final_context = service.build_context(
        SimpleNamespace(state={"temporal": {}})
    )
    assert len(final_context["messages"]) == 9
    assert final_context["messages"][0]["content"] == "My friend's name is Maria."
    assert final_context["messages"][-1]["content"] == "Turn 5, continue the plan from before."
    assert "Maria" in thread[-1]["text"]


def test_context_service_falls_back_to_canonical_conversation_history():
    bot = SimpleNamespace(conversation_history=lambda: [{"role": "user", "content": "CLI prior turn"}])
    service = StreamlitContextService(bot)

    context = service.build_context(SimpleNamespace(state={"temporal": {}}))

    assert context["messages"] == [{"role": "user", "content": "CLI prior turn"}]


def test_streamlit_context_retains_canonical_memory_and_temporal_sections(mocker):
    canonical = mocker.Mock(spec=ContextService)
    canonical.build_context.return_value = {
        "memory": "Relevant committed memory",
        "memory_layers": {"structured_tokens": 12},
        "temporal": {"wall_time": "2026-11-11T12:00:00"},
    }
    bot = SimpleNamespace(_streamlit_context_messages=[{"role": "user", "content": "Visible thread"}])
    service = StreamlitContextService(bot, context_service=canonical)
    turn = SimpleNamespace(state={"temporal": {"old": True}})
    result = service.build_context(turn)
    canonical.build_context.assert_called_once_with(turn)
    assert result == {
        **canonical.build_context.return_value,
        "messages": bot._streamlit_context_messages,
    }


def test_streamlit_context_propagates_canonical_memory_failure(mocker):
    canonical = mocker.Mock(spec=ContextService)
    canonical.build_context.side_effect = RuntimeError("memory snapshot unavailable")
    service = StreamlitContextService(context_service=canonical)
    with pytest.raises(RuntimeError, match="memory snapshot unavailable"):
        service.build_context(SimpleNamespace(state={}))


def test_runtime_thread_id_updates_canonical_session_key(bot):
    set_runtime_thread_id(bot, "thread-42")

    assert bot.runtime_state_manager.active_thread_id == bot.active_thread_id == "thread-42"


def test_sqlite_thread_store_survives_reopen_and_omits_audio_payload(tmp_path):
    db_path = tmp_path / "threads.sqlite3"
    store = SQLiteStreamlitThreadStore(db_path, tenant_id="tenant-a")
    store.save_thread(
        "thread-1",
        [
            {"role": "user", "text": "Hello", "time": "10:00"},
            {"role": "assistant", "text": "Hi", "time": "10:01", "audio_base64": "not-persisted"},
        ],
    )

    restored = SQLiteStreamlitThreadStore(db_path, tenant_id="tenant-a").load_threads()

    assert restored["thread-1"] == [
        {"role": "user", "text": "Hello", "time": "10:00"},
        {"role": "assistant", "text": "Hi", "time": "10:01"},
    ]


def test_sqlite_threads_are_tenant_scoped_and_do_not_import_unowned_rows(tmp_path):
    db_path = tmp_path / "threads.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE chat_threads(thread_id TEXT PRIMARY KEY, messages_json TEXT, updated_at TEXT)")
        connection.execute("INSERT INTO chat_threads VALUES ('legacy', '[]', 'old')")
    first = SQLiteStreamlitThreadStore(db_path, tenant_id="tenant-a")
    second = SQLiteStreamlitThreadStore(db_path, tenant_id="tenant-b")
    assert first.load_threads() == second.load_threads() == {}
    first.save_thread("same-id", [{"role": "user", "text": "First", "time": "10:00"}])
    second.save_thread("same-id", [{"role": "user", "text": "Second", "time": "10:01"}])
    first.save_thread("same-id", [])
    assert first.load_threads() == {"same-id": []}
    assert second.load_threads()["same-id"][0]["text"] == "Second"
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT thread_id FROM chat_threads").fetchall() == [("legacy",)]


@pytest.mark.parametrize("payload", ["not-json", "{}", '[{"role":"tool","text":"invalid"}]', "[null]"])
def test_corrupt_thread_history_is_reported_instead_of_replaced_with_empty_history(tmp_path, payload):
    db_path = tmp_path / "threads.sqlite3"
    store = SQLiteStreamlitThreadStore(db_path, tenant_id="tenant-a")
    store.save_thread("thread-1", [])
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE owned_chat_threads SET messages_json=? WHERE tenant_id=? AND thread_id=?",
            (payload, "tenant-a", "thread-1"),
        )
    with pytest.raises(ValueError):
        store.load_threads()


def test_thread_store_revalidates_active_tenant_authority(tmp_path, monkeypatch):
    store = SQLiteStreamlitThreadStore(tmp_path / "threads.sqlite3", tenant_id="tenant-a")

    def revoked(_tenant_id):
        raise PermissionError("Tenant authority revoked")

    monkeypatch.setattr("dadbot_system.delegation.require_tenant_authority_if_active", revoked)
    with pytest.raises(PermissionError, match="revoked"):
        store.load_threads()
    with pytest.raises(PermissionError, match="revoked"):
        store.save_thread("thread-1", [])


def test_reflection_service_delegates_to_internal_state_manager():
    calls = []

    class InternalStateManager:
        def reflect_after_turn(self, **kwargs):
            calls.append(kwargs)
            return {"turn_count": 5}

    bot = SimpleNamespace(last_saved_mood=lambda: "positive")
    service = StreamlitReflectionService(bot, InternalStateManager())
    result = service.reflect(
        SimpleNamespace(user_input="I got the job", state={"mood": "positive"}),
        ("That's great!", False),
    )

    assert result == {"turn_count": 5}
    assert calls == [{"user_input": "I got the job", "current_mood": "positive", "dad_reply": "That's great!"}]
