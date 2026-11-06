import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_runtime_thread_id_updates_canonical_session_key(bot):
    set_runtime_thread_id(bot, "thread-42")

    assert bot.runtime_state_manager.active_thread_id == bot.active_thread_id == "thread-42"


def test_sqlite_thread_store_survives_reopen_and_omits_audio_payload():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        db_path = Path(temp_dir) / "threads.sqlite3"
        store = SQLiteStreamlitThreadStore(db_path)
        store.save_thread(
            "thread-1",
            [
                {"role": "user", "text": "Hello", "time": "10:00"},
                {"role": "assistant", "text": "Hi", "time": "10:01", "audio_base64": "not-persisted"},
            ],
        )

        restored = SQLiteStreamlitThreadStore(db_path).load_threads()

        assert restored["thread-1"] == [
            {"role": "user", "text": "Hello", "time": "10:00"},
            {"role": "assistant", "text": "Hi", "time": "10:01"},
        ]


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