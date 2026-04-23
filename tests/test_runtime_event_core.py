from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dadbot.runtime_core import AgentRuntime, ConversationStore, EventBus, RuntimeEventAPI, FileEventJournal, new_event
from dadbot.runtime_core.services import UserMessageResult


@dataclass
class FakeServices:
    reply: str = "I hear you, buddy."
    mood: str = "neutral"
    should_end: bool = False
    memory_writes: list[tuple[str, dict]] | None = None

    def handle_user_message(self, *, thread_id: str, text: str, attachments: list[dict] | None = None) -> UserMessageResult:
        return UserMessageResult(
            reply=self.reply,
            should_end=self.should_end,
            mood=self.mood,
            pipeline={"final_path": "model_reply", "reply_source": "model_generation", "steps": []},
            active_rules=["be_warm", "be_grounded"],
        )

    def write_memory(self, *, thread_id: str, payload: dict) -> None:
        if self.memory_writes is None:
            self.memory_writes = []
        self.memory_writes.append((thread_id, dict(payload or {})))


def test_runtime_processes_user_message_into_assistant_reply_and_store_state():
    services = FakeServices()
    store = ConversationStore()
    runtime = AgentRuntime(services=services, store=store)
    bus = EventBus()

    bus.emit(new_event("user_message", thread_id="t-1", payload={"text": "Hey Dad"}))
    processed = runtime.run_until_idle(bus)

    assert [event.type for event in processed] == ["user_message", "assistant_reply", "thinking_update", "tts_request"]
    messages = store.thread_messages("t-1")
    thinking = store.thread_thinking("t-1")
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hey Dad"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "I hear you, buddy."
    assert thinking["final_path"] == "model_reply"
    assert thinking["active_rules"] == ["be_warm", "be_grounded"]


def test_runtime_emits_photo_request_for_supportive_mood_path():
    services = FakeServices(mood="sad")
    runtime = AgentRuntime(services=services, store=ConversationStore())
    bus = EventBus()

    bus.emit(new_event("user_message", thread_id="t-2", payload={"text": "I am overwhelmed"}))
    processed = runtime.run_until_idle(bus)

    assert [event.type for event in processed] == ["user_message", "assistant_reply", "thinking_update", "photo_request", "tts_request"]


def test_runtime_routes_memory_write_events_to_services():
    services = FakeServices()
    runtime = AgentRuntime(services=services, store=ConversationStore())
    bus = EventBus()

    bus.emit(new_event("memory_write", thread_id="t-3", payload={"summary": "Tony is preparing for interviews"}))
    runtime.run_until_idle(bus)

    assert services.memory_writes == [("t-3", {"summary": "Tony is preparing for interviews"})]


def test_runtime_reduces_assistant_attachment_events_into_last_assistant_message():
    services = FakeServices()
    store = ConversationStore()
    runtime = AgentRuntime(services=services, store=store)
    bus = EventBus()

    bus.emit(new_event("user_message", thread_id="t-4", payload={"text": "Hey Dad"}))
    runtime.run_until_idle(bus)
    bus.emit(
        new_event(
            "assistant_attachment_added",
            thread_id="t-4",
            payload={"attachment": {"type": "image", "note": "Dad took a quick photo for you"}},
        )
    )
    runtime.run_until_idle(bus)

    messages = store.thread_messages("t-4")
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["attachments"] == [{"type": "image", "note": "Dad took a quick photo for you"}]


def test_runtime_event_api_replays_durable_journal_into_projection(tmp_path: Path):
    journal_path = tmp_path / "runtime-events.jsonl"
    first_store = ConversationStore()

    first_api = RuntimeEventAPI(
        runtime=AgentRuntime(services=FakeServices(), store=first_store),
        store=first_store,
        bus=EventBus(),
        journal=FileEventJournal(journal_path),
    )
    first_api.emit_user_message(thread_id="t-5", text="Need advice")
    first_api.process_until_idle(max_events=16)
    first_api.emit_assistant_attachment(
        thread_id="t-5",
        attachment={"type": "image", "note": "Dad took a quick photo for you"},
    )
    first_api.process_until_idle(max_events=16)

    replay_store = ConversationStore()
    replay_api = RuntimeEventAPI(
        runtime=AgentRuntime(services=FakeServices(), store=replay_store),
        store=replay_store,
        bus=EventBus(),
        journal=FileEventJournal(journal_path),
    )

    view = replay_api.get_view("t-5")
    assert [message["role"] for message in view["messages"]] == ["user", "assistant"]
    assert view["messages"][-1]["attachments"] == [{"type": "image", "note": "Dad took a quick photo for you"}]
    assert view["thinking"]["final_path"] == "model_reply"
