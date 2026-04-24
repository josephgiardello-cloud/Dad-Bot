from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import pytest

from dadbot.consumers.streamlit import load_thread_projection
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


@dataclass
class StructuredPipelineServices(FakeServices):
    def handle_user_message(self, *, thread_id: str, text: str, attachments: list[dict] | None = None) -> UserMessageResult:
        _ = thread_id, text, attachments
        return UserMessageResult(
            reply=self.reply,
            should_end=self.should_end,
            mood=self.mood,
            pipeline={
                "final_path": "model_reply",
                "reply_source": "model_generation",
                "steps": [{"name": "seed", "kind": "reasoning", "depends_on": []}],
            },
            active_rules=["be_warm", "be_grounded"],
        )


def _execution_boundary_protocol(processed):
    return [
        {
            "type": event.type,
            "thread_id": event.thread_id,
            "payload": dict(event.payload or {}),
        }
        for event in processed
        if event.type in {"execution_region_started", "execution_region_completed"}
    ]


def _assert_plain_derived_data(value):
    assert not isinstance(value, AgentRuntime)
    assert not isinstance(value, ConversationStore)
    assert not isinstance(value, EventBus)
    assert not isinstance(value, RuntimeEventAPI)
    if isinstance(value, dict):
        for item in value.values():
            _assert_plain_derived_data(item)
        return
    if isinstance(value, list):
        for item in value:
            _assert_plain_derived_data(item)
        return
    if isinstance(value, tuple):
        for item in value:
            _assert_plain_derived_data(item)
        return
    assert value is None or isinstance(value, (str, int, float, bool))


def test_runtime_processes_user_message_into_assistant_reply_and_store_state():
    services = FakeServices()
    store = ConversationStore()
    runtime = AgentRuntime(services=services, store=store)
    bus = EventBus()

    bus.emit(new_event("user_message", thread_id="t-1", payload={"text": "Hey Dad"}))
    processed = runtime.run_until_idle(bus)

    assert [event.type for event in processed] == [
        "user_message",
        "execution_region_started",
        "execution_region_completed",
        "assistant_reply",
        "thinking_update",
        "tts_request",
    ]
    messages = store.thread_messages("t-1")
    thinking = store.thread_thinking("t-1")
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hey Dad"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "I hear you, buddy."
    assert thinking["final_path"] == "model_reply"
    assert thinking["active_rules"] == ["be_warm", "be_grounded"]
    assert [item["type"] for item in store.thread_execution_boundaries("t-1")] == [
        "execution_region_started",
        "execution_region_completed",
    ]


def test_runtime_emits_photo_request_for_supportive_mood_path():
    services = FakeServices(mood="sad")
    runtime = AgentRuntime(services=services, store=ConversationStore())
    bus = EventBus()

    bus.emit(new_event("user_message", thread_id="t-2", payload={"text": "I am overwhelmed"}))
    processed = runtime.run_until_idle(bus)

    assert [event.type for event in processed] == [
        "user_message",
        "execution_region_started",
        "execution_region_completed",
        "assistant_reply",
        "thinking_update",
        "photo_request",
        "tts_request",
    ]


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
    assert [item["type"] for item in view["execution_boundaries"]] == [
        "execution_region_started",
        "execution_region_completed",
    ]


def test_runtime_event_api_exposes_passive_execution_boundary_projection():
    store = ConversationStore()
    api = RuntimeEventAPI(
        runtime=AgentRuntime(services=StructuredPipelineServices(), store=store),
        store=store,
        bus=EventBus(),
    )

    api.emit_user_message(thread_id="t-api-boundary", text="Hello")
    api.process_until_idle(max_events=16)

    projected = api.thread_execution_boundaries("t-api-boundary")
    projected[0]["payload"]["semantic_fields"].append("illegal")

    refreshed = api.thread_execution_boundaries("t-api-boundary")
    assert [item["type"] for item in refreshed] == [
        "execution_region_started",
        "execution_region_completed",
    ]
    assert refreshed[0]["payload"]["semantic_fields"] == [
        "status",
        "result",
        "output",
        "output_hash",
        "execution_trace",
        "diagnostic",
        "metadata",
    ]


def test_runtime_event_api_observer_isolation_view_is_detached_and_stable():
    store = ConversationStore()
    api = RuntimeEventAPI(
        runtime=AgentRuntime(services=StructuredPipelineServices(), store=store),
        store=store,
        bus=EventBus(),
    )

    api.emit_user_message(thread_id="t-api-view", text="Hello")
    api.process_until_idle(max_events=16)

    first_view = api.get_view("t-api-view")
    second_view = api.get_view("t-api-view")

    _assert_plain_derived_data(first_view)
    _assert_plain_derived_data(second_view)
    assert first_view == second_view
    assert first_view is not second_view
    assert first_view["messages"] is not second_view["messages"]
    assert first_view["thinking"] is not second_view["thinking"]
    assert first_view["execution_boundaries"] is not second_view["execution_boundaries"]

    first_view["messages"].append({"role": "observer", "content": "mutation", "attachments": []})
    first_view["thinking"]["final_path"] = "observer_mutation"
    first_view["execution_boundaries"].reverse()

    refreshed = api.get_view("t-api-view")
    assert [message["role"] for message in refreshed["messages"]] == ["user", "assistant"]
    assert refreshed["thinking"]["final_path"] == "model_reply"
    assert [item["type"] for item in refreshed["execution_boundaries"]] == [
        "execution_region_started",
        "execution_region_completed",
    ]


def test_runtime_event_api_get_view_matches_canonical_store_projection():
    store = ConversationStore()
    api = RuntimeEventAPI(
        runtime=AgentRuntime(services=StructuredPipelineServices(), store=store),
        store=store,
        bus=EventBus(),
    )

    api.emit_user_message(thread_id="t-canonical-view", text="Hello")
    api.process_until_idle(max_events=16)

    api_view = api.get_view("t-canonical-view", version="v1")
    store_view = store.thread_view("t-canonical-view", version="v1")

    assert api_view == store_view
    assert api_view is not store_view
    assert api_view["messages"] is not store_view["messages"]
    assert api_view["thinking"] is not store_view["thinking"]
    assert api_view["execution_boundaries"] is not store_view["execution_boundaries"]


def test_runtime_event_api_default_view_is_v2_and_versions_are_additive_and_explicit():
    store = ConversationStore()
    api = RuntimeEventAPI(
        runtime=AgentRuntime(services=StructuredPipelineServices(), store=store),
        store=store,
        bus=EventBus(),
    )

    api.emit_user_message(thread_id="t-versioned-view", text="Hello")
    api.process_until_idle(max_events=16)

    default_view = api.get_view("t-versioned-view")
    v1_view = api.get_view("t-versioned-view", version="v1")
    v2_view = api.get_view("t-versioned-view", version="v2")

    assert default_view == v2_view
    assert set(v1_view) == {"thread_id", "messages", "thinking", "execution_boundaries"}
    assert set(v1_view).issubset(set(v2_view))
    assert v2_view["view_version"] == "v2"
    assert v2_view["schema_policy"] == api.view_schema_policy()
    assert v2_view["schema_policy"]["default_version"] == "v2"
    assert v2_view["schema_policy"]["live_version"] == "v2"
    assert v2_view["schema_policy"]["semantics_window"] == {
        "status": "frozen",
        "rules": [
            "no_new_v2_fields_unless_strictly_necessary",
            "no_semantic_reinterpretation_of_existing_fields",
            "only_additive_metadata_allowed",
        ],
    }
    assert v2_view["schema_policy"]["evolution_rule"] == "additive_only"
    assert v2_view["schema_policy"]["allowed_changes"] == [
        "field_additions",
        "optional_metadata_enrichment",
        "derived_fields_without_v1_output_changes",
    ]
    assert v2_view["schema_policy"]["forbidden_changes"] == [
        "remove_v1_fields",
        "reorder_v1_structures",
        "reinterpret_v1_values",
    ]
    assert v2_view["schema_policy"]["version_roles"] == {
        "v1": "replay_compatibility_contract",
        "v2": "current_live_projection_contract",
    }
    assert v2_view["schema_policy"]["compatibility_commitment"] == {
        "v1": "immutable_replay_audit_contract",
        "v2": "must_remain_additive_relative_to_v1",
    }
    assert v2_view["schema_policy"]["historical_reconstruction"] == {
        "v1": "deterministic_historical_reconstruction",
        "v2": "live_projection_with_additive_metadata",
    }
    assert v2_view["schema_policy"]["version_lifecycle"] == {
        "v1": {
            "change_policy": "never_changes",
            "roles": [
                "replay_contract",
                "audit_contract",
                "deterministic_historical_reconstruction",
            ],
        },
        "v2": {
            "change_policy": "additive_metadata_only_while_semantics_window_frozen",
            "compatible_additions": [
                "field_additions",
                "optional_metadata_enrichment",
                "derived_fields_without_v1_output_changes",
            ],
            "next_major_version_trigger": [
                "semantic_reinterpretation_of_existing_fields",
                "removal_of_existing_fields",
                "reordering_of_existing_structures",
                "breaking_shape_change",
            ],
        },
    }
    assert v2_view["schema_policy"]["consumer_rule"]["boundary_package"] == "dadbot/consumers"
    assert v2_view["schema_policy"]["consumer_rule"]["consumer_scopes"] == [
        "streamlit",
        "api_clients",
        "analytics",
    ]
    assert v2_view["schema_policy"]["consumer_rule"]["read_from"] == ["event_api.py", "store.py"]
    assert v2_view["schema_policy"]["consumer_rule"]["output"] == "derived_data_only"
    assert v2_view["thread_id"] == v1_view["thread_id"]
    assert v2_view["messages"] == v1_view["messages"]
    assert v2_view["thinking"] == v1_view["thinking"]
    assert v2_view["execution_boundaries"] == v1_view["execution_boundaries"]


def test_runtime_event_api_versioned_views_are_compatible_across_replay_history(tmp_path: Path):
    journal_path = tmp_path / "runtime-events-versioned.jsonl"
    first_store = ConversationStore()
    first_api = RuntimeEventAPI(
        runtime=AgentRuntime(services=FakeServices(), store=first_store),
        store=first_store,
        bus=EventBus(),
        journal=FileEventJournal(journal_path),
    )
    first_api.emit_user_message(thread_id="t-versioned-replay", text="Need advice")
    first_api.process_until_idle(max_events=16)
    first_api.emit_assistant_attachment(
        thread_id="t-versioned-replay",
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

    first_v1 = first_api.get_view("t-versioned-replay", version="v1")
    replay_v1 = replay_api.get_view("t-versioned-replay", version="v1")

    assert first_v1 == replay_v1
    assert set(first_v1) == {"thread_id", "messages", "thinking", "execution_boundaries"}
    assert replay_api.get_view("t-versioned-replay", version="v1") == first_api.get_view(
        "t-versioned-replay",
        version="v1",
    )
    assert replay_api.get_view("t-versioned-replay", version="v2") == first_api.get_view(
        "t-versioned-replay",
        version="v2",
    )


def test_runtime_event_api_replay_callers_must_request_v1_explicitly(tmp_path: Path):
    journal_path = tmp_path / "runtime-events-default-shift.jsonl"
    first_store = ConversationStore()
    first_api = RuntimeEventAPI(
        runtime=AgentRuntime(services=FakeServices(), store=first_store),
        store=first_store,
        bus=EventBus(),
        journal=FileEventJournal(journal_path),
    )
    first_api.emit_user_message(thread_id="t-default-shift", text="Need advice")
    first_api.process_until_idle(max_events=16)

    replay_store = ConversationStore()
    replay_api = RuntimeEventAPI(
        runtime=AgentRuntime(services=FakeServices(), store=replay_store),
        store=replay_store,
        bus=EventBus(),
        journal=FileEventJournal(journal_path),
    )

    assert replay_api.get_view("t-default-shift")["view_version"] == "v2"
    assert replay_api.get_view("t-default-shift", version="v1") == first_api.get_view(
        "t-default-shift",
        version="v1",
    )


def test_runtime_event_api_logs_unsupported_projection_version(caplog):
    api = RuntimeEventAPI(
        runtime=AgentRuntime(services=FakeServices(), store=ConversationStore()),
        store=ConversationStore(),
        bus=EventBus(),
    )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(ValueError, match="Unsupported thread view version"):
            api.get_view("t-unknown-view", version="v9")

    assert "Unsupported projection version requested: v9" in caplog.text


def test_streamlit_projection_access_logs_non_default_projection_request(caplog):
    store = ConversationStore()
    api = RuntimeEventAPI(
        runtime=AgentRuntime(services=StructuredPipelineServices(), store=store),
        store=store,
        bus=EventBus(),
    )
    api.emit_user_message(thread_id="t-streamlit-v1", text="Hello")
    api.process_until_idle(max_events=16)

    with caplog.at_level(logging.WARNING):
        view = load_thread_projection(api=api, thread_id="t-streamlit-v1", version="v1")

    assert set(view) == {"thread_id", "messages", "thinking", "execution_boundaries"}
    assert "Non-default projection version requested through streamlit consumer boundary: v1" in caplog.text


def test_runtime_event_api_rejects_unknown_view_version():
    api = RuntimeEventAPI(
        runtime=AgentRuntime(services=FakeServices(), store=ConversationStore()),
        store=ConversationStore(),
        bus=EventBus(),
    )

    with pytest.raises(ValueError, match="Unsupported thread view version"):
        api.get_view("t-unknown-view", version="v9")


def test_runtime_execution_boundary_allows_non_structural_pipeline_annotations():
    runtime = AgentRuntime(services=StructuredPipelineServices(), store=ConversationStore())
    bus = EventBus()

    def _annotate(*, event, result):
        _ = event
        result.pipeline["execution_trace"] = ["seed"]
        result.pipeline["steps"][0]["status"] = "verified"
        result.pipeline["steps"][0]["output_hash"] = "abc123"
        return result

    runtime._after_execution_region = _annotate
    bus.emit(new_event("user_message", thread_id="t-boundary-allow", payload={"text": "Hello"}))

    processed = runtime.run_until_idle(bus)

    started = next(event for event in processed if event.type == "execution_region_started")
    completed = next(event for event in processed if event.type == "execution_region_completed")
    assistant = next(event for event in processed if event.type == "assistant_reply")
    pipeline = dict(assistant.payload.get("pipeline") or {})

    assert started.payload["structural_fields"] == ["id", "name", "step", "tool_name", "kind", "depends_on"]
    assert completed.payload["semantic_fields"] == [
        "status",
        "result",
        "output",
        "output_hash",
        "execution_trace",
        "diagnostic",
        "metadata",
    ]
    assert completed.payload["structural_signature_before"] == completed.payload["structural_signature_after"]
    assert pipeline["execution_trace"] == ["seed"]
    assert pipeline["steps"][0]["status"] == "verified"
    assert pipeline["steps"][0]["output_hash"] == "abc123"


def test_runtime_emits_typed_execution_boundary_events():
    runtime = AgentRuntime(services=StructuredPipelineServices(), store=ConversationStore())
    bus = EventBus()

    bus.emit(new_event("user_message", thread_id="t-boundary-events", payload={"text": "Hello"}))
    processed = runtime.run_until_idle(bus)

    started = next(event for event in processed if event.type == "execution_region_started")
    completed = next(event for event in processed if event.type == "execution_region_completed")

    assert started.thread_id == "t-boundary-events"
    assert completed.thread_id == "t-boundary-events"
    assert started.payload["structural_snapshot"] == [
        {
            "id": "",
            "name": "seed",
            "step": "seed",
            "tool_name": "",
            "kind": "reasoning",
            "depends_on": (),
        }
    ]
    assert completed.payload["structural_snapshot"] == started.payload["structural_snapshot"]


def test_runtime_execution_boundary_protocol_is_deterministic_for_same_input():
    first_runtime = AgentRuntime(services=StructuredPipelineServices(), store=ConversationStore())
    first_bus = EventBus()
    first_bus.emit(new_event("user_message", thread_id="t-boundary-protocol", payload={"text": "Hello"}))

    second_runtime = AgentRuntime(services=StructuredPipelineServices(), store=ConversationStore())
    second_bus = EventBus()
    second_bus.emit(new_event("user_message", thread_id="t-boundary-protocol", payload={"text": "Hello"}))

    first_processed = first_runtime.run_until_idle(first_bus)
    second_processed = second_runtime.run_until_idle(second_bus)

    assert _execution_boundary_protocol(first_processed) == _execution_boundary_protocol(second_processed)


def test_execution_boundary_projection_is_passive_and_immutable_to_callers():
    store = ConversationStore()
    runtime = AgentRuntime(services=StructuredPipelineServices(), store=store)
    bus = EventBus()
    bus.emit(new_event("user_message", thread_id="t-boundary-view", payload={"text": "Hello"}))

    runtime.run_until_idle(bus)

    projected = store.thread_execution_boundaries("t-boundary-view")
    projected[0]["payload"]["structural_fields"].append("illegal")
    projected.append({"type": "observer_mutation"})

    refreshed = store.thread_execution_boundaries("t-boundary-view")
    assert [item["type"] for item in refreshed] == [
        "execution_region_started",
        "execution_region_completed",
    ]
    assert refreshed[0]["payload"]["structural_fields"] == [
        "id",
        "name",
        "step",
        "tool_name",
        "kind",
        "depends_on",
    ]


def test_runtime_execution_boundary_raises_on_structural_pipeline_mutation():
    runtime = AgentRuntime(services=StructuredPipelineServices(), store=ConversationStore())
    bus = EventBus()

    def _rewrite_structure(*, event, result):
        _ = event
        result.pipeline["steps"].append({"name": "illegal_followup", "kind": "reasoning", "depends_on": ["seed"]})
        return result

    runtime._after_execution_region = _rewrite_structure
    bus.emit(new_event("user_message", thread_id="t-boundary-block", payload={"text": "Hello"}))

    with pytest.raises(RuntimeError, match="Execution boundary violation"):
        runtime.run_until_idle(bus)
