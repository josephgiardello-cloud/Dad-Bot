import asyncio
from concurrent.futures import Future
import Dad

from Dad import DadBot
from dadbot_system import InMemoryEventBus, InMemoryStateStore
from dadbot_system.state import AppStateContainer


def test_dadbot_composes_split_services(bot):
    assert bot.config is not None
    assert bot.services is not None
    assert bot.context is bot.bot_context
    assert bot.service_registry is bot.services.registry
    assert bot.services.turn_service is bot.turn_service
    assert bot.memory is bot.services.memory_manager
    assert bot.relationship is bot.services.relationship_manager
    assert bot.mood_manager is bot.services.mood_manager
    assert bot.runtime_storage is bot.services.runtime_storage
    assert bot.profile_runtime is bot.services.profile_runtime
    assert bot.turn_service is bot.services.turn_service
    assert bot.reply_finalization is bot.services.reply_finalization
    assert bot.runtime_interface is bot.services.runtime_interface
    assert bot.status_reporting is bot.services.status_reporting
    assert bot.runtime_client is bot.services.runtime_client
    assert bot.model_runtime is bot.services.model_runtime
    assert bot.maintenance_scheduler is bot.services.maintenance_scheduler
    assert bot.PROFILE is bot.profile_runtime.profile
    assert bot.MEMORY_STORE is bot.memory.memory_store
    assert bot.memory_coordinator is not bot.memory_manager
    assert bot.profile_runtime is not None
    assert bot.long_term_signals is not None
    assert bot.multimodal_handler is not None
    assert bot.prompt_composer is bot.prompt_assembly
    assert bot.reply_generation is bot.turn_service.reply_generation
    assert bot.conversation_persistence is not None
    assert bot.runtime_client is not None
    assert bot.model_runtime is not None
    assert bot.runtime_orchestration is not None
    assert bot.runtime_storage is not None
    assert bot.session_summary_manager is not None
    assert bot.memory_query is not None
    assert bot.memory_commands is not None
    assert bot.reply_supervisor is not None
    assert bot.status_reporting is not None
    assert bot.runtime_interface is not None
    assert bot.bot_context.bot is bot
    assert bot.profile_context is not None
    assert bot.context_builder is not None
    assert bot.tone_context is not None
    assert bot.mood_manager is not None
    assert bot.relationship_manager is not None
    assert bot.internal_state_manager is not None
    assert bot.reply_finalization is not None
    assert bot.maintenance_scheduler is not None
    assert bot.runtime_state_manager is not None
    assert bot.tool_registry is not None
    assert bot.agentic_handler is not None


def test_shared_dadbot_context_is_live_and_used_by_extracted_managers(bot):
    assert bot.long_term_signals.bot is bot
    assert bot.multimodal_handler.context.bot is bot
    assert bot.profile_runtime.context.bot is bot
    assert bot.context_builder.context.bot is bot
    assert bot.tone_context.context.bot is bot
    assert bot.relationship_manager.context.bot is bot
    assert bot.prompt_assembly.context.bot is bot
    assert bot.turn_service.context.bot is bot
    assert bot.memory_manager.context.bot is bot
    assert bot.runtime_client.context.bot is bot
    assert bot.model_runtime.context.bot is bot
    assert bot.runtime_storage.context.bot is bot
    assert bot.session_summary_manager.context.bot is bot
    assert bot.memory_query.context.bot is bot
    assert bot.memory_commands.context.bot is bot
    assert bot.reply_supervisor.context.bot is bot
    assert bot.runtime_interface.context.bot is bot

    bot.ACTIVE_MODEL = "phase1-test-model"

    assert bot.bot_context.active_model == "phase1-test-model"
    assert bot.context_builder.context.active_model == "phase1-test-model"
    assert bot.tone_context.context.active_model == "phase1-test-model"
    assert bot.relationship_manager.context.active_model == "phase1-test-model"
    assert bot.prompt_assembly.context.active_model == "phase1-test-model"


def test_config_backed_runtime_fields_remain_mutable_via_facade(bot):
    bot.ACTIVE_MODEL = "config-model"
    bot.LLM_PROVIDER = "openai"
    bot.LLM_MODEL = "gpt-4o-mini"
    bot.ACTIVE_EMBEDDING_MODEL = "embed-small"
    bot.APPEND_SIGNOFF = False
    bot.LIGHT_MODE = True

    assert bot.config.active_model == "config-model"
    assert bot.config.llm_provider == "openai"
    assert bot.config.llm_model == "gpt-4o-mini"
    assert bot.config.active_embedding_model == "embed-small"
    assert bot.config.append_signoff is False
    assert bot.config.light_mode is True


def test_profile_and_memory_store_are_manager_backed(bot):
    profile = {"style": {"name": "Dad"}, "family": {}, "education": {}, "chat_routing": {"topic_rules": [], "core_fact_ids": []}, "facts": {}}
    memory_store = bot.default_memory_store()

    bot.PROFILE = profile
    bot.MEMORY_STORE = memory_store

    assert bot.profile_runtime.profile is profile
    assert bot.memory.memory_store is memory_store


def test_turn_orchestrator_property_uses_services_container(bot):
    sentinel = object()
    bot.services._turn_orchestrator = sentinel

    assert bot.turn_orchestrator is sentinel


def test_prompt_assembly_uses_extracted_context_services(bot, monkeypatch):
    monkeypatch.setattr(bot.context_builder, "build_core_persona_prompt", lambda: "CORE")
    monkeypatch.setattr(bot.context_builder, "build_dynamic_profile_context", lambda: "PROFILE")
    monkeypatch.setattr(bot.context_builder, "build_relationship_context", lambda: "RELATIONSHIP")
    monkeypatch.setattr(bot.tone_context, "build_mood_context", lambda mood: f"MOOD={mood}")
    monkeypatch.setattr(bot.tone_context, "build_daily_checkin_context", lambda mood: "CHECKIN")
    monkeypatch.setattr(bot, "build_active_tool_observation_context", lambda: "TOOL")
    monkeypatch.setattr(bot.context_builder, "build_cross_session_context", lambda user_input="": f"CROSS={user_input}")
    monkeypatch.setattr(bot.context_builder, "build_session_summary_context", lambda: "SUMMARY")
    monkeypatch.setattr(bot.context_builder, "build_relevant_context", lambda user_input: f"RELEVANT={user_input}")
    monkeypatch.setattr(bot.context_builder, "build_wisdom_context", lambda user_input: f"WISDOM={user_input}")
    monkeypatch.setattr(bot.context_builder, "build_memory_context", lambda user_input: f"MEMORY={user_input}")
    monkeypatch.setattr(bot.tone_context, "build_escalation_context", lambda current_mood, session_moods: "ESCALATE")

    prompt = bot.build_request_system_prompt("work is heavy", "stressed")

    assert "CORE" in prompt
    assert "PROFILE" in prompt
    assert "RELATIONSHIP" in prompt
    assert "MOOD=stressed" in prompt
    assert "CHECKIN" in prompt
    assert "TOOL" in prompt
    assert "CROSS=work is heavy" in prompt
    assert "SUMMARY" in prompt
    assert "RELEVANT=work is heavy" in prompt
    assert "WISDOM=work is heavy" in prompt
    assert "MEMORY=work is heavy" in prompt
    assert "ESCALATE" in prompt


def test_build_system_prompt_reuses_extracted_context_sections(bot, monkeypatch):
    monkeypatch.setattr(bot.context_builder, "build_core_persona_prompt", lambda: "CORE PERSONA")
    monkeypatch.setattr(bot.context_builder, "build_dynamic_profile_context", lambda: "PROFILE CONTEXT")

    prompt = bot.build_system_prompt()

    assert prompt == "CORE PERSONA\n\nPROFILE CONTEXT"


def test_prepare_final_reply_delegates_to_reply_finalization(bot, monkeypatch):
    monkeypatch.setattr(
        bot.reply_finalization,
        "finalize",
        lambda reply, current_mood, user_input=None: f"delegated::{reply}::{current_mood}::{user_input}",
    )

    reply = bot.prepare_final_reply("You did good, buddy.", "positive", "I got promoted.")

    assert reply == "delegated::You did good, buddy.::positive::I got promoted."


def test_prepare_final_reply_async_delegates_to_reply_finalization(bot, monkeypatch):
    async def fake_finalize(reply, current_mood, user_input=None):
        return f"async::{reply}::{current_mood}::{user_input}"

    monkeypatch.setattr(bot.reply_finalization, "finalize_async", fake_finalize)

    reply = asyncio.run(bot.prepare_final_reply_async("You did good, buddy.", "positive", "I got promoted."))

    assert reply == "async::You did good, buddy.::positive::I got promoted."


def test_turn_service_helper_methods_delegate_to_manager(bot, monkeypatch):
    monkeypatch.setattr(bot.turn_service, "should_offer_daily_checkin_for_turn", lambda: True)
    monkeypatch.setattr(bot.turn_service, "record_user_turn_state", lambda user_input, current_mood: (user_input, current_mood))
    monkeypatch.setattr(bot.turn_service, "direct_reply_for_input", lambda user_input, current_mood: f"reply::{user_input}::{current_mood}")

    should_offer = bot.should_offer_daily_checkin_for_turn()
    recorded = bot.record_user_turn_state("Just checking in.", "neutral")
    reply = bot.direct_reply_for_input("Where was I born?", "neutral")

    assert should_offer is True
    assert recorded == ("Just checking in.", "neutral")
    assert reply == "reply::Where was I born?::neutral"


def test_turn_service_uses_reply_generation_manager(bot, monkeypatch):
    monkeypatch.setattr(
        bot.reply_generation,
        "generate_validated_reply",
        lambda stripped_input, turn_text, current_mood, normalized_attachments, stream=False, chunk_callback=None: (
            f"generated::{stripped_input}::{turn_text}::{current_mood}::{stream}::{len(normalized_attachments)}"
        ),
    )
    monkeypatch.setattr(bot.turn_service, "prepare_user_turn", lambda user_input, attachments=None: ("neutral", None, False, user_input, attachments or []))
    monkeypatch.setattr(bot.turn_service, "finalize_user_turn", lambda stripped_input, current_mood, dad_reply, attachments=None: (dad_reply, False))

    reply, should_end = bot.process_user_message("Need a hand.")

    assert should_end is False
    assert reply == "generated::Need a hand.::Need a hand.::neutral::False::0"


def test_dependency_registry_can_override_runtime_interface(monkeypatch, tmp_path):
    class StubRuntimeInterface:
        def chat_loop(self):
            return "stub-chat-loop"

        def chat_loop_via_service(self, service_client, session_id=None):
            return "stub-chat-loop-service"

    monkeypatch.setenv("DADBOT_MEMORY_PATH", str(tmp_path / "memory.json"))
    monkeypatch.setenv("DADBOT_SEMANTIC_DB_PATH", str(tmp_path / "semantic.sqlite3"))
    monkeypatch.setenv("DADBOT_GRAPH_DB_PATH", str(tmp_path / "graph.sqlite3"))
    monkeypatch.setenv("DADBOT_SESSION_LOG_DIR", str(tmp_path / "session_logs"))

    stub_runtime_interface = StubRuntimeInterface()
    dependency_registry = {"runtime_interface": stub_runtime_interface}

    injected_bot = DadBot(dependency_registry=dependency_registry)
    try:
        assert injected_bot.runtime_interface is stub_runtime_interface
        assert injected_bot.chat_loop() == "stub-chat-loop"
        assert injected_bot.chat_loop_via_service(service_client=object()) == "stub-chat-loop-service"
    finally:
        injected_bot.shutdown()


def test_dependency_registry_can_override_runtime_state_bundle(monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_MEMORY_PATH", str(tmp_path / "memory.json"))
    monkeypatch.setenv("DADBOT_SEMANTIC_DB_PATH", str(tmp_path / "semantic.sqlite3"))
    monkeypatch.setenv("DADBOT_GRAPH_DB_PATH", str(tmp_path / "graph.sqlite3"))
    monkeypatch.setenv("DADBOT_SESSION_LOG_DIR", str(tmp_path / "session_logs"))

    custom_store = InMemoryStateStore()
    custom_event_bus = InMemoryEventBus()
    custom_container = AppStateContainer(
        "injected-runtime-state",
        DadBot.default_planner_debug_state,
        tenant_id="test-tenant",
        store=custom_store,
        event_bus=custom_event_bus,
    )
    dependency_registry = {
        "runtime_state_bundle": {
            "store": custom_store,
            "event_bus": custom_event_bus,
            "container": custom_container,
        }
    }

    injected_bot = DadBot(dependency_registry=dependency_registry)
    try:
        assert injected_bot._runtime_state_store is custom_store
        assert injected_bot._runtime_event_bus is custom_event_bus
        assert injected_bot._runtime_state is custom_container
        assert injected_bot.runtime_state_container is custom_container
    finally:
        injected_bot.shutdown()


def test_process_user_message_uses_graph_path_when_enabled(bot, monkeypatch):
    bot._turn_graph_enabled = True

    monkeypatch.setattr(bot, "_run_graph_turn_sync", lambda user_input, attachments=None: (f"graph::{user_input}", False))
    monkeypatch.setattr(bot.turn_service, "process_user_message", lambda user_input, attachments=None: ("legacy", False))

    reply, should_end = bot.process_user_message("Need a hand.")

    assert reply == "graph::Need a hand."
    assert should_end is False


def test_process_user_message_graph_failure_falls_back_to_legacy(bot, monkeypatch):
    bot._turn_graph_enabled = True

    def boom(user_input, attachments=None):
        raise RuntimeError("graph unavailable")

    monkeypatch.setattr(bot, "_run_graph_turn_sync", boom)
    monkeypatch.setattr(bot, "record_runtime_issue", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot.turn_service, "process_user_message", lambda user_input, attachments=None: (f"legacy::{user_input}", False))

    reply, should_end = bot.process_user_message("Need a hand.")

    assert reply == "legacy::Need a hand."
    assert should_end is False


def test_conversation_persistence_methods_delegate_to_manager(bot, monkeypatch):
    recorded = {"persist": 0, "snapshot": None, "log": None}

    monkeypatch.setattr(bot.conversation_persistence, "persist_conversation", lambda: recorded.__setitem__("persist", recorded["persist"] + 1))
    monkeypatch.setattr(bot.conversation_persistence, "persist_conversation_snapshot", lambda snapshot: recorded.__setitem__("snapshot", snapshot))
    monkeypatch.setattr(bot.conversation_persistence, "save_session_log", lambda history: recorded.__setitem__("log", history))

    bot.persist_conversation()
    bot.persist_conversation_snapshot({"history": [{"role": "user", "content": "hi"}]})
    bot.save_session_log([{"role": "assistant", "content": "hello"}])

    assert recorded["persist"] == 1
    assert recorded["snapshot"] == {"history": [{"role": "user", "content": "hi"}]}
    assert recorded["log"] == [{"role": "assistant", "content": "hello"}]


def test_status_reporting_methods_delegate_to_manager(bot, monkeypatch):
    monkeypatch.setattr(bot.status_reporting, "status_snapshot", lambda: {"active_model": "stub"})
    monkeypatch.setattr(bot.status_reporting, "service_status_snapshot", lambda: {"status": "ok"})
    monkeypatch.setattr(bot.status_reporting, "dashboard_status_snapshot", lambda: {"status": {"tenant_id": "tenant"}})
    monkeypatch.setattr(bot.status_reporting, "ui_shell_snapshot", lambda: {"health": {"level": "green"}})
    monkeypatch.setattr(bot.status_reporting, "format_status_snapshot", lambda: "status-line")
    monkeypatch.setattr(bot.status_reporting, "format_dad_snapshot", lambda: "dad-line")
    monkeypatch.setattr(bot.status_reporting, "format_proactive_snapshot", lambda: "proactive-line")

    assert bot.status_snapshot() == {"active_model": "stub"}
    assert bot.service_status_snapshot() == {"status": "ok"}
    assert bot.dashboard_status_snapshot() == {"status": {"tenant_id": "tenant"}}
    assert bot.ui_shell_snapshot() == {"health": {"level": "green"}}
    assert bot.format_status_snapshot() == "status-line"
    assert bot.format_dad_snapshot() == "dad-line"
    assert bot.format_proactive_snapshot() == "proactive-line"


def test_runtime_orchestration_methods_delegate_to_manager(bot, monkeypatch):
    monkeypatch.setattr(bot.runtime_orchestration, "record_background_task", lambda task_id, **kwargs: {"task_id": task_id, **kwargs})
    monkeypatch.setattr(bot.runtime_orchestration, "background_task_snapshot", lambda limit=8: {"tracked": limit})
    monkeypatch.setattr(bot.runtime_orchestration, "submit_background_task", lambda func, *args, **kwargs: (func, args, kwargs))
    monkeypatch.setattr(bot.runtime_orchestration, "persist_conversation_async", lambda: "persist-future")

    recorded = bot._record_background_task("task-1", task_kind="demo", status="queued", metadata={"x": 1})
    snapshot = bot.background_task_snapshot(limit=4)
    submitted = bot.submit_background_task(str, 123, task_kind="demo")
    persisted = bot.persist_conversation_async()

    assert recorded == {"task_id": "task-1", "task_kind": "demo", "status": "queued", "metadata": {"x": 1}, "error": ""}
    assert snapshot == {"tracked": 4}
    assert submitted[0] is str
    assert submitted[1] == (123,)
    assert submitted[2]["task_kind"] == "demo"
    assert persisted == "persist-future"


def test_background_task_manager_is_available_on_bot(bot):
    assert bot.background_tasks is not None


def test_internal_state_reflection_and_soft_reset(bot):
    bot.session_summary = "Tony felt overloaded by work this week."

    reflected = bot.reflect_internal_state("Work has been heavy", "stressed", "We can take this one step at a time.")
    assert reflected.get("turn_count", 0) >= 1
    assert "belief_vector" in reflected
    assert len(list(reflected.get("target_history") or [])) >= 1

    result = bot.soft_reset_session_context(preserve_recent_summary=True)
    assert result.get("mode") == "soft"
    assert bool(bot.session_summary)
    post_reset = bot.internal_state_snapshot()
    assert isinstance(post_reset, dict)
    assert len(list(post_reset.get("target_history") or [])) >= 1


def test_shutdown_flushes_memory_store_after_background_tasks(bot, monkeypatch):
    calls = []

    monkeypatch.setattr(bot.background_tasks, "shutdown", lambda wait=True: calls.append(("shutdown", wait)))
    monkeypatch.setattr(bot, "save_memory_store", lambda: calls.append(("save", None)))

    bot.shutdown()

    assert calls == [("shutdown", True), ("save", None)]


def test_profile_runtime_methods_delegate_to_manager(bot, monkeypatch):
    monkeypatch.setattr(bot.profile_runtime, "refresh_profile_runtime", lambda: "refreshed")
    monkeypatch.setattr(bot.profile_runtime, "cadence_settings", lambda: {"family_echo_turn_interval": 6})
    monkeypatch.setattr(bot.profile_runtime, "current_persona_preset", lambda: "coach")
    monkeypatch.setattr(bot.profile_runtime, "opening_message_candidates", lambda: ["Hey buddy"]) 
    monkeypatch.setattr(bot.profile_runtime, "apply_persona_preset", lambda preset_key, save=True: (preset_key, save))
    monkeypatch.setattr(bot.profile_runtime, "update_style_profile", lambda **kwargs: kwargs)
    monkeypatch.setattr(bot.profile_runtime, "update_cadence_profile", lambda cadence=None, save=True, **overrides: {"cadence": cadence, "save": save, **overrides})
    monkeypatch.setattr(bot.profile_runtime, "runtime_settings", lambda: {"stream_max_chars": 12000})
    monkeypatch.setattr(bot.profile_runtime, "update_runtime_profile", lambda settings=None, save=True, **overrides: {"settings": settings, "save": save, **overrides})
    monkeypatch.setattr(bot.profile_runtime, "update_opening_messages_profile", lambda opening_messages=None, save=True: {"opening_messages": opening_messages, "save": save})
    monkeypatch.setattr(bot.profile_runtime, "agentic_tool_settings", lambda: {"enabled": True})
    monkeypatch.setattr(bot.profile_runtime, "update_agentic_tool_profile", lambda settings=None, save=True, **overrides: {"settings": settings, "save": save, **overrides})
    monkeypatch.setattr(bot.profile_runtime, "relationship_calibration_settings", lambda: {"enabled": True, "opening_line": "Straight talk"})
    monkeypatch.setattr(bot.profile_runtime, "update_relationship_calibration_profile", lambda settings=None, save=True, **overrides: {"settings": settings, "save": save, **overrides})
    monkeypatch.setattr(bot.profile_runtime, "streamlit_security_settings", lambda: {"require_pin": True, "pin_hint": "digits"})
    monkeypatch.setattr(bot.profile_runtime, "verify_streamlit_pin", lambda pin_attempt: pin_attempt == "1234")

    refreshed = bot.refresh_profile_runtime()
    cadence = bot.cadence_settings()
    preset = bot.current_persona_preset()
    openings = bot.opening_message_candidates()
    applied = bot.apply_persona_preset("coach", save=False)
    updated_style = bot.update_style_profile(name="Dad", save=False)
    updated_cadence = bot.update_cadence_profile({"wisdom_turn_interval": 6}, save=False, family_echo_turn_interval=7)
    runtime = bot.runtime_settings()
    updated_runtime = bot.update_runtime_profile({"stream_max_chars": 12000}, save=False)
    updated_openings = bot.update_opening_messages_profile(["Hey buddy"], save=False)
    tools = bot.agentic_tool_settings()
    updated_tools = bot.update_agentic_tool_profile({"enabled": True}, save=False)
    calibration = bot.relationship_calibration_settings()
    updated_calibration = bot.update_relationship_calibration_profile({"enabled": False}, save=False)
    security = bot.streamlit_security_settings()
    pin_ok = bot.verify_streamlit_pin("1234")

    assert refreshed == "refreshed"
    assert cadence == {"family_echo_turn_interval": 6}
    assert preset == "coach"
    assert openings == ["Hey buddy"]
    assert applied == ("coach", False)
    assert updated_style["name"] == "Dad"
    assert updated_style["save"] is False
    assert updated_cadence == {"cadence": {"wisdom_turn_interval": 6}, "save": False, "family_echo_turn_interval": 7}
    assert runtime == {"stream_max_chars": 12000}
    assert updated_runtime == {"settings": {"stream_max_chars": 12000}, "save": False}
    assert updated_openings == {"opening_messages": ["Hey buddy"], "save": False}
    assert tools == {"enabled": True}
    assert updated_tools == {"settings": {"enabled": True}, "save": False}
    assert calibration == {"enabled": True, "opening_line": "Straight talk"}
    assert updated_calibration == {"settings": {"enabled": False}, "save": False}
    assert security == {"require_pin": True, "pin_hint": "digits"}
    assert pin_ok is True


def test_profile_context_methods_delegate_through_facade(bot, monkeypatch):
    monkeypatch.setattr(bot.profile_context, "age_on_date", lambda birthdate, today=None: 42)
    monkeypatch.setattr(bot.profile_context, "format_long_date", lambda value: "April 20th, 2026")
    monkeypatch.setattr(bot.profile_context, "template_context", lambda: {"listener_name": "Tony"})
    monkeypatch.setattr(bot.profile_context, "render_template", lambda template: f"rendered::{template}")
    monkeypatch.setattr(bot.profile_context, "get_fact_reply", lambda user_input: f"fact::{user_input}")
    monkeypatch.setattr(bot.profile_context, "expected_tokens_for_fact_ids", lambda fact_ids: {"dad", "rhode", *fact_ids})
    monkeypatch.setattr(bot.profile_context, "response_has_expected_anchor", lambda rule, reply: rule.get("name") == "ok" and reply == "reply")
    monkeypatch.setattr(bot.profile_context, "validate_reply", lambda user_input, reply: f"validated::{user_input}::{reply}")

    age = bot.age_on_date("ignored")
    formatted = bot.format_long_date("ignored")
    template = bot.template_context()
    rendered = bot.render_template("hello {listener_name}")
    fact_reply = bot.get_fact_reply("Where was I born?")
    expected_tokens = bot.expected_tokens_for_fact_ids(["dad_birthplace"])
    anchored = bot.response_has_expected_anchor({"name": "ok"}, "reply")
    validated = bot.validate_reply("Where was I born?", "Draft reply")

    assert age == 42
    assert formatted == "April 20th, 2026"
    assert template == {"listener_name": "Tony"}
    assert rendered == "rendered::hello {listener_name}"
    assert fact_reply == "fact::Where was I born?"
    assert expected_tokens == {"dad", "rhode", "dad_birthplace"}
    assert anchored is True
    assert validated == "validated::Where was I born?::Draft reply"


def test_multimodal_methods_delegate_to_manager(bot, monkeypatch):
    monkeypatch.setattr(bot.multimodal_handler, "normalize_chat_attachments", lambda attachments=None: [{"type": "audio", "transcript": "hello"}] if attachments else [])
    monkeypatch.setattr(bot.multimodal_handler, "compose_user_turn_text", lambda user_input, attachments=None: f"turn::{user_input}::{len(attachments or [])}")
    monkeypatch.setattr(bot.multimodal_handler, "build_user_request_message", lambda user_input, attachments=None: {"role": "user", "content": f"msg::{user_input}", "attachments": len(attachments or [])})
    monkeypatch.setattr(bot.multimodal_handler, "build_image_analysis_prompt", lambda note="", user_input="", attachment=None: f"prompt::{note}::{user_input}::{attachment.get('name') if attachment else ''}")
    monkeypatch.setattr(bot.multimodal_handler, "describe_image_attachment", lambda attachment, user_input="": f"describe::{attachment.get('name')}::{user_input}")
    monkeypatch.setattr(bot.multimodal_handler, "enrich_multimodal_attachments", lambda attachments=None, user_input="": [{"type": "image", "analysis": f"analysis::{user_input}"}])

    normalized_one = bot.normalize_chat_attachment({"type": "audio", "transcript": "hi"})
    normalized_many = bot.normalize_chat_attachments([{"type": "audio", "transcript": "hi"}])
    image_support = bot.model_supports_image_input("llava:7b")
    turn_text = bot.compose_user_turn_text("hello", [{"type": "audio", "transcript": "hi"}])
    request_message = bot.build_user_request_message("hello", [{"type": "image", "image_b64": "abc"}])
    history_metadata = bot.history_attachment_metadata({"type": "audio", "transcript": "hi", "name": "note.wav"})
    analysis_prompt = bot.build_image_analysis_prompt("note", user_input="help", attachment={"name": "photo.png"})
    described = bot.describe_image_attachment({"type": "image", "name": "photo.png"}, user_input="help")
    enriched = bot.enrich_multimodal_attachments([{"type": "image", "name": "photo.png"}], user_input="help")

    assert normalized_one == {"type": "audio", "name": "", "mime_type": "", "transcript": "hi"}
    assert normalized_many == [{"type": "audio", "transcript": "hello"}]
    assert image_support is True
    assert turn_text == "turn::hello::1"
    assert request_message == {"role": "user", "content": "msg::hello", "attachments": 1}
    assert history_metadata == {"type": "audio", "name": "note.wav", "mime_type": "", "transcript": "hi"}
    assert analysis_prompt == "prompt::note::help::photo.png"
    assert described == "describe::photo.png::help"
    assert enriched == [{"type": "image", "analysis": "analysis::help"}]


def test_runtime_client_methods_delegate_to_manager(bot, monkeypatch):
    async def fake_async_chat(messages, options=None, response_format=None, purpose="chat"):
        return {"async": True, "messages": messages, "options": options, "response_format": response_format, "purpose": purpose}

    async def fake_stream_async(messages, options=None, purpose="chat", chunk_callback=None):
        return {"stream": True, "messages": messages, "options": options, "purpose": purpose, "chunk_callback": chunk_callback is not None}

    monkeypatch.setattr(bot.runtime_client, "ollama_async_client", lambda: "async-client")
    monkeypatch.setattr(
        bot.runtime_client,
        "call_ollama_chat",
        lambda messages, options=None, response_format=None, purpose="chat": {
            "messages": messages,
            "options": options,
            "response_format": response_format,
            "purpose": purpose,
        },
    )
    monkeypatch.setattr(bot.runtime_client, "call_ollama_chat_async", fake_async_chat)
    monkeypatch.setattr(
        bot.runtime_client,
        "call_ollama_chat_with_model",
        lambda model_name, messages, options=None, response_format=None, purpose="chat": {
            "model": model_name,
            "messages": messages,
            "options": options,
            "response_format": response_format,
            "purpose": purpose,
        },
    )
    monkeypatch.setattr(
        bot.runtime_client,
        "call_ollama_chat_stream",
        lambda messages, options=None, purpose="chat", chunk_callback=None: {
            "messages": messages,
            "options": options,
            "purpose": purpose,
            "chunk_callback": chunk_callback is not None,
        },
    )
    monkeypatch.setattr(bot.runtime_client, "call_ollama_chat_stream_async", fake_stream_async)
    monkeypatch.setattr(bot.runtime_client, "available_model_names", lambda: ["llama3.2", "llava:7b"])
    monkeypatch.setattr(bot.runtime_client, "find_available_vision_model", lambda: "llava:7b")
    monkeypatch.setattr(bot.runtime_client, "vision_fallback_status", lambda: (True, "llava:7b is available"))
    monkeypatch.setattr(bot.runtime_client, "ensure_ollama_ready", lambda status_callback=None: status_callback is None)

    sync_response = bot.call_ollama_chat([{"role": "user", "content": "Hi"}], options={"temperature": 0.1}, response_format="json", purpose="reply")
    async_response = asyncio.run(bot.call_ollama_chat_async([{"role": "user", "content": "Hi"}], purpose="reply"))
    model_response = bot.call_ollama_chat_with_model("llava:7b", [{"role": "user", "content": "Describe"}], purpose="vision")
    stream_response = bot.call_ollama_chat_stream([{"role": "user", "content": "Hi"}], purpose="reply")
    async_stream_response = asyncio.run(bot.call_ollama_chat_stream_async([{"role": "user", "content": "Hi"}], purpose="reply"))
    available = bot.available_model_names()
    vision_model = bot.find_available_vision_model()
    vision_status = bot.vision_fallback_status()
    ready = bot.ensure_ollama_ready()

    assert bot.ollama_async_client() == "async-client"
    assert sync_response["purpose"] == "reply"
    assert async_response["async"] is True
    assert model_response["model"] == "llava:7b"
    assert stream_response["purpose"] == "reply"
    assert async_stream_response["stream"] is True
    assert available == ["llama3.2", "llava:7b"]
    assert vision_model == "llava:7b"
    assert vision_status == (True, "llava:7b is available")
    assert ready is True


def test_runtime_storage_methods_delegate_to_manager(bot, monkeypatch):
    monkeypatch.setattr(bot.runtime_storage, "capture_corrupt_json_snapshot", lambda source_path: f"snapshot::{source_path}")
    monkeypatch.setattr(bot.runtime_storage, "write_json_atomically", lambda destination, payload, backup=True: {"destination": destination, "payload": payload, "backup": backup})
    monkeypatch.setattr(bot.runtime_storage, "load_profile", lambda: {"style": {"name": "Dad"}})
    monkeypatch.setattr(bot.runtime_storage, "save_profile", lambda: "saved-profile")
    monkeypatch.setattr(bot.runtime_storage, "customer_persistence_status", lambda: {"enabled": True, "backend": "PostgresStateStore"})

    backup_path = bot.json_backup_path("dad_memory.json")
    corrupt_path = bot.corrupt_json_snapshot_path("dad_memory.json")
    captured = bot.capture_corrupt_json_snapshot("dad_memory.json")
    written = bot.write_json_atomically("dad_memory.json", {"ok": True}, backup=False)
    loaded = bot.runtime_storage.load_profile()
    saved = bot.save_profile()
    status = bot.customer_persistence_status()

    assert str(backup_path).endswith("dad_memory.json.bak")
    assert "dad_memory.corrupt-" in str(corrupt_path)
    assert captured == "snapshot::dad_memory.json"
    assert written == {"destination": "dad_memory.json", "payload": {"ok": True}, "backup": False}
    assert loaded == {"style": {"name": "Dad"}}
    assert saved == "saved-profile"
    assert status == {"enabled": True, "backend": "PostgresStateStore"}


def test_session_summary_methods_delegate_to_manager(bot, monkeypatch):
    monkeypatch.setattr(bot.session_summary_manager, "build_session_summary_prompt", lambda previous_summary, messages: f"prompt::{previous_summary}::{len(messages)}")
    monkeypatch.setattr(bot.session_summary_manager, "refresh_session_summary", lambda force=False: {"refreshed": True, "force": force})

    prompt = bot.build_session_summary_prompt("Earlier summary", [{"role": "user", "content": "Hey"}])
    refreshed = bot.refresh_session_summary(force=True)

    assert prompt == "prompt::Earlier summary::1"
    assert refreshed == {"refreshed": True, "force": True}


def test_long_term_signal_methods_delegate_to_manager(bot, monkeypatch):
    monkeypatch.setattr(bot.long_term_signals, "summarize_memory_graph", lambda: "graph::summary")
    monkeypatch.setattr(bot.long_term_signals, "should_generate_wisdom_insight", lambda user_input, force=False: (user_input, force) == ("input", True))
    monkeypatch.setattr(bot.long_term_signals, "build_wisdom_prompt", lambda user_input: f"wisdom::{user_input}")
    monkeypatch.setattr(bot.long_term_signals, "generate_wisdom_insight", lambda user_input, force=False: {"summary": user_input, "force": force})
    monkeypatch.setattr(bot.long_term_signals, "build_family_echo_prompt", lambda user_input, current_mood: f"family::{user_input}::{current_mood}")
    monkeypatch.setattr(bot.long_term_signals, "maybe_add_family_echo", lambda reply, user_input, current_mood: f"echo::{reply}::{user_input}::{current_mood}")
    monkeypatch.setattr(bot.long_term_signals, "add_memory_nodes_to_graph", lambda node_weights, node_types, edge_weights: node_weights.__setitem__("memory", 1))
    monkeypatch.setattr(bot.long_term_signals, "add_relationship_topics_to_graph", lambda node_weights, node_types: node_types.__setitem__("work", "topic"))
    monkeypatch.setattr(bot.long_term_signals, "add_archive_nodes_to_graph", lambda node_weights, node_types, edge_weights: edge_weights.__setitem__(("work", "stress"), 2))
    monkeypatch.setattr(bot.long_term_signals, "mark_memory_graph_dirty", lambda: setattr(bot, "_memory_graph_dirty", True))
    monkeypatch.setattr(bot.long_term_signals, "refresh_memory_graph", lambda force=False: {"force": force, "updated_at": "now"})

    node_weights = {}
    node_types = {}
    edge_weights = {}

    summary = bot.summarize_memory_graph()
    should_generate = bot.should_generate_wisdom_insight("input", force=True)
    prompt = bot.build_wisdom_prompt("input")
    generated = bot.generate_wisdom_insight("input", force=True)
    pattern_key = bot.pattern_identity({"topic": "Work", "day_hint": "Sunday", "mood": "stressed"})
    family_prompt = bot.build_family_echo_prompt("rough day", "stressed")
    echoed = bot.maybe_add_family_echo("reply", "rough day", "stressed")
    bot.add_memory_nodes_to_graph(node_weights, node_types, edge_weights)
    bot.add_relationship_topics_to_graph(node_weights, node_types)
    bot.add_archive_nodes_to_graph(node_weights, node_types, edge_weights)
    bot.mark_memory_graph_dirty()
    refreshed = bot.refresh_memory_graph(force=True)

    assert summary == "graph::summary"
    assert should_generate is True
    assert prompt == "wisdom::input"
    assert generated == {"summary": "input", "force": True}
    assert pattern_key == ("work", "sunday", "stressed")
    assert family_prompt == "family::rough day::stressed"
    assert echoed == "echo::reply::rough day::stressed"
    assert node_weights == {"memory": 1}
    assert node_types == {"work": "topic"}
    assert edge_weights == {("work", "stress"): 2}
    assert bot._memory_graph_dirty is True
    assert refreshed == {"force": True, "updated_at": "now"}


def test_reply_supervisor_methods_delegate_to_manager(bot, monkeypatch):
    monkeypatch.setattr(bot.reply_supervisor, "build_reply_critique_prompt", lambda user_input, draft_reply, current_mood: f"critique::{user_input}::{draft_reply}::{current_mood}")
    monkeypatch.setattr(bot.reply_supervisor, "build_reply_alignment_judge_prompt", lambda user_input, candidate_reply, current_mood: f"align::{candidate_reply}::{current_mood}")
    monkeypatch.setattr(bot.reply_supervisor, "build_reply_supervisor_prompt", lambda user_input, draft_reply, current_mood: f"prompt::{draft_reply}::{current_mood}")
    monkeypatch.setattr(bot.reply_supervisor, "apply_reply_supervisor_decision", lambda judgment, candidate_reply, stage="reply_supervisor": {"judgment": judgment, "reply": candidate_reply, "stage": stage})
    monkeypatch.setattr(bot.reply_supervisor, "run_reply_supervisor", lambda user_input, candidate_reply, current_mood, stage="reply_supervisor": f"run::{stage}::{candidate_reply}")
    monkeypatch.setattr(bot.reply_supervisor, "build_reply_supervisor_context", lambda current_mood: f"context::{current_mood}")
    monkeypatch.setattr(bot.reply_supervisor, "reply_supervisor_snapshot", lambda: {"enabled": True})
    monkeypatch.setattr(bot.reply_supervisor, "judge_reply_alignment", lambda user_input, candidate_reply, current_mood: f"judge::{candidate_reply}")
    monkeypatch.setattr(bot.reply_supervisor, "critique_reply", lambda user_input, draft_reply, current_mood: f"critique-reply::{draft_reply}")

    critique_prompt = bot.build_reply_critique_prompt("input", "draft", "stressed")
    alignment_prompt = bot.build_reply_alignment_judge_prompt("input", "draft", "stressed")
    supervisor_prompt = bot.build_reply_supervisor_prompt("input", "draft", "stressed")
    applied = bot.apply_reply_supervisor_decision({"ok": True}, "draft", stage="alignment_judge")
    ran = bot.run_reply_supervisor("input", "draft", "stressed", stage="alignment_judge")
    context = bot.build_reply_supervisor_context("stressed")
    snapshot = bot.reply_supervisor_snapshot()
    judged = bot.judge_reply_alignment("input", "draft", "stressed")
    critiqued = bot.critique_reply("input", "draft", "stressed")

    assert critique_prompt == "critique::input::draft::stressed"
    assert alignment_prompt == "align::draft::stressed"
    assert supervisor_prompt == "prompt::draft::stressed"
    assert applied == {"judgment": {"ok": True}, "reply": "draft", "stage": "alignment_judge"}
    assert ran == "run::alignment_judge::draft"
    assert context == "context::stressed"
    assert snapshot == {"enabled": True}
    assert judged == "judge::draft"
    assert critiqued == "critique-reply::draft"


def test_memory_query_methods_delegate_to_manager(bot, monkeypatch):
    monkeypatch.setattr(bot.memory_query, "memory_context_limit_for_input", lambda user_input: 2)
    monkeypatch.setattr(bot.memory_query, "retrieve_context", lambda user_input, strategy="hybrid", limit=4: {"user_input": user_input, "strategy": strategy, "limit": limit})
    monkeypatch.setattr(bot.memory_query, "relevant_archive_entries_for_input", lambda user_input, limit=2: [{"summary": user_input, "limit": limit}])
    monkeypatch.setattr(bot.memory_query, "relevant_memories_for_input", lambda user_input, limit=3, graph_result=None: [{"summary": user_input, "limit": limit, "graph_result": graph_result}])
    monkeypatch.setattr(bot.memory_query, "get_memory_reply", lambda user_input: f"reply::{user_input}")
    monkeypatch.setattr(bot.memory_query, "find_memory_matches", lambda query: [{"summary": query}])
    monkeypatch.setattr(bot.memory_query, "add_memory", lambda summary, category=None: {"summary": summary, "category": category or "general"})
    monkeypatch.setattr(bot.memory_query, "update_memory_entry", lambda original_summary, new_summary, category=None, mood=None: {"old": original_summary, "new": new_summary, "category": category, "mood": mood})
    monkeypatch.setattr(bot.memory_query, "delete_memory_entry", lambda summary: [{"summary": summary}])
    monkeypatch.setattr(bot.memory_query, "forget_memories", lambda query: [{"summary": query}])

    limit = bot.memory_context_limit_for_input("budget")
    context_bundle = bot.retrieve_context("budget", strategy="hybrid", limit=3)
    archive = bot.relevant_archive_entries_for_input("budget", limit=1)
    memories = bot.relevant_memories_for_input("budget", limit=4, graph_result={"graph": True})
    reply = bot.get_memory_reply("what do you remember")
    matches = bot.find_memory_matches("deadline")
    added = bot.add_memory("saving for emergencies", category="finance")
    updated = bot.update_memory_entry("old", "new", category="work", mood="stressed")
    deleted = bot.delete_memory_entry("saving for emergencies")
    forgotten = bot.forget_memories("deadline")

    assert limit == 2
    assert context_bundle == {"user_input": "budget", "strategy": "hybrid", "limit": 3}
    assert archive == [{"summary": "budget", "limit": 1}]
    assert memories == [{"summary": "budget", "limit": 4, "graph_result": {"graph": True}}]
    assert reply == "reply::what do you remember"
    assert matches == [{"summary": "deadline"}]
    assert added == {"summary": "saving for emergencies", "category": "finance"}
    assert updated == {"old": "old", "new": "new", "category": "work", "mood": "stressed"}
    assert deleted == [{"summary": "saving for emergencies"}]
    assert forgotten == [{"summary": "deadline"}]


def test_memory_contradiction_methods_delegate_to_manager(bot, monkeypatch):
    monkeypatch.setattr(
        bot.memory_coordinator,
        "consolidated_contradictions",
        lambda limit=8: [{"left_summary": "A", "right_summary": "B", "limit": limit}],
    )
    monkeypatch.setattr(
        bot.memory_coordinator,
        "resolve_consolidated_contradiction",
        lambda left_summary, right_summary, keep="auto", reason="user_review": {
            "left": left_summary,
            "right": right_summary,
            "keep": keep,
            "reason": reason,
        },
    )

    contradictions = bot.consolidated_contradictions(limit=5)
    resolved = bot.resolve_consolidated_contradiction("A", "B", keep="left", reason="test")

    assert contradictions == [{"left_summary": "A", "right_summary": "B", "limit": 5}]
    assert resolved == {"left": "A", "right": "B", "keep": "left", "reason": "test"}


def test_memory_command_methods_delegate_to_manager(bot, monkeypatch):
    monkeypatch.setattr(bot.memory_commands, "handle_memory_command", lambda user_input: f"handled::{user_input}")
    monkeypatch.setattr(bot.memory_commands, "build_memory_transcript", lambda history: f"transcript::{len(history)}")

    parsed = bot.parse_memory_command("remember this call the bank")
    handled = bot.handle_memory_command("remember this call the bank")
    transcript = bot.build_memory_transcript([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}])

    assert parsed == {"action": "remember", "summary": "call the bank"}
    assert handled == "handled::remember this call the bank"
    assert transcript == "transcript::2"


def test_runtime_interface_methods_delegate_to_manager(bot, monkeypatch):
    monkeypatch.setattr(bot.runtime_interface, "chat_loop", lambda: "chat-loop")
    monkeypatch.setattr(bot.runtime_interface, "chat_loop_via_service", lambda service_client, session_id=None: {"client": service_client, "session_id": session_id})

    loop_result = bot.chat_loop()
    service_result = bot.chat_loop_via_service("client", session_id="session-1")

    assert loop_result == "chat-loop"
    assert service_result == {"client": "client", "session_id": "session-1"}


def test_model_runtime_methods_delegate_to_manager(bot, monkeypatch):
    monkeypatch.setattr(bot.model_runtime, "ollama_show_payload", lambda model_name=None: {"model": model_name or "active"})
    monkeypatch.setattr(bot.model_runtime, "model_context_length", lambda model_name=None: 4096)
    monkeypatch.setattr(bot.model_runtime, "effective_context_token_budget", lambda model_name=None: 3072)
    monkeypatch.setattr(bot.model_runtime, "model_chars_per_token_estimate", lambda model_name=None: 3.25)
    monkeypatch.setattr(bot.model_runtime, "resolve_tiktoken_encoding_name", lambda model_name=None: "o200k_base")
    monkeypatch.setattr(bot.model_runtime, "initialize_tokenizer", lambda model_name=None: f"init::{model_name or 'active'}")
    monkeypatch.setattr(bot.model_runtime, "current_tokenizer", lambda model_name=None: f"current::{model_name or 'active'}")
    monkeypatch.setattr(bot.model_runtime, "model_candidates", lambda: ["primary", "fallback"])
    monkeypatch.setattr(bot.model_runtime, "dedicated_embedding_model_candidates", lambda: ["embed-a", "embed-b"])
    monkeypatch.setattr(bot.model_runtime, "fallback_embedding_model_candidates", lambda: ["chat-a", "chat-b"])
    monkeypatch.setattr(bot.model_runtime, "embedding_model_candidates", lambda: ["embed-a", "embed-b"])

    payload = bot.ollama_show_payload("llama3.2")
    context_length = bot.model_context_length("llama3.2")
    budget = bot.effective_context_token_budget("llama3.2")
    chars_per_token = bot.model_chars_per_token_estimate("llama3.2")
    encoding = bot.resolve_tiktoken_encoding_name("llama3.2")
    initialized = bot.initialize_tokenizer("llama3.2")
    current = bot.current_tokenizer("llama3.2")
    models = bot.model_candidates()
    dedicated_embeddings = bot.dedicated_embedding_model_candidates()
    fallback_embeddings = bot.fallback_embedding_model_candidates()
    embeddings = bot.embedding_model_candidates()

    assert payload == {"model": "llama3.2"}
    assert context_length == 4096
    assert budget == 3072
    assert chars_per_token == 3.25
    assert encoding == "o200k_base"
    assert initialized == "init::llama3.2"
    assert current == "current::llama3.2"
    assert models == ["primary", "fallback"]
    assert dedicated_embeddings == ["embed-a", "embed-b"]
    assert fallback_embeddings == ["chat-a", "chat-b"]
    assert embeddings == ["embed-a", "embed-b"]


def test_embedding_model_candidates_fall_back_to_chat_models_when_unset(bot):
    bot.ACTIVE_EMBEDDING_MODEL = None
    bot.PREFERRED_EMBEDDING_MODELS = ()
    bot.MODEL_NAME = "primary"
    bot.ACTIVE_MODEL = "primary"
    bot.FALLBACK_MODELS = ("fallback",)

    candidates = bot.model_runtime.embedding_model_candidates()

    assert candidates == ["primary", "fallback"]


def test_relationship_manager_snapshot_exposes_active_hypothesis(bot):
    bot.relationship_manager.update("I feel overwhelmed and needed to say that out loud.", "stressed")

    snapshot = bot.relationship_manager.snapshot()

    assert snapshot["active_hypothesis"]
    assert snapshot["active_hypothesis_label"]
    assert snapshot["hypotheses"]
    assert snapshot["emotional_momentum"] in {"steady", "warming", "heavy"}


def test_maintenance_methods_delegate_to_scheduler(bot, monkeypatch):
    monkeypatch.setattr(
        bot.maintenance_scheduler,
        "run_periodic_durable_synthesis",
        lambda trigger_text="", force=False: {"ran": True, "trigger_text": trigger_text, "force": force},
    )
    monkeypatch.setattr(
        bot.maintenance_scheduler,
        "maintenance_snapshot",
        lambda: {"last_background_synthesis_turn": 7, "latest_task": {"status": "completed"}},
    )

    synthesis = bot.run_periodic_durable_synthesis("checking in", force=True)
    snapshot = bot.maintenance_snapshot()

    assert synthesis == {"ran": True, "trigger_text": "checking in", "force": True}
    assert snapshot["last_background_synthesis_turn"] == 7
    assert snapshot["latest_task"]["status"] == "completed"


def test_scheduled_proactive_methods_delegate_to_scheduler(bot, monkeypatch):
    monkeypatch.setattr(
        bot.maintenance_scheduler,
        "run_scheduled_proactive_jobs",
        lambda force=False, reference_time=None: {"queued_total": 1, "force": force, "reference_time": reference_time},
    )

    result = bot.run_scheduled_proactive_jobs(force=True, reference_time="2026-04-20T09:00:00")

    assert result == {"queued_total": 1, "force": True, "reference_time": "2026-04-20T09:00:00"}


def test_mood_detection_cache_methods_delegate_to_mood_manager(bot, monkeypatch):
    monkeypatch.setattr(bot.mood_manager, "get_cached_mood_detection", lambda user_input, recent_history=None: ("cache-key", "stressed"))
    monkeypatch.setattr(bot.mood_manager, "remember_mood_detection", lambda cache_key, mood: f"{cache_key}:{mood}")

    cached = bot.get_cached_mood_detection("I am overloaded")
    remembered = bot.remember_mood_detection("cache-key", "stressed")

    assert cached == ("cache-key", "stressed")
    assert remembered == "cache-key:stressed"


def test_runtime_state_methods_delegate_to_runtime_manager(bot, monkeypatch):
    monkeypatch.setattr(bot.runtime_state_manager, "create_chat_thread", lambda title="": {"thread_id": "t-1", "title": title})
    monkeypatch.setattr(bot.runtime_state_manager, "planner_debug_snapshot", lambda: {"final_path": "planner"})
    monkeypatch.setattr(bot.runtime_state_manager, "message_token_cost", lambda message: 42 if message.get("content") == "cached" else 7)
    monkeypatch.setattr(bot.runtime_state_manager, "prompt_history", lambda: [{"role": "user", "content": "kept"}])
    monkeypatch.setattr(bot.runtime_state_manager, "prompt_history_token_budget", lambda system_prompt, user_input: 123)
    monkeypatch.setattr(bot.runtime_state_manager, "trim_text_to_token_budget", lambda text, token_budget: f"text::{token_budget}::{text}")
    monkeypatch.setattr(
        bot.runtime_state_manager,
        "trim_message_to_token_budget",
        lambda message, token_budget: {**message, "content": f"trim::{token_budget}::{message.get('content', '')}"},
    )
    monkeypatch.setattr(bot.runtime_state_manager, "token_budgeted_prompt_history", lambda system_prompt, user_input: [{"role": "assistant", "content": "budgeted"}])
    monkeypatch.setattr(bot.runtime_state_manager, "session_turn_count", lambda: 9)

    thread = bot.create_chat_thread("Project")
    debug = bot.planner_debug_snapshot()
    token_cost = bot.message_token_cost({"role": "user", "content": "cached"})
    prompt_history = bot.prompt_history()
    budget = bot.prompt_history_token_budget("sys", "user")
    trimmed_text = bot.trim_text_to_token_budget("hello", 5)
    trimmed_message = bot.trim_message_to_token_budget({"role": "user", "content": "hello"}, 7)
    selected = bot.token_budgeted_prompt_history("sys", "user")
    turns = bot.session_turn_count()

    assert thread == {"thread_id": "t-1", "title": "Project"}
    assert debug == {"final_path": "planner"}
    assert token_cost == 42
    assert prompt_history == [{"role": "user", "content": "kept"}]
    assert budget == 123
    assert trimmed_text == "text::5::hello"
    assert trimmed_message == {"role": "user", "content": "trim::7::hello"}
    assert selected == [{"role": "assistant", "content": "budgeted"}]
    assert turns == 9


def test_tool_methods_delegate_to_agentic_services(bot, monkeypatch):
    monkeypatch.setattr(bot.tool_registry, "get_available_tools", lambda: [{"type": "function", "function": {"name": "stub"}}])
    monkeypatch.setattr(bot.agentic_handler, "handle_tool_command", lambda user_input: f"tool::{user_input}")

    tools = bot.get_available_tools()
    result = bot.handle_tool_command("/status")

    assert tools == [{"type": "function", "function": {"name": "stub"}}]
    assert result == "tool::/status"


def test_memory_pipeline_methods_delegate_to_memory_coordinator(bot, monkeypatch):
    monkeypatch.setattr(bot.memory_coordinator, "consolidate_memories", lambda force=False: [{"summary": "steady progress", "force": force}])
    monkeypatch.setattr(bot.memory_coordinator, "update_memory_store", lambda history: len(history))

    consolidated = bot.consolidate_memories(force=True)
    updated = bot.update_memory_store([{"role": "user", "content": "remember this"}])

    assert consolidated == [{"summary": "steady progress", "force": True}]
    assert updated == 1


def test_record_runtime_issue_delegates_to_health_manager(bot, monkeypatch):
    captured = {}

    def fake_record_runtime_issue(purpose, fallback, exc=None, *, level=None, metadata=None):
        captured["purpose"] = purpose
        captured["fallback"] = fallback
        captured["exc"] = exc
        captured["level"] = level
        return "ok"

    monkeypatch.setattr(bot.health_manager, "record_runtime_issue", fake_record_runtime_issue)

    result = bot.record_runtime_issue("turn_graph", "turn_service_sync_fallback", exc=RuntimeError("boom"))

    assert result == "ok"
    assert captured["purpose"] == "turn_graph"
    assert captured["fallback"] == "turn_service_sync_fallback"
    assert isinstance(captured["exc"], RuntimeError)


def test_validate_managers_smoke_true_passes(bot):
    bot._validate_managers(smoke=True)


def test_smoke_init_under_pytest_skips_graph_init_background_task(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests::test_smoke_init")
    monkeypatch.setenv("DADBOT_VALIDATE_MANAGERS_SMOKE", "1")
    monkeypatch.setenv("DADBOT_MEMORY_PATH", str(tmp_path / "memory.json"))
    monkeypatch.setenv("DADBOT_SEMANTIC_DB_PATH", str(tmp_path / "semantic.sqlite3"))
    monkeypatch.setenv("DADBOT_GRAPH_DB_PATH", str(tmp_path / "graph.sqlite3"))
    monkeypatch.setenv("DADBOT_SESSION_LOG_DIR", str(tmp_path / "session_logs"))

    submit_calls = []

    def fake_submit(self, func, *args, **kwargs):
        submit_calls.append({"func": getattr(func, "__name__", str(func)), "kwargs": kwargs})
        future = Future()
        future.set_result(None)
        return future

    monkeypatch.setattr(Dad.BackgroundTaskManager, "submit", fake_submit)

    injected_bot = DadBot()
    try:
        assert all(call["kwargs"].get("task_kind") != "graph-init" for call in submit_calls)
    finally:
        injected_bot.shutdown()


def test_runtime_client_tries_fallback_models_sync(bot, monkeypatch):
    from dadbot.managers import runtime_client as runtime_client_module

    attempted = []

    def fake_chat(**kwargs):
        model_name = kwargs.get("model")
        attempted.append(model_name)
        if model_name == "primary":
            raise ConnectionError("primary unavailable")
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(runtime_client_module.ollama, "chat", fake_chat)
    bot.ACTIVE_MODEL = "primary"
    monkeypatch.setattr(bot, "model_candidates", lambda: ["primary", "fallback"])

    response = bot.runtime_client.call_ollama_chat([{"role": "user", "content": "hello"}], purpose="chat")

    assert response == {"message": {"content": "ok"}}
    assert attempted == ["primary", "fallback"]
    assert bot.ACTIVE_MODEL == "fallback"


def test_runtime_client_tries_fallback_models_async(bot, monkeypatch):
    from dadbot.managers import runtime_client as runtime_client_module

    attempted = []

    def fake_chat(**kwargs):
        model_name = kwargs.get("model")
        attempted.append(model_name)
        if model_name == "primary":
            raise ConnectionError("primary unavailable")
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(runtime_client_module.ollama, "chat", fake_chat)
    monkeypatch.setattr(bot.runtime_client, "ollama_async_client", lambda: None)
    bot.ACTIVE_MODEL = "primary"
    monkeypatch.setattr(bot, "model_candidates", lambda: ["primary", "fallback"])

    response = asyncio.run(bot.runtime_client.call_ollama_chat_async([{"role": "user", "content": "hello"}], purpose="chat"))

    assert response == {"message": {"content": "ok"}}
    assert attempted == ["primary", "fallback"]
    assert bot.ACTIVE_MODEL == "fallback"