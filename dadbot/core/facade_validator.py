from __future__ import annotations


def validate_dadbot_facade(bot, *, smoke: bool = False) -> None:
    """Validate facade wiring after init.

    This is intentionally a shallow structural check that fails fast when a
    manager is missing or exposes an unexpected surface.
    """
    required_managers = {
        "runtime_storage": ("load_profile",),
        "profile_runtime": ("refresh_profile_runtime",),
        "memory_manager": ("_load_memory_store", "save_memory_store"),
        "long_term_signals": ("refresh_memory_graph",),
        "memory_query": ("relevant_memories_for_input",),
        "memory_commands": ("handle_memory_command",),
        "safety_support": ("detect_crisis_signal",),
        "profile_context": ("template_context",),
        "context_builder": ("build_core_persona_prompt",),
        "tone_context": ("build_mood_context",),
        "mood_manager": ("detect",),
        "relationship_manager": ("snapshot", "get_relationship_view"),
        "internal_state_manager": ("snapshot", "reflect_after_turn"),
        "prompt_assembly": ("build_chat_request_messages",),
        "multimodal_handler": ("build_user_request_message",),
        "turn_service": ("process_user_message",),
        "reply_supervisor": ("run_reply_supervisor",),
        "reply_finalization": ("finalize",),
        "conversation_persistence": ("persist_conversation",),
        "runtime_orchestration": ("submit_background_task",),
        "status_reporting": ("dashboard_status_snapshot",),
        "maintenance_scheduler": ("run_post_turn_maintenance",),
        "runtime_interface": ("chat_loop",),
        "tool_registry": ("parse_tool_command",),
        "agentic_handler": ("handle_tool_command",),
        "memory_coordinator": ("update_memory_store",),
        "model_runtime": ("initialize_tokenizer",),
        "runtime_client": ("call_ollama_chat",),
        "runtime_state_manager": ("prompt_history",),
        "session_summary_manager": ("refresh_session_summary",),
    }

    failures: list[str] = []
    for manager_name, expected_methods in required_managers.items():
        manager = getattr(bot, manager_name, None)
        if manager is None:
            failures.append(f"{manager_name}: missing")
            continue
        for method_name in expected_methods:
            if not callable(getattr(manager, method_name, None)):
                failures.append(f"{manager_name}.{method_name}: missing")

    reply_generation = getattr(bot, "reply_generation", None)
    if reply_generation is None:
        failures.append("reply_generation: missing")
    elif not callable(getattr(reply_generation, "generate_validated_reply", None)):
        failures.append("reply_generation.generate_validated_reply: missing")

    if bot.prompt_composer is not bot.prompt_assembly:
        failures.append("prompt_composer alias mismatch")

    if smoke and not failures:
        smoke_checks = {
            "status_reporting": ("status_snapshot", (), {}),
            "runtime_state_manager": ("session_turn_count", (), {}),
            "profile_runtime": ("current_persona_preset", (), {}),
            "memory_manager": ("memory_catalog", (), {}),
            "memory_query": ("recent_memory_topics", (), {"limit": 1}),
            "memory_commands": (
                "build_memory_transcript",
                ([{"role": "user", "content": "hi"}],),
                {},
            ),
            "safety_support": ("moderation_settings", (), {}),
            "profile_context": ("template_context", (), {}),
            "context_builder": ("build_core_persona_prompt", (), {}),
            "tone_context": ("build_mood_context", ("neutral",), {}),
            "mood_manager": ("fastpath_detect", ("quick check",), {}),
            "relationship_manager": ("snapshot", (), {}),
            "internal_state_manager": ("snapshot", (), {}),
            "multimodal_handler": (
                "build_user_request_message",
                ("quick check", None),
                {},
            ),
            "turn_service": ("should_offer_daily_checkin_for_turn", (), {}),
            "reply_supervisor": ("reply_supervisor_snapshot", (), {}),
            "reply_finalization": (
                "should_calibrate_pushback",
                ("quick check", "neutral"),
                {},
            ),
            "runtime_orchestration": ("background_task_snapshot", (), {"limit": 1}),
            "maintenance_scheduler": ("maintenance_snapshot", (), {}),
            "tool_registry": ("get_available_tools", (), {}),
            "agentic_handler": ("format_reminder_list", ([],), {}),
            "memory_coordinator": (
                "build_active_consolidated_context",
                ("quick check",),
                {},
            ),
            "model_runtime": ("model_candidates", (), {}),
            "session_summary_manager": ("build_session_summary_prompt", (None, []), {}),
        }
        for manager_name, (method_name, args, kwargs) in smoke_checks.items():
            manager = getattr(bot, manager_name, None)
            method = getattr(manager, method_name, None)
            if not callable(method):
                failures.append(f"{manager_name}.{method_name}: missing for smoke")
                continue
            try:
                method(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{manager_name}.{method_name}: smoke failed ({exc})")

    if failures:
        details = "; ".join(failures)
        raise RuntimeError(f"DadBot manager validation failed: {details}")
