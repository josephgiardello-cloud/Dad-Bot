import asyncio
import contextlib
import json
import os
import time
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import ollama

from Dad import DadBot
from dadbot.core.execution_trace_context import ExecutionTraceRecorder, bind_execution_trace
from dadbot.core.graph import LedgerMutationOp, TurnContext
from dadbot_system.state import InMemoryStateStore


class DadBotRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.cleanup_temp_dir)

        self.bot = DadBot()
        self.addCleanup(self.bot.shutdown)
        self.addCleanup(self.bot.wait_for_semantic_index_idle, 5)

        temp_path = Path(self.temp_dir.name)
        self.bot.MEMORY_PATH = temp_path / "dad_memory.json"
        self.bot.SEMANTIC_MEMORY_DB_PATH = temp_path / "dad_memory_semantic.sqlite3"
        self.bot.SESSION_LOG_DIR = temp_path / "session_logs"
        self.bot.MEMORY_STORE = self.bot.default_memory_store()
        self.bot.save_memory_store()
        self.bot.embed_texts = self.fake_embed_texts

    def cleanup_temp_dir(self):
        for attempt in range(3):
            try:
                self.temp_dir.cleanup()
                return
            except (PermissionError, OSError):
                if attempt == 2:
                    return
                time.sleep(0.1 * (attempt + 1))

    @staticmethod
    def fake_embed_texts(texts, purpose="semantic retrieval"):
        items = [texts] if isinstance(texts, str) else list(texts)
        vectors = []

        for item in items:
            lowered = str(item).lower()
            vector = [0.0] * 12
            vector[0] = 1.0 if any(token in lowered for token in ("save", "saving", "budget", "money")) else 0.0
            vector[1] = 1.0 if any(token in lowered for token in ("work", "career", "boss")) else 0.0
            vector[2] = 1.0 if "stress" in lowered or "anxious" in lowered else 0.0
            vectors.append(vector)

        return vectors

    @contextlib.contextmanager
    def _save_commit_context(self):
        previous_commit_active = bool(getattr(self.bot, "_graph_commit_active", False))
        turn_context = TurnContext(user_input="test")
        turn_context.temporal = SimpleNamespace(
            wall_time=datetime.now().isoformat(timespec="seconds"),
            wall_date=date.today().isoformat(),
        )
        turn_context.state["_active_graph_stage"] = "save"
        try:
            self.bot._graph_commit_active = True
            yield turn_context
        finally:
            self.bot._graph_commit_active = previous_commit_active

    def test_split_reminder_details_parses_relative_datetime(self):
        title, due_text = self.bot.split_reminder_details("call the dentist tomorrow at 3pm")

        self.assertEqual(title, "call the dentist")
        self.assertTrue(due_text.startswith((date.today() + timedelta(days=1)).isoformat()))
        self.assertIn("3:00 PM", due_text)

    def test_split_reminder_details_without_date_signal_returns_original_text(self):
        title, due_text = self.bot.split_reminder_details("call the dentist sometime")

        self.assertEqual(title, "call the dentist sometime")
        self.assertEqual(due_text, "")

    def test_split_reminder_details_parses_next_week(self):
        title, due_text = self.bot.split_reminder_details("submit the form next week")

        self.assertEqual(title, "submit the form")
        self.assertTrue(due_text.startswith((date.today() + timedelta(days=7)).isoformat()))

    def test_parse_model_json_content_handles_code_fences(self):
        parsed = self.bot.parse_model_json_content("```json\n{\"approved\": true, \"revised_reply\": null}\n```")

        self.assertEqual(parsed["approved"], True)
        self.assertIsNone(parsed["revised_reply"])

    def test_refresh_session_summary_updates_summary_and_coverage_window(self):
        self.bot.RECENT_HISTORY_WINDOW = 1
        self.bot.SUMMARY_TRIGGER_MESSAGES = 2
        self.bot.history = [
            {"role": "system", "content": "Dad system prompt"},
            {"role": "user", "content": "Work felt heavy today."},
            {"role": "assistant", "content": "I am here with you, buddy."},
            {"role": "user", "content": "I think I handled it better than last week."},
        ]

        captured = {}

        def fake_chat(messages, options=None, response_format=None, purpose="chat"):
            captured["messages"] = messages
            captured["options"] = options
            captured["response_format"] = response_format
            captured["purpose"] = purpose
            return {"message": {"content": "- Tony felt stretched at work but steadier than before."}}

        with patch.object(self.bot, "call_ollama_chat", side_effect=fake_chat):
            summary = self.bot.refresh_session_summary(force=True)

        self.assertEqual(summary, "- Tony felt stretched at work but steadier than before.")
        self.assertEqual(self.bot.session_summary, "- Tony felt stretched at work but steadier than before.")
        self.assertEqual(self.bot.session_summary_covered_messages, 2)
        self.assertIsNotNone(self.bot.session_summary_updated_at)
        self.assertEqual(captured["purpose"], "session summarization")
        self.assertEqual(captured["options"], {"temperature": 0.1})
        self.assertIn("Tony: Work felt heavy today.", captured["messages"][0]["content"])
        self.assertIn("Dad: I am here with you, buddy.", captured["messages"][0]["content"])

    def test_handle_memory_command_remembers_and_formats_saved_memory(self):
        reply = self.bot.handle_memory_command("please remember I'm saving money for emergencies")

        self.assertIsNotNone(reply)
        self.assertIn("I'll remember that under finance", reply)
        self.assertIn("Tony is saving money for emergencies", reply)
        memories = self.bot.memory_catalog()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["category"], "finance")

    def test_embedding_candidates_prioritize_bge_m3(self):
        candidates = self.bot.embedding_model_candidates()

        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0], "bge-m3")
        self.assertIn("nomic-embed-text", candidates)

    def test_normalize_mood_maps_positive_alias(self):
        self.assertEqual(self.bot.normalize_mood("happy"), "positive")

    def test_normalize_mood_maps_phrase_alias(self):
        self.assertEqual(self.bot.normalize_mood("I feel burned out lately"), "tired")

    def test_normalize_mood_defaults_unknown_to_neutral(self):
        self.assertEqual(self.bot.normalize_mood("curious"), "neutral")

    def test_decay_relationship_level_without_last_updated_returns_clamped_score(self):
        self.assertEqual(self.bot.decay_relationship_level(140, None), 100)

    def test_decay_relationship_level_moves_high_score_toward_midpoint(self):
        last_updated = (date.today() - timedelta(days=10)).isoformat()

        decayed = self.bot.decay_relationship_level(90, last_updated)

        self.assertLess(decayed, 90)
        self.assertGreater(decayed, 50)

    def test_decay_relationship_level_moves_low_score_toward_midpoint(self):
        last_updated = (date.today() - timedelta(days=10)).isoformat()

        decayed = self.bot.decay_relationship_level(10, last_updated)

        self.assertGreater(decayed, 10)
        self.assertLess(decayed, 50)

    def test_normalize_confidence_rewards_recency_when_inferred(self):
        recent = self.bot.normalize_confidence(None, source_count=2, contradiction_count=0, updated_at=date.today().isoformat())
        older = self.bot.normalize_confidence(None, source_count=2, contradiction_count=0, updated_at=(date.today() - timedelta(days=60)).isoformat())

        self.assertGreater(recent, older)

    def test_normalize_confidence_penalizes_contradictions_when_inferred(self):
        stable = self.bot.normalize_confidence(None, source_count=4, contradiction_count=0, updated_at=date.today().isoformat())
        conflicted = self.bot.normalize_confidence(None, source_count=4, contradiction_count=2, updated_at=date.today().isoformat())

        self.assertLess(conflicted, stable)

    def test_normalize_confidence_invalid_explicit_value_falls_back_to_midpoint(self):
        confidence = self.bot.normalize_confidence("not-a-number")

        self.assertEqual(confidence, 0.5)

    def test_memory_quality_score_rewards_specific_memory(self):
        score = self.bot.memory_quality_score({"summary": "Tony is saving money for a trip budget."})

        self.assertGreaterEqual(score, 40)

    def test_memory_quality_score_penalizes_generic_memory(self):
        score = self.bot.memory_quality_score({"summary": "Tony shared that personal struggles."})

        self.assertLess(score, 5)

    def test_is_high_quality_memory_rejects_low_signal_summary(self):
        self.assertFalse(self.bot.is_high_quality_memory({"summary": "Tony shared that emotional state.", "category": "health"}))

    def test_prepare_final_reply_blends_before_signoff(self):
        self.bot._pending_daily_checkin_context = True

        reply = self.bot.prepare_final_reply("You're doing okay, buddy.", "neutral")

        self.assertEqual(reply, "You're doing okay, buddy. How's your day shaping up so far? Love you, buddy.")

    def test_create_chat_thread_keeps_conversation_state_isolated(self):
        original_thread_id = self.bot.active_thread_id
        self.bot.history.append({"role": "user", "content": "Thread one is about work."})
        self.bot.history.append({"role": "assistant", "content": "Dad heard the work update."})
        self.bot.sync_active_thread_snapshot()

        new_thread = self.bot.create_chat_thread()

        self.assertNotEqual(new_thread["thread_id"], original_thread_id)
        self.assertEqual(self.bot.conversation_history(), [])

        self.bot.history.append({"role": "user", "content": "Thread two is about family."})
        self.bot.sync_active_thread_snapshot()

        self.bot.switch_chat_thread(original_thread_id)

        restored_history = self.bot.conversation_history()
        self.assertEqual(restored_history[-2]["content"], "Thread one is about work.")
        self.assertEqual(restored_history[-1]["content"], "Dad heard the work update.")

    def test_snapshot_session_state_roundtrip_preserves_chat_threads(self):
        first_thread_id = self.bot.active_thread_id
        self.bot.history.append({"role": "user", "content": "Keep this in thread one."})
        self.bot.sync_active_thread_snapshot()

        second_thread = self.bot.create_chat_thread()
        self.bot.history.append({"role": "user", "content": "This belongs to thread two."})
        self.bot.mark_chat_thread_closed(closed=True)
        snapshot = self.bot.snapshot_session_state()

        self.bot.reset_session_state()
        restored = self.bot.load_session_state_snapshot(snapshot)

        self.assertEqual(len(restored["chat_threads"]), 2)
        self.assertEqual(restored["active_thread_id"], second_thread["thread_id"])
        self.assertTrue(any(thread["closed"] for thread in restored["chat_threads"] if thread["thread_id"] == second_thread["thread_id"]))

        self.bot.switch_chat_thread(first_thread_id)
        self.assertEqual(self.bot.conversation_history()[-1]["content"], "Keep this in thread one.")

        self.bot.switch_chat_thread(second_thread["thread_id"])
        self.assertEqual(self.bot.conversation_history()[-1]["content"], "This belongs to thread two.")

    def test_finalize_reply_can_skip_signoff(self):
        self.bot.APPEND_SIGNOFF = False

        reply = self.bot.finalize_reply("You're doing okay, buddy.")

        self.assertEqual(reply, "You're doing okay, buddy.")

    def test_prepare_final_reply_can_add_family_echo(self):
        self.bot._pending_daily_checkin_context = False
        self.bot.session_turn_count = lambda: 4
        self.bot.family_echo = lambda *_args, **_kwargs: "Carrie would probably tell you to breathe and let the good news land."

        reply = self.bot.prepare_final_reply("That's great news, buddy.", "positive", "I got promoted today.")

        self.assertIn("Carrie would probably tell you", reply)

    def test_cadence_settings_can_delay_family_echo(self):
        self.bot.CADENCE = {"family_echo_turn_interval": 6}
        self.bot._pending_daily_checkin_context = False
        self.bot.session_turn_count = lambda: 4
        self.bot.family_echo = lambda *_args, **_kwargs: "Carrie would probably tell you to breathe and let the good news land."

        reply = self.bot.prepare_final_reply("That's great news, buddy.", "positive", "I got promoted today.")

        self.assertNotIn("Carrie would probably tell you", reply)

    def test_apply_persona_preset_updates_style_without_save(self):
        changed = self.bot.apply_persona_preset("coach", save=False)

        self.assertTrue(changed)
        self.assertEqual(self.bot.current_persona_preset(), "coach")
        self.assertEqual(self.bot.STYLE["name"], "Coach Dad")
        self.assertEqual(self.bot.STYLE["signoff"], "Proud of you, buddy.")

    def test_update_cadence_profile_normalizes_runtime_settings(self):
        updated = self.bot.update_cadence_profile(
            {
                "wisdom_turn_interval": 6,
                "family_echo_turn_interval": "7",
                "life_pattern_confidence_threshold": 0,
                "life_pattern_queue_limit": "invalid",
            },
            save=False,
        )

        self.assertEqual(updated["wisdom_turn_interval"], 6)
        self.assertEqual(updated["family_echo_turn_interval"], 7)
        self.assertEqual(updated["life_pattern_confidence_threshold"], 1)
        self.assertEqual(updated["life_pattern_queue_limit"], 2)
        self.assertEqual(self.bot.CADENCE["wisdom_turn_interval"], 6)
        self.assertEqual(self.bot.cadence_settings()["family_echo_turn_interval"], 7)

    def test_update_runtime_profile_normalizes_runtime_preferences(self):
        updated = self.bot.update_runtime_profile(
            {
                "preferred_embedding_models": ["mxbai-embed-large", "", "bge-m3", "mxbai-embed-large"],
                "max_thinking_time_seconds": "75",
                "stream_max_chars": "16000",
                "graph_refresh_debounce_seconds": "-4",
            },
            save=False,
        )

        self.assertEqual(updated["preferred_embedding_models"], ["mxbai-embed-large", "bge-m3"])
        self.assertEqual(updated["max_thinking_time_seconds"], 75)
        self.assertEqual(updated["stream_max_chars"], 16000)
        self.assertEqual(updated["graph_refresh_debounce_seconds"], 0)
        self.assertEqual(self.bot.PREFERRED_EMBEDDING_MODELS, ("mxbai-embed-large", "bge-m3"))
        self.assertEqual(self.bot.STREAM_TIMEOUT_SECONDS, 75)
        self.assertEqual(self.bot.STREAM_MAX_CHARS, 16000)
        self.assertEqual(self.bot.GRAPH_REFRESH_DEBOUNCE_SECONDS, 0)

    def test_agentic_tool_settings_defaults_to_enabled(self):
        self.bot.AGENTIC_TOOLS = {}

        settings = self.bot.agentic_tool_settings()

        self.assertTrue(settings["enabled"])
        self.assertTrue(settings["auto_reminders"])
        self.assertTrue(settings["auto_web_lookup"])

    def test_get_available_tools_exposes_reminder_and_web_search_definitions(self):
        tools = self.bot.get_available_tools()

        tool_names = [tool["function"]["name"] for tool in tools]
        self.assertIn("set_reminder", tool_names)
        self.assertIn("web_search", tool_names)
        reminder_tool = next(t for t in tools if t["function"]["name"] == "set_reminder")
        web_search_tool = next(t for t in tools if t["function"]["name"] == "web_search")
        self.assertEqual(reminder_tool["type"], "function")
        self.assertIn("title", reminder_tool["function"]["parameters"]["properties"])
        self.assertIn("query", web_search_tool["function"]["parameters"]["properties"])

    def test_update_agentic_tool_profile_persists_runtime_values(self):
        updated = self.bot.update_agentic_tool_profile(
            {
                "enabled": True,
                "auto_reminders": False,
                "auto_web_lookup": True,
            },
            save=False,
        )

        self.assertEqual(updated["auto_reminders"], False)
        self.assertEqual(updated["auto_web_lookup"], True)
        self.assertEqual(self.bot.agentic_tool_settings()["auto_reminders"], False)

    def test_autonomous_tool_result_can_set_reminder(self):
        self.bot.update_agentic_tool_profile({"enabled": True, "auto_reminders": True, "auto_web_lookup": False}, save=False)

        reply, observation = self.bot.autonomous_tool_result_for_input(
            "I need to remember to call the bank tomorrow at 9am",
            "neutral",
        )

        self.assertIsNone(observation)
        self.assertIsInstance(reply, str)
        self.assertIn("turned that into a reminder", reply.lower())
        reminders = self.bot.reminder_catalog()
        self.assertEqual(len(reminders), 1)
        self.assertIn("call the bank", reminders[0]["title"])

    def test_autonomous_tool_result_can_add_web_observation(self):
        self.bot.update_agentic_tool_profile({"enabled": True, "auto_reminders": False, "auto_web_lookup": True}, save=False)
        self.bot.lookup_web = lambda _query: {
            "heading": "Weather",
            "summary": "Light rain expected tonight.",
            "source_label": "example.com",
        }

        reply, observation = self.bot.autonomous_tool_result_for_input(
            "What's the weather in Boston tonight?",
            "neutral",
        )

        self.assertIsNone(reply)
        self.assertIn("Light rain expected tonight.", observation)

    def test_plan_agentic_tools_can_set_reminder_from_model_plan(self):
        self.bot.update_agentic_tool_profile({"enabled": True, "auto_reminders": True, "auto_web_lookup": False}, save=False)
        self.bot.call_ollama_chat = lambda *args, **kwargs: {
            "message": {"content": json.dumps({
                "needs_tool": True,
                "tool": "set_reminder",
                "parameters": {"title": "call the bank", "due_text": "tomorrow at 9:00 AM"},
                "reason": "Tony asked to remember something later.",
            })}
        }

        reply, observation = self.bot.turn_service.plan_agentic_tools("I need to remember to call the bank tomorrow at 9am", "neutral")

        self.assertIsNone(observation)
        self.assertIn("set that reminder", reply)
        self.assertEqual(len(self.bot.reminder_catalog()), 1)
        snapshot = self.bot.planner_debug_snapshot()
        self.assertEqual(snapshot["planner_status"], "used_tool")
        self.assertEqual(snapshot["planner_tool"], "set_reminder")
        self.assertEqual(snapshot["final_path"], "planner_tool")

    def test_prepare_user_turn_uses_planner_before_heuristic_tool_logic(self):
        self.bot.mood_manager.detect = lambda *_args, **_kwargs: "neutral"
        self.bot.direct_reply_for_input = lambda *_args, **_kwargs: None
        self.bot.call_ollama_chat = lambda *args, **kwargs: {
            "message": {"content": json.dumps({
                "needs_tool": True,
                "tool": "web_search",
                "parameters": {"query": "weather in Boston tonight"},
                "reason": "This needs current weather information.",
            })}
        }
        self.bot.lookup_web = lambda query: {
            "heading": "Weather",
            "summary": f"Rain expected for {query}.",
            "source_label": "example.com",
        }
        self.bot.autonomous_tool_result_for_input = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("heuristic fallback should not run when planner already found a tool"))

        current_mood, dad_reply, should_end, turn_text, _attachments = self.bot.turn_service.prepare_user_turn(
            "What's the weather in Boston tonight?"
        )

        self.assertEqual(current_mood, "neutral")
        self.assertIsNone(dad_reply)
        self.assertFalse(should_end)
        self.assertEqual(turn_text, "What's the weather in Boston tonight?")
        self.assertIn("Rain expected", self.bot.build_active_tool_observation_context())
        snapshot = self.bot.planner_debug_snapshot()
        self.assertEqual(snapshot["planner_status"], "used_tool")
        self.assertEqual(snapshot["planner_tool"], "web_search")
        self.assertEqual(snapshot["final_path"], "planner_tool")

    def test_prepare_user_turn_records_heuristic_fallback_after_planner_no_tool(self):
        self.bot.mood_manager.detect = lambda *_args, **_kwargs: "neutral"
        self.bot.direct_reply_for_input = lambda *_args, **_kwargs: None
        self.bot.call_ollama_chat = lambda *args, **kwargs: {
            "message": {"content": json.dumps({
                "needs_tool": False,
                "tool": None,
                "parameters": None,
                "reason": "No external tool is required from the planner's view.",
            })}
        }
        self.bot.lookup_web = lambda _query: {
            "heading": "Weather",
            "summary": "Light rain expected tonight.",
            "source_label": "example.com",
        }

        current_mood, dad_reply, should_end, _turn_text, _attachments = self.bot.turn_service.prepare_user_turn(
            "What's the weather in Boston tonight?"
        )

        self.assertEqual(current_mood, "neutral")
        self.assertIsNone(dad_reply)
        self.assertFalse(should_end)
        snapshot = self.bot.planner_debug_snapshot()
        self.assertEqual(snapshot["planner_status"], "no_tool")
        self.assertEqual(snapshot["fallback_status"], "used_tool")
        self.assertEqual(snapshot["fallback_tool"], "web_search")
        self.assertEqual(snapshot["final_path"], "heuristic_tool")

    def test_planner_debug_snapshot_normalizes_invalid_parameters_payload(self):
        self.bot.runtime_state_manager.planner_debug = {
            "updated_at": "2026-04-20T12:00:00",
            "user_input": "Need help",
            "current_mood": "happy",
            "planner_status": "used_tool",
            "planner_parameters": "not-a-dict",
            "final_path": "planner_tool",
        }

        snapshot = self.bot.planner_debug_snapshot()

        self.assertEqual(snapshot["current_mood"], "positive")
        self.assertEqual(snapshot["planner_parameters"], {})

    def test_enrich_multimodal_attachments_adds_analysis_for_text_models(self):
        self.bot.ACTIVE_MODEL = "llama3.2:3b"
        self.bot.find_available_vision_model = lambda: "llava:7b"
        self.bot.call_ollama_chat_with_model = lambda *args, **kwargs: {
            "message": {"content": "A laptop screen shows a traceback in a Python terminal."}
        }

        enriched = self.bot.enrich_multimodal_attachments([
            {"type": "image", "image_b64": "ZmFrZQ==", "name": "debug.png", "note": "error screenshot"}
        ])

        self.assertEqual(len(enriched), 1)
        self.assertIn("analysis", enriched[0])
        self.assertIn("traceback", enriched[0]["analysis"].lower())

    def test_build_request_system_prompt_adds_visual_debug_context_for_image_turn(self):
        prompt = self.bot.build_request_system_prompt(
            "Can you help me figure out this traceback screenshot?",
            "stressed",
            attachments=[{"type": "image", "image_b64": "ZmFrZQ==", "name": "traceback.png", "note": "python terminal error"}],
        )

        self.assertIn("Visual mode for this turn: Debug Screenshot", prompt)
        self.assertIn("visible error text", prompt.lower())

    def test_build_image_analysis_prompt_routes_homework_images(self):
        prompt = self.bot.build_image_analysis_prompt(
            "math worksheet",
            user_input="Can you help me solve this homework problem?",
            attachment={"type": "image", "name": "worksheet.png"},
        )

        self.assertIn("Homework Help", prompt)
        self.assertIn("equations", prompt)

    def test_add_reminder_persists_due_at_for_scheduler(self):
        reminder = self.bot.add_reminder("call the bank", "2026-04-20 03:00 PM")

        self.assertIsNotNone(reminder)
        self.assertEqual(reminder["due_text"], "2026-04-20 03:00 PM")
        self.assertTrue(reminder["due_at"].startswith("2026-04-20T15:00"))

    def test_run_scheduled_proactive_jobs_queues_due_reminder_message(self):
        reference_time = datetime(2026, 4, 20, 14, 40)
        self.bot.add_reminder("call the bank", "2026-04-20 03:00 PM")

        result = self.bot.run_scheduled_proactive_jobs(reference_time=reference_time)

        # Phase 4: background patches are queued; flush before reading MEMORY_STORE state.
        bg_queue = getattr(self.bot, "_background_memory_store_patch_queue", None)
        if isinstance(bg_queue, list):
            for patch in list(bg_queue):
                if isinstance(patch, dict):
                    self.bot.mutate_memory_store(**patch)
            bg_queue.clear()

        self.assertEqual(result["queued_reminders"], 1)
        queued = self.bot.pending_proactive_messages()
        self.assertTrue(any("call the bank" in item["message"].lower() for item in queued))
        reminder = self.bot.reminder_catalog()[0]
        self.assertEqual(reminder["notification_count"], 1)
        self.assertIsNotNone(reminder["last_notified_at"])

    def test_run_scheduled_proactive_jobs_queues_matching_day_pattern_once(self):
        self.bot.MEMORY_STORE["life_patterns"] = [
            {
                "summary": "Tony often carries work stress on Mondays.",
                "topic": "work",
                "mood": "stressed",
                "day_hint": "Monday",
                "confidence": 88,
                "last_seen_at": "2026-04-18T20:00:00",
                "proactive_message": "Mondays seem to carry extra work weight for you lately.",
            }
        ]

        result = self.bot.run_scheduled_proactive_jobs(reference_time=datetime(2026, 4, 20, 9, 0))

        # Phase 4: background patches are queued; flush before reading MEMORY_STORE state.
        bg_queue = getattr(self.bot, "_background_memory_store_patch_queue", None)
        if isinstance(bg_queue, list):
            for patch in list(bg_queue):
                if isinstance(patch, dict):
                    self.bot.mutate_memory_store(**patch)
            bg_queue.clear()

        self.assertEqual(result["queued_patterns"], 1)
        self.assertTrue(any(item["source"] == "scheduled-pattern" for item in self.bot.pending_proactive_messages()))
        updated_pattern = self.bot.life_patterns()[0]
        self.assertEqual(updated_pattern["last_proactive_at"][:10], "2026-04-20")

    def test_relationship_calibration_profile_can_disable_pushback(self):
        self.bot.update_relationship_calibration_profile(
            {
                "enabled": False,
            },
            save=False,
        )

        should_push = self.bot.should_calibrate_pushback("I keep procrastinating", "neutral")

        self.assertFalse(should_push)

    def test_relationship_calibration_profile_uses_custom_opening_line(self):
        self.bot.update_relationship_calibration_profile(
            {
                "enabled": True,
                "protected_moods": ["sad"],
                "trigger_patterns": [r"\bi keep procrastinating\b"],
                "opening_line": "Straight talk, buddy:",
            },
            save=False,
        )

        calibrated = self.bot.apply_calibrated_pushback("Let's make a plan.", "I keep procrastinating.", "neutral")

        self.assertTrue(calibrated.startswith("Straight talk, buddy:"))

    def test_cadence_settings_come_from_runtime_config_defaults(self):
        self.bot.CADENCE = {}

        cadence = self.bot.cadence_settings()

        self.assertEqual(cadence["persona_evolution_min_sessions"], self.bot.runtime_config.cadence_defaults["persona_evolution_min_sessions"])
        self.assertEqual(cadence["family_echo_turn_interval"], self.bot.runtime_config.cadence_defaults["family_echo_turn_interval"])

    def test_memory_manager_normalize_memory_store_applies_configured_limits(self):
        store = self.bot.default_memory_store()
        store["persona_evolution"] = [
            {
                "trait": f"trait-{index}",
                "reason": "earned over time",
                "announcement": "",
                "session_count": index,
                "applied_at": f"2026-04-{index + 1:02d}T20:00:00",
            }
            for index in range(12)
        ]
        store["pending_proactive_messages"] = [
            {
                "message": f"message-{index}",
                "source": "life-pattern",
                "created_at": f"2026-04-{index + 1:02d}T20:00:00",
            }
            for index in range(10)
        ]
        store["session_archive"] = [
            {
                "summary": f"Session {index}",
                "topics": ["work"],
                "dominant_mood": "happy",
                "turn_count": 2,
                "created_at": f"2026-04-{(index % 28) + 1:02d}T20:00:00",
                "id": str(index),
            }
            for index in range(30)
        ]
        store["last_mood"] = "happy"

        normalized = self.bot.memory_manager.normalize_memory_store(store)

        self.assertEqual(len(normalized["persona_evolution"]), self.bot.runtime_config.store_limits["persona_evolution"])
        self.assertEqual(len(normalized["pending_proactive_messages"]), self.bot.runtime_config.store_limits["pending_proactive_messages"])
        self.assertEqual(len(normalized["session_archive"]), self.bot.runtime_config.store_limits["session_archive"])
        self.assertEqual(normalized["last_mood"], "positive")

    def test_long_term_signals_manager_summarize_memory_graph_handles_empty_graph(self):
        self.bot.MEMORY_STORE["memory_graph"] = {"nodes": [], "edges": [], "updated_at": None}

        summary = self.bot.long_term_signals.summarize_memory_graph()

        self.assertEqual(summary, "No strong graph links yet.")

    def test_long_term_signals_manager_build_pattern_message_prefers_proactive_text(self):
        message = self.bot.long_term_signals.build_pattern_message(
            {"summary": "Tony often carries work stress on Sundays.", "proactive_message": "Sundays seem heavy for work lately."}
        )

        self.assertEqual(message, "Sundays seem heavy for work lately.")

    def test_long_term_signals_manager_should_offer_family_echo_respects_mom_reference(self):
        self.assertFalse(self.bot.long_term_signals.should_offer_family_echo("I already talked to mom about it.", "stressed"))
        self.assertTrue(self.bot.long_term_signals.should_offer_family_echo("Work was brutal today.", "stressed"))

    def test_long_term_signals_manager_wisdom_generation_can_trigger_from_topic_overlap(self):
        self.bot.CADENCE = {"wisdom_min_archived_sessions": 2, "wisdom_turn_interval": 9}
        self.bot.MEMORY_STORE["session_archive"] = [
            {"summary": "A", "topics": ["work"], "dominant_mood": "neutral", "turn_count": 2, "created_at": "2026-04-01T20:00:00", "id": "a"},
            {"summary": "B", "topics": ["work"], "dominant_mood": "neutral", "turn_count": 2, "created_at": "2026-04-02T20:00:00", "id": "b"},
        ]
        self.bot.MEMORY_STORE["memory_graph"] = {
            "nodes": [{"id": "category:work", "label": "work", "type": "category", "weight": 3}],
            "edges": [{"source": "work", "target": "stressed", "weight": 3}],
            "updated_at": date.today().isoformat(),
        }
        self.bot.session_turn_count = lambda: 2
        self.bot.relationship.top_topics = lambda *_args, **_kwargs: ["work"]

        should_generate = self.bot.long_term_signals.should_generate_wisdom_insight("Work still feels heavy.")

        self.assertTrue(should_generate)

    def test_long_term_signals_manager_detect_life_patterns_skips_existing_summary(self):
        self.bot.MEMORY_STORE["life_patterns"] = [
            {
                "summary": "Tony often carries work stressed on Saturdays.",
                "topic": "work",
                "mood": "stressed",
                "day_hint": "Saturday",
                "confidence": 85,
                "last_seen_at": "2026-04-11T20:00:00",
                "proactive_message": "Saturdays seem heavy for work lately.",
            }
        ]
        self.bot.MEMORY_STORE["session_archive"] = [
            {"summary": "Work felt heavy again.", "topics": ["work"], "dominant_mood": "stressed", "turn_count": 4, "created_at": "2026-04-04T20:00:00", "id": "a"},
            {"summary": "Saturday work dread again.", "topics": ["work"], "dominant_mood": "stressed", "turn_count": 4, "created_at": "2026-04-11T20:00:00", "id": "b"},
            {"summary": "Another tough Saturday night about work.", "topics": ["work"], "dominant_mood": "stressed", "turn_count": 4, "created_at": "2026-04-18T20:00:00", "id": "c"},
            {"summary": "Still talking about work stress.", "topics": ["work"], "dominant_mood": "stressed", "turn_count": 4, "created_at": "2026-04-25T20:00:00", "id": "d"},
        ]

        detected = self.bot.long_term_signals.detect_life_patterns(force=True)

        self.assertEqual(detected, [])
        self.assertEqual(len(self.bot.life_patterns()), 1)

    def test_safety_support_manager_settings_uses_profile_configuration(self):
        self.bot.CRISIS_SUPPORT = {
            "enabled": False,
            "high_risk_phrases": ["jump off a bridge"],
            "grounding_lines": ["Stay with me.", "Call someone now."],
            "resource_line": "Call emergency services now.",
        }

        settings = self.bot.safety_support.settings()

        self.assertFalse(settings["enabled"])
        self.assertEqual(settings["high_risk_phrases"], ["jump off a bridge"])
        self.assertEqual(settings["grounding_lines"], ["Stay with me.", "Call someone now."])
        self.assertEqual(settings["resource_line"], "Call emergency services now.")

    def test_safety_support_manager_direct_reply_returns_finalized_support_message(self):
        reply = self.bot.safety_support.direct_reply_for_input("I want to kill myself tonight.")

        self.assertIsNotNone(reply)
        self.assertIn("988", reply)
        self.assertTrue(reply.endswith("Love you, buddy."))

    def test_safety_support_manager_direct_reply_returns_none_without_crisis_signal(self):
        reply = self.bot.safety_support.direct_reply_for_input("I had a long day and feel off.")

        self.assertIsNone(reply)

    def test_safety_support_manager_negated_reassurance_matches_common_variants(self):
        self.assertTrue(self.bot.safety_support.has_negated_reassurance("i won't hurt myself"))
        self.assertTrue(self.bot.safety_support.has_negated_reassurance("not suicidal just overwhelmed"))
        self.assertFalse(self.bot.safety_support.has_negated_reassurance("i want to hurt myself"))

    def test_memory_manager_normalize_consolidated_entry_limits_lists_and_recomputes_confidence(self):
        normalized = self.bot.memory_manager.normalize_consolidated_memory_entry(
            {
                "summary": "i'm saving money for a house",
                "category": "finance",
                "source_count": 3,
                "confidence": None,
                "supporting_summaries": [
                    "i'm saving money",
                    "i've been sticking to a budget",
                    "saving matters to me",
                    "trying to spend less",
                    "extra summary should drop",
                ],
                "contradictions": ["A", "B", "C", "D", "E"],
                "updated_at": date.today().isoformat(),
            }
        )

        self.assertEqual(normalized["summary"], "Tony is saving money for a house.")
        self.assertEqual(len(normalized["supporting_summaries"]), self.bot.runtime_config.window("supporting_summaries", 4))
        self.assertEqual(len(normalized["contradictions"]), self.bot.runtime_config.window("contradictions", 4))
        self.assertLess(normalized["confidence"], 0.7)

    def test_memory_manager_normalize_relationship_state_decays_levels_and_limits_recent_checkins(self):
        stale_date = (date.today() - timedelta(days=200)).isoformat()
        normalized = self.bot.memory_manager.normalize_relationship_state(
            {
                "trust_level": 95,
                "openness_level": 5,
                "emotional_momentum": "stormy",
                "hypotheses": [
                    {"name": "guarded_distance", "label": "Guarded Distance", "summary": "Needs room.", "probability": 0.75},
                    {"name": "supportive_baseline", "label": "Supportive Baseline", "summary": "Stable.", "probability": 0.25},
                ],
                "active_hypothesis": "guarded_distance",
                "last_hypothesis_updated": stale_date,
                "recurring_topics": {"work": "3", "": 9, "family": 2},
                "recent_checkins": [
                    {"date": f"2026-04-{(index % 28) + 1:02d}", "mood": "happy", "topic": f"topic-{index}"}
                    for index in range(40)
                ],
                "last_reflection": "Still showing up.",
                "last_updated": stale_date,
            }
        )

        self.assertLess(normalized["trust_level"], 95)
        self.assertGreater(normalized["openness_level"], 5)
        self.assertEqual(normalized["emotional_momentum"], "steady")
        self.assertEqual(len(normalized["recent_checkins"]), self.bot.runtime_config.store_limits["recent_checkins"])
        self.assertEqual(set(normalized["recurring_topics"].keys()), {"work", "family"})
        self.assertEqual(normalized["active_hypothesis"], "guarded_distance")
        self.assertEqual(len(normalized["hypotheses"]), 2)
        self.assertAlmostEqual(sum(item["probability"] for item in normalized["hypotheses"]), 1.0, places=2)

    def test_update_relationship_state_tracks_active_hypothesis(self):
        snapshot = self.bot.relationship.current_state()

        self.assertIsInstance(snapshot, dict)
        self.assertIn("active_hypothesis", snapshot)
        self.assertIn("trust_level", snapshot)

    def test_memory_manager_detected_contradictions_deduplicate_repeat_pairs(self):
        contradictions = self.bot.detect_memory_contradictions(
            memories=[
                {"summary": "Tony is saving money for a car.", "category": "finance"},
                {"summary": "Tony is not saving money for a car right now.", "category": "finance"},
            ],
            existing_insights=[
                {"summary": "Tony is saving money for a car.", "category": "finance"},
                {"summary": "Tony is not saving money for a car right now.", "category": "finance"},
            ],
        )

        self.assertEqual(len(contradictions), 1)

    def test_living_dad_snapshot_surfaces_latest_long_term_signals(self):
        self.bot.MEMORY_STORE["persona_evolution"] = [
            {
                "trait": "more coach-like",
                "reason": "Tony responds well to structure",
                "announcement": "",
                "session_count": 10,
                "applied_at": "2026-04-10T20:00:00",
            }
        ]
        self.bot.MEMORY_STORE["wisdom_insights"] = [
            {
                "summary": "Slow the moment down before work runs you.",
                "topic": "work",
                "trigger": "Work is getting to me.",
                "created_at": "2026-04-11T20:00:00",
            }
        ]
        self.bot.MEMORY_STORE["life_patterns"] = [
            {
                "summary": "Tony often carries work stress on Sundays.",
                "topic": "work",
                "mood": "stressed",
                "day_hint": "Sunday",
                "confidence": 85,
                "last_seen_at": "2026-04-12T20:00:00",
                "proactive_message": "Sundays seem heavy for work lately.",
            }
        ]
        self.bot.MEMORY_STORE["pending_proactive_messages"] = [
            {
                "message": "I've noticed Sundays seem heavy for work lately.",
                "source": "life-pattern",
                "created_at": "2026-04-13T20:00:00",
            }
        ]

        snapshot = self.bot.living_dad_snapshot(limit=3)

        self.assertEqual(snapshot["counts"]["persona_shifts"], 1)
        self.assertEqual(snapshot["counts"]["wisdom"], 1)
        self.assertEqual(snapshot["counts"]["patterns"], 1)
        self.assertEqual(snapshot["counts"]["proactive_queue"], 1)
        self.assertEqual(snapshot["persona_shifts"][0]["trait"], "more coach-like")
        self.assertEqual(snapshot["wisdom"][0]["topic"], "work")
        self.assertEqual(snapshot["patterns"][0]["day_hint"], "Sunday")
        self.assertEqual(snapshot["proactive_queue"][0]["source"], "life-pattern")

    def test_evolve_persona_stores_trait_and_queues_announcement(self):
        responses = iter([
            {"message": {"content": json.dumps({"new_trait": "more coach-like", "reason": "Tony responds well to structured encouragement"})}},
            {"message": {"content": json.dumps({"score": 8, "approved": True, "feedback": "Strong and grounded.", "suggested_refinement": None})}},
        ])
        self.bot.call_ollama_chat = lambda *args, **kwargs: next(responses)

        entry = self.bot.evolve_persona(force=True)

        self.assertIsNotNone(entry)
        self.assertEqual(self.bot.persona_evolution_history()[-1]["trait"], "more coach-like")
        self.assertEqual(entry["critique_score"], 8)
        self.assertTrue(self.bot.pending_proactive_messages())

    def test_cadence_settings_can_delay_persona_evolution(self):
        self.bot.CADENCE = {"persona_evolution_min_sessions": 12, "persona_evolution_session_gap": 12}
        self.bot.MEMORY_STORE["session_archive"] = [
            {
                "summary": f"Session {index}",
                "topics": ["work"],
                "dominant_mood": "neutral",
                "turn_count": 2,
                "created_at": f"2026-04-{index + 1:02d}T20:00:00",
                "id": str(index),
            }
            for index in range(10)
        ]

        self.assertFalse(self.bot.should_evolve_persona())

    def test_should_evolve_persona_when_cadence_gap_is_met(self):
        self.bot.CADENCE = {"persona_evolution_min_sessions": 10, "persona_evolution_session_gap": 3}
        self.bot.MEMORY_STORE["session_archive"] = [
            {
                "summary": f"Session {index}",
                "topics": ["work"],
                "dominant_mood": "neutral",
                "turn_count": 2,
                "created_at": f"2026-04-{index + 1:02d}T20:00:00",
                "id": str(index),
            }
            for index in range(12)
        ]
        self.bot.MEMORY_STORE["persona_evolution"] = [
            {
                "trait": "more reflective",
                "reason": "Tony has been opening up more.",
                "announcement": "",
                "session_count": 9,
                "applied_at": "2026-04-09T20:00:00",
            }
        ]

        self.assertTrue(self.bot.should_evolve_persona())

    def test_generate_wisdom_insight_stores_special_wisdom_entry(self):
        today_stamp = date.today().isoformat()
        self.bot.MEMORY_STORE["consolidated_memories"] = [
            {
                "summary": "Tony has recurring stress around work deadlines.",
                "category": "work",
                "source_count": 2,
                "confidence": 0.74,
                "supporting_summaries": [],
                "contradictions": [],
                "updated_at": today_stamp,
            }
        ]
        self.bot.MEMORY_STORE["memory_graph"] = {
            "nodes": [{"id": "category:work", "label": "work", "type": "category", "weight": 3}],
            "edges": [{"source": "work", "target": "stressed", "weight": 3}],
            "updated_at": today_stamp,
        }
        self.bot.call_ollama_chat = lambda *args, **kwargs: {
            "message": {"content": json.dumps({"summary": "When work starts shrinking your breathing room, slow the moment down before it runs you.", "topic": "work"})}
        }

        entry = self.bot.generate_wisdom_insight("Work is getting to me.", force=True)

        self.assertIsNotNone(entry)
        self.assertEqual(entry["topic"], "work")
        self.assertEqual(len(self.bot.wisdom_catalog()), 1)

    def test_cadence_settings_can_delay_wisdom_generation(self):
        self.bot.CADENCE = {"wisdom_min_archived_sessions": 2, "wisdom_turn_interval": 5}
        self.bot.MEMORY_STORE["session_archive"] = [
            {"summary": "A", "topics": ["work"], "dominant_mood": "neutral", "turn_count": 2, "created_at": "2026-04-01T20:00:00", "id": "a"},
            {"summary": "B", "topics": ["work"], "dominant_mood": "neutral", "turn_count": 2, "created_at": "2026-04-02T20:00:00", "id": "b"},
        ]
        self.bot.MEMORY_STORE["memory_graph"] = {
            "nodes": [{"id": "category:work", "label": "work", "type": "category", "weight": 3}],
            "edges": [{"source": "work", "target": "stressed", "weight": 3}],
            "updated_at": date.today().isoformat(),
        }
        self.bot.session_turn_count = lambda: 3
        self.bot.relationship.top_topics = lambda *_args, **_kwargs: []

        self.assertFalse(self.bot.should_generate_wisdom_insight("Just checking in."))

    def test_detect_life_patterns_queues_proactive_message(self):
        self.bot.MEMORY_STORE["session_archive"] = [
            {"summary": "Work felt heavy again.", "topics": ["work"], "dominant_mood": "stressed", "turn_count": 4, "created_at": "2026-04-05T20:00:00", "id": "a"},
            {"summary": "Sunday work dread again.", "topics": ["work"], "dominant_mood": "stressed", "turn_count": 4, "created_at": "2026-04-12T20:00:00", "id": "b"},
            {"summary": "Another tough Sunday night about work.", "topics": ["work"], "dominant_mood": "stressed", "turn_count": 4, "created_at": "2026-04-19T20:00:00", "id": "c"},
            {"summary": "Still talking about work stress.", "topics": ["work"], "dominant_mood": "stressed", "turn_count": 4, "created_at": "2026-04-26T20:00:00", "id": "d"},
        ]

        patterns = self.bot.detect_life_patterns(force=True)

        self.assertTrue(patterns)
        self.assertIn("Sundays", patterns[0]["summary"])
        self.assertTrue(self.bot.pending_proactive_messages())

    def test_cadence_settings_can_raise_pattern_threshold(self):
        self.bot.CADENCE = {
            "life_pattern_min_archived_sessions": 4,
            "life_pattern_window": 12,
            "life_pattern_min_occurrences": 5,
            "life_pattern_confidence_threshold": 80,
            "life_pattern_queue_limit": 2,
        }
        self.bot.MEMORY_STORE["session_archive"] = [
            {"summary": "Work felt heavy again.", "topics": ["work"], "dominant_mood": "stressed", "turn_count": 4, "created_at": "2026-04-05T20:00:00", "id": "a"},
            {"summary": "Sunday work dread again.", "topics": ["work"], "dominant_mood": "stressed", "turn_count": 4, "created_at": "2026-04-12T20:00:00", "id": "b"},
            {"summary": "Another tough Sunday night about work.", "topics": ["work"], "dominant_mood": "stressed", "turn_count": 4, "created_at": "2026-04-19T20:00:00", "id": "c"},
            {"summary": "Still talking about work stress.", "topics": ["work"], "dominant_mood": "stressed", "turn_count": 4, "created_at": "2026-04-26T20:00:00", "id": "d"},
        ]

        self.assertEqual(self.bot.detect_life_patterns(force=True), [])

    def test_opening_message_consumes_queued_proactive_message(self):
        self.bot.MEMORY_STORE["last_mood_updated_at"] = (date.today() - timedelta(days=1)).isoformat()
        self.bot.queue_proactive_message("I've noticed Sundays weigh on you lately.", source="life-pattern")

        opening = self.bot.opening_message("Fallback")

        self.assertIn("Sundays weigh on you", opening)
        self.assertIn("How's your day going so far?", opening)
        self.assertFalse(self.bot.pending_proactive_messages())

    def test_opening_message_uses_profile_opening_messages_when_idle(self):
        self.bot.PROFILE["opening_messages"] = ["Hey buddy, good to see you."]
        self.bot.refresh_profile_runtime()
        self.bot.MEMORY_STORE["last_mood_updated_at"] = date.today().isoformat()

        opening = self.bot.opening_message("Fallback")

        self.assertEqual(opening, "Hey buddy, good to see you.")

    def test_refresh_memory_graph_debounces_recent_refreshes(self):
        sync_calls = []
        self.bot.GRAPH_REFRESH_DEBOUNCE_SECONDS = 30
        self.bot._memory_graph_dirty = True
        self.bot._last_memory_graph_refresh_monotonic = time.monotonic()
        self.bot.sync_graph_store = lambda: sync_calls.append("sync") or {"nodes": [], "edges": [], "updated_at": "2026-04-19T12:00:00"}
        self.bot.memory_manager.preview_memory_graph = lambda snapshot: {"nodes": [], "edges": [], "updated_at": snapshot["updated_at"]}

        self.bot.refresh_memory_graph()
        self.assertEqual(sync_calls, [])
        self.assertTrue(self.bot._memory_graph_dirty)

        self.bot._last_memory_graph_refresh_monotonic = time.monotonic() - 31
        self.bot.refresh_memory_graph()

        self.assertEqual(sync_calls, ["sync"])
        self.assertFalse(self.bot._memory_graph_dirty)

    def test_refresh_memory_graph_records_issue_and_preserves_previous_snapshot_on_failure(self):
        self.bot.GRAPH_REFRESH_DEBOUNCE_SECONDS = 0
        self.bot._memory_graph_dirty = True
        self.bot.MEMORY_STORE["memory_graph"] = {"nodes": [{"label": "existing"}], "edges": [], "updated_at": "before"}
        self.bot.sync_graph_store = lambda: (_ for _ in ()).throw(RuntimeError("graph backend unavailable"))

        graph = self.bot.refresh_memory_graph()

        self.assertEqual(graph["updated_at"], "before")
        self.assertTrue(self.bot._memory_graph_dirty)
        self.assertEqual(self.bot._recent_runtime_issues[-1]["purpose"], "memory graph refresh")

    def test_compose_user_turn_text_includes_voice_note_transcript(self):
        turn_text = self.bot.compose_user_turn_text(
            "",
            attachments=[{"type": "audio", "transcript": "I had a rough day at work.", "name": "note.wav"}],
        )

        self.assertIn("Voice note transcript:", turn_text)
        self.assertIn("rough day at work", turn_text)

    def test_build_chat_request_messages_attaches_images_for_multimodal_models(self):
        self.bot.ACTIVE_MODEL = "llava:7b"

        messages = self.bot.build_chat_request_messages(
            "Take a look at this.",
            "neutral",
            attachments=[{"type": "image", "image_b64": "ZmFrZQ==", "name": "photo.png"}],
        )

        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["images"], ["ZmFrZQ=="])

    def test_validate_managers_raises_when_required_manager_method_is_missing(self):
        original_runtime_client = self.bot.runtime_client
        self.bot.runtime_client = object()
        self.addCleanup(setattr, self.bot, "runtime_client", original_runtime_client)

        with self.assertRaises(RuntimeError):
            self.bot._validate_managers()

    def test_validate_managers_smoke_mode_raises_when_probe_fails(self):
        with patch.object(self.bot.status_reporting, "status_snapshot", side_effect=RuntimeError("smoke boom")):
            with self.assertRaises(RuntimeError):
                self.bot._validate_managers(smoke=True)

    def test_process_user_message_applies_top_level_prompt_guard_before_runtime_client(self):
        self.bot.effective_context_token_budget = lambda _model_name=None: 320
        self.bot.RESERVED_RESPONSE_TOKENS = 120
        self.bot.mood_manager.detect = lambda *_args, **_kwargs: "neutral"
        self.bot.critique_reply = lambda *_args, **_kwargs: "I am right here with you, buddy."
        self.bot.schedule_post_turn_maintenance = lambda *_args, **_kwargs: {}

        self.bot.history = [{"role": "system", "content": "Dad system prompt"}]
        for index in range(12):
            self.bot.history.append(
                {
                    "role": "user",
                    "content": f"Long check-in {index}: " + ("work stress and money pressure " * 16),
                }
            )
            self.bot.history.append(
                {
                    "role": "assistant",
                    "content": "I hear you, buddy. " + ("Let's take one step at a time. " * 12),
                }
            )

        captured = {}

        def fake_runtime_chat(model_name, messages, options=None, response_format=None, purpose="chat"):
            if purpose == "chat response":
                captured["messages"] = messages
                captured["purpose"] = purpose
            return {"message": {"content": "I am right here with you, buddy."}}

        with patch.object(self.bot.runtime_client, "call_ollama_chat_with_model", side_effect=fake_runtime_chat):
            reply, should_end = self.bot.process_user_message("I'm still stressed about work and bills.")

        self.assertFalse(should_end)
        self.assertIn("buddy", reply.lower())
        self.assertIn("messages", captured)
        self.assertEqual(captured["purpose"], "chat response")
        sent_tokens = sum(self.bot.message_token_cost(message) for message in captured["messages"])
        self.assertLessEqual(sent_tokens, 200)

    def test_guard_chat_request_messages_replaces_oversized_system_prompt_with_minimal_fallback(self):
        self.bot.effective_context_token_budget = lambda _model_name=None: 320
        self.bot.RESERVED_RESPONSE_TOKENS = 120
        huge_system = "Dad profile context " * 600
        messages = [
            {"role": "system", "content": huge_system},
            {"role": "user", "content": "I need help with work stress and money pressure."},
        ]

        guarded = self.bot.guard_chat_request_messages(messages, purpose="unit guard")

        prompt_budget = max(128, max(256, int(self.bot.effective_context_token_budget() or 0)) - max(64, int(self.bot.RESERVED_RESPONSE_TOKENS or 0)))
        guarded_tokens = sum(self.bot.message_token_cost(message) for message in guarded)
        self.assertLessEqual(guarded_tokens, prompt_budget)
        self.assertEqual(guarded[0]["role"], "system")
        self.assertIn("warm, grounded dad", guarded[0]["content"])

    def test_process_user_message_combines_memory_pruning_and_prompt_guard_under_pressure(self):
        self.bot.effective_context_token_budget = lambda _model_name=None: 360
        self.bot.RESERVED_RESPONSE_TOKENS = 160
        self.bot.mood_manager.detect = lambda *_args, **_kwargs: "stressed"
        self.bot.critique_reply = lambda *_args, **_kwargs: "I hear you, buddy. Let us take one calm step."
        self.bot.schedule_post_turn_maintenance = lambda *_args, **_kwargs: {}

        today = date.today().isoformat()
        self.bot.MEMORY_STORE["memories"] = [
            {
                "summary": f"Tony shared long work and budget pressure thread {index}: " + ("work money stress " * 35),
                "category": "work",
                "mood": "stressed",
                "created_at": today,
                "updated_at": today,
            }
            for index in range(18)
        ]

        baseline_messages = self.bot.build_chat_request_messages(
            "Work and bills have both been heavy this week.",
            "stressed",
        )
        prompt_budget = max(128, max(256, int(self.bot.effective_context_token_budget() or 0)) - max(64, int(self.bot.RESERVED_RESPONSE_TOKENS or 0)))
        baseline_tokens = sum(self.bot.message_token_cost(message) for message in baseline_messages)

        captured = {}

        def fake_runtime_chat(model_name, messages, options=None, response_format=None, purpose="chat"):
            if purpose == "chat response":
                captured["messages"] = messages
            return {"message": {"content": "I hear you, buddy. Let us take one calm step."}}

        with patch.object(self.bot.runtime_client, "call_ollama_chat_with_model", side_effect=fake_runtime_chat):
            reply, should_end = self.bot.process_user_message("Work and bills have both been heavy this week.")

        self.assertFalse(should_end)
        self.assertIn("buddy", reply.lower())
        self.assertGreater(baseline_tokens, prompt_budget)
        self.assertIn("messages", captured)
        sent_tokens = sum(self.bot.message_token_cost(message) for message in captured["messages"])
        self.assertLessEqual(sent_tokens, prompt_budget)

    def test_reflection_retry_loop_stays_within_prompt_guard_budget(self):
        self.bot.effective_context_token_budget = lambda _model_name=None: 320
        self.bot.RESERVED_RESPONSE_TOKENS = 120
        long_observation = "Initial web observation: " + ("market stress context " * 140)
        captured_token_counts = []

        responses = iter(
            [
                {"message": {"content": '{"sufficient": false, "refined_query": "better work stress coping query", "reason": "Need a more specific source."}'}},
                {"message": {"content": '{"sufficient": true, "refined_query": null, "reason": "The refined source is good enough."}'}},
            ]
        )

        def fake_runtime_chat(model_name, messages, options=None, response_format=None, purpose="chat"):
            if purpose == "agentic reflection":
                captured_token_counts.append(sum(self.bot.message_token_cost(message) for message in messages))
            return next(responses)

        self.bot.lookup_web = lambda query: {
            "heading": "Coping guide",
            "summary": f"Refined result for {query}",
            "source_label": "Example Source",
        }

        with patch.object(self.bot.runtime_client, "call_ollama_chat_with_model", side_effect=fake_runtime_chat):
            refined_observation = self.bot.turn_service._reflect_on_web_observation(
                "Can you look this up?",
                "work stress help",
                long_observation,
                settings={"web_lookup": True},
                max_retries=2,
            )

        prompt_budget = max(128, max(256, int(self.bot.effective_context_token_budget() or 0)) - max(64, int(self.bot.RESERVED_RESPONSE_TOKENS or 0)))
        self.assertGreaterEqual(len(captured_token_counts), 2)
        self.assertTrue(all(count <= prompt_budget for count in captured_token_counts))
        self.assertIn("Coping guide", refined_observation)

    def test_reflection_retry_loop_logs_retry_attempts(self):
        responses = iter(
            [
                {"message": {"content": '{"sufficient": false, "refined_query": "better work stress coping query", "reason": "Need a more specific source."}'}},
                {"message": {"content": '{"sufficient": true, "refined_query": null, "reason": "Looks good."}'}},
            ]
        )

        def fake_runtime_chat(model_name, messages, options=None, response_format=None, purpose="chat"):
            return next(responses)

        self.bot.lookup_web = lambda query: {
            "heading": "Coping guide",
            "summary": f"Refined result for {query}",
            "source_label": "Example Source",
        }

        with patch.object(self.bot.runtime_client, "call_ollama_chat_with_model", side_effect=fake_runtime_chat):
            with self.assertLogs("dadbot.managers.turn_processing", level="INFO") as captured_logs:
                self.bot.turn_service._reflect_on_web_observation(
                    "Can you look this up?",
                    "work stress help",
                    "Initial web observation",
                    settings={"web_lookup": True},
                    max_retries=2,
                )

        joined = "\n".join(captured_logs.output)
        self.assertIn("retry 1/2", joined)
        self.assertIn("completed after 2/2 attempts with 1 retries", joined)

    def test_reflect_relationship_state_updates_store_with_mocked_reflection_json(self):
        self.bot.history = [
            {"role": "system", "content": "Dad system prompt"},
            {"role": "user", "content": "Work has been heavy this week."},
            {"role": "assistant", "content": "I am with you through it, buddy."},
            {"role": "user", "content": "I feel calmer after talking."},
            {"role": "assistant", "content": "That is a strong step forward."},
            {"role": "user", "content": "I think I can keep going tomorrow."},
        ]

        snapshot = self.bot.relationship.current_state()

        self.assertIsInstance(snapshot, dict)
        self.assertIn("active_hypothesis", snapshot)

    def test_validate_reply_returns_unfinalized_fact_fallback(self):
        self.bot.get_memory_reply = lambda *_args, **_kwargs: None
        self.bot.get_fact_reply = lambda *_args, **_kwargs: "Fact fallback"

        reply = self.bot.validate_reply("Where was I born?", "Draft reply")

        self.assertEqual(reply, "Fact fallback")

    def test_critique_reply_uses_alignment_judge_revision(self):
        calls = []

        def fake_call(*args, **kwargs):
            calls.append(kwargs.get("purpose"))
            return {
                "message": {
                    "content": (
                        '{"approved": false, "score": 6, "dad_likeness": 5, '
                        '"groundedness": 9, "emotional_fit": 6, '
                        '"issues": ["too generic"], '
                        '"revised_reply": "I know this has been weighing on you, buddy. Take one breath and we will handle it together."}'
                    )
                }
            }

        self.bot.call_ollama_chat = fake_call

        revised = self.bot.critique_reply("I am overwhelmed at work.", "You can do it.", "stressed")

        self.assertEqual(calls, ["reply supervisor"])
        self.assertEqual(revised, "I know this has been weighing on you, buddy. Take one breath and we will handle it together.")

    def test_judge_reply_alignment_uses_unified_supervisor_call(self):
        calls = []

        def fake_call(*args, **kwargs):
            calls.append(kwargs.get("purpose"))
            return {
                "message": {
                    "content": (
                        '{"approved": false, "score": 7, "dad_likeness": 7, '
                        '"groundedness": 9, "emotional_fit": 8, '
                        '"issues": ["tighten warmth"], '
                        '"revised_reply": "I am with you, buddy. Let us take this one steady step at a time."}'
                    )
                }
            }

        self.bot.call_ollama_chat = fake_call

        revised = self.bot.judge_reply_alignment("Work still feels heavy.", "I am here for you.", "stressed")

        self.assertEqual(calls, ["reply supervisor"])
        self.assertEqual(revised, "I am with you, buddy. Let us take this one steady step at a time.")

    def test_reply_supervisor_prompt_includes_anchor_examples_and_hidden_reasoning_instruction(self):
        prompt = self.bot.build_reply_supervisor_prompt(
            "Work still feels heavy.",
            "I am here for you, buddy.",
            "stressed",
        )

        self.assertIn("Scoring guide:", prompt)
        self.assertIn("Good reply:", prompt)
        self.assertIn("Bad reply:", prompt)
        self.assertIn("Think silently through groundedness, dad_likeness, and emotional_fit", prompt)

    def test_reply_supervisor_snapshot_records_duration_ms(self):
        self.bot.call_ollama_chat = lambda *args, **kwargs: {
            "message": {
                "content": (
                    '{"approved": true, "score": 8, "dad_likeness": 8, '
                    '"groundedness": 9, "emotional_fit": 8, '
                    '"issues": [], "revised_reply": null}'
                )
            }
        }

        self.bot.critique_reply("Work still feels heavy.", "I am here for you, buddy.", "stressed")
        snapshot = self.bot.reply_supervisor_snapshot()

        self.assertIn("duration_ms", snapshot["last_decision"])
        self.assertGreaterEqual(snapshot["last_decision"]["duration_ms"], 0)

    def test_apply_reply_supervisor_decision_clamps_scores_and_trims_issues(self):
        revised = self.bot.apply_reply_supervisor_decision(
            {
                "approved": False,
                "score": 99,
                "dad_likeness": "9",
                "groundedness": 0,
                "emotional_fit": "7",
                "issues": ["  too generic  ", ""],
                "revised_reply": "  I am with you, buddy. Let us take it one steady step at a time.  ",
            },
            "Draft reply",
            stage="alignment_judge",
        )

        snapshot = self.bot.reply_supervisor_snapshot()

        self.assertEqual(revised, "I am with you, buddy. Let us take it one steady step at a time.")
        self.assertEqual(snapshot["last_decision"]["stage"], "alignment_judge")
        self.assertEqual(snapshot["last_decision"]["score"], 10)
        self.assertEqual(snapshot["last_decision"]["groundedness"], 1)
        self.assertEqual(snapshot["last_decision"]["issues"], ["too generic"])
        self.assertTrue(snapshot["last_decision"]["revised"])

    def test_memory_extraction_prompt_requires_json_array(self):
        prompt = self.bot.memory_extraction_prompt()

        self.assertIn("Return ONLY a JSON array of objects.", prompt)
        self.assertIn('"summary"', prompt)

    def test_build_cross_session_context_includes_consolidated_memories(self):
        self.bot.MEMORY_STORE["consolidated_memories"] = [
            {
                "summary": "Tony has been working steadily on saving money goals.",
                "category": "finance",
                "source_count": 2,
                "updated_at": date.today().isoformat(),
            }
        ]

        context = self.bot.build_cross_session_context()

        self.assertIsNotNone(context)
        self.assertIn("Consolidated long-term insights", context)
        self.assertIn("saving money goals", context)

    def test_select_active_consolidated_memories_uses_model_selected_indices(self):
        today = date.today().isoformat()
        self.bot.MEMORY_STORE["consolidated_memories"] = [
            {
                "summary": "Tony has been carrying steady work stress lately.",
                "category": "work",
                "confidence": 0.82,
                "source_count": 2,
                "updated_at": today,
            },
            {
                "summary": "Tony has been saving money for an emergency fund.",
                "category": "finance",
                "confidence": 0.91,
                "source_count": 3,
                "updated_at": today,
            },
            {
                "summary": "Tony keeps trying to sleep earlier during stressful weeks.",
                "category": "health",
                "confidence": 0.67,
                "source_count": 2,
                "updated_at": today,
            },
        ]
        self.bot.call_ollama_chat = lambda *args, **kwargs: {"message": {"content": "[2, 1]"}}

        selected = self.bot.select_active_consolidated_memories("I am stressed about money right now.", max_items=2)

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]["summary"], "Tony has been saving money for an emergency fund.")
        self.assertEqual(selected[1]["summary"], "Tony has been carrying steady work stress lately.")

    def test_select_active_consolidated_memories_reuses_cached_selection(self):
        today = date.today().isoformat()
        self.bot.MEMORY_STORE["consolidated_memories"] = [
            {
                "summary": "Tony has been carrying steady work stress lately.",
                "category": "work",
                "confidence": 0.82,
                "source_count": 2,
                "updated_at": today,
            },
            {
                "summary": "Tony has been saving money for an emergency fund.",
                "category": "finance",
                "confidence": 0.91,
                "source_count": 3,
                "updated_at": today,
            },
        ]
        calls = []
        self.bot.call_ollama_chat = lambda *args, **kwargs: calls.append(kwargs.get("purpose")) or {"message": {"content": "[2]"}}

        first = self.bot.select_active_consolidated_memories("I am stressed about money right now.", max_items=1)
        second = self.bot.select_active_consolidated_memories("I am stressed about money right now.", max_items=1)

        self.assertEqual([entry["summary"] for entry in first], [entry["summary"] for entry in second])
        self.assertEqual(calls, ["active memory selection"])

    def test_build_cross_session_context_prioritizes_positive_traits_and_recent_notes(self):
        today = date.today().isoformat()
        self.bot.MEMORY_STORE["persona_evolution"] = [
            {
                "trait": "more intense when you stall",
                "reason": "",
                "announcement": "",
                "session_count": 12,
                "applied_at": today + "T08:00:00",
                "last_reinforced_at": today + "T08:00:00",
                "strength": 2.6,
                "impact_score": -1.5,
            },
            {
                "trait": "gentler when you are hard on yourself",
                "reason": "",
                "announcement": "",
                "session_count": 12,
                "applied_at": today + "T09:00:00",
                "last_reinforced_at": today + "T09:00:00",
                "strength": 1.2,
                "impact_score": 2.4,
            },
        ]
        self.bot.MEMORY_STORE["session_archive"] = [
            {
                "summary": "Work was heavy but Tony kept showing up.",
                "topics": ["work"],
                "dominant_mood": "stressed",
                "turn_count": 4,
                "created_at": today + "T10:00:00",
                "id": "a",
            }
        ]
        self.bot.MEMORY_STORE["consolidated_memories"] = [
            {
                "summary": "Tony has been carrying steady work stress lately.",
                "category": "work",
                "source_count": 2,
                "updated_at": today,
            }
        ]

        context = self.bot.build_cross_session_context()

        self.assertIsNotNone(context)
        self.assertLess(context.index("gentler when you are hard on yourself"), context.index("more intense when you stall"))
        self.assertLess(context.index("Recent prior session notes"), context.index("Consolidated long-term insights"))

    def test_consolidate_memories_stores_merged_insights(self):
        today_stamp = date.today().isoformat()
        self.bot.MEMORY_STORE["memories"] = [
            {
                "summary": "Tony is saving money for a car.",
                "category": "finance",
                "mood": "positive",
                "created_at": today_stamp,
                "updated_at": today_stamp,
            },
            {
                "summary": "Tony has been stressed about work deadlines.",
                "category": "work",
                "mood": "stressed",
                "created_at": today_stamp,
                "updated_at": today_stamp,
            },
            {
                "summary": "Tony is trying to exercise more consistently.",
                "category": "health",
                "mood": "positive",
                "created_at": today_stamp,
                "updated_at": today_stamp,
            },
        ]
        self.bot.call_ollama_chat = lambda *args, **kwargs: {
            "message": {
                "content": json.dumps([
                    {
                        "summary": "Tony is building stronger saving habits.",
                        "category": "finance",
                        "source_count": 2,
                        "confidence": 0.84,
                        "supporting_summaries": ["Tony is saving money for a car."],
                        "contradictions": [],
                    },
                    {
                        "summary": "Tony has recurring stress around work deadlines.",
                        "category": "work",
                        "source_count": 2,
                        "confidence": 0.74,
                        "supporting_summaries": ["Tony has been stressed about work deadlines."],
                        "contradictions": [],
                    },
                ])
            }
        }

        with self._save_commit_context() as turn_context:
            consolidated = self.bot.consolidate_memories(force=True, turn_context=turn_context)

        self.assertEqual(len(consolidated), 2)
        self.assertEqual(self.bot.MEMORY_STORE["last_consolidated_at"], today_stamp)
        self.assertTrue(any("saving habits" in entry["summary"].lower() for entry in consolidated))
        self.assertTrue(any(entry["confidence"] >= 0.74 for entry in consolidated))

    def test_detect_memory_contradictions_flags_negated_overlap(self):
        contradictions = self.bot.detect_memory_contradictions(
            memories=[
                {"summary": "Tony is saving money for a car.", "category": "finance"},
                {"summary": "Tony is not saving money for a car right now.", "category": "finance"},
            ]
        )

        self.assertEqual(len(contradictions), 1)
        self.assertIn("opposite polarity", contradictions[0]["reason"])

    def test_detect_memory_contradictions_flags_opposing_states(self):
        contradictions = self.bot.detect_memory_contradictions(
            memories=[
                {"summary": "Tony is single right now.", "category": "relationships"},
                {"summary": "Tony is married now.", "category": "relationships"},
            ]
        )

        self.assertEqual(len(contradictions), 1)
        self.assertIn("opposing states", contradictions[0]["reason"])

    def test_merge_consolidated_memories_marks_superseded_conflicts(self):
        self.bot.MEMORY_STORE["consolidated_memories"] = [
            {
                "summary": "Tony is single right now.",
                "category": "relationships",
                "source_count": 2,
                "confidence": 0.82,
                "updated_at": "2024-01-01",
            }
        ]
        with self._save_commit_context() as turn_context:
            consolidated = self.bot.merge_consolidated_memories(
                [
                    {
                        "summary": "Tony is married now.",
                        "category": "relationships",
                        "source_count": 3,
                        "confidence": 0.86,
                        "supporting_summaries": ["Tony is married now."],
                        "contradictions": [],
                    }
                ],
                turn_context=turn_context,
            )
        active = [entry for entry in consolidated if not entry.get("superseded")]
        superseded = [entry for entry in consolidated if entry.get("superseded")]

        self.assertGreaterEqual(len(active), 1)
        self.assertGreaterEqual(len(superseded), 1)
        self.assertTrue(any(entry.get("superseded_reason") for entry in superseded))
        self.assertTrue(all("importance_score" in entry for entry in consolidated))
        self.assertTrue(all("version" in entry for entry in consolidated))

    def test_retrieve_context_returns_hybrid_bundle(self):
        today_stamp = date.today().isoformat()
        self.bot.MEMORY_STORE["memories"] = [
            {
                "summary": "Tony is preparing for a hard deadline at work.",
                "category": "work",
                "mood": "stressed",
                "created_at": today_stamp,
                "updated_at": today_stamp,
                "impact_score": 2.0,
            }
        ]
        self.bot.MEMORY_STORE["consolidated_memories"] = [
            {
                "summary": "Tony has recurring stress around work deadlines.",
                "category": "work",
                "source_count": 2,
                "confidence": 0.8,
                "importance_score": 0.7,
                "updated_at": today_stamp,
            }
        ]

        payload = self.bot.retrieve_context("Can you help me with this work deadline?", strategy="hybrid", limit=3)

        self.assertEqual(payload["strategy"], "hybrid")
        self.assertIn("bundle", payload)
        self.assertIn("semantic_memories", payload)
        self.assertIn("consolidated_memories", payload)
        self.assertGreaterEqual(len(payload["semantic_memories"]), 1)

    def test_extract_mood_label_maps_alias_to_known_category(self):
        self.assertEqual(self.bot.mood_manager.extract_label("Mood: overwhelmed\nReason: Too much going on."), "stressed")

    def test_detect_crisis_signal_flags_high_risk_language(self):
        self.assertTrue(self.bot.detect_crisis_signal("I want to kill myself tonight."))

    def test_detect_crisis_signal_ignores_negated_reassurance(self):
        self.assertFalse(self.bot.detect_crisis_signal("I'm not suicidal, just exhausted and overwhelmed."))

    def test_process_user_message_short_circuits_to_crisis_support_reply(self):
        self.bot.mood_manager.detect = lambda *_args, **_kwargs: "sad"
        self.bot.call_ollama_chat = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called for crisis routing"))

        dad_reply, should_end = self.bot.process_user_message("I want to kill myself tonight.")

        self.assertFalse(should_end)
        self.assertIn("988", dad_reply)
        self.assertIn("real person", dad_reply.lower())

    def test_handle_tool_command_status_snapshot_reports_core_runtime(self):
        self.bot.MEMORY_STORE["memories"] = [
            {"summary": "Tony is saving money.", "category": "finance", "mood": "positive", "created_at": date.today().isoformat(), "updated_at": date.today().isoformat()}
        ]
        self.bot.MEMORY_STORE["pending_proactive_messages"] = [
            {"message": "I've noticed Sundays weigh on you lately.", "source": "life-pattern", "created_at": "2026-04-13T20:00:00"}
        ]

        reply = self.bot.handle_tool_command("/status")

        self.assertIn("model=", reply)
        self.assertIn("saved memories=1", reply)
        self.assertIn("pending proactive=1", reply)

    def test_dashboard_status_snapshot_reports_operational_details(self):
        self.bot.MEMORY_STORE["memories"] = [
            {"summary": "Tony is saving money.", "category": "finance", "mood": "positive", "created_at": date.today().isoformat(), "updated_at": date.today().isoformat()}
        ]
        self.bot.begin_planner_debug("Can you help me with work stress?", "stressed")
        self.bot.update_planner_debug(planner_status="used", planner_tool="web_lookup", final_path="tool")
        self.bot._last_reply_supervisor = {
            "stage": "alignment_judge",
            "approved": True,
            "score": 8,
            "dad_likeness": 8,
            "groundedness": 9,
            "emotional_fit": 8,
            "issues": ["slightly generic"],
            "revised": False,
            "source": "llm",
        }

        with patch(
            "Dad.DadServiceClient.health",
            return_value={
                "status": "ok",
                "workers": 2,
                "queue_backend": "local",
                "state_backend": "memory",
                "service_name": "dadbot",
            },
        ), patch("Dad.DadServiceClient._port_is_open", return_value=True):
            snapshot = self.bot.dashboard_status_snapshot()

        self.assertEqual(snapshot["service"]["status"], "ok")
        self.assertTrue(snapshot["service"]["reachable"])
        self.assertEqual(snapshot["planner_debug"]["planner_tool"], "web_lookup")
        self.assertEqual(snapshot["status"]["saved_memories"], 1)
        self.assertGreaterEqual(snapshot["threads"]["total"], 1)
        self.assertIn("background_tasks", snapshot)
        self.assertIn("semantic_memory", snapshot)
        self.assertIn("maintenance", snapshot)
        self.assertIn("supervisor", snapshot)
        self.assertIn("health", snapshot)
        self.assertIn("hypotheses", snapshot["relationship"])

    def test_runtime_health_snapshot_warns_for_high_pressure_signals(self):
        self.bot.record_memory_context_stats(
            tokens=3050,
            budget_tokens=3200,
            selected_sections=4,
            total_sections=6,
            pruned=True,
            user_input="long stress check-in",
        )
        self.bot._prompt_guard_stats["trim_count"] = 9
        self.bot.record_runtime_issue("prompt guard", "trimmed prompt context", RuntimeError("context window exceeded"))

        snapshot = self.bot.runtime_health_snapshot(log_warnings=False)

        self.assertEqual(snapshot["level"], "red")
        self.assertGreaterEqual(snapshot["memory_context_ratio"], 0.9)
        self.assertGreaterEqual(snapshot["prompt_guard_trim_count"], 9)
        self.assertGreaterEqual(snapshot["recent_runtime_issue_count"], 1)
        self.assertIn("health_score", snapshot)
        self.assertLess(snapshot["reasoning_confidence"], 0.6)
        self.assertTrue(snapshot["clarification_recommended"])
        self.assertTrue(snapshot["clarification_message"])
        self.assertIn("projected_minutes_to_red", snapshot)

    def test_runtime_health_snapshot_red_reduces_worker_limit_in_light_mode(self):
        self.bot.LIGHT_MODE = True
        self.bot.record_memory_context_stats(
            tokens=960,
            budget_tokens=1000,
            selected_sections=3,
            total_sections=5,
            pruned=True,
            user_input="high pressure check",
        )
        self.bot._prompt_guard_stats["trim_count"] = 24

        snapshot = self.bot.runtime_health_snapshot(log_warnings=False, persist=False)

        self.assertEqual(snapshot["level"], "red")
        self.assertEqual(self.bot.adaptive_background_worker_limit(snapshot), 1)
        self.assertTrue(self.bot.should_delay_noncritical_maintenance(snapshot))

    def test_current_runtime_health_snapshot_uses_cached_value_until_stale(self):
        call_counter = {"count": 0}
        original = self.bot.runtime_health_snapshot

        def _wrapped(*args, **kwargs):
            call_counter["count"] += 1
            return original(*args, **kwargs)

        with patch.object(self.bot, "runtime_health_snapshot", side_effect=_wrapped):
            first = self.bot.current_runtime_health_snapshot(force=True, log_warnings=False, persist=False)
            second = self.bot.current_runtime_health_snapshot(force=False, log_warnings=False, persist=False, max_age_seconds=999)

        self.assertEqual(call_counter["count"], 1)
        self.assertEqual(first["updated_at"], second["updated_at"])

    def test_process_user_message_refreshes_health_snapshot_at_turn_end(self):
        self.bot.mood_manager.detect = lambda *_args, **_kwargs: "neutral"

        with patch.object(self.bot, "current_runtime_health_snapshot", wraps=self.bot.current_runtime_health_snapshot) as health_mock:
            dad_reply, should_end = self.bot.process_user_message("remember I need to call the dentist tomorrow")

        self.assertFalse(should_end)
        self.assertIsInstance(dad_reply, str)
        self.assertGreaterEqual(health_mock.call_count, 1)

    def test_status_snapshot_includes_health_payload(self):
        snapshot = self.bot.status_snapshot()

        self.assertIn("health", snapshot)
        self.assertIn("level", snapshot["health"])
        self.assertIn("health_score", snapshot["health"])

    def test_apply_relationship_feedback_records_history(self):
        before = len(self.bot.relationship_history(limit=200))
        updated = self.bot.relationship.current_state()

        self.assertIn("trust_level", updated)
        self.assertIn("openness_level", updated)
        self.assertEqual(len(self.bot.relationship_history(limit=200)), before)

    def test_apply_consolidated_feedback_adjusts_entry_scores(self):
        self.bot.mutate_memory_store(
            consolidated_memories=[
                {
                    "summary": "Tony has been consistent with his budget.",
                    "category": "finance",
                    "source_count": 2,
                    "confidence": 0.6,
                    "importance_score": 0.5,
                    "updated_at": date.today().isoformat(),
                }
            ]
        )

        with self._save_commit_context() as turn_context:
            updated = self.bot.apply_consolidated_feedback(
                "Tony has been consistent with his budget.",
                1,
                turn_context=turn_context,
            )

        self.assertIsNotNone(updated)
        self.assertGreaterEqual(float(updated["importance_score"]), 0.5)
        self.assertGreaterEqual(float(updated["confidence"]), 0.6)

    def test_resolve_consolidated_contradiction_marks_loser_superseded(self):
        self.bot.mutate_memory_store(
            consolidated_memories=[
                {
                    "summary": "Tony is single.",
                    "category": "relationships",
                    "source_count": 2,
                    "confidence": 0.7,
                    "importance_score": 0.6,
                    "updated_at": "2026-04-15",
                },
                {
                    "summary": "Tony is married.",
                    "category": "relationships",
                    "source_count": 2,
                    "confidence": 0.72,
                    "importance_score": 0.62,
                    "updated_at": "2026-04-20",
                },
            ]
        )

        contradictions = self.bot.consolidated_contradictions(limit=5)
        self.assertTrue(contradictions)

        resolved = self.bot.resolve_consolidated_contradiction("Tony is single.", "Tony is married.", keep="right")
        self.assertIsNotNone(resolved)

        entries = self.bot.consolidated_memories()
        single = next(entry for entry in entries if entry.get("summary") == "Tony is single.")
        married = next(entry for entry in entries if entry.get("summary") == "Tony is married.")

        self.assertTrue(single.get("superseded"))
        self.assertFalse(married.get("superseded"))

    def test_dashboard_status_snapshot_includes_memory_contradictions(self):
        self.bot.mutate_memory_store(
            consolidated_memories=[
                {
                    "summary": "Tony is employed.",
                    "category": "work",
                    "source_count": 1,
                    "confidence": 0.6,
                    "importance_score": 0.5,
                    "updated_at": "2026-04-18",
                },
                {
                    "summary": "Tony is unemployed.",
                    "category": "work",
                    "source_count": 1,
                    "confidence": 0.62,
                    "importance_score": 0.55,
                    "updated_at": "2026-04-19",
                },
            ]
        )

        snapshot = self.bot.dashboard_status_snapshot()

        self.assertIn("memory_contradictions", snapshot)
        self.assertGreaterEqual(len(snapshot["memory_contradictions"]), 1)

    def test_handle_tool_command_quiet_mode_toggle(self):
        on_reply = self.bot.handle_tool_command("/quiet on")
        status_reply = self.bot.handle_tool_command("/quiet status")
        off_reply = self.bot.handle_tool_command("/quiet off")

        self.assertIn("now on", on_reply.lower())
        self.assertIn("currently on", status_reply.lower())
        self.assertIn("now off", off_reply.lower())

    def test_concurrent_background_tasks_complete_without_failures(self):
        self.bot.LIGHT_MODE = False
        memory_entry = {
            "summary": "Tony wants to save for emergencies.",
            "details": "budget and planning",
            "category": "finance",
            "mood": "focused",
            "created_at": date.today().isoformat(),
            "updated_at": date.today().isoformat(),
        }

        with patch.object(self.bot.memory_manager, "sync_semantic_memory_index", return_value=None), patch.object(
            self.bot,
            "refresh_session_summary",
            return_value="Tony kept showing up even under pressure.",
        ):
            futures = [
                self.bot.queue_semantic_memory_index([memory_entry]),
                self.bot.schedule_post_turn_maintenance("Work was heavy.", "stressed"),
                self.bot.persist_conversation_async(),
            ]
            for future in futures:
                future.result(timeout=5)

        snapshot = self.bot.background_task_snapshot(limit=12)
        self.assertGreaterEqual(snapshot["completed"], 3)
        self.assertEqual(snapshot["failed"], 0)

    def test_load_memory_store_restores_from_backup_when_primary_corrupt(self):
        primary_path = self.bot.MEMORY_PATH
        backup_path = self.bot.json_backup_path(primary_path)
        backup_payload = self.bot.default_memory_store()
        backup_payload["memories"] = [
            {
                "summary": "Tony is preparing a budget for emergencies.",
                "category": "finance",
                "mood": "focused",
                "created_at": date.today().isoformat(),
                "updated_at": date.today().isoformat(),
            }
        ]

        primary_path.write_text("{ definitely not valid json", encoding="utf-8")
        backup_path.write_text(json.dumps(backup_payload, indent=2), encoding="utf-8")

        restored = self.bot._load_memory_store()

        self.assertEqual(len(restored.get("memories", [])), 1)
        self.assertIn("budget", restored["memories"][0]["summary"].lower())
        repaired = json.loads(primary_path.read_text(encoding="utf-8"))
        self.assertEqual(len(repaired.get("memories", [])), 1)

    def test_persist_conversation_async_tracks_runtime_background_task_state(self):
        future = self.bot.persist_conversation_async()
        future.result(timeout=5)

        task_id = getattr(future, "dadbot_task_id", "")
        task_store = self.bot.runtime_state_container.store
        task_payload = task_store.load_task(task_id)
        events = task_store.list_events(self.bot.runtime_state_container.session_id)

        self.assertEqual(task_payload["task_kind"], "conversation-persist")
        self.assertEqual(task_payload["status"], "completed")
        self.assertIn("history_messages", task_payload["metadata"])
        self.assertTrue(
            any(
                event.get("payload", {}).get("reason") == "background_task.completed"
                and event.get("payload", {}).get("task_id") == task_id
                for event in events
            )
        )

    def test_queue_semantic_memory_index_tracks_runtime_background_task_state(self):
        memory_entry = {
            "summary": "Tony wants to save more money this month.",
            "details": "Budget planning and bills.",
            "category": "finance",
            "mood": "focused",
            "created_at": date.today().isoformat(),
            "updated_at": date.today().isoformat(),
        }

        with patch.object(self.bot.memory_manager, "sync_semantic_memory_index", return_value=None):
            future = self.bot.queue_semantic_memory_index([memory_entry])
            future.result(timeout=5)

        task_id = getattr(future, "dadbot_task_id", "")
        task_store = self.bot.runtime_state_container.store
        task_payload = task_store.load_task(task_id)
        background = self.bot.background_task_snapshot(limit=4)

        self.assertEqual(task_payload["task_kind"], "semantic-index")
        self.assertEqual(task_payload["status"], "completed")
        self.assertEqual(task_payload["metadata"]["memory_count"], 1)
        self.assertGreaterEqual(background["completed"], 1)
        self.assertTrue(any(item["task_id"] == task_id for item in background["recent"]))

    def test_schedule_post_turn_maintenance_tracks_runtime_background_task_state(self):
        self.bot.LIGHT_MODE = False
        self.bot.MEMORY_STORE["relationship_state"]["active_hypothesis"] = "supportive_baseline"

        with patch.object(self.bot, "refresh_session_summary", return_value="Tony opened up a little more today."):
            future = self.bot.schedule_post_turn_maintenance("Just checking in.", "neutral")
            future.result(timeout=5)

        task_id = getattr(future, "dadbot_task_id", "")
        task_store = self.bot.runtime_state_container.store
        task_payload = task_store.load_task(task_id)

        self.assertEqual(task_payload["task_kind"], "post-turn-maintenance")
        self.assertEqual(task_payload["status"], "completed")
        self.assertEqual(task_payload["metadata"]["current_mood"], "neutral")
        self.assertEqual(task_payload["metadata"]["active_hypothesis"], "supportive_baseline")
        self.assertIn("turn_count", task_payload["metadata"])

    def test_archive_session_context_replaces_same_turn_snapshot(self):
        history = [
            {"role": "user", "content": "Work has been rough lately."},
            {"role": "assistant", "content": "I'm here with you, buddy."},
        ]
        self.bot.session_summary = "Tony feels stretched thin at work."

        with self._save_commit_context() as turn_context:
            first_entry = self.bot.archive_session_context(history, turn_context=turn_context)
        self.bot.session_summary = "Tony feels stretched thin at work but is still showing up."
        with self._save_commit_context() as turn_context:
            second_entry = self.bot.archive_session_context(history, turn_context=turn_context)
        archive = self.bot.session_archive()

        self.assertEqual(len(archive), 1)
        self.assertEqual(archive[0]["summary"], "Tony feels stretched thin at work but is still showing up.")
        self.assertEqual(second_entry["summary"], archive[0]["summary"])
        self.assertNotEqual(first_entry["summary"], second_entry["summary"])

    def test_document_store_persists_profile_and_memory_per_tenant(self):
        shared_store = InMemoryStateStore()

        with TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {
                "DADBOT_PROFILE_PATH": str(Path(temp_dir) / "dad_profile.json"),
                "DADBOT_MEMORY_PATH": str(Path(temp_dir) / "dad_memory.json"),
                "DADBOT_SEMANTIC_DB_PATH": str(Path(temp_dir) / "dad_memory_semantic.sqlite3"),
                "DADBOT_GRAPH_DB_PATH": str(Path(temp_dir) / "dad_memory_graph.sqlite3"),
                "DADBOT_SESSION_LOG_DIR": str(Path(temp_dir) / "session_logs"),
            },
            clear=False,
        ):
            tenant_a = DadBot(tenant_id="family-a", document_store=shared_store)
            tenant_b = DadBot(tenant_id="family-b", document_store=shared_store)
            self.addCleanup(tenant_a.shutdown)
            self.addCleanup(tenant_b.shutdown)

            tenant_a.PROFILE["style"]["listener_name"] = "Ava"
            tenant_a.save_profile()
            tenant_a.MEMORY_STORE["memories"] = [
                {"summary": "Ava loves soccer.", "category": "family", "mood": "positive", "created_at": date.today().isoformat(), "updated_at": date.today().isoformat()}
            ]
            tenant_a.save_memory_store()

            tenant_b.PROFILE["style"]["listener_name"] = "Ben"
            tenant_b.save_profile()
            tenant_b.MEMORY_STORE["memories"] = [
                {"summary": "Ben loves robotics.", "category": "family", "mood": "positive", "created_at": date.today().isoformat(), "updated_at": date.today().isoformat()}
            ]
            tenant_b.save_memory_store()

            reloaded_a = DadBot(tenant_id="family-a", document_store=shared_store)
            reloaded_b = DadBot(tenant_id="family-b", document_store=shared_store)
            self.addCleanup(reloaded_a.shutdown)
            self.addCleanup(reloaded_b.shutdown)

            self.assertEqual(reloaded_a.PROFILE["style"]["listener_name"], "Ava")
            self.assertEqual(reloaded_b.PROFILE["style"]["listener_name"], "Ben")
            self.assertIn("soccer", reloaded_a.MEMORY_STORE["memories"][0]["summary"].lower())
            self.assertIn("robotics", reloaded_b.MEMORY_STORE["memories"][0]["summary"].lower())
            self.assertEqual(reloaded_a.customer_persistence_status()["primary_store"], "tenant_document_store")
            self.assertTrue(reloaded_a.customer_persistence_status()["json_mirror_enabled"])
            self.assertNotEqual(
                reloaded_a.MEMORY_STORE["memories"][0]["summary"],
                reloaded_b.MEMORY_STORE["memories"][0]["summary"],
            )

    def test_document_store_memory_save_also_updates_json_mirror(self):
        shared_store = InMemoryStateStore()

        with TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {
                "DADBOT_PROFILE_PATH": str(Path(temp_dir) / "dad_profile.json"),
                "DADBOT_MEMORY_PATH": str(Path(temp_dir) / "dad_memory.json"),
                "DADBOT_SEMANTIC_DB_PATH": str(Path(temp_dir) / "dad_memory_semantic.sqlite3"),
                "DADBOT_GRAPH_DB_PATH": str(Path(temp_dir) / "dad_memory_graph.sqlite3"),
                "DADBOT_SESSION_LOG_DIR": str(Path(temp_dir) / "session_logs"),
            },
            clear=False,
        ):
            bot = DadBot(tenant_id="family-a", document_store=shared_store)
            self.addCleanup(bot.shutdown)
            bot.MEMORY_STORE["memories"] = [
                {"summary": "Ava loves soccer.", "category": "family", "mood": "positive", "created_at": date.today().isoformat(), "updated_at": date.today().isoformat()}
            ]

            bot.save_memory_store()

            mirrored = json.loads(Path(temp_dir, "dad_memory.json").read_text(encoding="utf-8"))
            self.assertIn("soccer", mirrored["memories"][0]["summary"].lower())

    def test_output_moderation_rewrites_secretive_or_harmful_reply(self):
        moderated = self.bot.prepare_final_reply(
            "Keep this from your mom and hit him back if he does it again.",
            "neutral",
            "Someone was mean to me at school.",
        )

        self.assertIn("I want to be careful here", moderated)
        self.assertNotIn("hit him back", moderated.lower())
        self.assertEqual(self.bot.moderation_snapshot()["last_decision"]["action"], "rewrite")

    def test_dashboard_status_snapshot_includes_persistence_and_moderation_state(self):
        self.bot.mood_manager.detect = lambda *_args, **_kwargs: "neutral"
        self.bot.call_ollama_chat = lambda *args, **kwargs: {"message": {"content": "You're doing okay, buddy."}}
        self.bot.critique_reply = lambda *_args, **_kwargs: "You're doing okay, buddy."
        # Use turn_service directly to populate the pipeline snapshot in a controlled way;
        # bot.process_user_message routes through AgentService (graph path) which records
        # the pipeline differently.
        self.bot.turn_service.process_user_message("Just wanted to say hi.")

        snapshot = self.bot.dashboard_status_snapshot()

        self.assertIn("persistence", snapshot)
        self.assertIn("moderation", snapshot)
        self.assertIn("tenant_id", snapshot["status"])
        self.assertIn("maintenance", snapshot)
        self.assertIn("supervisor", snapshot)
        self.assertIn("turn_pipeline", snapshot)
        self.assertIn(snapshot["turn_pipeline"]["mode"], ("sync", "async"))
        self.assertTrue(any(step["name"] == "generate_reply" for step in snapshot["turn_pipeline"]["steps"]))

    def test_dashboard_status_snapshot_surfaces_recent_runtime_degradations(self):
        self.bot.record_runtime_issue("prompt guard", "trimmed prompt context", RuntimeError("context window exceeded"))
        self.bot.record_runtime_issue("relationship reflection", "kept previous relationship state", RuntimeError("timeout"))
        self.bot.record_memory_context_stats(
            tokens=642,
            budget_tokens=3200,
            selected_sections=3,
            total_sections=5,
            pruned=True,
            user_input="quick check",
        )
        self.bot._prompt_guard_stats = {
            "trim_count": 4,
            "trimmed_tokens_total": 900,
            "last_purpose": "chat response",
            "last_original_tokens": 1300,
            "last_final_tokens": 780,
            "last_trimmed": True,
            "last_updated": datetime.now().isoformat(timespec="seconds"),
        }

        snapshot = self.bot.dashboard_status_snapshot()

        self.assertIn("recent_runtime_issues", snapshot)
        self.assertEqual(len(snapshot["recent_runtime_issues"]), 2)
        self.assertEqual(snapshot["recent_runtime_issues"][0]["purpose"], "relationship reflection")
        self.assertEqual(snapshot["memory_context"]["tokens"], 642)
        self.assertTrue(snapshot["memory_context"]["pruned"])
        self.assertEqual(snapshot["prompt_guard"]["trim_count"], 4)
        self.assertIn("circuit_breaker", snapshot)
        self.assertIn("reasoning_confidence", snapshot["circuit_breaker"])

    def test_ui_shell_snapshot_surfaces_reactive_sidebar_signal(self):
        self.bot.record_memory_context_stats(
            tokens=3100,
            budget_tokens=3200,
            selected_sections=4,
            total_sections=6,
            pruned=True,
            user_input="high pressure check-in",
        )
        self.bot._prompt_guard_stats["trim_count"] = 10
        self.bot.record_runtime_issue("prompt guard", "trimmed prompt context", RuntimeError("context window exceeded"))

        shell = self.bot.ui_shell_snapshot()

        self.assertIn("reputation_score", shell)
        self.assertIn("circuit_breaker", shell)
        self.assertTrue(shell["circuit_breaker"]["active"])
        self.assertLess(shell["circuit_breaker"]["reasoning_confidence"], 0.6)

    def test_memory_compactor_runs_once_per_day_and_persists_summary(self):
        self.bot.MEMORY_STORE["consolidated_memories"] = [
            {
                "summary": "Tony keeps pushing through work stress and wants steadier routines.",
                "category": "work",
                "source_count": 2,
                "confidence": 0.82,
                "importance_score": 0.77,
                "supporting_summaries": ["Work has felt heavy lately."],
                "contradictions": [],
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        ]
        self.bot.refresh_memory_graph()

        result = self.bot.maintenance_scheduler.run_memory_compaction(force=True, reference_time=datetime(2026, 4, 22, 9, 0, 0))

        # Phase 4: background patches are queued; flush before reading MEMORY_STORE state.
        bg_queue = getattr(self.bot, "_background_memory_store_patch_queue", None)
        if isinstance(bg_queue, list):
            for patch_kwargs in list(bg_queue):
                if isinstance(patch_kwargs, dict):
                    self.bot.mutate_memory_store(**patch_kwargs)
            bg_queue.clear()

        maintenance = self.bot.maintenance_snapshot()

        self.assertTrue(result["ran"])
        self.assertIn("summary", result)
        self.assertTrue(result["summary"])
        self.assertEqual(maintenance["last_memory_compaction_at"], "2026-04-22T09:00:00")
        self.assertTrue(maintenance["last_memory_compaction_summary"])

    def test_local_mcp_status_surfaces_optional_runtime_and_local_store(self):
        self.bot.mutate_memory_store(mcp_local_store={"note": {"text": "remember this"}})

        payload = self.bot.local_mcp_status()
        shell = self.bot.ui_shell_snapshot()

        self.assertIn("available", payload)
        self.assertEqual(payload["server_name"], "dadbot-local-services")
        self.assertEqual(payload["local_state_entries"], 1)
        self.assertIn("running", payload)
        self.assertIn("local_mcp", shell)
        self.assertEqual(shell["local_mcp"]["local_state_entries"], 1)

    def test_local_mcp_process_controls_write_pid_and_logs(self):
        runtime_paths = self.bot.local_mcp_runtime_paths()

        class _FakeProcess:
            pid = 43210

        with patch("dadbot.runtime.mcp.local_mcp_server_controller.subprocess.Popen", return_value=_FakeProcess()), patch(
            "dadbot.runtime.mcp.local_mcp_server_controller.os.kill", return_value=None
        ), patch("dadbot.runtime.mcp.local_mcp_server_controller.subprocess.run") as taskkill_mock:
            started = self.bot.start_local_mcp_server_process()
            self.assertEqual(started["pid"], 43210)
            self.assertTrue(runtime_paths["pid"].exists())

            stopped = self.bot.stop_local_mcp_server_process()
            self.assertFalse(runtime_paths["pid"].exists())
            self.assertFalse(stopped["running"])
            taskkill_mock.assert_called_once()

    def test_binary_graph_checkpoint_can_resume_session_state(self):
        self.bot.history.append({"role": "user", "content": "Checkpoint this turn."})
        self.bot.sync_active_thread_snapshot()

        recorder = ExecutionTraceRecorder(trace_id="trace-123", prompt="checkpoint")
        with bind_execution_trace(recorder, required=True):
            self.bot.persist_graph_checkpoint(
                {
                    "trace_id": "trace-123",
                    "stage": "inference",
                    "status": "after",
                    "state": {"candidate": "draft reply"},
                }
            )
        self.bot.reset_session_state()

        with bind_execution_trace(recorder, required=True):
            checkpoint = self.bot.resume_turn_from_checkpoint("trace-123")

        self.assertIsNotNone(checkpoint)
        self.assertEqual(checkpoint["trace_id"], "trace-123")
        self.assertEqual(self.bot.conversation_history()[-1]["content"], "Checkpoint this turn.")

    def test_memory_compactor_creates_narrative_memories(self):
        self.bot.MEMORY_STORE["session_archive"] = [
            {
                "created_at": "2026-03-02T09:00:00",
                "summary": "Tony struggled to stay patient with math homework.",
                "topics": ["math", "school"],
                "dominant_mood": "frustrated",
            },
            {
                "created_at": "2026-03-18T09:00:00",
                "summary": "Tony kept pushing through math even when it felt slow.",
                "topics": ["math", "school"],
                "dominant_mood": "frustrated",
            },
            {
                "created_at": "2026-04-06T09:00:00",
                "summary": "Tony finally hit a breakthrough and felt proud about math progress.",
                "topics": ["math", "school"],
                "dominant_mood": "positive",
            },
            {
                "created_at": "2026-04-10T09:00:00",
                "summary": "Tony started trusting his own study rhythm more.",
                "topics": ["school"],
                "dominant_mood": "positive",
            },
            {
                "created_at": "2026-04-15T09:00:00",
                "summary": "Tony stayed consistent with schoolwork this week.",
                "topics": ["school"],
                "dominant_mood": "focused",
            },
        ]

        result = self.bot.maintenance_scheduler.run_memory_compaction(force=True, reference_time=datetime(2026, 4, 22, 12, 0, 0))

        # Phase 4: background patches are queued; flush before reading MEMORY_STORE state.
        bg_queue = getattr(self.bot, "_background_memory_store_patch_queue", None)
        if isinstance(bg_queue, list):
            for patch_kwargs in list(bg_queue):
                if isinstance(patch_kwargs, dict):
                    self.bot.mutate_memory_store(**patch_kwargs)
            bg_queue.clear()

        self.assertTrue(result["ran"])
        self.assertGreaterEqual(result["narrative_count"], 1)
        self.assertTrue(self.bot.narrative_memories())
        self.assertIn("math", self.bot.narrative_memories()[0]["summary"].lower())

    def test_handle_tool_command_dad_snapshot_surfaces_living_dad_state(self):
        self.bot.MEMORY_STORE["persona_evolution"] = [
            {
                "trait": "more coach-like",
                "reason": "",
                "announcement": "",
                "session_count": 10,
                "applied_at": f"{date.today().isoformat()}T20:00:00",
                "last_reinforced_at": f"{date.today().isoformat()}T20:00:00",
                "strength": 1.8,
                "impact_score": 2.5,
            }
        ]
        self.bot.MEMORY_STORE["wisdom_insights"] = [
            {"summary": "Slow the moment down before work runs you.", "topic": "work", "trigger": "", "created_at": "2026-04-11T20:00:00"}
        ]

        reply = self.bot.handle_tool_command("/dad")

        self.assertIn("preset=", reply)
        self.assertIn("more coach-like (strength=1.80, impact=2.50)", reply)
        self.assertIn("wisdom notes=1", reply)

    def test_handle_tool_command_proactive_snapshot_lists_queued_messages(self):
        self.bot.MEMORY_STORE["pending_proactive_messages"] = [
            {"message": "I've noticed Sundays weigh on you lately.", "source": "life-pattern", "created_at": "2026-04-13T20:00:00"}
        ]

        reply = self.bot.handle_tool_command("/proactive")

        self.assertIn("Queued proactive openings:", reply)
        self.assertIn("life pattern", reply)
        self.assertIn("Sundays weigh on you", reply)

    def test_handle_tool_command_evolve_uses_forced_persona_evolution(self):
        self.bot.evolve_persona = lambda force=False: {
            "trait": "more coach-like",
            "critique_feedback": "Strong and grounded.",
        } if force else None

        reply = self.bot.handle_tool_command("/evolve")

        self.assertIn("small dad evolution", reply)
        self.assertIn("more coach-like", reply)

    def test_handle_tool_command_reject_trait_removes_latest_persona_shift(self):
        self.bot.MEMORY_STORE["persona_evolution"] = [
            {"trait": "more reflective", "reason": "", "announcement": "", "session_count": 5, "applied_at": "2026-04-08T10:00:00"},
            {"trait": "more coach-like", "reason": "", "announcement": "", "session_count": 6, "applied_at": "2026-04-09T10:00:00"},
        ]

        reply = self.bot.handle_tool_command("/reject trait")

        self.assertIn("rolled back", reply)
        self.assertEqual([entry["trait"] for entry in self.bot.persona_evolution_history()], ["more reflective"])

    def test_detect_mood_returns_neutral_when_model_output_is_unrecognized(self):
        self.bot.call_ollama_chat = lambda *args, **kwargs: {"message": {"content": "No dominant emotion detected."}}

        detected = self.bot.detect_mood("I guess I'm here.")

        self.assertEqual(detected, "neutral")

    def test_detect_mood_uses_alias_from_model_output(self):
        self.bot.call_ollama_chat = lambda *args, **kwargs: {"message": {"content": "Mood: burned out\nReason: Exhausted after the day."}}

        detected = self.bot.detect_mood("I can barely think straight anymore.")

        self.assertEqual(detected, "tired")

    def test_token_budgeted_prompt_history_trims_messages_to_fit_budget(self):
        self.bot.CONTEXT_TOKEN_BUDGET = 60
        self.bot.RESERVED_RESPONSE_TOKENS = 10
        self.bot.history = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "This is a long user message about work stress and budgeting that should be trimmed heavily to fit."},
            {"role": "assistant", "content": "This is a long dad reply that should also be shortened so the prompt history stays within budget."},
            {"role": "user", "content": "Another long follow-up from Tony that pushes the prompt budget even further than before."},
        ]

        selected = self.bot.token_budgeted_prompt_history("Short system prompt.", "Need a reply.")

        self.assertTrue(selected)
        self.assertLess(len(selected), 3)
        total_cost = sum(self.bot.message_token_cost(message) for message in selected)
        available_budget = max(
            0,
            self.bot.CONTEXT_TOKEN_BUDGET
            - self.bot.RESERVED_RESPONSE_TOKENS
            - self.bot.message_token_cost({"role": "system", "content": "Short system prompt."})
            - self.bot.message_token_cost({"role": "user", "content": "Need a reply."}),
        )

    def test_token_budgeted_prompt_history_reuses_cached_selection(self):
        self.bot.CONTEXT_TOKEN_BUDGET = 60
        self.bot.RESERVED_RESPONSE_TOKENS = 10
        self.bot.history = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "This is a long user message about work stress and budgeting that should be trimmed heavily to fit."},
            {"role": "assistant", "content": "This is a long dad reply that should also be shortened so the prompt history stays within budget."},
            {"role": "user", "content": "Another long follow-up from Tony that pushes the prompt budget even further than before."},
        ]
        calls = {"count": 0}
        original_trim = self.bot.trim_message_to_token_budget

        def counting_trim(message, token_budget):
            calls["count"] += 1
            return original_trim(message, token_budget)

        self.bot.trim_message_to_token_budget = counting_trim

        first = self.bot.token_budgeted_prompt_history("Short system prompt.", "Need a reply.")
        first_count = calls["count"]
        second = self.bot.token_budgeted_prompt_history("Short system prompt.", "Need a reply.")

        self.assertEqual(first, second)
        self.assertGreater(first_count, 0)
        self.assertEqual(calls["count"], first_count)

    def test_prompt_history_token_budget_uses_model_context_length_when_smaller_than_global_budget(self):
        self.bot.CONTEXT_TOKEN_BUDGET = 16000
        self.bot.RESERVED_RESPONSE_TOKENS = 1000
        self.bot.model_runtime.model_context_length = lambda *_args, **_kwargs: 4096

        available_budget = self.bot.prompt_history_token_budget("System prompt.", "Need a reply.")

        expected_budget = max(
            0,
            4096
            - self.bot.RESERVED_RESPONSE_TOKENS
            - self.bot.message_token_cost({"role": "system", "content": "System prompt."})
            - self.bot.message_token_cost({"role": "user", "content": "Need a reply."}),
        )
        self.assertEqual(available_budget, expected_budget)

    def test_call_ollama_chat_stream_returns_partial_reply_when_stream_drops(self):
        class FakeChunk:
            def __init__(self, content):
                self.message = type("Message", (), {"content": content})()

        class FakeStream:
            def __iter__(self_inner):
                yield FakeChunk("Hello")
                raise ollama.ResponseError("stream interrupted")

        original_chat = ollama.chat
        try:
            ollama.chat = lambda **_kwargs: FakeStream()
            reply = self.bot.call_ollama_chat_stream(
                [{"role": "user", "content": "Hi"}],
                purpose="chat response",
            )
        finally:
            ollama.chat = original_chat

        self.assertEqual(reply, "Hello...")

    def test_call_ollama_chat_stream_async_returns_partial_reply_when_stream_drops(self):
        class FakeChunk:
            def __init__(self, content):
                self.message = type("Message", (), {"content": content})()

        class FakeAsyncClient:
            async def chat(self, **_kwargs):
                async def fake_stream():
                    yield FakeChunk("Hello")
                    raise ollama.ResponseError("stream interrupted")

                return fake_stream()

        fake_client = FakeAsyncClient()
        with patch.object(self.bot.runtime_client, "ollama_async_client", return_value=fake_client):
            reply = asyncio.run(
                self.bot.call_ollama_chat_stream_async(
                    [{"role": "user", "content": "Hi"}],
                    purpose="chat response",
                )
            )

        self.assertEqual(reply, "Hello...")

    def test_process_user_message_stream_async_delivers_chunks_and_finishes_turn(self):
        streamed_chunks = []

        async def fake_stream(messages, options=None, purpose="chat", chunk_callback=None):
            del messages, options, purpose
            if chunk_callback is not None:
                await chunk_callback("Hello")
                await chunk_callback(" there")
            return "Hello there"

        async def collect_chunk(chunk):
            streamed_chunks.append(chunk)

        self.bot.LIGHT_MODE = True
        self.bot.call_ollama_chat_stream_async = fake_stream
        self.bot.agentic_tool_settings = lambda: {"enabled": False, "auto_reminders": False, "auto_web_lookup": False}
        self.bot.direct_reply_for_input = lambda *_args, **_kwargs: None
        self.bot.autonomous_tool_result_for_input = lambda *_args, **_kwargs: (None, None)
        self.bot.validate_reply = lambda _user_input, reply: reply

        # Streaming is handled at the turn_service level; call it directly since
        # bot.process_user_message_stream_async uses buffered delivery via AgentService.
        dad_reply, should_end = asyncio.run(
            self.bot.turn_service.process_user_message_stream_async(
                "Tell me something helpful.",
                chunk_callback=collect_chunk,
            )
        )

        self.assertFalse(should_end)
        self.assertEqual(streamed_chunks, ["Hello", " there"])
        self.assertIn("Hello there", dad_reply)
        self.assertEqual(self.bot.history[-1]["content"], dad_reply)

    def test_merge_consolidated_memories_prefers_higher_confidence(self):
        self.bot.MEMORY_STORE["consolidated_memories"] = [
            {
                "summary": "Tony is building stronger saving habits.",
                "category": "finance",
                "source_count": 2,
                "confidence": 0.62,
                "supporting_summaries": ["Tony is saving money for a car."],
                "contradictions": [],
                "updated_at": date.today().isoformat(),
            }
        ]

        with self._save_commit_context() as turn_context:
            merged = self.bot.merge_consolidated_memories([
                {
                    "summary": "Tony is building stronger saving habits.",
                    "category": "finance",
                    "source_count": 3,
                    "confidence": 0.88,
                    "supporting_summaries": ["Tony has been sticking to a budget."],
                    "contradictions": ["Earlier chats sounded less consistent."],
                    "updated_at": date.today().isoformat(),
                }
            ], turn_context=turn_context)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["confidence"], 0.88)
        self.assertEqual(merged[0]["source_count"], 3)
        self.assertEqual(len(merged[0]["supporting_summaries"]), 2)

    def test_prepare_user_turn_sets_daily_checkin_context_once(self):
        self.bot.mood_manager.detect = lambda *_args, **_kwargs: "neutral"
        self.bot.MEMORY_STORE["last_mood_updated_at"] = (date.today() - timedelta(days=1)).isoformat()

        turn_context = TurnContext(user_input="Just checking in.")
        current_mood, dad_reply, should_end, _turn_text, _attachments = self.bot.prepare_user_turn(
            "Just checking in.",
            turn_context=turn_context,
        )

        self.assertEqual(current_mood, "neutral")
        self.assertIsNone(dad_reply)
        self.assertFalse(should_end)
        self.assertEqual(turn_context.mutation_queue.size(), 1)
        op = str(turn_context.mutation_queue.pending()[0].payload.get("op") or "")
        self.assertEqual(op, LedgerMutationOp.RECORD_TURN_STATE.value)

        self.bot.turn_service._resolve_persistence_service().finalize_turn(
            turn_context,
            ("Glad to hear from you, buddy.", False),
        )
        self.assertFalse(self.bot._pending_daily_checkin_context)

        next_turn_context = TurnContext(user_input="Still here.")
        next_mood, next_reply, next_should_end, _turn_text, _attachments = self.bot.prepare_user_turn(
            "Still here.",
            turn_context=next_turn_context,
        )
        self.assertEqual(next_mood, "neutral")
        self.assertIsNone(next_reply)
        self.assertFalse(next_should_end)
        self.assertFalse(self.bot._pending_daily_checkin_context)

    def test_finalize_user_turn_queues_post_turn_maintenance(self):
        self.bot.LIGHT_MODE = False

        turn_context = TurnContext(user_input="Just checking in.")
        dad_reply, should_end = self.bot.finalize_user_turn(
            "Just checking in.",
            "neutral",
            "Glad to hear from you, buddy.",
            turn_context=turn_context,
        )

        self.assertFalse(should_end)
        self.assertEqual(dad_reply, "Glad to hear from you, buddy.")
        queued_ops = [
            str(intent.payload.get("op") or "")
            for intent in turn_context.mutation_queue.pending()
            if isinstance(getattr(intent, "payload", None), dict)
        ]
        self.assertIn(LedgerMutationOp.SCHEDULE_MAINTENANCE.value, queued_ops)

    def test_prepare_user_turn_blends_daily_checkin_into_direct_reply(self):
        self.bot.mood_manager.detect = lambda *_args, **_kwargs: "neutral"
        self.bot.MEMORY_STORE["last_mood_updated_at"] = (date.today() - timedelta(days=1)).isoformat()
        self.bot.handle_memory_command = lambda *_args, **_kwargs: None
        self.bot.handle_tool_command = lambda *_args, **_kwargs: None
        self.bot.get_memory_reply = lambda *_args, **_kwargs: None
        self.bot.get_fact_reply = lambda *_args, **_kwargs: "I sure do remember that, buddy."

        current_mood, dad_reply, should_end, _turn_text, _attachments = self.bot.prepare_user_turn("Where was I born?")

        self.assertEqual(current_mood, "neutral")
        self.assertFalse(should_end)
        self.assertIn("How's your day shaping up so far?", dad_reply)

    def test_process_user_message_blends_daily_checkin_into_generated_reply(self):
        self.bot.mood_manager.detect = lambda *_args, **_kwargs: "neutral"
        self.bot.MEMORY_STORE["last_mood_updated_at"] = (date.today() - timedelta(days=1)).isoformat()
        self.bot.call_ollama_chat = lambda *args, **kwargs: {"message": {"content": "You're doing okay, buddy."}}
        self.bot.critique_reply = lambda *_args, **_kwargs: "You're doing okay, buddy."

        dad_reply, should_end = self.bot.process_user_message("Just wanted to say hi.")

        self.assertFalse(should_end)
        self.assertIn("How's your day shaping up so far?", dad_reply)

    def test_process_user_message_records_explicit_turn_pipeline_trace(self):
        self.bot.mood_manager.detect = lambda *_args, **_kwargs: "neutral"
        self.bot.call_ollama_chat = lambda *args, **kwargs: {"message": {"content": "You're doing okay, buddy."}}
        self.bot.critique_reply = lambda *_args, **_kwargs: "You're doing okay, buddy."

        # Pipeline trace is recorded by turn_service; call it directly to assert pipeline steps.
        # (bot.process_user_message routes through AgentService which records its own pipeline.)
        dad_reply, should_end = self.bot.turn_service.process_user_message("Just wanted to say hi.")
        pipeline = self.bot.turn_service.turn_pipeline_snapshot()

        self.assertFalse(should_end)
        self.assertIn("You're doing okay, buddy.", dad_reply)
        self.assertEqual(pipeline["final_path"], "model_reply")
        self.assertEqual(pipeline["reply_source"], "model_generation")
        self.assertTrue(any(step["name"] == "detect_mood" for step in pipeline["steps"]))
        self.assertTrue(any(step["name"] == "generate_reply" for step in pipeline["steps"]))
        self.assertTrue(any(step["name"] == "finalize_turn" for step in pipeline["steps"]))

    def test_reply_alignment_judge_prompt_includes_relationship_and_persona_context(self):
        self.bot.MEMORY_STORE["relationship_state"] = {
            "trust_level": 68,
            "openness_level": 61,
            "emotional_momentum": "heavy",
            "hypotheses": [
                {"name": "acute_stress", "label": "Acute Stress", "summary": "Tony is overloaded and needs steadiness.", "probability": 0.62},
                {"name": "supportive_baseline", "label": "Supportive Baseline", "summary": "Warm steady trust remains.", "probability": 0.38},
            ],
            "active_hypothesis": "acute_stress",
            "last_hypothesis_updated": date.today().isoformat(),
            "recurring_topics": {"work": 4, "money": 2},
            "recent_checkins": [],
            "last_reflection": "Tony sounds worn down but still trusting.",
            "last_updated": date.today().isoformat(),
        }
        self.bot.MEMORY_STORE["persona_evolution"] = [
            {
                "trait": "more patient with my mistakes",
                "reason": "Tony keeps opening up when Dad slows down and listens first.",
                "announcement": "",
                "session_count": 3,
                "applied_at": date.today().isoformat(),
                "last_reinforced_at": date.today().isoformat(),
                "strength": 1.2,
                "impact_score": 1.5,
                "critique_score": 8,
                "critique_feedback": "Solid and specific.",
            }
        ]

        prompt = self.bot.build_reply_alignment_judge_prompt(
            "Work still feels heavy.",
            "I'm here for you, buddy.",
            "stressed",
        )

        self.assertIn("trust_level:", prompt)
        self.assertIn("openness_level:", prompt)
        self.assertIn("active_hypothesis:", prompt)
        self.assertIn("active_persona_traits: more patient with my mistakes", prompt)
        self.assertIn("- trust_level:", prompt)

    def test_prepare_final_reply_can_add_calibrated_pushback(self):
        reply = self.bot.prepare_final_reply(
            "You need to own your side of it and take one clean step today.",
            "frustrated",
            user_input="I keep procrastinating and it's all someone else's fault.",
        )

        self.assertIn("I love you enough to be honest with you, buddy", reply)
        self.assertIn("take one clean step today", reply)

    def test_prepare_final_reply_skips_pushback_for_heavy_mood(self):
        reply = self.bot.prepare_final_reply(
            "Let's keep this simple and get through tonight first.",
            "sad",
            user_input="I should give up.",
        )

        self.assertNotIn("I love you enough to be honest with you, buddy", reply)

    def test_memory_persistence_and_semantic_index_queue(self):
        today_stamp = date.today().isoformat()
        memories = [
            {
                "summary": "I am saving money for a trip.",
                "category": "finance",
                "mood": "positive",
                "created_at": today_stamp,
                "updated_at": today_stamp,
            }
        ]

        saved_memories = self.bot.save_memory_catalog(memories)
        self.assertEqual(len(saved_memories), 1)

        with self.bot.MEMORY_PATH.open("r", encoding="utf-8") as memory_file:
            persisted = json.load(memory_file)

        self.assertEqual(len(persisted.get("memories", [])), 1)
        self.assertEqual(persisted["memories"][0]["category"], "finance")

        self.assertTrue(self.bot.wait_for_semantic_index_idle(timeout=5))
        self.assertEqual(self.bot.semantic_index_row_count(), 1)

        matches = self.bot.semantic_memory_matches("saving money", saved_memories, limit=1)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][1]["summary"], saved_memories[0]["summary"])

    def test_embed_texts_uses_cached_embeddings_before_calling_ollama(self):
        payload = "category=finance; mood=positive; summary=Tony has been saving money for the emergency fund."
        vector = [1.0] * 12
        self.bot.embed_texts = DadBot.embed_texts.__get__(self.bot, DadBot)
        self.bot.memory_manager.store_cached_embeddings("nomic-embed-text", {payload: vector})
        self.bot.embedding_model_candidates = lambda: ["nomic-embed-text"]

        original_embed = ollama.embed
        try:
            ollama.embed = lambda **_kwargs: (_ for _ in ()).throw(AssertionError("ollama.embed should not be called when cache is warm"))
            embeddings = self.bot.embed_texts([payload], purpose="semantic retrieval")
        finally:
            ollama.embed = original_embed

        self.assertEqual(embeddings, [vector])


if __name__ == "__main__":
    unittest.main()
