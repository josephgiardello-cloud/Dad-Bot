import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from test_streamlit_smoke import _prepare_streamlit_sandbox


@pytest.mark.parametrize("durable", [False, True])
def test_canonical_streamlit_boot_and_turn(tmp_path, durable):
    repo_root = Path(__file__).resolve().parents[1]
    sandbox = tmp_path / "phase1_app"
    sandbox.mkdir()
    _prepare_streamlit_sandbox(repo_root, sandbox)
    shutil.copy2(sandbox / "dad_profile.template.json", sandbox / "dad_profile.json")
    code = textwrap.dedent(
        """
        from streamlit.testing.v1 import AppTest
        from dadbot.services.agent_service import AgentService
        from dadbot.services.maintenance_service import MaintenanceService
        from dadbot.services.safety_service import SafetyService

        app = AppTest.from_file("dad_streamlit.py").run(timeout=45)
        assert not app.exception, list(app.exception)
        bot = app.session_state["bot"]
        registry = bot.services.registry
        assert bot.service_registry is registry
        orchestrator = bot._get_turn_orchestrator()
        assert orchestrator is bot.services.turn_orchestrator
        assert orchestrator.registry is registry
        assert orchestrator.graph.registry is registry
        assert registry.get("turn_graph").registry is registry
        persistence = registry.get("persistence_service")
        assert persistence is bot.services.persistence_service
        assert persistence is registry.get("persistence")
        assert persistence is registry.get("storage")
        assert persistence is orchestrator.registry.get("persistence_service")
        assert bot.model_port is not None
        agent_service = registry.get("agent_service")
        assert isinstance(agent_service, AgentService)
        assert agent_service.bot is bot
        maintenance = registry.get("maintenance_service")
        safety = registry.get("safety_service")
        assert isinstance(maintenance, MaintenanceService)
        assert isinstance(safety, SafetyService)
        assert registry.get("health") is maintenance
        assert registry.get("safety") is safety
        maintenance_calls = []
        safety_calls = []
        context_calls = []
        context_service = registry.get("context_service")
        canonical_context = context_service.context_service
        assert canonical_context is not None
        assert registry.get("memory") is context_service
        assert canonical_context.semantic_index is bot.memory_manager._semantic_manager._semantic_index_backend
        snapshot_sessions = []
        get_snapshot = canonical_context.get_snapshot
        def capture_snapshot(session_id):
            snapshot_sessions.append(session_id)
            return get_snapshot(session_id)
        canonical_context.get_snapshot = capture_snapshot
        build_context = context_service.build_context
        def capture_context(turn_context):
            result = build_context(turn_context)
            context_calls.append(turn_context)
            return result
        context_service.build_context = capture_context
        run_proactive = maintenance.maintenance.run_scheduled_proactive_jobs
        enforce = safety.enforce_policies
        def run_scheduled(*args, **kwargs):
            maintenance_calls.append(True)
            return run_proactive(*args, **kwargs)
        def enforce_policies(*args, **kwargs):
            safety_calls.append(True)
            return enforce(*args, **kwargs)
        maintenance.maintenance.run_scheduled_proactive_jobs = run_scheduled
        safety.enforce_policies = enforce_policies
        assert bot.profile_runtime.profile
        assert bot.memory.memory_projection() is not None
        requests = []

        async def generate_async(messages, **kwargs):
            requests.append(kwargs)
            return "Start by clearing one corner of your desk."

        bot.model_port.generate_async = generate_async
        app.session_state["auto_speak_replies"] = False
        next(button for button in app.button if button.label == "New chat").click().run(timeout=45)
        thread_id = app.session_state["active_thread_id"]
        assert thread_id.startswith("thread-")
        app.chat_input[0].set_value("Help me organize my desk.").run(timeout=45)
        assert not app.exception, list(app.exception)
        assert requests, "The canonical turn never invoked the model"
        assert any(request.get("purpose") == "chat" for request in requests), requests
        assert maintenance_calls, "Canonical proactive maintenance was bypassed"
        assert safety_calls, "Canonical safety policies were bypassed"
        assert len(context_calls) == 1, "Memory must be evaluated exactly once per turn"
        snapshot = context_calls[0].state.get("memory_snapshot")
        assert isinstance(snapshot, dict), "UI bypassed the canonical memory snapshot"
        assert isinstance(snapshot.get("memory_recent_buffer"), list)
        assert isinstance(snapshot.get("memory_rolling_summary"), str)
        assert isinstance(snapshot.get("memory_structured"), dict)
        assert snapshot.get("memory_full_history_id"), "Canonical history provenance is missing"
        assert context_calls[0].state.get("_memory_eval_count") == 1
        assert snapshot_sessions == [thread_id], "Memory snapshot used the wrong thread"
        history = app.session_state["chat_threads"][thread_id]
        assert history[-1]["role"] == "assistant", history
        assert history[-1]["text"] == "Start by clearing one corner of your desk.", history[-1]["text"]
        assert not app.session_state["message_outbox"], app.session_state["message_outbox"]
        assert bot.services.registry is registry
        assert bot.service_registry is registry
        reconnected = AppTest.from_file("dad_streamlit.py")
        reconnected.query_params["thread"] = thread_id
        reconnected.run(timeout=45)
        assert not reconnected.exception, list(reconnected.exception)
        assert reconnected.session_state["bot"] is bot, "Reload created a second WAL runtime"
        assert reconnected.session_state["chat_history"] == history
        assert not bot._streamlit_context_messages
        trace_count = sum(event.get("type") == "PolicyTraceEvent" for event in bot._execution_ledger.read())
        assert trace_count > 0, "Safety policy trace was not committed to the execution ledger"
        traces = [event for event in bot._execution_ledger.read() if event.get("type") == "PolicyTraceEvent"]
        assert all(event.get("session_id") == thread_id for event in traces), "Audit trace used the wrong thread"
        assert all(event.get("tenant_id") == bot.config.tenant_id for event in traces), "Audit tenant mismatch"
        reconnected.session_state["auto_speak_replies"] = False
        reconnected.chat_input[0].set_value("Help me organize my desk.").run(timeout=45)
        assert not reconnected.exception, list(reconnected.exception)
        assert not reconnected.session_state["message_outbox"]
        assert sum(event.get("type") == "PolicyTraceEvent" for event in bot._execution_ledger.read()) > trace_count
        scoped_types = {
            "JOB_QUEUED", "JOB_STARTED", "TURN_EVENT", "PolicyTraceEvent",
            "GRAPH_CHECKPOINT", "EFFECT_BEGIN", "EFFECT_COMMIT", "JOB_COMPLETED",
            "SESSION_STATE_UPDATED",
        }
        scoped_events = [event for event in bot._execution_ledger.read() if event.get("type") in scoped_types]
        assert scoped_types <= {event["type"] for event in scoped_events}
        assert all(event.get("session_id") == thread_id for event in scoped_events), "Canonical thread mismatch"
        assert all(
            event.get("tenant_id") == bot.config.tenant_id for event in scoped_events
        ), "Canonical tenant mismatch"
        assert sum(event["type"] == "JOB_COMPLETED" for event in scoped_events) == 2
        assert len(context_calls) == 2
        assert snapshot_sessions == [thread_id, thread_id]
        from dadbot.ui import interaction_controller
        def forbidden_standalone():
            raise AssertionError("Voice navigation built a second runtime")
        interaction_controller._standalone_runtime = forbidden_standalone
        strict_mode = bot.config.strict_graph_mode
        navigation = next(radio for radio in app.radio if radio.label == "Navigation")
        navigation.set_value("Smart Home").run(timeout=45)
        assert not app.exception, list(app.exception)
        rendered_copy = "\\n".join(str(item.value) for item in app.markdown)
        assert "Home Assistant MCP controls with an explicit tool allowlist." in rendered_copy
        assert "MQTT controls" not in rendered_copy
        navigation = next(radio for radio in app.radio if radio.label == "Navigation")
        navigation.set_value("Voice").run(timeout=45)
        assert not app.exception, list(app.exception)
        assert any(item.label == "Save an oral history" for item in app.expander)
        assert bot.config.strict_graph_mode == strict_mode
        bot.shutdown()
        """
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("DADBOT_")}
    env["PYTHONIOENCODING"] = "utf-8"
    env["DADBOT_TENANT_ID"] = "tenant-streamlit-smoke"
    if durable:
        env["DADBOT_DURABLE_EXECUTION_LEDGER"] = "1"
        env["DADBOT_EXECUTION_LEDGER_BACKEND"] = "file"
        env["DADBOT_MEMORY_INDEX_BACKEND"] = "sqlite"
        env["DADBOT_SESSION_LOG_DIR"] = str(sandbox / "wal")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=sandbox,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        env=env,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "coerced malformed memory_snapshot" not in result.stderr, result.stderr


def test_source_mode_approval_refusal_and_thread_isolation(tmp_path):
    from dadbot.core.dadbot import DadBot

    repo_root = Path(__file__).resolve().parents[1]
    sandbox = tmp_path / "source_app"
    sandbox.mkdir()
    _prepare_streamlit_sandbox(repo_root, sandbox)
    (sandbox / "dad_profile.json").write_text(
        json.dumps(DadBot.default_profile()), encoding="utf-8",
    )
    code = textwrap.dedent(
        """
        from datetime import date
        from streamlit.testing.v1 import AppTest
        from dadbot.services.source_grounding import SOURCE_REFUSAL

        app = AppTest.from_file("dad_streamlit.py").run(timeout=45)
        assert not app.exception, list(app.exception)
        bot = app.session_state["bot"]
        bot.PROFILE = {**bot.PROFILE, "facts": {"observatory": "The observatory opens at noon."}}
        bot.mutate_memory_store(memories=[{
            "summary": "The telescope is on the oak shelf.",
            "category": "home", "pinned": True,
            "created_at": date.today().isoformat(),
            "updated_at": date.today().isoformat(),
        }])
        calls = []
        def forbidden(*args, **kwargs):
            calls.append("called")
            raise AssertionError("Source mode called a model or embedding transport")
        bot.model_port.generate = forbidden
        bot.model_port.generate_async = forbidden
        bot.embed_texts = forbidden
        def checkbox(label):
            return next(item for item in app.checkbox if item.label == label)
        def button(label):
            return next(item for item in app.button if item.label == label)
        def send(text):
            app.chat_input[0].set_value(text).run(timeout=45)
            assert not app.exception, list(app.exception)
            assert not app.session_state["message_outbox"]
            return app.session_state["chat_history"][-1]["text"]
        checkbox("Jor-El source mode").check().run(timeout=45)
        assert checkbox("Auto-speak replies").disabled
        assert send("telescope") == SOURCE_REFUSAL
        button("Approve for citations").click().run(timeout=45)
        assert "oak shelf" in send("telescope")
        button("Revoke citation approval").click().run(timeout=45)
        assert send("telescope") == SOURCE_REFUSAL
        assert "50% (partial match)" in send("observatory volcanoes")
        grounding = bot.services.registry.get("source_grounding_service")
        def forbidden_maintenance(*args, **kwargs):
            raise AssertionError("Source mode ran proactive maintenance")
        maintenance = bot.services.registry.get("maintenance_service").maintenance
        maintenance.run_scheduled_proactive_jobs = forbidden_maintenance
        answer = grounding.answer
        def unavailable(*args, **kwargs):
            raise OSError("Temporary synthetic evidence-store outage")
        grounding.answer = unavailable
        app.chat_input[0].set_value("observatory").run(timeout=45)
        assert not app.exception, list(app.exception)
        assert len(app.session_state["message_outbox"]) == 1
        assert app.session_state["message_outbox"][0]["source_mode"] is True
        checkbox("Jor-El source mode").uncheck().run(timeout=45)
        next(item for item in app.radio if item.label == "Navigation").set_value("Workshop").run(timeout=45)
        button("Retry queued").click().run(timeout=45)
        assert not app.exception, list(app.exception)
        assert len(app.session_state["message_outbox"]) == 1
        assert app.session_state["message_outbox"][0]["source_mode"] is True
        grounding.answer = answer
        button("Retry queued").click().run(timeout=45)
        assert not app.exception, list(app.exception)
        assert not app.session_state["message_outbox"]
        assert "The observatory opens at noon." in app.session_state["chat_history"][-1]["text"]
        next(item for item in app.radio if item.label == "Navigation").set_value("Chat").run(timeout=45)
        checkbox("Jor-El source mode").check().run(timeout=45)
        button("New chat").click().run(timeout=45)
        assert not checkbox("Jor-El source mode").value
        assert not app.session_state["chat_history"]
        next(item for item in app.selectbox if item.label == "Active conversation").select("default").run(timeout=45)
        assert checkbox("Jor-El source mode").value
        assert app.session_state["chat_history"]
        assert calls == [], calls
        bot.shutdown()
        """
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("DADBOT_")}
    env["PYTHONIOENCODING"] = "utf-8"
    env["DADBOT_TENANT_ID"] = "tenant-streamlit-smoke"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=sandbox, capture_output=True,
        text=True, encoding="utf-8", timeout=180, env=env, check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
