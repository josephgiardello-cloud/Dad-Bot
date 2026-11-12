import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from test_streamlit_smoke import _prepare_streamlit_sandbox


@pytest.mark.parametrize("approved,outcome", [(True, "succeeded"), (False, "rejected"), (True, "uncertain")])
def test_streamlit_mcp_decision_has_canonical_history_job_and_checkpoint(tmp_path, approved, outcome):
    root = Path(__file__).resolve().parents[1]
    sandbox = tmp_path / "mcp_app"
    sandbox.mkdir()
    _prepare_streamlit_sandbox(root, sandbox)
    shutil.copy2(sandbox / "dad_profile.template.json", sandbox / "dad_profile.json")
    code = textwrap.dedent(
        """
        import asyncio
        import json
        import os
        from contextlib import asynccontextmanager
        from streamlit.testing.v1 import AppTest
        from dadbot.core.graph_pipeline_nodes import PlannerNode
        from dadbot.core.graph_context import TurnContext
        from dadbot.core.orchestrator import DadBotOrchestrator
        from dadbot.core.tool_recording import ToolIOLedger
        from dadbot.runtime.mcp import approval_service

        approved = os.environ["TEST_APPROVED"] == "1"
        outcome = os.environ["TEST_OUTCOME"]
        app = AppTest.from_file("dad_streamlit.py").run(timeout=60)
        assert not app.exception, list(app.exception)
        app.session_state["auto_speak_replies"] = False
        next(button for button in app.button if button.label == "New chat").click().run(timeout=60)
        bot = app.session_state["bot"]
        thread_id = app.session_state["active_thread_id"]
        service = bot.services.registry.get("mcp_tool_service")
        contexts = []
        build_context = bot.services.registry.get("context_service").build_context
        def capture_context(context):
            contexts.append(context)
            return build_context(context)
        bot.services.registry.get("context_service").build_context = capture_context
        safety_calls = []
        safety = bot.services.registry.get("safety_service")
        enforce = safety.enforce_policies
        def capture_safety(*args, **kwargs):
            safety_calls.append(True)
            return enforce(*args, **kwargs)
        safety.enforce_policies = capture_safety
        actions = []
        connections = []
        class Client:
            @asynccontextmanager
            async def connect(self):
                connections.append(True)
                yield object()
            async def list_allowed_tools(self, session, allowed):
                return [{
                    "name": "light_control", "input_schema": {
                        "type": "object", "properties": {"entity_id": {"type": "string"}},
                        "required": ["entity_id"], "additionalProperties": False,
                    },
                }]
            async def call_tool(self, session, name, arguments):
                actions.append((name, arguments))
                if outcome == "uncertain":
                    raise TimeoutError("Synthetic remote timeout")
                return {"is_error": False, "content": "Synthetic light is off."}
        approval_service.MCPToolClient = lambda *args, **kwargs: Client()
        outputs = iter([
            json.dumps({
                "should_continue": True, "action_input": "Review this action.",
                "reasoning": "User explicitly requested light_control", "confidence": 1.0,
                "tool_calls": [{"name": "light_control", "arguments": {"entity_id": "light.kitchen"}}],
            }),
            json.dumps({
                "should_continue": False, "action_input": "The approved action completed.",
                "reasoning": "Confirmed result", "confidence": 1.0, "tool_calls": [],
            }),
        ])
        service._model_call = lambda *args, **kwargs: next(outputs)
        app.chat_input[0].set_value("Use light_control to turn off light.kitchen.").run(timeout=60)
        assert not app.exception, list(app.exception)
        assert not app.session_state["message_outbox"]
        pending = service.get_pending_mcp_approval(thread_id)
        assert pending is not None
        parent = service._pending[thread_id][3]
        assert len(contexts) == 1
        assert not actions
        source = next(item for item in app.checkbox if item.label == "Jor-El source mode")
        source.check().run(timeout=60)
        assert next(button for button in app.button if button.label == "Approve this action").disabled
        assert next(button for button in app.button if button.label == "Reject this action").disabled
        assert not actions
        next(item for item in app.checkbox if item.label == "Jor-El source mode").uncheck().run(timeout=60)
        def forbidden(*args, **kwargs):
            raise AssertionError("Approval continuation ran fresh planning or ordinary inference")
        PlannerNode.run = forbidden
        bot.model_port.generate_async = forbidden
        before_safety = len(safety_calls)
        label = "Approve this action" if approved else "Reject this action"
        next(button for button in app.button if button.label == label).click().run(timeout=60)
        assert not app.exception, list(app.exception)
        assert not app.error, list(app.error)
        assert len(contexts) == 2
        assert len(safety_calls) > before_safety
        decision = contexts[-1]
        assert decision.trace_id != parent.trace_id
        assert decision.metadata["mcp_approval_parent_trace_id"] == parent.trace_id
        assert decision.metadata["mcp_approval_resolution"] == {
            "request_id": pending.request_id, "approved": approved,
        }
        assert decision.state["memory_snapshot"]["memory_full_history_id"]
        assert decision.fidelity.full_pipeline
        assert decision.state["tool_ir"] == {"requests": []}
        history = app.session_state["chat_threads"][thread_id]
        assert len(history) == 4, [(item["role"], item["text"]) for item in history]
        assert history[-2]["role"] == "user"
        assert history[-2]["text"] == service.decision_text(approved=approved)
        assert history[-1]["role"] == "assistant"
        canonical = bot.prompt_history()
        assert any(item.get("content") == history[-2]["text"] for item in canonical), canonical
        events = bot._execution_ledger.read()
        assert sum(event["type"] == "JOB_COMPLETED" for event in events) == 2
        scoped = [event for event in events if event.get("trace_id") == decision.trace_id]
        assert {"JOB_STARTED", "JOB_COMPLETED", "PolicyTraceEvent", "GRAPH_CHECKPOINT"} <= {
            event["type"] for event in scoped
        }
        assert all(
            event["tenant_id"] == bot.config.tenant_id and event["session_id"] == thread_id for event in scoped
        )
        checkpoints = [
            event for event in bot._execution_ledger.read_checkpoints(payload_trace_id=decision.trace_id)
            if event["tenant_id"] == bot.config.tenant_id and event["session_id"] == thread_id
        ]
        checkpoint = checkpoints[-1]["payload"]["checkpoint"]
        loaded = bot.services.registry.get("persistence_service").persistence_manager.load_latest_graph_checkpoint(
            trace_token=decision.trace_id, session_id=thread_id, tenant_id=bot.config.tenant_id,
        )
        assert loaded == checkpoint
        assert (
            checkpoint["execution_state"]["metadata"]["mcp_approval_resolution"]
            == decision.metadata["mcp_approval_resolution"]
        )
        if approved:
            assert len(actions) == 1
            io = ToolIOLedger.from_checkpoint(checkpoint["tool_io_ledger"])
            assert io.records[0].status == outcome
            if outcome == "succeeded":
                assert io.records[0].output_payload["content"] == "Synthetic light is off."
            else:
                assert "uncertain" in history[-1]["text"]
            assert io.records[0].metadata["scope"]["trace_id"] == decision.trace_id
            restored = TurnContext(user_input=decision.user_input, trace_id=decision.trace_id)
            DadBotOrchestrator._hydrate_context_from_checkpoint(restored, checkpoint)
            assert restored._tool_io_ledger.to_dict() == io.to_dict()
        else:
            assert not actions
            assert "tool_io_ledger" not in checkpoint
        reloaded = AppTest.from_file("dad_streamlit.py")
        reloaded.query_params["thread"] = thread_id
        reloaded.run(timeout=60)
        assert not reloaded.exception, list(reloaded.exception)
        assert reloaded.session_state["chat_history"] == history
        assert reloaded.session_state["bot"] is bot
        assert service.get_pending_mcp_approval(thread_id) is None
        before_connections = len(connections)
        asyncio.run(service.resolve_mcp_approval(thread_id, approved=approved))
        assert len(connections) == before_connections
        assert len(actions) == int(approved)
        if outcome == "uncertain":
            history_before_review = list(history)
            next(item for item in reloaded.radio if item.label == "Navigation").set_value("Workshop").run(timeout=60)
            assert not reloaded.exception, list(reloaded.exception)
            review_label = "I verified no action happened"
            assert next(button for button in reloaded.button if button.label == review_label).disabled
            source_modes = dict(reloaded.session_state["thread_source_modes"])
            reloaded.session_state["thread_source_modes"] = {**source_modes, thread_id: True}
            reloaded.run(timeout=60)
            assert not reloaded.exception, list(reloaded.exception)
            assert next(
                checkbox for checkbox in reloaded.checkbox
                if checkbox.label == "I checked this exact action in Home Assistant"
            ).disabled
            assert next(button for button in reloaded.button if button.label == review_label).disabled
            assert next(
                button for button in reloaded.button if button.label == "I verified the action happened"
            ).disabled
            reloaded.session_state["thread_source_modes"] = source_modes
            reloaded.run(timeout=60)
            next(
                checkbox for checkbox in reloaded.checkbox
                if checkbox.label == "I checked this exact action in Home Assistant"
            ).check().run(timeout=60)
            assert not next(button for button in reloaded.button if button.label == review_label).disabled
            connections_before_review = len(connections)
            safety_before_review = len(safety_calls)
            journal = service._approval_journal()
            write_event = journal.writer.write_event
            def unavailable(event_type, **kwargs):
                if event_type == "TOOL_ACTION_REVIEWED":
                    raise OSError("Synthetic action-review audit outage")
                return write_event(event_type, **kwargs)
            journal.writer.write_event = unavailable
            next(button for button in reloaded.button if button.label == review_label).click().run(timeout=60)
            assert not reloaded.exception, list(reloaded.exception)
            assert any("could not finish" in error.value for error in reloaded.error)
            assert len(connections) == connections_before_review and len(actions) == 1
            assert reloaded.session_state["chat_threads"][thread_id] == history_before_review
            assert service.get_mcp_action_history(thread_id)[0].review_outcome == ""
            failed_review = contexts[-1]
            journal.writer.write_event = write_event
            next(button for button in reloaded.button if button.label == review_label).click().run(timeout=60)
            assert not reloaded.exception, list(reloaded.exception)
            assert not reloaded.error, list(reloaded.error)
            assert len(connections) == connections_before_review and len(actions) == 1
            assert len(safety_calls) > safety_before_review
            review = contexts[-1]
            assert review.metadata["request_id"] != failed_review.metadata["request_id"]
            assert review.fidelity.full_pipeline
            assert review.state["decision_outcome"] == "no_tool_needed"
            assert review.state["tool_ir"] == {"requests": []}
            reviewed_history = reloaded.session_state["chat_threads"][thread_id]
            assert reviewed_history[:4] == history_before_review
            assert len(reviewed_history) == 6
            assert "your confirmation" in reviewed_history[-1]["text"]
            assert "fresh approval" in reviewed_history[-1]["text"]
            assert "I checked Home Assistant" in reviewed_history[-2]["text"]
            events_after_review = bot._execution_ledger.read()
            review_events = [event for event in events_after_review if event["type"] == "TOOL_ACTION_REVIEWED"]
            assert len(review_events) == 1
            assert review_events[0]["trace_id"] == review.trace_id
            assert review_events[0]["session_id"] == thread_id
            assert review_events[0]["tenant_id"] == bot.config.tenant_id
            assert review_events[0]["payload"]["review_source"] == "user_confirmation"
            assert sum(event["type"] == "JOB_COMPLETED" for event in events_after_review) == 3
            review_checkpoint = next(
                event["payload"]["checkpoint"] for event in reversed(
                    bot._execution_ledger.read_checkpoints(payload_trace_id=review.trace_id)
                )
                if event["tenant_id"] == bot.config.tenant_id and event["session_id"] == thread_id
            )
            assert "tool_io_ledger" not in review_checkpoint
            assert "user_confirmation" not in str(review_checkpoint.get("tool_io_ledger"))
            restored_service = approval_service.MCPApprovalService(bot)
            assert restored_service.get_mcp_action_history(thread_id)[0].review_outcome == "not_performed"
            assert restored_service.get_mcp_action_history(thread_id)[0].status == "uncertain"
            reloaded.run(timeout=60)
            assert not reloaded.exception, list(reloaded.exception)
            assert not any(button.label == review_label for button in reloaded.button)
            next(item for item in reloaded.radio if item.label == "Navigation").set_value("Chat").run(timeout=60)
            assert reloaded.session_state["chat_history"] == reviewed_history
        bot.shutdown()
        """
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("DADBOT_")}
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "DADBOT_TENANT_ID": "tenant-streamlit-mcp",
        "DADBOT_DURABLE_EXECUTION_LEDGER": "1",
        "DADBOT_EXECUTION_LEDGER_BACKEND": "file",
        "DADBOT_MEMORY_INDEX_BACKEND": "sqlite",
        "DADBOT_SESSION_LOG_DIR": str(sandbox / "wal"),
        "DADBOT_HA_MCP_URL": "http://127.0.0.1:9999/mcp",
        "DADBOT_HA_MCP_ALLOWED_TOOLS": "light_control",
        "TEST_APPROVED": "1" if approved else "0",
        "TEST_OUTCOME": outcome,
    })
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=sandbox, capture_output=True,
        text=True, encoding="utf-8", timeout=240, env=env, check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "State divergence detected" not in result.stderr, result.stderr
