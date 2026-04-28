from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from dadbot.core.execution_boundary import (
    MemoryWriteSurfaceViolation,
    ModelGatewayViolation,
    RuntimeExecutionViolation,
)
from dadbot.runtime import launcher


@pytest.mark.phase4
def test_alternate_runtime_launcher_fails_fast() -> None:
    args = SimpleNamespace(
        no_signoff=False,
        light=False,
        tenant_id="",
        init_profile=False,
        stop_streamlit=False,
        serve_api=False,
        clear_memory=False,
        export_memory=None,
        cli=False,
        disable_service_client=False,
        model=None,
        service_url="",
    )

    with pytest.raises(RuntimeExecutionViolation):
        launcher.dispatch_runtime_mode(
            args,
            dadbot_cls=object,
            ensure_streamlit_app_file=lambda _path: False,
            build_service_state_store=lambda _config: None,
        )


@pytest.mark.phase4
def test_runtime_client_rejects_spoofed_modelport(bot) -> None:
    with pytest.raises(ModelGatewayViolation):
        bot.runtime_client.call_llm(
            [{"role": "user", "content": "hello"}],
            caller="ModelPort",
        )


@pytest.mark.phase4
def test_memory_storage_rejects_spoofed_owner(bot) -> None:
    with pytest.raises(MemoryWriteSurfaceViolation):
        bot.memory._storage.mutate_memory_store(owner="MemoryManager", memories=[])


@pytest.mark.phase4
def test_terminal_state_persisted_for_turn(bot, monkeypatch) -> None:
    orchestrator = bot.turn_orchestrator

    async def _fake_execute(context, audit_mode=False):
        context.state["memory_structured"] = {"pref": "coffee"}
        context.state["memory_full_history_id"] = "history-1"
        context.state["memory_retrieval_set"] = [{"memory_id": "m1", "score": 0.9}]
        context.state["capability_audit_report"] = {"allowed": True}
        context.state["safety_check_result"] = {"safe": True, "violations": []}
        context.state["tony_level"] = "steady"
        context.state["tony_score"] = 55
        context.metadata["kernel_policy"] = {"enforced": True, "step": "inference"}
        context.metadata["tool_execution_graph_hash"] = "tool-hash"
        return ("kernel closed", False)

    monkeypatch.setattr(orchestrator.graph, "execute", _fake_execute)

    asyncio.run(orchestrator.handle_turn("hello dad"))

    session = orchestrator.session_registry.get_or_create("default")
    session_state = dict(session.get("state") or {})
    terminal_state = dict(session_state.get("last_terminal_state") or {})
    trace_context = dict(session_state.get("last_execution_trace_context") or {})

    required_keys = {
        "schema_version",
        "final_output",
        "final_memory_view",
        "final_trace_hash",
        "execution_dag_hash",
        "policy_snapshot",
        "model_output_hashes",
        "memory_retrieval_hash",
        "policy_hash",
        "determinism_closure_hash",
    }
    assert required_keys.issubset(set(terminal_state.keys()))
    assert terminal_state["final_trace_hash"] == str(trace_context.get("final_hash") or "")
    assert terminal_state["final_memory_view"]["memory_full_history_id"] == "history-1"
