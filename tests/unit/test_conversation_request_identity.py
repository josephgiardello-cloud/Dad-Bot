import asyncio
from types import SimpleNamespace

import pytest

from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
from dadbot.core.execution_contract import ExecutionMode, SovereignContext, TurnDelivery, live_turn_request
from dadbot.core.kernel_signals import TracingContext
from dadbot.core.turn_mixin import DadBotTurnMixin

pytestmark = pytest.mark.unit


def _runtime(plane, monkeypatch):
    runtime = DadBotTurnMixin()
    for name, value in {
        "_execute_turn_session_id": "chat",
        "_execute_turn_request_metadata": {},
        "_get_turn_orchestrator": lambda: SimpleNamespace(control_plane=plane),
        "_get_policy_store": lambda: None,
        "_get_prompt_builder": lambda: None,
        "_get_memory_ledger": lambda: None,
        "_get_relationship_snapshotter": lambda: None,
        "_get_world_model_store": lambda: None,
    }.items():
        monkeypatch.setattr(runtime, name, value, raising=False)
    return runtime


def test_repeated_wording_is_a_new_request_but_retry_confluence_is_enforced(monkeypatch):
    async def run():
        calls = []

        async def execute(session, job):
            calls.append(job)
            variant = job.metadata["variant"]
            session.setdefault("state", {})["last_terminal_state"] = {
                "execution_dag_hash": "dag",
                "policy_hash": "policy",
                "post_commit_mutation_effects_hash": variant,
                "determinism_closure_hash": "closure",
            }
            return f"reply-{variant}", False

        plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=execute)
        runtime = _runtime(plane, monkeypatch)
        runtime._execute_turn_request_metadata = {"tenant_id": "tenant-a", "request_id": "first", "variant": "a"}
        assert await DadBotTurnMixin._run_graph_turn_async(runtime, "How are you?") == ("reply-a", False)
        runtime._execute_turn_request_metadata = {"tenant_id": "tenant-a", "request_id": "second", "variant": "b"}
        assert await DadBotTurnMixin._run_graph_turn_async(runtime, "How are you?") == ("reply-b", False)
        assert calls[0].metadata["confluence_key"] != calls[1].metadata["confluence_key"]

        runtime._execute_turn_request_metadata = {"tenant_id": "tenant-a", "request_id": "second", "variant": "c"}
        assert await DadBotTurnMixin._run_graph_turn_async(runtime, "How are you?") == ("reply-b", False)
        assert len(calls) == 2

        with pytest.raises(RuntimeError, match="confluence"):
            await plane.submit_turn(
                session_id="replay-session",
                user_input="How are you?",
                metadata={"confluence_key": calls[1].metadata["confluence_key"], "variant": "c"},
            )
        assert calls[-1].metadata["confluence_key"] == calls[1].metadata["confluence_key"]

    asyncio.run(run())


def test_request_factory_allocates_identity_once_and_preserves_explicit_retry_identity():
    first = live_turn_request("hello")
    second = live_turn_request("hello")
    assert first.context is not None
    assert second.context is not None
    assert first.context.request_id
    assert first.context.request_id != second.context.request_id
    retry = live_turn_request("hello", context=first.context)
    assert retry.context is not None
    assert retry.context.request_id == first.context.request_id


@pytest.mark.parametrize("changed", [
    {"tenant_id": "tenant-b"}, {"session_id": "other"},
    {"attachments": [{"type": "image", "path": "different"}]},
])
def test_request_confluence_preserves_tenant_session_and_attachment_isolation(changed):
    params = {
        "tenant_id": "tenant-a", "session_id": "chat", "request_id": "request",
        "user_input": "hello", "attachments": [],
    }
    original = DadBotTurnMixin._derive_confluence_key(**params)
    assert original != DadBotTurnMixin._derive_confluence_key(**(params | changed))

@pytest.mark.parametrize("request_id", [None, "", " "])
def test_direct_graph_turns_without_identity_get_distinct_request_identities(monkeypatch, request_id):
    async def run():
        keys = []

        async def submit(**kwargs):
            keys.append(kwargs["metadata"]["confluence_key"])
            return "ok", False

        runtime = _runtime(SimpleNamespace(submit_turn=submit), monkeypatch)
        runtime._execute_turn_request_metadata = {"request_id": request_id}
        await DadBotTurnMixin._run_graph_turn_async(runtime, "hello")
        await DadBotTurnMixin._run_graph_turn_async(runtime, "hello")
        assert len(set(keys)) == 2

    asyncio.run(run())


def test_new_turn_identity_is_not_reused_from_an_enclosing_trace(monkeypatch):
    runtime = DadBotTurnMixin()
    monkeypatch.setattr(TracingContext, "current_trace_id", lambda: "enclosing-trace")
    first = runtime._build_sovereign_context(mode=ExecutionMode.LIVE)
    second = runtime._build_sovereign_context(mode=ExecutionMode.LIVE)
    assert first.trace_id == second.trace_id == "enclosing-trace"
    assert first.request_id != second.request_id


def test_new_turn_context_uses_active_tenant_and_thread_instead_of_dataclass_defaults(monkeypatch):
    runtime = DadBotTurnMixin()
    monkeypatch.setattr(runtime, "tenant_id", "tenant-a", raising=False)
    monkeypatch.setattr(runtime, "active_thread_id", "thread-selected", raising=False)
    context = runtime._build_sovereign_context(mode=ExecutionMode.LIVE)
    assert context.tenant_id == "tenant-a"
    assert context.session_id == "thread-selected"


def test_explicit_default_context_is_not_rebound_to_the_active_thread(monkeypatch):
    runtime = DadBotTurnMixin()
    monkeypatch.setattr(runtime, "tenant_id", "tenant-a", raising=False)
    monkeypatch.setattr(runtime, "active_thread_id", "thread-selected", raising=False)
    context = runtime._build_sovereign_context(mode=ExecutionMode.LIVE, base_context=SovereignContext())
    assert context.tenant_id == "default"
    assert context.session_id == "default"


@pytest.mark.parametrize("delivery", [TurnDelivery.SYNC, TurnDelivery.ASYNC])
def test_live_turn_propagates_trusted_request_identity(delivery, caplog, monkeypatch):
    observed = []

    def record():
        observed.append(dict(runtime._execute_turn_request_metadata))
        return "ok", False

    async def run_async(*_args, **_kwargs):
        return record()

    runtime = DadBotTurnMixin()
    for name, value in {
        "_execute_turn_session_id": "",
        "_execute_turn_request_metadata": {},
        "_begin_turn_event_run": lambda: "",
        "_emit_turn_event": lambda *_args, **_kwargs: None,
        "_run_graph_turn_sync": lambda *_args, **_kwargs: record(),
        "_run_graph_turn_async": run_async,
        "_checkpoint_turn_state": lambda: None,
    }.items():
        monkeypatch.setattr(runtime, name, value, raising=False)
    context = SovereignContext(tenant_id="tenant-a", session_id="chat", trace_id="trace-a", request_id="request-a")
    request = live_turn_request("private user text", context=context, delivery=delivery)
    with caplog.at_level("INFO", logger="dadbot.core.turn_mixin"):
        response = asyncio.run(DadBotTurnMixin._execute_live_turn_async(runtime, request))
    assert response.reply == "ok"
    assert observed[0]["request_id"] == "request-a"
    assert "seconds=" in caplog.text
    assert "private user text" not in caplog.text
    assert runtime._execute_turn_request_metadata == {}
