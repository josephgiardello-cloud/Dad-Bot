import asyncio
from types import SimpleNamespace

import pytest

from dadbot.core.core_state import CoreState
from dadbot.core.execution_context import (
    close_core_state_scope,
    get_active_core_state,
    open_core_state_scope,
    push_core_state_event,
)
from dadbot.core.graph import TurnContext, TurnGraph
from dadbot.core.graph_pipeline_nodes import ControlPlaneNode, InferenceNode, MCPToolNode

pytestmark = pytest.mark.unit


class _Registry:
    def __init__(self, service: object) -> None:
        self._services = {"agent_service": service}

    def get(self, name: str) -> object:
        return self._services[name]


class _ControlPlaneStub:
    def __init__(self) -> None:
        self.calls = 0

    async def execute_from_graph_context(self, _turn_context, _rich_context):
        self.calls += 1
        return ("control-plane output", False)


class _OrchestratorStub:
    def __init__(self, control_plane: _ControlPlaneStub) -> None:
        self.control_plane = control_plane


class _BotStub:
    def __init__(self, control_plane: _ControlPlaneStub) -> None:
        self.turn_orchestrator = _OrchestratorStub(control_plane)


def _make_node() -> InferenceNode:
    node = InferenceNode()
    node._run_critique_check = lambda *_args, **_kwargs: True
    return node


def test_pipeline_inference_rejects_sync_run_agent_only_service() -> None:
    def _run_agent(_context: TurnContext, _rich: dict) -> tuple[str, bool]:
        return ("sync candidate", False)

    registry = _Registry(SimpleNamespace(run_agent=_run_agent))
    context = TurnContext(user_input="sync inference")

    with pytest.raises(RuntimeError, match="agent_service.generate_candidate"):
        asyncio.run(_make_node().execute(registry, context))


def test_pipeline_inference_rejects_async_run_agent_only_service() -> None:
    async def _run_agent(_context: TurnContext, _rich: dict) -> tuple[str, bool]:
        return ("async candidate", False)

    registry = _Registry(SimpleNamespace(run_agent=_run_agent))
    context = TurnContext(user_input="async inference")

    with pytest.raises(RuntimeError, match="agent_service.generate_candidate"):
        asyncio.run(_make_node().execute(registry, context))


def test_pipeline_inference_uses_registered_candidate_service_only() -> None:
    control_plane = _ControlPlaneStub()

    def _run_agent(_context: TurnContext, _rich: dict) -> tuple[str, bool]:
        return ("legacy candidate", False)

    async def _generate_candidate(_context: TurnContext, _rich: dict) -> tuple[str, bool]:
        return ("inference candidate", False)

    registry = _Registry(
        SimpleNamespace(
            bot=_BotStub(control_plane),
            run_agent=_run_agent,
            generate_candidate=_generate_candidate,
        ),
    )
    context = TurnContext(user_input="control plane inference")

    asyncio.run(_make_node().execute(registry, context))

    assert context.state["candidate"] == ("inference candidate", False)
    assert control_plane.calls == 0


def test_control_plane_node_selects_candidate_before_safety() -> None:
    control_plane = _ControlPlaneStub()
    finalized = []

    async def _generate_candidate(_context: TurnContext, _rich: dict) -> tuple[str, bool]:
        return ("inference candidate", False)

    service = SimpleNamespace(
        bot=SimpleNamespace(
            turn_orchestrator=SimpleNamespace(control_plane=control_plane),
            reply_finalization=SimpleNamespace(
                finalize=lambda reply, _mood, *, user_input: finalized.append(
                    (reply, user_input),
                ) or f"finalized: {reply}",
            ),
        ),
        generate_candidate=_generate_candidate,
    )
    registry = _Registry(service)
    context = TurnContext(user_input="select the response")
    context.state["candidate"] = ("inference candidate", False)

    asyncio.run(ControlPlaneNode().execute(registry, context))

    assert context.state["candidate"] == ("finalized: control-plane output", False)
    assert context.state["response_authority"] == "response_engine"
    assert control_plane.calls == 1
    assert finalized == [("control-plane output", "select the response")]


def test_control_plane_node_propagates_worker_core_state_to_parent_context() -> None:
    reminder = {"id": "reminder-1", "title": "Keep the durable write"}

    def execute(_turn_context, _rich_context):
        push_core_state_event("MemoryUpdated", {"key": "reminders", "value": [reminder]})
        return ("control-plane output", False)

    control_plane = SimpleNamespace(execute_from_graph_context=execute)
    service = SimpleNamespace(
        bot=SimpleNamespace(
            turn_orchestrator=SimpleNamespace(control_plane=control_plane),
            reply_finalization=SimpleNamespace(finalize=lambda reply, _mood, *, user_input: reply),
        ),
    )
    registry = _Registry(service)
    context = TurnContext(user_input="create a reminder")

    async def run() -> CoreState | None:
        token = open_core_state_scope(CoreState())
        try:
            await ControlPlaneNode().execute(registry, context)
            return get_active_core_state()
        finally:
            close_core_state_scope(token)

    active = asyncio.run(run())

    assert active is not None
    assert active.memory_kv["reminders"] == [reminder]


def test_default_turn_graph_places_control_plane_before_safety() -> None:
    stages = [node.name for node in TurnGraph().nodes]

    assert (
        stages.index("mcp_tools")
        < stages.index("inference")
        < stages.index("control_plane")
        < stages.index("safety")
    )


def test_mcp_tool_stage_uses_registered_service_and_skips_model_inference() -> None:
    async def _generate_candidate(_context: TurnContext, _rich: dict) -> tuple[str, bool]:
        return ("approval candidate", False)

    async def _unexpected_inference(_context: TurnContext, _rich: dict) -> tuple[str, bool]:
        raise AssertionError("MCP candidate should bypass duplicate model inference")

    service = SimpleNamespace(
        is_enabled=True,
        generate_candidate=_generate_candidate,
    )
    registry = _Registry(SimpleNamespace(generate_candidate=_unexpected_inference))
    registry._services["mcp_tool_service"] = service
    context = TurnContext(user_input="request an approved action")
    context.state["rich_context"] = {}

    asyncio.run(MCPToolNode().execute(registry, context))
    asyncio.run(_make_node().execute(registry, context))

    assert context.state["candidate"] == ("approval candidate", False)
    assert "mcp_tool_candidate_ready" not in context.state
