import asyncio
from types import SimpleNamespace

import pytest

from dadbot.core.graph import TurnContext
from dadbot.core.graph_pipeline_nodes import InferenceNode

pytestmark = pytest.mark.unit


class _Registry:
    def __init__(self, service: object) -> None:
        self._service = service

    def get(self, name: str) -> object:
        assert name == "agent_service"
        return self._service


class _ControlPlaneStub:
    def __init__(self) -> None:
        self.calls = 0

    async def execute_from_graph_context(self, _turn_context, _rich_context):
        self.calls += 1
        return ("control-plane candidate", False)


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

    with pytest.raises(RuntimeError, match="legacy agent_service.run_agent fallback is disabled"):
        asyncio.run(_make_node().execute(registry, context))


def test_pipeline_inference_rejects_async_run_agent_only_service() -> None:
    async def _run_agent(_context: TurnContext, _rich: dict) -> tuple[str, bool]:
        return ("async candidate", False)

    registry = _Registry(SimpleNamespace(run_agent=_run_agent))
    context = TurnContext(user_input="async inference")

    with pytest.raises(RuntimeError, match="legacy agent_service.run_agent fallback is disabled"):
        asyncio.run(_make_node().execute(registry, context))


def test_pipeline_inference_prefers_control_plane_helper() -> None:
    control_plane = _ControlPlaneStub()

    def _run_agent(_context: TurnContext, _rich: dict) -> tuple[str, bool]:
        return ("legacy candidate", False)

    registry = _Registry(SimpleNamespace(bot=_BotStub(control_plane), run_agent=_run_agent))
    context = TurnContext(user_input="control plane inference")

    asyncio.run(_make_node().execute(registry, context))

    assert context.state["candidate"] == ("control-plane candidate", False)
    assert control_plane.calls == 1
