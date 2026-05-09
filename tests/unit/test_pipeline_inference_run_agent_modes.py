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


def _make_node() -> InferenceNode:
    node = InferenceNode()
    node._run_critique_check = lambda *_args, **_kwargs: True
    return node


def test_pipeline_inference_accepts_sync_run_agent_callable() -> None:
    def _run_agent(_context: TurnContext, _rich: dict) -> tuple[str, bool]:
        return ("sync candidate", False)

    registry = _Registry(SimpleNamespace(run_agent=_run_agent))
    context = TurnContext(user_input="sync inference")

    asyncio.run(_make_node().execute(registry, context))

    assert context.state["candidate"] == ("sync candidate", False)


def test_pipeline_inference_accepts_async_run_agent_callable() -> None:
    async def _run_agent(_context: TurnContext, _rich: dict) -> tuple[str, bool]:
        return ("async candidate", False)

    registry = _Registry(SimpleNamespace(run_agent=_run_agent))
    context = TurnContext(user_input="async inference")

    asyncio.run(_make_node().execute(registry, context))

    assert context.state["candidate"] == ("async candidate", False)
