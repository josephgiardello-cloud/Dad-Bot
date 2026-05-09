import asyncio
from types import SimpleNamespace

import pytest

from dadbot.core.graph import TurnContext
from dadbot.core.nodes import ContextBuilderNode

pytestmark = pytest.mark.unit


def test_context_builder_accepts_sync_query_callable():
    manager = SimpleNamespace(query=lambda prompt: [{"id": "m1", "text": prompt}])
    context = TurnContext(user_input="memory ping")

    asyncio.run(ContextBuilderNode(manager).run(context))

    assert context.state["memories"] == [{"id": "m1", "text": "memory ping"}]
    assert context.state["rich_context"]["memories"] == [{"id": "m1", "text": "memory ping"}]


def test_context_builder_accepts_async_query_callable():
    async def _query(prompt: str):
        return [{"id": "m2", "text": prompt.upper()}]

    manager = SimpleNamespace(query=_query)
    context = TurnContext(user_input="memory pong")

    asyncio.run(ContextBuilderNode(manager).run(context))

    assert context.state["memories"] == [{"id": "m2", "text": "MEMORY PONG"}]
    assert context.state["rich_context"]["memories"] == [{"id": "m2", "text": "MEMORY PONG"}]
