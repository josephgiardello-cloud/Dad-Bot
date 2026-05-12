from __future__ import annotations

import pytest

from dadbot.core.graph import TurnContext, TurnGraph
from dadbot.core.nodes import TemporalNode
from dadbot.core.runtime_errors import InvariantViolation

pytestmark = pytest.mark.unit


class _MutateTraceIdNode:
    async def run(self, context: TurnContext) -> TurnContext:
        context.trace_id = "mutated-trace-id"
        return context


class _ReplaceManifestNode:
    async def run(self, context: TurnContext) -> TurnContext:
        context.determinism_manifest = {"replaced": True}
        return context


class _SaveNode:
    async def run(self, context: TurnContext) -> TurnContext:
        context.state["safe_result"] = ("done", False)
        return context


def _graph_with(node) -> TurnGraph:
    graph = TurnGraph(registry={})
    graph.add_node("temporal", TemporalNode())
    graph.add_node("inference", node)
    graph.add_node("save", _SaveNode())
    graph.set_edge("temporal", "inference")
    graph.set_edge("inference", "save")
    return graph


@pytest.mark.asyncio
async def test_identity_tuple_blocks_trace_id_mutation() -> None:
    graph = _graph_with(_MutateTraceIdNode())
    context = TurnContext(user_input="hi", trace_id="stable-trace", kernel_step_id="stable-step")

    with pytest.raises(InvariantViolation):
        await graph.execute(context)


@pytest.mark.asyncio
async def test_identity_tuple_blocks_manifest_replacement() -> None:
    graph = _graph_with(_ReplaceManifestNode())
    context = TurnContext(user_input="hi", trace_id="stable-trace", kernel_step_id="stable-step")

    with pytest.raises(InvariantViolation):
        await graph.execute(context)
