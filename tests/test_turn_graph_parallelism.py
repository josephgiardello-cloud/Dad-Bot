import asyncio
import time
from types import SimpleNamespace

from dadbot.core.graph import TurnContext, TurnGraph
from dadbot.core.nodes import TemporalNode


class SlowNode:
    def __init__(self, key, value, delay):
        self.key = key
        self.value = value
        self.delay = delay

    async def run(self, context):
        await asyncio.sleep(self.delay)
        context.state[self.key] = self.value
        return context


class FinalNode:
    async def run(self, context):
        context.state["safe_result"] = (
            f"{context.state.get('health')}|{context.state.get('rich_context')}",
            False,
        )
        return context


def test_turn_graph_parallel_stage_merges_results_and_runs_concurrently():
    graph = TurnGraph(registry=None)
    graph.add_node("temporal", TemporalNode())
    graph.add_node(
        "fanout",
        (
            SlowNode("health", "ok", 0.05),
            SlowNode("rich_context", {"memory": "loaded"}, 0.05),
        ),
    )
    graph.add_node("save", FinalNode())
    graph.set_edge("temporal", "fanout")
    graph.set_edge("fanout", "save")

    started = time.perf_counter()
    result = asyncio.run(graph.execute(TurnContext(user_input="hello")))
    elapsed = time.perf_counter() - started

    assert result == ("ok|{'memory': 'loaded'}", False)
    assert elapsed < 0.09


def test_turn_graph_emits_before_and_after_checkpoints():
    recorded = []

    class Registry(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    persistence = SimpleNamespace(save_graph_checkpoint=lambda payload, **_kw: recorded.append(payload))
    graph = TurnGraph(registry=Registry({"persistence_service": persistence}))
    graph.add_node("temporal", TemporalNode())
    graph.add_node("fanout", SlowNode("health", "ok", 0.0))
    graph.add_node("save", FinalNode())
    graph.set_edge("temporal", "fanout")
    graph.set_edge("fanout", "save")

    result = asyncio.run(graph.execute(TurnContext(user_input="hello")))

    assert result == ("ok|None", False)
    assert [item["status"] for item in recorded] == [
        "before",
        "after",
        "edge",
        "before",
        "after",
        "edge",
        "before",
        "after",
    ]
    assert [item["stage"] for item in recorded] == [
        "temporal",
        "temporal",
        "temporal\u2192fanout",
        "fanout",
        "fanout",
        "fanout\u2192save",
        "save",
        "save",
    ]
    assert recorded[-1]["state"]["safe_result"] == ["ok|None", False]
    assert all(item["trace_id"] == recorded[0]["trace_id"] for item in recorded)
