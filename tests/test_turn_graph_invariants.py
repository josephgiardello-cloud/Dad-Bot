from __future__ import annotations

import asyncio

import pytest

from dadbot.core.graph import FatalTurnError, MutationIntent, MutationQueue, TurnContext, TurnGraph
from dadbot.core.nodes import TemporalNode


def test_mutation_queue_retains_pending_after_failed_drain() -> None:
    queue = MutationQueue()
    queue.bind_owner("trace-a")
    temporal = {"wall_time": "2026-01-01T00:00:00", "wall_date": "2026-01-01"}
    queue.queue(
        MutationIntent(
            type="memory",
            payload={"op": "save_mood_state", "mood": "neutral", "temporal": temporal},
        )
    )
    queue.queue(
        MutationIntent(
            type="relationship",
            payload={"op": "update", "user_input": "x", "mood": "neutral", "temporal": temporal},
        )
    )

    def _fail_first(_intent: MutationIntent) -> None:
        raise RuntimeError("boom")

    with pytest.raises(FatalTurnError):
        queue.drain(_fail_first, hard_fail_on_error=True)

    # Failed drain must keep intents pending for explicit visibility.
    assert queue.is_empty() is False
    assert queue.size() == 2


def test_mutation_queue_rejects_cross_turn_owner_rebind() -> None:
    queue = MutationQueue()
    queue.bind_owner("trace-a")
    with pytest.raises(RuntimeError, match="cross-turn reuse"):
        queue.bind_owner("trace-b")


def test_turn_graph_raises_when_save_stage_missing() -> None:
    graph = TurnGraph(registry={})
    graph.add_node("temporal", TemporalNode())

    context = TurnContext(user_input="hello")

    with pytest.raises(RuntimeError, match="Structural turn invariant violated: SaveNode did not execute"):
        asyncio.run(graph.execute(context))
