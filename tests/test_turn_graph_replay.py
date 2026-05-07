import asyncio
from types import SimpleNamespace

import pytest

from dadbot.core.execution_trace_context import ExecutionTraceRecorder, RuntimeTraceViolation, bind_execution_trace
from dadbot.core.graph import TurnContext, TurnGraph
from dadbot.core.nodes import TemporalNode


class _PassNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value

    async def run(self, context):
        context.state[self.key] = self.value
        return context


class _SaveNode:
    async def run(self, context):
        context.state["safe_result"] = ("done", False)
        return context


def test_turn_graph_records_deterministic_phase_transitions():
    checkpoints = []
    events = []

    class Registry(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    persistence = SimpleNamespace(
        save_graph_checkpoint=lambda payload, **_kw: checkpoints.append(payload),
        save_turn_event=lambda payload: events.append(payload),
    )
    graph = TurnGraph(registry=Registry({"persistence_service": persistence}))
    graph.add_node("temporal", TemporalNode())
    graph.add_node("preflight", _PassNode("health", "ok"))
    graph.add_node("inference", _PassNode("candidate", "draft"))
    graph.add_node("safety", _PassNode("safe_result", ("safe", False)))
    graph.add_node("reflection", _PassNode("reflection", "ok"))
    graph.add_node("save", _SaveNode())
    graph.set_edge("temporal", "preflight")
    graph.set_edge("preflight", "inference")
    graph.set_edge("inference", "safety")
    graph.set_edge("safety", "reflection")
    graph.set_edge("reflection", "save")

    context = TurnContext(user_input="hello")
    context.metadata["determinism"] = {
        "state_machine": "PLAN_ACT_OBSERVE_RESPOND",
        "enforced": True,
        "lock_hash": "abc123",
        "lock_id": "det-abc123",
    }
    result = asyncio.run(graph.execute(context))

    assert result == ("done", False)
    assert checkpoints
    phase_events = [evt for evt in events if evt.get("event_type") == "phase_transition"]
    assert [evt["transition"]["to"] for evt in phase_events] == ["ACT", "OBSERVE", "RESPOND"]
    assert all(int(evt.get("sequence") or 0) > 0 for evt in events)
    checkpoint_events = [evt for evt in events if evt.get("event_type") == "graph_checkpoint"]
    assert checkpoint_events
    assert checkpoint_events[0].get("determinism_lock", {}).get("lock_hash") == "abc123"
    assert checkpoint_events[0].get("occurred_at") == context.temporal.wall_time


def test_temporal_node_exposes_canonical_turn_time_to_context_builder():
    from dadbot.core.nodes import ContextBuilderNode, TemporalNode

    class MemoryService:
        def build_context(self, context):
            return {"source": "ok"}

    context = TurnContext(user_input="hello")
    asyncio.run(TemporalNode().run(context))
    asyncio.run(ContextBuilderNode(MemoryService()).run(context))

    assert context.state["temporal"]["turn_started_at"] == context.temporal.turn_started_at
    assert context.state["rich_context"]["temporal"]["turn_started_at"] == context.temporal.turn_started_at


def test_graph_checkpoint_events_are_replayable(bot):
    recorder = ExecutionTraceRecorder(trace_id="trace-replay-001", prompt="replay")
    with bind_execution_trace(recorder, required=True):
        bot.persist_graph_checkpoint(
            {
                "trace_id": "trace-replay-001",
                "stage": "inference",
                "status": "after",
                "phase": "ACT",
                "state": {"candidate": "draft reply"},
                "metadata": {"determinism": {"enforced": True, "lock_hash": "det-lock-001"}},
            }
        )

        events = bot.list_turn_events("trace-replay-001")
        replay = bot.replay_turn_events("trace-replay-001")

    assert len(events) >= 1
    assert replay["trace_id"] == "trace-replay-001"
    assert replay["event_count"] >= 1
    checkpoint_events = [evt for evt in events if evt.get("event_type") == "graph_checkpoint"]
    assert checkpoint_events
    checkpoint_payload = dict(checkpoint_events[0].get("checkpoint") or {})
    assert "state" not in checkpoint_payload
    assert "metadata" not in checkpoint_payload
    assert "session_state" not in checkpoint_payload
    assert checkpoint_payload.get("stage") == "inference"
    assert checkpoint_payload.get("status") == "after"
    assert replay["determinism"]["consistent"] is True
    assert replay["determinism"]["lock_hash"] == "det-lock-001"


def test_validate_replay_determinism_detects_expected_hash(bot):
    recorder = ExecutionTraceRecorder(trace_id="trace-replay-002", prompt="replay")
    with bind_execution_trace(recorder, required=True):
        bot.persist_graph_checkpoint(
            {
                "trace_id": "trace-replay-002",
                "stage": "safety",
                "status": "after",
                "phase": "OBSERVE",
                "state": {"safe_result": "ok"},
                "metadata": {"determinism": {"enforced": True, "lock_hash": "det-lock-xyz"}},
            }
        )

        valid = bot.validate_replay_determinism("trace-replay-002", expected_lock_hash="det-lock-xyz")
        invalid = bot.validate_replay_determinism("trace-replay-002", expected_lock_hash="det-lock-mismatch")

    assert valid["consistent"] is True
    assert valid["matches_expected"] is True
    assert invalid["matches_expected"] is False


def test_persistence_replay_operations_require_active_trace(bot):
    with pytest.raises(RuntimeTraceViolation):
        bot.persist_graph_checkpoint(
            {
                "trace_id": "trace-no-context",
                "stage": "inference",
                "status": "after",
                "state": {"candidate": "draft"},
            }
        )
