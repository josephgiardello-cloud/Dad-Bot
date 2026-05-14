from __future__ import annotations

import asyncio

import pytest

from dadbot.core.graph import TurnGraph
from dadbot.core.graph_context import TurnContext


class _NodeA:
    name = "alpha"

    async def run(self, _registry, ctx: TurnContext) -> TurnContext:
        ctx.state["alpha"] = True
        return ctx


class _NodeB:
    name = "beta"

    async def run(self, _registry, ctx: TurnContext) -> TurnContext:
        ctx.state["safe_result"] = ("ok", False)
        return ctx


class _NodeFail:
    name = "fail"

    async def run(self, _registry, _ctx: TurnContext) -> TurnContext:
        raise RuntimeError("forced stage failure")


@pytest.mark.unit
def test_graph_records_node_and_edge_trace_events() -> None:
    graph = TurnGraph(registry=None, nodes=[_NodeA(), _NodeB()])
    ctx = TurnContext(user_input="trace enrichment")

    result = asyncio.run(graph.execute(ctx))

    assert result == ("ok", False)
    trace = list(ctx.state.get("execution_trace") or [])

    node_start = [e for e in trace if str(e.get("event_type") or "") == "node_start"]
    node_complete = [e for e in trace if str(e.get("event_type") or "") == "node_complete"]
    edge_events = [e for e in trace if str(e.get("event_type") or "") == "edge_transition"]

    assert len(node_start) >= 2
    assert len(node_complete) >= 2
    assert len(edge_events) >= 1

    first_complete = node_complete[0]
    detail = dict(first_complete.get("detail") or {})
    assert "duration_ms" in detail
    assert str(detail.get("status") or "") in {"ok", "error"}

    first_edge = edge_events[0]
    edge_detail = dict(first_edge.get("detail") or {})
    assert str(edge_detail.get("from") or "")
    assert str(edge_detail.get("to") or "")

    # Unified trace envelope should now be emitted by TurnGraph.
    turn_trace = dict(ctx.state.get("turn_trace") or {})
    assert str(turn_trace.get("trace_id") or "") == str(ctx.trace_id)
    assert bool(turn_trace.get("completed")) is True
    assert len(list(turn_trace.get("nodes") or [])) >= 2
    assert len(list(turn_trace.get("trace_events") or [])) >= len(trace)
    trace_meta = dict(turn_trace.get("metadata") or {})
    assert str(trace_meta.get("source_of_truth") or "") == "execution_trace+stage_traces"
    assert str(trace_meta.get("projection_version") or "") == "graph-execution-v1"
    assert bool(turn_trace.get("_sink_only")) is True

    # Derived projection contract: one unified trace event per execution_trace event.
    projected_events = list(turn_trace.get("trace_events") or [])
    assert len(projected_events) == len(trace)
    assert [str(item.get("event_type") or "") for item in projected_events] == [
        str(item.get("event_type") or "") for item in trace
    ]


@pytest.mark.unit
def test_graph_turn_trace_records_failed_node_status() -> None:
    graph = TurnGraph(registry=None, nodes=[_NodeA(), _NodeFail()])
    ctx = TurnContext(user_input="trace failure enrichment")

    with pytest.raises(RuntimeError, match="forced stage failure"):
        asyncio.run(graph.execute(ctx))

    turn_trace = dict(ctx.state.get("turn_trace") or {})
    assert str(turn_trace.get("trace_id") or "") == str(ctx.trace_id)
    assert bool(turn_trace.get("completed")) is True
    assert str(turn_trace.get("error") or "")

    failed_nodes = [
        node for node in list(turn_trace.get("nodes") or []) if str(node.get("status") or "") == "failed"
    ]
    assert len(failed_nodes) >= 1
    assert "forced stage failure" in str(failed_nodes[0].get("error") or "")


@pytest.mark.unit
def test_graph_turn_trace_ignores_preseeded_snapshot_authority() -> None:
    graph = TurnGraph(registry=None, nodes=[_NodeA(), _NodeB()])
    ctx = TurnContext(user_input="sink contract")
    ctx.state["turn_trace"] = {"poison": True}
    ctx.metadata["turn_trace"] = {"poison": True}

    result = asyncio.run(graph.execute(ctx))
    assert result == ("ok", False)

    out = dict(ctx.state.get("turn_trace") or {})
    assert bool(out.get("poison")) is False
    assert str(out.get("trace_id") or "") == str(ctx.trace_id)
    assert bool(out.get("_sink_only")) is True


@pytest.mark.unit
def test_graph_fails_when_projected_trace_event_count_diverges(monkeypatch) -> None:
    graph = TurnGraph(registry=None, nodes=[_NodeA(), _NodeB()])
    ctx = TurnContext(user_input="projection mismatch")

    original_projector = graph._project_unified_trace_events

    def _mismatch_projector(turn_ctx: TurnContext):
        projected = list(original_projector(turn_ctx))
        return projected[:-1] if projected else projected

    monkeypatch.setattr(graph, "_project_unified_trace_events", _mismatch_projector)

    with pytest.raises(RuntimeError, match="Unified trace projection mismatch"):
        asyncio.run(graph.execute(ctx))

    # Divergent projection must fail before sink persistence.
    assert "turn_trace" not in ctx.state
