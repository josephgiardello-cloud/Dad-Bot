from __future__ import annotations

import pytest

from dadbot.core.determinism import DeterminismBoundary
from dadbot.core.graph import TurnContext, TurnGraph
from dadbot.core.nodes import TemporalNode, ToolExecutorNode
from dadbot.core.tool_ir import deterministic_tool_id

pytestmark = pytest.mark.unit


class _InferencePlanNode:
    async def run(self, context: TurnContext) -> TurnContext:
        args: dict[str, str] = {}
        context.state["tool_ir"] = {
            "execution_plan": [
                {
                    "sequence": 0,
                    "tool_name": "current_time",
                    "args": args,
                    "intent": "time_lookup",
                    "expected_output": "iso timestamp",
                    "priority": 0,
                    "deterministic_id": deterministic_tool_id("current_time", args),
                },
            ],
        }
        context.state["candidate"] = "draft"
        return context


class _SafetyNode:
    async def run(self, context: TurnContext) -> TurnContext:
        results = list(context.state.get("tool_results") or [])
        stamp = ""
        if results:
            stamp = str(results[0].get("output") or "")
        context.state["safe_result"] = (f"safe:{stamp}", False)
        return context


class _SaveNode:
    async def run(self, context: TurnContext) -> TurnContext:
        context.state["safe_result"] = tuple(context.state.get("safe_result") or ("", False))
        return context


def _build_graph() -> TurnGraph:
    graph = TurnGraph(registry={})
    graph.add_node("temporal", TemporalNode())
    graph.add_node("inference", _InferencePlanNode())
    graph.add_node("tool_executor", ToolExecutorNode())
    graph.add_node("safety", _SafetyNode())
    graph.add_node("save", _SaveNode())
    graph.set_edge("temporal", "inference")
    graph.set_edge("inference", "tool_executor")
    graph.set_edge("tool_executor", "safety")
    graph.set_edge("safety", "save")
    return graph


def _stage_transitions(context: TurnContext) -> list[dict[str, object]]:
    return [
        {
            "stage": str(trace.stage or ""),
            "duration_ms": float(trace.duration_ms or 0.0),
            "error": str(trace.error or ""),
        }
        for trace in list(context.stage_traces or [])
    ]


def _trace_log(context: TurnContext) -> list[dict[str, object]]:
    return [dict(item) for item in list(context.state.get("execution_trace") or [])]


@pytest.mark.asyncio
async def test_replay_equivalence_harness_detects_no_divergence() -> None:
    graph = _build_graph()

    first = TurnContext(
        user_input="same input",
        trace_id="replay-equivalence-trace",
        kernel_step_id="kernel-step-001",
    )
    first_result = await graph.execute(first)
    first_snapshot = first.determinism_boundary.snapshot()

    replay_boundary = DeterminismBoundary.from_snapshot(first_snapshot)
    replay_boundary.seal()

    second = TurnContext(
        user_input="same input",
        trace_id="replay-equivalence-trace",
        kernel_step_id="kernel-step-001",
        determinism_manifest={},
        determinism_boundary=replay_boundary,
    )
    second_result = await graph.execute(second)

    assert first_result == second_result, "output divergence detected"
    assert _stage_transitions(first) == _stage_transitions(second), "state transition divergence detected"
    assert _trace_log(first) == _trace_log(second), "trace log divergence detected"


def test_replay_equivalence_harness_fails_on_divergence() -> None:
    left = {
        "output": ("ok", False),
        "transitions": [{"stage": "temporal", "duration_ms": 1.0, "error": ""}],
        "trace": [{"sequence": 1, "event_type": "node_start", "stage": "temporal", "phase": "PLAN"}],
    }
    right = {
        "output": ("ok", False),
        "transitions": [{"stage": "temporal", "duration_ms": 2.0, "error": ""}],
        "trace": [{"sequence": 1, "event_type": "node_start", "stage": "temporal", "phase": "PLAN"}],
    }

    with pytest.raises(AssertionError, match="state transition divergence"):
        assert left["transitions"] == right["transitions"], "state transition divergence"
