import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.graph_context import TurnContext
from dadbot.core.graph_pipeline_nodes import ContextBuilderNode, MCPToolNode, PlannerNode
from dadbot.core.runtime_errors import ReplayInvariantViolation
from dadbot.runtime.mcp.approval_service import MCPApprovalService


def _context(**metadata):
    return TurnContext(
        user_input="Approve this Home Assistant action.",
        metadata={
            "tenant_id": "tenant-a", "session_id": "thread-a",
            "mcp_approval_resolution": {"request_id": "approval-a", "approved": True},
            **metadata,
        },
    )


def test_continuation_builds_memory_context_without_fresh_tool_planning(monkeypatch):
    planner = AsyncMock(side_effect=AssertionError("A decision must not run a fresh planner"))
    monkeypatch.setattr(PlannerNode, "run", planner)
    calls = []
    def build_context(context):
        calls.append(context)
        return {"messages": [{"role": "user", "content": "existing history"}]}
    registry = SimpleNamespace(get=lambda *_args, **_kwargs: SimpleNamespace(build_context=build_context))
    context = _context()
    asyncio.run(ContextBuilderNode().execute(registry, context))
    assert calls == [context]
    assert context.state["rich_context"]["messages"]
    assert context.state["tool_ir"] == {"requests": []}
    assert context.state["turn_plan"]["strategy"] == "mcp_approval_continuation"
    planner.assert_not_awaited()


def test_continuation_cannot_silently_fall_back_when_service_is_absent():
    registry = SimpleNamespace(get=lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="graph resolver"):
        asyncio.run(MCPToolNode().execute(registry, _context()))


@pytest.mark.parametrize("node", [ContextBuilderNode(), MCPToolNode()])
def test_source_mode_cannot_execute_an_approval_continuation(node):
    registry = SimpleNamespace(get=lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError, match="source mode"):
        asyncio.run(node.execute(registry, _context(source_mode=True)))


def test_replay_uses_matching_checkpoint_result_without_dispatch(monkeypatch):
    service = MCPApprovalService(SimpleNamespace(
        config=SimpleNamespace(tenant_id="tenant-a"), active_thread_id="thread-a",
        _execution_ledger=ExecutionLedger(),
    ))
    forbidden = AsyncMock(side_effect=AssertionError("Replay attempted resolution"))
    monkeypatch.setattr(service, "_resolve_pending", forbidden)
    context = _context(replay_mode=True)
    context.state["mcp_approval_resolution"] = dict(context.metadata["mcp_approval_resolution"])
    context.state["mcp_approval_result"] = "The recorded action completed."
    registry = SimpleNamespace(get=lambda *_args, **_kwargs: service)
    asyncio.run(MCPToolNode().execute(registry, context))
    assert context.state["candidate"] == ("The recorded action completed.", False)
    forbidden.assert_not_awaited()
    context.state["mcp_approval_resolution"]["request_id"] = "different-approval"
    with pytest.raises(ReplayInvariantViolation, match="matching checkpoint"):
        asyncio.run(MCPToolNode().execute(registry, context))
    forbidden.assert_not_awaited()


@pytest.mark.parametrize("resolution", [
    {"request_id": "a", "review_outcome": "performed"},
    {"request_id": "a", "review_outcome": "performed", "outcome_event_id": ""},
    {"request_id": "a", "review_outcome": "unknown", "outcome_event_id": "event"},
    {"request_id": "a", "review_outcome": "performed", "outcome_event_id": "event", "approved": True},
    {"request_id": "a", "approved": "true"},
])
def test_graph_action_resolution_rejects_ambiguous_or_incomplete_decisions(resolution):
    service = MCPApprovalService(SimpleNamespace(
        config=SimpleNamespace(tenant_id="tenant-a"), active_thread_id="thread-a",
        _execution_ledger=ExecutionLedger(),
    ))
    with pytest.raises(ValueError, match="Graph MCP resolution"):
        asyncio.run(service.resolve_in_graph(_context(mcp_approval_resolution=resolution)))


@pytest.mark.parametrize("selected_tool,event_tool,consistent", [
    ("light_control", "light_control", True),
    ("light_control", "different_tool", False),
    ("light_control", None, False),
    (None, "light_control", False),
])
def test_mcp_execution_truth_requires_matching_turn_selection_and_kernel_evidence(
    selected_tool, event_tool, consistent,
):
    from dadbot.services._tool_pipeline_mixin import ToolPipelineMixin

    class Service(ToolPipelineMixin):
        bot: SimpleNamespace

    service = Service()
    service.bot = SimpleNamespace(planner_debug_snapshot=lambda: {
        "planner_tool": "unrelated_prior_tool", "planner_status": "execution_contract_violation",
    })
    context = _context()
    context.state["decision_outcome"] = "executed_tool"
    context.state["tool_execution"] = {"selected_tool": selected_tool, "status": "succeeded"}
    context.state["sovereign_events"] = [{"payload": {"tool_name": event_tool}}] if event_tool else []
    assert service.build_execution_truth_contract(context)["consistent"] is consistent
