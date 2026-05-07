"""Failure soak tests for ValidationGateNode.

Verifies that the gate:
  1. Passes through valid tool requests unchanged.
  2. Detects CONTRACT_VIOLATION when a required arg is missing, emits a
     CognitionEnvelope, and strips the violating request from tool_ir.
  3. Triggers the repair loop (re-runs PlannerNode) and, if the planner fixes
     the args, allows the corrected request through on the second pass.
  4. After the maximum repair iteration, strips any still-violating requests
     rather than letting them reach the execution layer.
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.unit

from dadbot.core.graph import TurnContext
from dadbot.core.graph_pipeline_nodes import ValidationGateNode
from dadbot.core.tool_ir import ToolStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(requests: list[dict]) -> TurnContext:
    ctx = TurnContext(user_input="test")
    ctx.state["tool_ir"] = {"requests": requests}
    return ctx


def _cognition_violations(ctx: TurnContext) -> list[dict]:
    stream = list(ctx.state.get("cognition_stream") or [])
    return [e for e in stream if "CONTRACT_VIOLATION" in (e.get("thought_trace") or "")]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_valid_request_passes_through():
    """memory_lookup with 'query' present → no violation, tool_ir unchanged."""
    ctx = _make_ctx([{"tool_name": "memory_lookup", "args": {"query": "find goals"}}])
    node = ValidationGateNode()
    # PlannerNode.run is a no-op here because validation passes immediately
    asyncio.run(node.execute(None, ctx))

    assert _cognition_violations(ctx) == []
    requests = ctx.state["tool_ir"]["requests"]
    assert len(requests) == 1
    assert requests[0]["tool_name"] == "memory_lookup"


def test_missing_required_arg_emits_violation_and_strips():
    """memory_lookup missing 'query' → violation emitted, request stripped."""
    ctx = _make_ctx([{"tool_name": "memory_lookup", "args": {}}])
    node = ValidationGateNode()

    # PlannerNode.run — stub that does NOT fix the args (simulates LLM still failing)
    async def _planner_noop(self, turn_context):
        pass  # Leave tool_ir unchanged so violations persist

    import dadbot.core.graph_pipeline_nodes as gpn
    original_run = gpn.PlannerNode.run
    gpn.PlannerNode.run = _planner_noop
    try:
        asyncio.run(node.execute(None, ctx))
    finally:
        gpn.PlannerNode.run = original_run

    # A CognitionEnvelope marking CONTRACT_VIOLATION must be emitted
    violations = _cognition_violations(ctx)
    assert len(violations) >= 1
    assert "memory_lookup" in violations[0]["thought_trace"]
    assert violations[0]["confidence_score"] == 0.0

    # The violating request must be stripped from tool_ir
    remaining = ctx.state["tool_ir"].get("requests", [])
    assert all(r.get("tool_name") != "memory_lookup" for r in remaining)

    # The gate must record what was stripped
    gate_record = ctx.state["tool_ir"].get("validation_gate", {})
    assert "memory_lookup" in gate_record.get("stripped_tools", [])


def test_repair_loop_succeeds_when_planner_fixes_args():
    """Planner fixes the missing arg on the repair pass → request reaches inference."""
    ctx = _make_ctx([{"tool_name": "memory_lookup", "args": {}}])
    node = ValidationGateNode()
    repaired = False

    async def _planner_fix(self, turn_context):
        nonlocal repaired
        if not repaired:
            # Simulate planner repairing the tool call on first repair attempt
            repaired = True
            turn_context.state["tool_ir"]["requests"] = [
                {"tool_name": "memory_lookup", "args": {"query": "repaired query"}}
            ]

    import dadbot.core.graph_pipeline_nodes as gpn
    original_run = gpn.PlannerNode.run
    gpn.PlannerNode.run = _planner_fix
    try:
        asyncio.run(node.execute(None, ctx))
    finally:
        gpn.PlannerNode.run = original_run

    # The repaired request must survive (not stripped)
    remaining = ctx.state["tool_ir"].get("requests", [])
    assert any(
        r.get("tool_name") == "memory_lookup" and r.get("args", {}).get("query") == "repaired query"
        for r in remaining
    ), f"Expected repaired request in tool_ir; got: {remaining}"
    # No extra stripping record should exist
    assert "validation_gate" not in ctx.state["tool_ir"]
    repair = dict(ctx.state.get("_validation_gate_repair") or {})
    remediation = dict(repair.get("remediation") or {})
    assert remediation.get("action") == "replan"


def test_current_time_no_required_args_passes():
    """current_time has no required args → always valid."""
    ctx = _make_ctx([{"tool_name": "current_time", "args": {}}])
    node = ValidationGateNode()
    asyncio.run(node.execute(None, ctx))
    assert _cognition_violations(ctx) == []


def test_unknown_tool_skips_validation():
    """Tools not in the schema dict are not validated (open for extension)."""
    ctx = _make_ctx([{"tool_name": "some_future_tool", "args": {}}])
    node = ValidationGateNode()
    asyncio.run(node.execute(None, ctx))
    assert _cognition_violations(ctx) == []
    assert len(ctx.state["tool_ir"]["requests"]) == 1


def test_empty_tool_ir_is_noop():
    """No tool_ir in state → gate is silent."""
    ctx = TurnContext(user_input="hello")
    node = ValidationGateNode()
    asyncio.run(node.execute(None, ctx))
    assert _cognition_violations(ctx) == []


def test_memory_lookup_contract_result_on_missing_query():
    """Direct dispatch: memory_lookup with no query returns ToolContractResult(CONTRACT_VIOLATION)."""
    from dadbot.core.nodes import dispatch_registered_tool
    from dadbot.core.tool_ir import ToolContractResult, ToolStatus

    ctx = TurnContext(user_input="test")
    result = dispatch_registered_tool("memory_lookup", {}, ctx)

    assert isinstance(result, ToolContractResult)
    assert result.status == ToolStatus.CONTRACT_VIOLATION
    assert "query" in result.repair_hint
    assert result.data is None


def test_memory_lookup_contract_result_on_valid_query():
    """Direct dispatch: memory_lookup with query returns ToolContractResult(SUCCESS)."""
    from dadbot.core.nodes import dispatch_registered_tool
    from dadbot.core.tool_ir import ToolContractResult, ToolStatus

    ctx = TurnContext(user_input="test")
    ctx.state["memories"] = [{"text": "some memory"}]
    result = dispatch_registered_tool("memory_lookup", {"query": "find goals"}, ctx)

    assert isinstance(result, ToolContractResult)
    assert result.status == ToolStatus.SUCCESS
    assert result.data is not None


def test_missing_required_arg_records_downgrade_after_repair_exhaustion():
    ctx = _make_ctx([{"tool_name": "memory_lookup", "args": {}}])
    node = ValidationGateNode()

    async def _planner_noop(self, turn_context):
        pass

    import dadbot.core.graph_pipeline_nodes as gpn
    original_run = gpn.PlannerNode.run
    gpn.PlannerNode.run = _planner_noop
    try:
        asyncio.run(node.execute(None, ctx))
    finally:
        gpn.PlannerNode.run = original_run

    gate_record = dict(ctx.state["tool_ir"].get("validation_gate") or {})
    remediation = dict(gate_record.get("remediation") or {})
    assert remediation.get("action") == "downgrade"
    assert remediation.get("failure_class") == "validation_contract_violation"
