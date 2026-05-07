from __future__ import annotations

import asyncio
import uuid

import pytest

from dadbot.core.graph import TurnContext
from dadbot.core.nodes import (
    ToolExecutorNode,
    ToolRouterNode,
    dispatch_registered_tool,
    register_tool,
)
from dadbot.core.tool_ir import ToolContractResult, ToolStatus

pytestmark = pytest.mark.unit


def _tool_name(prefix: str = "unit_tool") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def test_custom_tool_can_be_registered_and_executed_without_core_edits():
    tool_name = _tool_name()

    def _handler(args, _context):
        return str(args.get("text") or "").upper()

    register_tool(
        tool_name,
        handler=_handler,
        required_args={"text"},
        allowed_intents={"session_memory_fetch"},
    )

    ctx = TurnContext(user_input="hello")
    ctx.state["tool_ir"] = {
        "requests": [
            {
                "tool_name": tool_name,
                "args": {"text": "hello"},
                "intent": "session_memory_fetch",
                "expected_output": "uppercase text",
                "priority": 1,
            }
        ]
    }

    asyncio.run(ToolRouterNode().run(ctx))
    asyncio.run(ToolExecutorNode().run(ctx))

    plan = list(ctx.state["tool_ir"].get("execution_plan") or [])
    assert len(plan) == 1
    assert plan[0]["tool_name"] == tool_name

    results = list(ctx.state.get("tool_results") or [])
    assert len(results) == 1
    assert results[0]["status"] == "ok"
    assert results[0]["output"] == "HELLO"


def test_registered_tool_missing_required_arg_returns_contract_violation():
    tool_name = _tool_name()

    def _handler(args, _context):
        return args

    register_tool(tool_name, handler=_handler, required_args={"required"})
    ctx = TurnContext(user_input="test")

    result = dispatch_registered_tool(tool_name, {}, ctx)

    assert isinstance(result, ToolContractResult)
    assert result.status == ToolStatus.CONTRACT_VIOLATION
    assert result.error_context.get("missing_args") == ["required"]


def test_registered_tool_retryable_exception_maps_to_retry_status():
    tool_name = _tool_name()

    def _handler(_args, _context):
        raise TimeoutError("transient timeout")

    register_tool(
        tool_name,
        handler=_handler,
        retryable_exceptions=(TimeoutError,),
    )
    ctx = TurnContext(user_input="test")

    result = dispatch_registered_tool(tool_name, {"ok": True}, ctx)

    assert isinstance(result, ToolContractResult)
    assert result.status == ToolStatus.RETRY
    assert "transient timeout" in str(result.error_context.get("exception") or "")


def test_registered_tool_output_validator_enforces_output_shape():
    tool_name = _tool_name()

    def _handler(_args, _context):
        return "not a dict"

    register_tool(
        tool_name,
        handler=_handler,
        output_validator=lambda output: isinstance(output, dict) and "value" in output,
    )
    ctx = TurnContext(user_input="test")

    result = dispatch_registered_tool(tool_name, {"x": 1}, ctx)

    assert isinstance(result, ToolContractResult)
    assert result.status == ToolStatus.CONTRACT_VIOLATION
    assert result.error_context.get("output_validation") == "failed"


def test_tool_executor_retries_once_and_records_remediation():
    tool_name = _tool_name()
    attempts = {"count": 0}

    def _handler(_args, _context):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("retry me once")
        return {"value": "ok"}

    register_tool(
        tool_name,
        handler=_handler,
        allowed_intents={"session_memory_fetch"},
        retryable_exceptions=(TimeoutError,),
    )

    ctx = TurnContext(user_input="hello")
    ctx.state["tool_ir"] = {
        "requests": [
            {
                "tool_name": tool_name,
                "args": {},
                "intent": "session_memory_fetch",
                "expected_output": "dict with value",
                "priority": 1,
            }
        ]
    }

    asyncio.run(ToolRouterNode().run(ctx))
    asyncio.run(ToolExecutorNode().run(ctx))

    results = list(ctx.state.get("tool_results") or [])
    assert len(results) == 1
    assert results[0]["status"] == "ok"
    remediation = list(results[0].get("remediation") or [])
    assert remediation
    assert remediation[0]["action"] == "retry"
    assert attempts["count"] == 2
