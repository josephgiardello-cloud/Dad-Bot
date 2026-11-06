from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from dadbot.runtime.mcp.tool_client import (
    MCPApprovalRequest,
    MCPToolClient,
    append_mcp_unavailable_notice,
    configured_ha_mcp_tools,
    run_mcp_tool_loop,
    run_mcp_with_fallback,
    resume_mcp_with_fallback,
)


def test_configured_ha_mcp_tools_are_explicit_allowlist(monkeypatch):
    monkeypatch.setenv("DADBOT_HA_MCP_ALLOWED_TOOLS", "light_control, thermostat_set, ,")

    assert configured_ha_mcp_tools() == {"light_control", "thermostat_set"}


def test_unavailable_notice_prevents_claiming_smart_home_actions():
    messages = append_mcp_unavailable_notice(
        [{"role": "system", "content": "Dad persona."}, {"role": "user", "content": "Turn off a light."}]
    )

    assert "Do not claim to control" in messages[0]["content"]


def test_mcp_client_only_exposes_allowlisted_server_tools():
    class FakeSession:
        async def list_tools(self):
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(name="light_control", description="Control a light", inputSchema={"type": "object"}),
                    SimpleNamespace(name="unlock_door", description="Unlock a door", inputSchema={"type": "object"}),
                ]
            )

    tools = asyncio.run(MCPToolClient.list_allowed_tools(FakeSession(), {"light_control"}))

    assert [tool["name"] for tool in tools] == ["light_control"]


@pytest.mark.parametrize("max_tool_calls", [2, 4])
def test_mcp_tool_loop_executes_ordered_allowlisted_calls_then_returns_final_text(max_tool_calls):
    responses = iter(
        [
            json.dumps(
                {
                    "should_continue": True,
                    "action_input": "I will adjust those now.",
                    "reasoning": "Two requested actions.",
                    "tool_calls": [
                        {"name": "light_control", "arguments": {"entity_id": "light.kitchen", "action": "turn_off"}},
                        {"name": "thermostat_set", "arguments": {"temperature": 68}},
                    ],
                    "confidence": 0.95,
                }
            ),
            json.dumps(
                {
                    "should_continue": False,
                    "action_input": "The kitchen light is off and the thermostat is set to 68.",
                    "reasoning": "Both tools confirmed success.",
                    "tool_calls": [],
                    "confidence": 1.0,
                }
            ),
        ]
    )
    calls = []
    model_inputs = []

    def model_call(messages):
        model_inputs.append(messages)
        assert "light_control" in str(messages[0]["content"])
        return next(responses)

    async def tool_call(name, arguments):
        calls.append((name, arguments))
        return {"is_error": False, "result": "success"}

    result = asyncio.run(
        run_mcp_tool_loop(
            [{"role": "system", "content": "Persona prompt."}, {"role": "user", "content": "Do both."}],
            [
                {"name": "light_control", "description": "Control a light", "input_schema": {"type": "object"}},
                {"name": "thermostat_set", "description": "Set thermostat", "input_schema": {"type": "object"}},
            ],
            model_call=model_call,
            tool_call=tool_call,
            approve_tool_call=lambda _name, _arguments: True,
            max_tool_calls=max_tool_calls,
        )
    )

    assert result == "The kitchen light is off and the thermostat is set to 68."
    assert [name for name, _arguments in calls] == ["light_control", "thermostat_set"]
    assert len(model_inputs) == 2
    assert any("MCP tool results" in message["content"] for message in model_inputs[1])
    assert "final answer in the Dad persona" in model_inputs[1][0]["content"]


def test_unreachable_mcp_falls_back_to_plain_chat_and_discloses_no_action():
    fallback_contexts = []

    class UnreachableClient:
        @asynccontextmanager
        async def connect(self):
            raise ConnectionError("server offline")
            yield None

    def fallback_call(messages):
        fallback_contexts.append(messages)
        return "I can still talk through what you wanted to do."

    result = asyncio.run(
        run_mcp_with_fallback(
            [{"role": "system", "content": "Dad persona."}, {"role": "user", "content": "Turn off a light."}],
            UnreachableClient(),
            {"light_control"},
            model_call=lambda _messages: "unused",
            fallback_call=fallback_call,
        )
    )

    assert "No device action was performed." in result
    assert "server offline" not in result
    assert "Do not claim to control" in fallback_contexts[0][0]["content"]


def test_mcp_allowlist_mismatch_falls_back_without_claiming_success():
    fallback_contexts = []

    class NoMatchingToolsClient:
        @asynccontextmanager
        async def connect(self):
            yield object()

        async def list_allowed_tools(self, _session, _allowed_names):
            return []

    result = asyncio.run(
        run_mcp_with_fallback(
            [{"role": "system", "content": "Dad persona."}, {"role": "user", "content": "Turn off a light."}],
            NoMatchingToolsClient(),
            {"light_control"},
            model_call=lambda _messages: "unused",
            fallback_call=lambda messages: fallback_contexts.append(messages) or "We can talk about the request.",
        )
    )

    assert "No device action was performed." in result
    assert "Do not claim to control" in fallback_contexts[0][0]["content"]


def test_tool_error_stops_chain_and_reports_no_action_performed():
    model_calls = []
    tool_calls = []
    response = json.dumps(
        {
            "should_continue": True,
            "action_input": "Trying the requested action.",
            "reasoning": "The tool is needed.",
            "tool_calls": [
                {"name": "light_control", "arguments": {"entity_id": "light.kitchen", "action": "turn_off"}},
                {"name": "thermostat_set", "arguments": {"temperature": 68}},
            ],
            "confidence": 0.9,
        }
    )

    async def failing_tool(name, arguments):
        tool_calls.append((name, arguments))
        return {"is_error": True, "result": "Home Assistant rejected the request."}

    result = asyncio.run(
        run_mcp_tool_loop(
            [{"role": "system", "content": "Dad persona."}, {"role": "user", "content": "Do both."}],
            [
                {"name": "light_control", "description": "Control light", "input_schema": {"type": "object"}},
                {"name": "thermostat_set", "description": "Set temperature", "input_schema": {"type": "object"}},
            ],
            model_call=lambda messages: model_calls.append(messages) or response,
            tool_call=failing_tool,
            approve_tool_call=lambda _name, _arguments: True,
        )
    )

    assert "outcome is uncertain" in result
    assert "completed successfully" not in result.lower()
    assert len(model_calls) == 1
    assert [name for name, _arguments in tool_calls] == ["light_control"]


@pytest.mark.parametrize("max_tool_calls", [1, 4])
@pytest.mark.parametrize(
    "final_response",
    [
        '{"is_error": false, "result": {"state": "off"}}',
        '{"action_input":',
        "",
        json.dumps(
            {
                "action_input": "I will turn off another light.",
                "tool_calls": [{"name": "light_control", "arguments": {}}],
            }
        ),
    ],
)
def test_invalid_post_tool_reply_never_leaks_payload_or_claims_no_action(max_tool_calls, final_response):
    initial_plan = json.dumps(
        {
            "action_input": "I'll check the light.",
            "tool_calls": [{"name": "light_control", "arguments": {}}],
        }
    )
    responses = iter([initial_plan] * max_tool_calls + [final_response])
    calls = []

    async def tool_call(name, arguments):
        calls.append((name, arguments))
        return {"is_error": False, "result": {"state": "off"}}

    result = asyncio.run(
        run_mcp_tool_loop(
            [{"role": "system", "content": "Dad persona."}],
            [{"name": "light_control", "input_schema": {"type": "object"}}],
            model_call=lambda _messages: next(responses),
            tool_call=tool_call,
            approve_tool_call=lambda _name, _arguments: True,
            max_tool_calls=max_tool_calls,
        )
    )

    assert result
    assert not result.startswith(("{", "["))
    assert "couldn't" in result.lower()
    assert "no device action was performed" not in result.lower()
    assert "completed" not in result.lower()
    assert len(calls) == max_tool_calls


@pytest.mark.parametrize("failure", ["invalid", "empty", "exception"])
@pytest.mark.parametrize("tool_steps", [1, 4])
def test_post_action_model_failure_does_not_use_no_action_fallback(failure, tool_steps):
    calls = []

    class FakeClient:
        @asynccontextmanager
        async def connect(self):
            yield object()

        async def list_allowed_tools(self, _session, _allowed_names):
            return [{"name": "light_control", "input_schema": {"type": "object"}}]

        async def call_tool(self, _session, name, arguments):
            calls.append((name, arguments))
            return {"is_error": False, "result": {"state": "off"}}

    def model_call(_messages):
        if len(calls) < tool_steps:
            return json.dumps(
                {
                    "action_input": "I'll check the light.",
                    "tool_calls": [{"name": "light_control", "arguments": {}}],
                }
            )
        if failure == "exception":
            raise RuntimeError("model unavailable")
        return '{"result": "off"}' if failure == "invalid" else ""

    def fallback_call(_messages):
        pytest.fail("A post-action failure must not use the no-action fallback")

    result = asyncio.run(
        run_mcp_with_fallback(
            [{"role": "system", "content": "Dad persona."}],
            FakeClient(),
            {"light_control"},
            model_call=model_call,
            fallback_call=fallback_call,
            approve_tool_call=lambda _name, _arguments: True,
        )
    )

    assert "couldn't" in result.lower()
    assert "actions may already have occurred" in result
    assert "no device action was performed" not in result.lower()
    assert len(calls) == tool_steps


def test_tool_loop_returns_exact_pending_call_without_executing():
    calls = []
    response = json.dumps(
        {
            "should_continue": True,
            "action_input": "I can turn the light off.",
            "reasoning": "A device action was requested.",
            "tool_calls": [{"name": "light_control", "arguments": {"entity_id": "light.kitchen", "action": "turn_off"}}],
            "confidence": 0.9,
        }
    )

    async def tool_call(name, arguments):
        calls.append((name, arguments))

    result = asyncio.run(
        run_mcp_tool_loop(
            [{"role": "user", "content": "Turn off light.kitchen"}],
            [{"name": "light_control", "input_schema": {"type": "object"}}],
            model_call=lambda _messages: response,
            tool_call=tool_call,
        )
    )

    assert isinstance(result, MCPApprovalRequest)
    assert result.tool_name == "light_control"
    assert result.arguments == {"entity_id": "light.kitchen", "action": "turn_off"}
    assert not calls


def test_approval_revalidates_current_schema_and_executes_once():
    calls = []
    initial_plan = json.dumps(
        {
            "should_continue": True,
            "action_input": "I can turn the light off.",
            "reasoning": "A device action was requested.",
            "tool_calls": [{"name": "light_control", "arguments": {"entity_id": "light.kitchen", "action": "turn_off"}}],
            "confidence": 0.9,
        }
    )
    final_answer = json.dumps(
        {
            "should_continue": False,
            "action_input": "The kitchen light is off.",
            "reasoning": "Home Assistant confirmed the result.",
            "tool_calls": [],
            "confidence": 1.0,
        }
    )
    request = asyncio.run(
        run_mcp_tool_loop(
            [{"role": "user", "content": "Turn off light.kitchen"}],
            [{"name": "light_control", "input_schema": {"type": "object"}}],
            model_call=lambda _messages: initial_plan,
            tool_call=lambda _name, _arguments: pytest.fail("The initial plan must not execute"),
        )
    )
    assert isinstance(request, MCPApprovalRequest)

    class FakeClient:
        @asynccontextmanager
        async def connect(self):
            yield object()

        async def list_allowed_tools(self, _session, _allowed_names):
            return [{
                "name": "light_control",
                "input_schema": {
                    "type": "object",
                    "properties": {"entity_id": {"type": "string"}, "action": {"type": "string"}},
                    "required": ["entity_id", "action"],
                    "additionalProperties": False,
                },
            }]

        async def call_tool(self, _session, name, arguments):
            calls.append((name, arguments))
            return {"is_error": False, "result": "success"}

    result = asyncio.run(
        resume_mcp_with_fallback(
            request,
            FakeClient(),
            {"light_control"},
            approved=True,
            model_call=lambda _messages: final_answer,
        )
    )

    assert result == "The kitchen light is off."
    assert calls == [("light_control", {"entity_id": "light.kitchen", "action": "turn_off"})]
    replay_result = asyncio.run(
        resume_mcp_with_fallback(
            request,
            FakeClient(),
            {"light_control"},
            approved=True,
            model_call=lambda _messages: final_answer,
        )
    )
    assert "already been resolved" in replay_result
    assert len(calls) == 1


def test_rejected_approval_never_connects_or_executes():
    initial_plan = json.dumps(
        {
            "should_continue": True,
            "action_input": "I can turn the light off.",
            "reasoning": "A device action was requested.",
            "tool_calls": [{"name": "light_control", "arguments": {}}],
            "confidence": 0.9,
        }
    )
    request = asyncio.run(
        run_mcp_tool_loop(
            [{"role": "user", "content": "Turn off a light."}],
            [{"name": "light_control", "input_schema": {"type": "object"}}],
            model_call=lambda _messages: initial_plan,
        )
    )
    assert isinstance(request, MCPApprovalRequest)

    class NoConnectClient:
        @asynccontextmanager
        async def connect(self):
            pytest.fail("Rejecting an action must not connect to MCP")
            yield None

    result = asyncio.run(
        resume_mcp_with_fallback(
            request,
            NoConnectClient(),
            {"light_control"},
            approved=False,
            model_call=lambda _messages: pytest.fail("A rejected action must not call the model"),
        )
    )
    assert "not approved" in result