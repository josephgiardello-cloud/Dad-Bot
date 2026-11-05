from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

from dadbot.runtime.mcp.tool_client import (
    MCPToolClient,
    append_mcp_unavailable_notice,
    configured_ha_mcp_tools,
    run_mcp_tool_loop,
    run_mcp_with_fallback,
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


def test_mcp_tool_loop_executes_ordered_allowlisted_calls_then_returns_final_text():
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
        )
    )

    assert result == "The kitchen light is off and the thermostat is set to 68."
    assert [name for name, _arguments in calls] == ["light_control", "thermostat_set"]
    assert len(model_inputs) == 2
    assert "MCP tool results" in model_inputs[1][-1]["content"]


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
        )
    )

    assert "No device action was performed." in result
    assert "completed successfully" not in result.lower()
    assert len(model_calls) == 1
    assert [name for name, _arguments in tool_calls] == ["light_control"]