"""Unit tests for dadbot.runtime.structured_output (Task 1)."""

from __future__ import annotations

import json

import pytest

from dadbot.runtime.structured_output import (
    AgentPlan,
    SchemaValidationError,
    ToolCall,
    build_llm_reflection_hook,
    build_reflection_prompt,
    parse_agent_plan,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# parse_agent_plan
# ---------------------------------------------------------------------------


def _valid_plan_json(**overrides: object) -> str:
    base = {
        "should_continue": True,
        "action_input": "do the next thing",
        "reasoning": "because",
        "tool_calls": [],
        "confidence": 0.9,
    }
    base.update(overrides)
    return json.dumps(base)


class TestParseAgentPlan:
    def test_parses_valid_json(self) -> None:
        plan = parse_agent_plan(_valid_plan_json())
        assert isinstance(plan, AgentPlan)
        assert plan.action_input == "do the next thing"
        assert plan.should_continue is True
        assert plan.confidence == pytest.approx(0.9)

    def test_parses_fenced_json(self) -> None:
        fenced = "```json\n" + _valid_plan_json() + "\n```"
        plan = parse_agent_plan(fenced)
        assert plan.action_input == "do the next thing"

    def test_parses_json_embedded_in_prose(self) -> None:
        prose = "Here is my plan:\n" + _valid_plan_json() + "\nEnd."
        plan = parse_agent_plan(prose)
        assert plan.should_continue is True

    def test_raises_on_missing_action_input(self) -> None:
        bad = json.dumps({"should_continue": True, "confidence": 1.0})
        with pytest.raises(SchemaValidationError):
            parse_agent_plan(bad)

    def test_raises_on_blank_action_input(self) -> None:
        bad = _valid_plan_json(action_input="   ")
        with pytest.raises(SchemaValidationError):
            parse_agent_plan(bad)

    def test_raises_on_no_json_in_text(self) -> None:
        with pytest.raises(SchemaValidationError):
            parse_agent_plan("Hello, no JSON here at all.")

    def test_raises_on_malformed_json(self) -> None:
        with pytest.raises(SchemaValidationError):
            parse_agent_plan("{not: valid json}")

    def test_raises_on_disallowed_tool(self) -> None:
        plan_json = _valid_plan_json(
            tool_calls=[{"name": "forbidden_tool", "arguments": {}}]
        )
        with pytest.raises(SchemaValidationError, match="not in allowed list"):
            parse_agent_plan(plan_json, allowed_tools=["safe_tool"])

    def test_allows_valid_tool(self) -> None:
        plan_json = _valid_plan_json(
            tool_calls=[{"name": "search", "arguments": {"q": "hello"}}]
        )
        plan = parse_agent_plan(plan_json, allowed_tools=["search", "memory_write"])
        assert plan.tool_calls[0].name == "search"

    def test_defaults_should_continue_true(self) -> None:
        minimal = json.dumps({"action_input": "go"})
        plan = parse_agent_plan(minimal)
        assert plan.should_continue is True
        assert plan.confidence == pytest.approx(1.0)

    def test_should_continue_false_accepted(self) -> None:
        plan = parse_agent_plan(_valid_plan_json(should_continue=False))
        assert plan.should_continue is False

    def test_confidence_clamped_range(self) -> None:
        plan = parse_agent_plan(_valid_plan_json(confidence=0.0))
        assert plan.confidence == pytest.approx(0.0)
        plan2 = parse_agent_plan(_valid_plan_json(confidence=1.0))
        assert plan2.confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# build_reflection_prompt
# ---------------------------------------------------------------------------


class TestBuildReflectionPrompt:
    def test_returns_two_message_list(self) -> None:
        ctx = {
            "turn_index": 3,
            "last_reply": "something happened",
            "initial_observation": "start state",
            "records": [object(), object()],
        }
        msgs = build_reflection_prompt(ctx, allowed_tools=["tool_a"])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_user_message_is_json_parseable(self) -> None:
        ctx = {"turn_index": 1, "last_reply": "done", "initial_observation": "hi", "records": []}
        msgs = build_reflection_prompt(ctx)
        payload = json.loads(msgs[1]["content"])
        assert "last_reply" in payload
        assert "turn_index" in payload

    def test_tool_names_appear_in_system_prompt(self) -> None:
        ctx = {"turn_index": 1, "last_reply": "", "initial_observation": "", "records": []}
        msgs = build_reflection_prompt(ctx, allowed_tools=["magic_tool"])
        assert "magic_tool" in msgs[0]["content"]


# ---------------------------------------------------------------------------
# build_llm_reflection_hook
# ---------------------------------------------------------------------------


class _FakeLLMClient:
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[dict] = []

    def call_llm(self, messages: list, **kwargs: object) -> str:
        self.calls.append({"messages": messages, **kwargs})
        return self._response


class TestBuildLLMReflectionHook:
    def _ctx(self, turn_index: int = 1) -> dict:
        return {
            "turn_index": turn_index,
            "last_reply": "previous output",
            "initial_observation": "starting task",
            "records": [],
        }

    def test_hook_returns_valid_plan_dict(self) -> None:
        raw = json.dumps(
            {
                "should_continue": True,
                "action_input": "check status",
                "reasoning": "need more info",
                "tool_calls": [],
                "confidence": 0.85,
            }
        )
        client = _FakeLLMClient(raw)
        hook = build_llm_reflection_hook(client, tool_names=["search"])
        result = hook(self._ctx())
        assert result["should_continue"] is True
        assert result["action_input"] == "check status"
        assert result["confidence"] == pytest.approx(0.85)

    def test_hook_calls_llm_once(self) -> None:
        raw = json.dumps({"action_input": "next step", "should_continue": True})
        client = _FakeLLMClient(raw)
        hook = build_llm_reflection_hook(client)
        hook(self._ctx())
        assert len(client.calls) == 1

    def test_hook_passes_json_format(self) -> None:
        raw = json.dumps({"action_input": "x", "should_continue": True})
        client = _FakeLLMClient(raw)
        hook = build_llm_reflection_hook(client)
        hook(self._ctx())
        assert client.calls[0].get("response_format") == "json"

    def test_hook_fallback_on_bad_json(self) -> None:
        client = _FakeLLMClient("This is not JSON at all.")
        hook = build_llm_reflection_hook(client, fallback_on_error=True)
        result = hook(self._ctx())
        # Fallback: should_continue stays True, confidence is 0
        assert result["should_continue"] is True
        assert result["confidence"] == pytest.approx(0.0)
        assert result["action_input"]  # non-empty fallback from last_reply

    def test_hook_raises_on_bad_json_when_no_fallback(self) -> None:
        client = _FakeLLMClient("garbage")
        hook = build_llm_reflection_hook(client, fallback_on_error=False)
        with pytest.raises(SchemaValidationError):
            hook(self._ctx())

    def test_hook_validates_tool_names(self) -> None:
        raw = json.dumps(
            {
                "action_input": "run it",
                "should_continue": True,
                "tool_calls": [{"name": "bad_tool", "arguments": {}}],
            }
        )
        client = _FakeLLMClient(raw)
        hook = build_llm_reflection_hook(client, tool_names=["good_tool"], fallback_on_error=False)
        with pytest.raises(SchemaValidationError):
            hook(self._ctx())

    def test_hook_surfaces_system_observation_on_tool_hallucination(self) -> None:
        raw = json.dumps(
            {
                "action_input": "run it",
                "should_continue": True,
                "tool_calls": [{"name": "bad_tool", "arguments": {}}],
            }
        )
        client = _FakeLLMClient(raw)
        hook = build_llm_reflection_hook(client, tool_names=["good_tool"], fallback_on_error=True)
        result = hook(self._ctx())
        assert result["action_input"] == ""
        assert "system_observation" in result
        assert "allowed" in result["system_observation"].lower()

    def test_hook_stops_loop_on_llm_say_so(self) -> None:
        raw = json.dumps({"action_input": "wrap up", "should_continue": False})
        client = _FakeLLMClient(raw)
        hook = build_llm_reflection_hook(client)
        result = hook(self._ctx())
        assert result["should_continue"] is False
