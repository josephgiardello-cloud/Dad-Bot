"""Tests for Bayesian governing authority over tool selection."""
import asyncio
from types import SimpleNamespace

import pytest
from dadbot.services.turn_service import TurnService


class _BotStub:
    """Minimal bot stub for TurnService Bayesian gate tests."""

    def __init__(self, *, tool_bias: str = "planner_default"):
        self._tool_bias = tool_bias
        self._planner_debug: dict = {
            "planner_status": "idle",
            "planner_reason": "",
            "planner_tool": "",
            "planner_parameters": {},
            "bayesian_tool_bias": tool_bias,
        }

    def planner_debug_snapshot(self) -> dict:
        return dict(self._planner_debug)

    def update_planner_debug(self, **kwargs) -> None:
        self._planner_debug.update(kwargs)

    def agentic_tool_settings(self) -> dict:
        return {
            "enabled": True,
            "auto_reminders": True,
            "auto_web_lookup": True,
        }

    def get_available_tools(self) -> list:
        return [
            {"function": {"name": "set_reminder", "description": "Set a reminder"}},
            {"function": {"name": "web_search", "description": "Search the web"}},
        ]


def _make_turn_service(tool_bias: str) -> TurnService:
    bot = _BotStub(tool_bias=tool_bias)
    ts = object.__new__(TurnService)
    ts.bot = bot
    ts.context = SimpleNamespace(bot=bot)
    ts.reply_generation = None
    return ts


def test_planner_default_bias_permits_both_tools():
    ts = _make_turn_service("planner_default")
    allowed_reminder, _ = ts._bayesian_tool_gate(
        tool_name="set_reminder", tool_bias="planner_default", plan_reason="test"
    )
    allowed_web, _ = ts._bayesian_tool_gate(
        tool_name="web_search", tool_bias="planner_default", plan_reason="test"
    )
    assert allowed_reminder is True
    assert allowed_web is True


def test_minimal_tools_bias_blocks_web_search():
    ts = _make_turn_service("minimal_tools")
    allowed, reason = ts._bayesian_tool_gate(
        tool_name="web_search", tool_bias="minimal_tools", plan_reason="test"
    )
    assert allowed is False
    assert "minimal_tools" in reason


def test_minimal_tools_bias_permits_set_reminder():
    ts = _make_turn_service("minimal_tools")
    allowed, _ = ts._bayesian_tool_gate(
        tool_name="set_reminder", tool_bias="minimal_tools", plan_reason="test"
    )
    assert allowed is True


def test_defer_tools_bias_blocks_all_tools():
    ts = _make_turn_service("defer_tools_unless_explicit")
    for tool in ("set_reminder", "web_search"):
        allowed, reason = ts._bayesian_tool_gate(
            tool_name=tool, tool_bias="defer_tools_unless_explicit", plan_reason=""
        )
        assert allowed is False, f"Expected {tool!r} to be blocked"
        assert "blocks all tools" in reason


def test_unknown_bias_falls_back_to_default_permissions():
    ts = _make_turn_service("unknown_bias_xyz")
    allowed, _ = ts._bayesian_tool_gate(
        tool_name="set_reminder", tool_bias="unknown_bias_xyz", plan_reason=""
    )
    assert allowed is True


def test_permitted_tools_for_bias_returns_expected_sets():
    assert "web_search" not in TurnService._permitted_tools_for_bias("minimal_tools")
    assert "set_reminder" in TurnService._permitted_tools_for_bias("minimal_tools")
    assert len(TurnService._permitted_tools_for_bias("defer_tools_unless_explicit")) == 0
    assert "web_search" in TurnService._permitted_tools_for_bias("planner_default")
