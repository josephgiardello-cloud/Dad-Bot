from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from dadbot.core.graph import InferenceNode, TurnContext

pytestmark = pytest.mark.unit


class _RegistryStub:
    def __init__(self, agent_service) -> None:
        self._agent_service = agent_service

    def get(self, name: str):
        if name == "agent_service":
            return self._agent_service
        raise KeyError(name)


class _AgentServiceStub:
    def __init__(self) -> None:
        self.calls = 0

    async def run_agent(self, _turn_context, _rich_context):
        self.calls += 1
        return ("agent-output", False)


def test_goal_alignment_guard_interrupts_and_skips_agent() -> None:
    node = InferenceNode()
    service = _AgentServiceStub()
    registry = _RegistryStub(service)
    ctx = TurnContext(user_input="what grill recipe should I cook tonight")
    ctx.state["goal_alignment_guard_enabled"] = True
    ctx.state["session_goals"] = [
        {"id": "goal-1", "description": "finish python persistence unit tests"},
    ]
    ctx.state["turn_plan"] = {}

    asyncio.run(node.execute(registry, ctx))

    candidate = ctx.state.get("candidate")
    assert isinstance(candidate, tuple)
    assert "off our agreed direction" in str(candidate[0])
    assert service.calls == 0
    guard = dict(ctx.state.get("goal_alignment_guard") or {})
    assert guard.get("triggered") is True


def test_goal_alignment_guard_allows_agent_when_aligned() -> None:
    node = InferenceNode()
    service = _AgentServiceStub()
    registry = _RegistryStub(service)
    ctx = TurnContext(user_input="finish python persistence unit tests now")
    ctx.state["goal_alignment_guard_enabled"] = True
    ctx.state["session_goals"] = [
        {"id": "goal-1", "description": "finish python persistence unit tests"},
    ]
    ctx.state["turn_plan"] = {}

    asyncio.run(node.execute(registry, ctx))

    candidate = ctx.state.get("candidate")
    assert candidate == ("agent-output", False)
    assert service.calls == 1


def test_goal_alignment_guard_enforces_mandatory_halt_after_three_diversions() -> None:
    node = InferenceNode()
    service = _AgentServiceStub()
    registry = _RegistryStub(service)

    ctx = TurnContext(user_input="plan a fishing trip this weekend")
    ctx.state["goal_alignment_guard_enabled"] = True
    ctx.state["session_goals"] = [
        {"id": "goal-1", "description": "finish python persistence unit tests"},
    ]
    ctx.state["turn_plan"] = {}

    asyncio.run(node.execute(registry, ctx))
    assert "off our agreed direction" in str(ctx.state.get("candidate", ("",))[0])
    assert int(ctx.state.get("goal_alignment_diversion_streak") or 0) == 1

    ctx.user_input = "what is the best steak marinade"
    asyncio.run(node.execute(registry, ctx))
    assert int(ctx.state.get("goal_alignment_diversion_streak") or 0) == 2

    ctx.user_input = "tell me about baseball batting order"
    asyncio.run(node.execute(registry, ctx))
    candidate = ctx.state.get("candidate")
    assert isinstance(candidate, tuple)
    assert "MANDATORY_HALT" in str(candidate[0])
    guard = dict(ctx.state.get("goal_alignment_guard") or {})
    assert guard.get("mandatory_halt") is True
    assert guard.get("diversion_streak") == 3
    assert service.calls == 0


def test_goal_alignment_guard_clears_halt_after_realign_confirmation() -> None:
    node = InferenceNode()
    service = _AgentServiceStub()
    registry = _RegistryStub(service)

    ctx = TurnContext(user_input="realign please, return to goal and finish python persistence unit tests")
    ctx.state["goal_alignment_guard_enabled"] = True
    ctx.state["goal_alignment_mandatory_halt"] = True
    ctx.state["goal_alignment_diversion_streak"] = 3
    ctx.state["session_goals"] = [
        {"id": "goal-1", "description": "finish python persistence unit tests"},
    ]
    ctx.state["turn_plan"] = {}

    asyncio.run(node.execute(registry, ctx))

    assert ctx.state.get("candidate") == ("agent-output", False)
    assert ctx.state.get("goal_alignment_mandatory_halt") is False
    assert int(ctx.state.get("goal_alignment_diversion_streak") or 0) == 0
    assert service.calls == 1
