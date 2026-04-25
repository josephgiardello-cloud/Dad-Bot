import asyncio
from types import SimpleNamespace

from dadbot.services.agent_service import AgentService


class _TurnServiceStub:
    def __init__(self, *, mood="neutral", early_reply=None, should_end=False, turn_text="hello", attachments=None):
        self._payload = (mood, early_reply, should_end, turn_text, attachments)

    async def prepare_user_turn_async(self, _stripped, _attachments, **_kw):
        return self._payload


class _RelationshipStub:
    def current_state(self):
        return {"score": 61, "level": "growing"}

    def build_hypothesis_posteriors(self, _state, _turn_text, _mood):
        return [
            {"name": "supportive_baseline", "probability": 0.72},
            {"name": "acute_stress", "probability": 0.28},
        ]


def test_agent_service_requires_bayesian_step_before_returning_direct_reply():
    bot = SimpleNamespace(
        turn_service=_TurnServiceStub(mood="stressed", early_reply="tool handled", should_end=False),
        relationship_manager=_RelationshipStub(),
        planner_debug_snapshot=lambda: {
            "planner_status": "used_tool",
            "planner_tool": "web_search",
            "planner_parameters": {"query": "weather"},
            "planner_observation": "Weather is sunny",
            "final_path": "planner_tool",
        },
    )
    service = AgentService(bot)
    turn_context = SimpleNamespace(
        user_input="check weather",
        attachments=None,
        state={},
        metadata={},
    )

    result = asyncio.run(service.run_agent(turn_context, rich_context={}))

    assert result == ("tool handled", False)
    assert turn_context.state["bayesian_state"]["required"] is True
    assert turn_context.state["bayesian_state"]["applied"] is True
    assert turn_context.state["bayesian_state"]["policy"] == "supportive_problem_solving"
    assert turn_context.state["tool_execution_envelope"]["selected_tool"] == "web_search"
    assert turn_context.state["tool_execution_envelope"]["deterministic"] is True


def test_agent_service_attaches_tool_envelope_on_model_generation_path():
    bot = SimpleNamespace(
        turn_service=_TurnServiceStub(mood="neutral", early_reply=None, should_end=False, turn_text="hi"),
        relationship_manager=_RelationshipStub(),
        planner_debug_snapshot=lambda: {
            "planner_status": "no_tool",
            "planner_tool": "",
            "planner_parameters": {},
            "planner_observation": "",
            "final_path": "model_reply",
        },
        reply_generation=SimpleNamespace(generate_validated_reply=lambda *_args, **_kwargs: "model reply"),
    )
    service = AgentService(bot)
    turn_context = SimpleNamespace(
        user_input="say hi",
        attachments=None,
        state={},
        metadata={},
    )

    result = asyncio.run(service.run_agent(turn_context, rich_context={}))

    assert result == ("model reply", False)
    assert turn_context.state["tool_execution_envelope"]["final_path"] == "model_reply"
    assert turn_context.state["tool_execution_envelope"]["selected_tool"] is None
