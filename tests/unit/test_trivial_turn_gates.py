import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dadbot.mood import MoodManager
from dadbot.services._tool_pipeline_mixin import ToolPipelineMixin

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("asynchronous", [False, True])
def test_factual_turn_skips_mood_model(asynchronous):
    bot = SimpleNamespace(
        LIGHT_MODE=False, runtime_config=SimpleNamespace(mood_fastpath_confidence_threshold=2),
        normalize_mood=lambda mood: mood,
        normalize_mood_detection_key=lambda text, history: text,
        _recent_mood_detections={}, call_ollama_chat=Mock(), call_ollama_chat_async=AsyncMock(),
    )
    manager = MoodManager(bot)
    mood = asyncio.run(manager.detect_async("2+2")) if asynchronous else manager.detect("2+2")
    assert mood == "neutral"
    bot.call_ollama_chat.assert_not_called()
    bot.call_ollama_chat_async.assert_not_called()


@pytest.mark.parametrize("text", [
    "I felt uneasy", "I'm having a hard day", "I'm happy", "I'm stressed",
    "I can barely think straight anymore", "I can't keep up",
], ids=["felt", "hard-day", "happy", "pressure", "fatigue", "implicit-pressure"])
def test_emotional_turns_remain_eligible_for_mood_inference(text):
    assert MoodManager.needs_model_detection(text)


def test_mood_followup_retains_recent_user_emotional_context():
    assert MoodManager.needs_model_detection("still the same", [
        {"role": "user", "content": "I felt sad yesterday"},
    ])
    assert not MoodManager.needs_model_detection("2+2", [
        {"role": "assistant", "content": "Are you sad?"},
    ])
    assert not MoodManager.needs_model_detection("What is 2+2?", [
        {"role": "user", "content": "I felt sad yesterday"},
    ])


@pytest.mark.parametrize("asynchronous", [False, True])
def test_factual_turn_skips_planner_transport(asynchronous):
    service = ToolPipelineMixin()
    service.bot = SimpleNamespace(
        agentic_tool_settings=lambda: {"enabled": True, "auto_web_lookup": True},
        get_available_tools=lambda: [{"function": {"name": "web_search"}}],
        parse_tool_command=lambda _text: None, prompt_history=lambda: [],
        update_planner_debug=Mock(),
    )
    service._llm_adapter = SimpleNamespace(call=Mock(), call_async=AsyncMock())
    method = service.plan_agentic_tools_async if asynchronous else service.plan_agentic_tools
    result = method("2+2", "neutral")
    if asynchronous:
        result = asyncio.run(result)
    assert result == (None, None)
    service._llm_adapter.call.assert_not_called()
    service._llm_adapter.call_async.assert_not_called()
    assert service.bot.update_planner_debug.call_args.kwargs["planner_status"] == "skipped"


@pytest.mark.parametrize("text", [
    "remind me tomorrow", "schedule a meeting", "turn on the lights", "add milk",
    "check my calendar", "email my notes", "search the web", "use inventory",
])
def test_tool_intents_keep_planning(text):
    service = ToolPipelineMixin()
    service.bot = SimpleNamespace(prompt_history=lambda: [])
    assert service._needs_tool_planning(text, [{"function": {"name": "get_inventory"}}])


def test_tool_followups_and_attachments_keep_planning():
    service = ToolPipelineMixin()
    service.bot = SimpleNamespace(prompt_history=lambda: [
        {"role": "user", "content": "schedule a meeting"},
        {"role": "assistant", "content": "Tomorrow?"},
    ])
    assert service._needs_tool_planning("yes please", [])
    assert service._needs_tool_planning("this one", [], attachments=[{"type": "image"}])
    assert not service._needs_tool_planning("2+2", [])


def test_configured_mcp_skips_planning_for_factual_turn(monkeypatch):
    from dadbot.runtime.mcp.approval_service import MCPApprovalService

    monkeypatch.setenv("DADBOT_HA_MCP_URL", "http://127.0.0.1:9999/mcp")
    monkeypatch.setenv("DADBOT_HA_MCP_ALLOWED_TOOLS", "light_control")
    generate = Mock(side_effect=AssertionError("Factual turns must not invoke the MCP planner"))
    service = MCPApprovalService(SimpleNamespace(model_port=SimpleNamespace(generate=generate)))
    result = asyncio.run(service.generate_candidate(
        SimpleNamespace(user_input="2+2", attachments=None, state={}),
        {"messages": [{"role": "user", "content": "2+2"}]},
    ))
    assert result is None
    generate.assert_not_called()


def test_mcp_graph_node_allows_service_to_decline_non_tool_turn():
    from dadbot.core.graph_pipeline_nodes import MCPToolNode

    service = SimpleNamespace(is_enabled=True, generate_candidate=AsyncMock(return_value=None))
    registry = SimpleNamespace(get=lambda *_args, **_kwargs: service)
    context = SimpleNamespace(state={})
    asyncio.run(MCPToolNode().execute(registry, context))
    assert "candidate" not in context.state
    assert "mcp_tool_candidate_ready" not in context.state


def test_trimming_accounts_for_suffix_tokens_without_stalling():
    from dadbot.managers.state_prompt_budget_mixin import _StatePromptBudgetMixin

    manager = SimpleNamespace(bot=SimpleNamespace(
        ACTIVE_MODEL="test-model",
        current_tokenizer=lambda **_kwargs: SimpleNamespace(encode=list, decode="".join),
        estimate_token_count=len,
    ))
    trimmed = _StatePromptBudgetMixin.trim_text_to_token_budget(manager, "abcdefghi", 6)
    assert trimmed == "abcdef"
    assert len(trimmed) <= 6


@pytest.mark.parametrize("strategy,single_pass", [
    ("direct_answer", True), ("emotional_support", True), ("empathy_first", True),
    ("task_plan", False), ("clarify", False),
])
def test_simple_strategies_keep_critique_but_skip_revision(strategy, single_pass):
    from dadbot.core.nodes import InferenceNode

    critique = Mock(return_value=SimpleNamespace(
        passed=False, score=0.1, issues=["missing_detail"], revision_hint="Expand the answer",
    ))
    node = InferenceNode(None, critique_engine=SimpleNamespace(critique=critique))
    context = SimpleNamespace(user_input="hello", state={"turn_plan": {"strategy": strategy}})
    assert node._run_critique_check(context, ("hello", False), 0) is single_pass
    critique.assert_called_once()
    assert context.state["critique_record"]["passed"] is False
    assert ("_critique_revision_context" in context.state) is not single_pass


@pytest.mark.parametrize("payload", [
    '{"mood":"invented","reply":"hello"}', '{"mood":"sad","reply":""}',
    '{"mood":"sad","reply":"hello","extra":"hidden"}', "plain text",
])
def test_batched_mood_requires_valid_structured_output(payload):
    from pydantic import ValidationError
    from dadbot.services.agent_service import _EmotionalAnswer

    with pytest.raises(ValidationError):
        _EmotionalAnswer.model_validate_json(payload)
