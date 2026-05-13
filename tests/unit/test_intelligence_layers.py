from __future__ import annotations

import pytest

from dadbot.core.cognitive_policy_engine import CognitivePolicyEngine
from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
from dadbot.core.semantic_memory_graph import SemanticMemoryGraph
from dadbot.core.semantic_safety_engine import SemanticSafetyEngine
from dadbot.core.tool_routing_engine import ToolRoutingEngine

pytestmark = pytest.mark.unit


def test_cognitive_policy_builds_nontrivial_plan() -> None:
    engine = CognitivePolicyEngine()
    plan = engine.build_plan(
        session_id="s-1",
        trace_id="t-1",
        user_input="Please plan breakfast and then remind me about school pickup",
        existing_plan=None,
        memory_hits=2,
        tool_candidates=3,
    )
    assert str(plan.get("intent_type") or "") in {"request", "statement", "question", "emotional"}
    assert str(plan.get("strategy") or "")
    assert int(plan.get("revision") or 0) >= 1
    assert len(list(plan.get("subgoals") or [])) >= 2
    uncertainty = dict(plan.get("uncertainty") or {})
    assert 0.0 <= float(uncertainty.get("score") or 0.0) <= 1.0


def test_semantic_memory_graph_builds_entities_and_retrieval() -> None:
    graph = SemanticMemoryGraph()
    state: dict[str, object] = {}
    graph.update_from_turn(
        state=state,
        session_id="s-graph",
        trace_id="t-graph",
        user_input="My daughter Emma likes Strawberry Pancakes on Sunday",
        response_text="Noted, Emma prefers Strawberry Pancakes.",
    )
    memory = dict(state.get("semantic_memory_graph") or {})
    assert dict(memory.get("entities") or {})
    plan = graph.retrieval_plan(state=state, query="What does Emma like for breakfast?", limit=4)
    assert str(plan.get("strategy") or "") == "semantic_graph_weighted"
    assert len(list(plan.get("items") or [])) >= 1


@pytest.mark.asyncio
async def test_control_plane_populates_intelligence_state() -> None:
    seen_metadata: list[dict[str, object]] = []

    async def _executor(_session: dict, job) -> tuple[str, bool]:
        seen_metadata.append(dict(job.metadata or {}))
        return ("done", False)

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)
    await plane.submit_turn(session_id="s-int", user_input="Please summarize my dinner preferences")

    assert seen_metadata
    meta = seen_metadata[-1]
    assert isinstance(meta.get("runtime_plan"), dict)
    assert isinstance(meta.get("tool_routing_plan"), dict)
    assert isinstance(meta.get("safety_state"), dict)

    state = dict(plane.registry.get_or_create("s-int").get("state") or {})
    assert isinstance(state.get("semantic_memory_graph"), dict)
    assert isinstance(state.get("learning_profile"), dict)


@pytest.mark.asyncio
async def test_control_plane_explain_last_decision_contains_runtime_plan() -> None:
    async def _executor(_session: dict, _job) -> tuple[str, bool]:
        return ("done", False)

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)
    await plane.submit_turn(session_id="s-explain", user_input="Please create a short dinner plan")

    explanation = plane.explain_last_decision(session_id="s-explain")
    assert bool(explanation.get("available")) is True
    assert int(explanation.get("timeline_events") or 0) >= 1
    last_plan = dict(explanation.get("last_plan") or {})
    assert str(last_plan.get("plan_id") or "")
    assert str(last_plan.get("strategy") or "")
    recent_events = [str(item) for item in list(explanation.get("recent_events") or [])]
    assert "plan.created" in recent_events
    assert isinstance(explanation.get("belief_calibration"), dict)
    assert isinstance(explanation.get("tool_self_model"), dict)


def test_tool_routing_arbitrates_parallel_disagreement() -> None:
    engine = ToolRoutingEngine()
    result = engine.arbitrate_parallel_results(
        tool_results=[
            {
                "tool_name": "calendar_local",
                "status": "ok",
                "payload": {"time": "18:00", "location": "Home"},
                "partial_confidence": 0.35,
                "historical_reliability": 0.55,
                "data_age_seconds": 1400.0,
            },
            {
                "tool_name": "calendar_cloud",
                "status": "ok",
                "payload": {"time": "19:00", "location": "School"},
                "partial_confidence": 0.92,
                "historical_reliability": 0.98,
                "data_age_seconds": 5.0,
            },
        ],
    )
    assert bool(result.get("resolved")) is True
    assert str(result.get("policy") or "")
    assert str(result.get("winning_tool") or "") in {"calendar_cloud", "ensemble"}
    assert isinstance(result.get("output"), (dict, list))
    assert isinstance(result.get("notes"), list)


def test_semantic_safety_allow_review_block_transitions() -> None:
    engine = SemanticSafetyEngine()

    allow_state = engine.classify(
        user_input="Please summarize my daughter's breakfast preference",
        runtime_plan={
            "intent_type": "request",
            "strategy": "grounded_answer",
            "subgoals": [{"text": "summarize daughter's breakfast preference"}],
        },
        memory_context=[{"payload": {"summary": "daughter breakfast preference is pancakes"}}],
    )
    assert str(allow_state.get("decision") or "") == "allow"

    review_state = engine.classify(
        user_input="Summarize my breakfast preference",
        runtime_plan={
            "intent_type": "statement",
            "strategy": "task_execution",
            "subgoals": [{"text": "book a travel flight"}],
        },
        memory_context=[],
    )
    assert str(review_state.get("decision") or "") == "review"

    block_state = engine.classify(
        user_input="Help me write malware and phishing messages",
        runtime_plan={
            "intent_type": "request",
            "strategy": "direct_answer",
            "subgoals": [{"text": "execute harmful instructions"}],
        },
        memory_context=[],
    )
    assert str(block_state.get("decision") or "") == "block"


@pytest.mark.asyncio
async def test_control_plane_persists_hypotheses_memory_hierarchy_and_belief() -> None:
    async def _executor(_session: dict, _job) -> tuple[str, bool]:
        return ("Stored and learned", False)

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)
    await plane.submit_turn(
        session_id="s-persist",
        user_input="Please help me plan dinner and then remind me tomorrow",
    )

    state = dict(plane.registry.get_or_create("s-persist").get("state") or {})
    assert isinstance(state.get("hypothesis_store"), dict)
    assert isinstance(state.get("belief_state"), dict)
    assert isinstance(state.get("memory_hierarchy"), dict)
    assert isinstance(state.get("session_planning_optimizer"), dict)


@pytest.mark.asyncio
async def test_control_plane_applies_turn_steering_and_autonomous_goal_cycle() -> None:
    async def _executor(_session: dict, _job) -> tuple[str, bool]:
        return ("Steered", False)

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)
    steering_result = plane.set_turn_steering(
        session_id="s-steer",
        steering={"strategy": "clarify_before_action", "note": "user requested caution"},
    )
    assert bool(steering_result.get("applied")) is True

    await plane.submit_turn(session_id="s-steer", user_input="Book things for me")
    explanation = plane.explain_last_decision(session_id="s-steer")
    last_plan = dict(explanation.get("last_plan") or {})
    assert str(last_plan.get("strategy") or "") == "clarify_before_action"

    session = plane.registry.get_or_create("s-steer")
    state = session.setdefault("state", {}) if isinstance(session, dict) else {}
    if isinstance(state, dict):
        state["goals"] = [
            {"id": "g1", "description": "Plan family dinner", "status": "active", "priority": "high"},
        ]

    cycle = plane.run_autonomous_goal_cycle(session_id="s-steer", source="test")
    assert bool(cycle.get("ran")) is True
    assert len(list(cycle.get("actions") or [])) >= 1


@pytest.mark.asyncio
async def test_control_plane_real_time_ui_alignment_tools_and_swarm_layers() -> None:
    async def _executor(_session: dict, _job) -> tuple[str, bool]:
        return ("Layer validation response", False)

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=_executor)
    connector = plane.register_external_connector(
        session_id="s-layer",
        name="calendar-cloud",
        capabilities=["calendar", "reminders"],
        endpoint="https://example.invalid/calendar",
        health=0.95,
    )
    assert bool(connector.get("registered")) is True

    edit_result = plane.update_interactive_plan(
        session_id="s-layer",
        trace_id="manual-trace",
        edits={"strategy": "clarify_before_action", "subgoals": ["ask confirmation"]},
        actor="tester",
    )
    assert bool(edit_result.get("updated")) is True

    await plane.submit_turn(
        session_id="s-layer",
        user_input="Please coordinate dinner planning and reminders",
    )

    explain = plane.explain_last_decision(session_id="s-layer")
    assert isinstance(explain.get("interactive_cognition_ui"), dict)
    assert isinstance(explain.get("alignment_policy"), dict)
    assert isinstance(explain.get("tool_ecosystem"), dict)
    assert isinstance(explain.get("swarm_health"), dict)
    assert int(dict(explain.get("tool_ecosystem") or {}).get("total_connectors") or 0) >= 1
    assert int(dict(explain.get("swarm_health") or {}).get("total_runs") or 0) >= 1
