from datetime import date, datetime

import pytest
pytestmark = pytest.mark.unit
from pydantic import ValidationError

from dadbot.models import ActiveThreadSnapshot, AgenticToolPlan, BackgroundTaskOverview, BackgroundTaskRecord, ChatThreadState, ConsolidatedMemory, DashboardStatusSnapshot, MemoryEntry, MemoryStore, ModerationSnapshot, OutputModerationDecision, PersonaTrait, PersistenceStatusSnapshot, PlannerDebugState, RelationshipState, RelationshipStatusSnapshot, ReplySupervisorSnapshot, RuntimeStatusSnapshot, SecurityStatusSnapshot, ServiceStatusSnapshot, SessionStatusSnapshot, StatusTraitMetric, SupervisorDecisionState, SupervisorJudgment, ThreadRuntimeSnapshot, ThreadsStatusSnapshot, TurnPipelineSnapshot, TurnPipelineStep, VisionStatusSnapshot, WisdomInsight


def test_memory_entry_defaults_and_json_dump():
    entry = MemoryEntry(
        summary="Tony wants to save more money.",
        created_at=datetime(2026, 4, 20, 9, 30),
        updated_at=datetime(2026, 4, 20, 9, 30),
    )

    dumped = entry.model_dump(mode="json")

    assert entry.category == "general"
    assert entry.mood == "neutral"
    assert entry.confidence == 0.5
    assert dumped["created_at"] == "2026-04-20T09:30:00"
    assert dumped["contradictions"] == []


def test_relationship_state_defaults():
    state = RelationshipState(
        last_hypothesis_updated=datetime(2026, 4, 20, 12, 0),
        last_updated=datetime(2026, 4, 20, 12, 0),
    )

    assert state.trust_level == 50
    assert state.openness_level == 50
    assert state.emotional_momentum == "steady"
    assert state.active_hypothesis == "supportive_baseline"


def test_relationship_state_accepts_runtime_warming_vocab():
    state = RelationshipState(
        emotional_momentum="warming",
        last_hypothesis_updated=date(2026, 4, 20),
        last_updated=date(2026, 4, 20),
    )

    dumped = state.model_dump(mode="json")

    assert state.emotional_momentum == "warming"
    assert dumped["last_updated"] == "2026-04-20"


def test_supervisor_judgment_enforces_score_bounds():
    with pytest.raises(ValidationError):
        SupervisorJudgment(
            approved=True,
            score=11,
            dad_likeness=8,
            groundedness=8,
            emotional_fit=8,
        )


def test_persona_trait_and_wisdom_insight_models_construct():
    trait = PersonaTrait(
        trait="steady reassurance",
        announcement="I have become a little steadier with you.",
        session_count=8,
        applied_at=datetime(2026, 4, 20, 15, 0),
        last_reinforced_at=datetime(2026, 4, 20, 15, 30),
        strength=1.7,
        impact_score=1.2,
        reason="Repeated positive relationship feedback.",
        critique_score=8,
        critique_feedback="Specific and steady.",
    )
    insight = WisdomInsight(
        summary="Tony responds well to calm validation before advice.",
        topic="support style",
        trigger="stress check-in",
        created_at=datetime(2026, 4, 20, 15, 5),
    )

    assert trait.trait == "steady reassurance"
    assert trait.session_count == 8
    assert insight.topic == "support style"


def test_memory_store_model_serializes_nested_runtime_shapes():
    store = MemoryStore(
        memories=[
            MemoryEntry(
                summary="Tony wants to save more money.",
                category="finance",
                mood="positive",
                created_at=date(2026, 4, 20),
                updated_at=date(2026, 4, 20),
            )
        ],
        consolidated_memories=[
            ConsolidatedMemory(
                summary="Tony has been saving for an emergency fund.",
                category="finance",
                source_count=2,
                confidence=0.84,
                supporting_summaries=["Tony has been budgeting weekly."],
                updated_at=date(2026, 4, 20),
            )
        ],
        relationship_state=RelationshipState(
            emotional_momentum="warming",
            last_hypothesis_updated=date(2026, 4, 20),
            last_updated=date(2026, 4, 20),
        ),
        last_mood_updated_at=date(2026, 4, 20),
    )

    dumped = store.model_dump(mode="json")

    assert dumped["consolidated_memories"][0]["source_count"] == 2
    assert dumped["relationship_state"]["emotional_momentum"] == "warming"
    assert dumped["last_mood_updated_at"] == "2026-04-20"


def test_operational_runtime_models_serialize_cleanly():
    planner = PlannerDebugState(
        updated_at="2026-04-20T15:00:00",
        user_input="Can you check the weather?",
        current_mood="stressed",
        planner_status="used_tool",
        planner_tool="web_search",
        planner_parameters={"query": "weather in Boston"},
        final_path="planner_tool",
    )
    moderation = OutputModerationDecision(
        approved=False,
        action="rewrite",
        category="unsafe_output",
        source="heuristic",
        reason="Matched blocked pattern.",
        revised_reply="I want to be careful here, buddy.",
    )
    task = BackgroundTaskRecord(
        task_id="task-1",
        session_id="session-1",
        task_kind="semantic-index",
        status="completed",
        metadata={"memory_count": 1},
        created_at="2026-04-20T15:00:00",
        updated_at="2026-04-20T15:00:01",
        completed_at="2026-04-20T15:00:01",
    )
    thread = ThreadRuntimeSnapshot(
        history=[{"role": "system", "content": "Dad system prompt"}],
        planner_debug=planner,
    )
    chat_thread = ChatThreadState(
        thread_id="thread-1",
        title="Chat 1",
        created_at="2026-04-20T15:00:00",
        updated_at="2026-04-20T15:00:01",
    )
    tool_plan = AgenticToolPlan(
        needs_tool=True,
        tool="web_search",
        parameters={"query": "weather in Boston"},
        reason="Needs current information.",
    )
    supervisor_snapshot = ReplySupervisorSnapshot(
        enabled=True,
        last_decision=SupervisorDecisionState(stage="alignment_judge", score=8, dad_likeness=8, groundedness=9, emotional_fit=8),
    )
    dashboard = DashboardStatusSnapshot(
        status=RuntimeStatusSnapshot(active_model="llama3.2", tenant_id="default", top_trait_metrics=[StatusTraitMetric(trait="steady", strength=1.2, impact_score=2.0)]),
        service=ServiceStatusSnapshot(status="ok", reachable=True),
        persistence=PersistenceStatusSnapshot(enabled=True, primary_store="tenant_document_store", profile_backend="tenant_document_store", memory_backend="tenant_document_store", json_mirror_enabled=True),
        moderation=ModerationSnapshot(last_decision=moderation),
        background_tasks=BackgroundTaskOverview(recent=[task]),
        security=SecurityStatusSnapshot(require_pin=False, has_pin_hint=False),
        session=SessionStatusSnapshot(turn_count=1, history_messages=2),
        threads=ThreadsStatusSnapshot(total=1, open=1, closed=0),
        active_thread=ActiveThreadSnapshot(thread_id="thread-1", title="Chat 1"),
        relationship=RelationshipStatusSnapshot(),
        supervisor=supervisor_snapshot,
        vision=VisionStatusSnapshot(ready=False, message="fallback"),
        turn_pipeline=TurnPipelineSnapshot(mode="sync", user_input="hello", steps=[TurnPipelineStep(name="prepare", status="completed")]),
    )

    assert planner.model_dump(mode="json")["planner_tool"] == "web_search"
    assert moderation.model_dump(mode="json")["action"] == "rewrite"
    assert task.model_dump(mode="json")["metadata"]["memory_count"] == 1
    assert thread.model_dump(mode="json")["planner_debug"]["planner_status"] == "used_tool"
    assert chat_thread.turn_count == 0
    assert tool_plan.parameters["query"] == "weather in Boston"
    assert dashboard.model_dump(mode="json")["persistence"]["primary_store"] == "tenant_document_store"
    assert dashboard.model_dump(mode="json")["turn_pipeline"]["steps"][0]["name"] == "prepare"