"""Cognition primitive tests: PlannerNode, Goal System, CritiqueEngine, GoalAwareRanker.

Four independent test classes — one per primitive — plus an integration class that
verifies the full pipeline wiring (preflight → planner → inference → save).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import pytest

from dadbot.core.goals import (
    GoalPriority,
    GoalRecord,
    GoalStatus,
    GoalStore,
    detect_goal_in_input,
)
from dadbot.core.planner import (
    ComplexityLevel,
    IntentType,
    PlannerNode,
    ReplyStrategy,
    TurnPlan,
)
from dadbot.core.critic import CritiqueEngine, CritiqueResult, PASS_THRESHOLD
from dadbot.core.goal_scorer import GoalAwareRanker
from dadbot.core.graph import TurnContext
from dadbot.core.nodes import ToolExecutorNode, ToolRouterNode
from dadbot.core.orchestrator import DadBotOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(user_input: str = "hello", **state_kw) -> TurnContext:
    ctx = TurnContext(user_input=user_input, state=dict(state_kw))
    # Minimal temporal stub required by nodes that check for it.
    ctx.temporal = type("_Temporal", (), {
        "wall_time": "12:00:00",
        "wall_date": "2025-01-01",
        "timezone": "UTC",
        "utc_offset_minutes": 0,
        "epoch_seconds": time.time(),
        "is_virtual": False,
        "step_index": 0,
        "base_epoch": time.time(),
        "step_size_seconds": 1.0,
    })()
    return ctx


def _make_goal(description: str = "learn Python", **kw) -> GoalRecord:
    return GoalRecord(description=description, **kw)


# ---------------------------------------------------------------------------
# 1. Goal System
# ---------------------------------------------------------------------------

class TestGoalSystem:
    """Verify GoalRecord, GoalStore, and detection helper."""

    def test_goal_record_round_trip(self):
        """GoalRecord serialises to dict and back without data loss."""
        original = _make_goal("become a better developer", priority=GoalPriority.HIGH)
        original.progress_notes.append("Completed first chapter")
        original.tags.append("learning")

        d = original.to_dict()
        restored = GoalRecord.from_dict(d)

        assert restored.id == original.id
        assert restored.description == original.description
        assert restored.status == GoalStatus.ACTIVE
        assert restored.priority == GoalPriority.HIGH
        assert restored.progress_notes == ["Completed first chapter"]
        assert restored.tags == ["learning"]

    def test_goal_store_upsert_and_active_goals(self):
        """Upserted goals appear in active_goals until completed."""
        store = GoalStore()
        g = _make_goal("run a marathon")
        store.upsert(g)

        assert len(store.active_goals()) == 1
        assert store.active_goals()[0].id == g.id

    def test_goal_store_complete(self):
        """Completing a goal removes it from active_goals."""
        store = GoalStore()
        g = _make_goal("write a novel")
        store.upsert(g)

        completed = store.complete(g.id, turn_id="turn-42")

        assert completed is True
        assert len(store.active_goals()) == 0
        assert store.get(g.id).status == GoalStatus.COMPLETED
        assert store.get(g.id).completion_turn == "turn-42"

    def test_goal_store_abandon(self):
        """Abandoning a goal marks it abandoned, not deleted."""
        store = GoalStore()
        g = _make_goal("start a podcast")
        store.upsert(g)

        abandoned = store.abandon(g.id)

        assert abandoned is True
        assert store.get(g.id).status == GoalStatus.ABANDONED
        assert len(store.active_goals()) == 0

    def test_goal_store_add_note(self):
        """Progress notes accumulate on the goal record."""
        store = GoalStore()
        g = _make_goal("learn guitar")
        store.upsert(g)

        store.add_note(g.id, "Practiced chords for 30 minutes")
        store.add_note(g.id, "Learned first song")

        assert len(store.get(g.id).progress_notes) == 2

    def test_goal_store_serialise_roundtrip(self):
        """to_state / load_from_state is a lossless round-trip."""
        store = GoalStore()
        for desc in ["Goal A", "Goal B", "Goal C"]:
            store.upsert(_make_goal(desc))
        store.complete(store.active_goals()[0].id)

        serialised = store.to_state()

        restored_store = GoalStore()
        restored_store.load_from_state(serialised)

        assert restored_store.size() == 3
        # One completed, two active.
        assert len(restored_store.active_goals()) == 2

    def test_detect_goal_in_input_positive(self):
        """Goal detection triggers on common goal-marker phrases."""
        positive_cases = [
            "I want to get better at public speaking",
            "My goal is to save enough money for a house",
            "I'm trying to improve my relationship with my kids",
            "I need to lose some weight before summer",
            "Help me find a better work-life balance",
        ]
        for text in positive_cases:
            result = detect_goal_in_input(text)
            assert result is not None, f"Expected goal detection for: {text!r}"
            assert len(result.description) >= 5

    def test_detect_goal_in_input_negative(self):
        """Trivially short or non-goal sentences do not trigger detection."""
        negative_cases = [
            "Hi Dad",
            "How are you?",
            "Thanks!",
            "I want to",  # no continuation
        ]
        for text in negative_cases:
            result = detect_goal_in_input(text)
            assert result is None, f"Unexpected goal detection for: {text!r}"


# ---------------------------------------------------------------------------
# 2. PlannerNode
# ---------------------------------------------------------------------------

class TestPlannerNode:
    """Verify intent classification, plan generation, and goal detection."""

    @pytest.mark.asyncio
    async def test_question_intent_classified(self):
        """Questions are classified as QUESTION with DIRECT_ANSWER strategy."""
        planner = PlannerNode()
        ctx = _make_context("How do I get better at chess?")

        result = await planner.run(ctx)

        plan = result.state.get("turn_plan")
        assert plan is not None
        assert plan["intent_type"] == IntentType.QUESTION
        assert plan["strategy"] == ReplyStrategy.DIRECT_ANSWER

    @pytest.mark.asyncio
    async def test_emotional_intent_classified(self):
        """Emotional inputs are classified as EMOTIONAL_SHARE with EMPATHY_FIRST strategy."""
        planner = PlannerNode()
        ctx = _make_context("I've been feeling really anxious and stressed about my job")

        result = await planner.run(ctx)

        plan = result.state.get("turn_plan")
        assert plan["intent_type"] == IntentType.EMOTIONAL_SHARE
        assert plan["strategy"] == ReplyStrategy.EMPATHY_FIRST

    @pytest.mark.asyncio
    async def test_goal_intent_classified(self):
        """Goal-bearing inputs are classified as GOAL_ORIENTED with GOAL_TRACK strategy."""
        planner = PlannerNode()
        ctx = _make_context("I want to become a better father to my kids")

        result = await planner.run(ctx)

        plan = result.state.get("turn_plan")
        assert plan["intent_type"] == IntentType.GOAL_ORIENTED
        assert plan["strategy"] == ReplyStrategy.GOAL_TRACK

    @pytest.mark.asyncio
    async def test_casual_intent_classified(self):
        """Short greetings are classified as CASUAL."""
        planner = PlannerNode()
        ctx = _make_context("Hey")

        result = await planner.run(ctx)

        plan = result.state.get("turn_plan")
        assert plan["intent_type"] == IntentType.CASUAL

    @pytest.mark.asyncio
    async def test_planner_ran_metadata_flag(self):
        """PlannerNode stamps planner_ran=True in metadata."""
        planner = PlannerNode()
        ctx = _make_context("What time is it?")

        result = await planner.run(ctx)

        assert result.metadata.get("planner_ran") is True

    @pytest.mark.asyncio
    async def test_new_goal_detected_and_stored(self):
        """PlannerNode writes a new GoalRecord to context.state['new_goals']."""
        planner = PlannerNode()
        ctx = _make_context("I want to learn how to cook healthy meals")

        result = await planner.run(ctx)

        new_goals = result.state.get("new_goals", [])
        assert len(new_goals) >= 1
        assert isinstance(new_goals[0], dict)
        assert "description" in new_goals[0]
        assert result.state["turn_plan"]["new_goal_detected"] is True

    @pytest.mark.asyncio
    async def test_active_goals_matched_in_plan(self):
        """PlannerNode matches user input to active goals by keyword overlap."""
        planner = PlannerNode()
        goal_id = "abc123"
        ctx = _make_context(
            "I've been working on cooking, trying some new recipes",
            session_goals=[{"id": goal_id, "description": "learn to cook healthy meals", "status": "active"}],
        )

        result = await planner.run(ctx)

        plan = result.state["turn_plan"]
        assert goal_id in plan["active_goal_ids"]

    @pytest.mark.asyncio
    async def test_complex_multi_step_input(self):
        """Long, multi-clause inputs are classified as COMPLEX."""
        planner = PlannerNode()
        ctx = _make_context(
            "I want to first improve my diet and then start exercising regularly "
            "and also work on my sleep schedule so that I have more energy throughout the day"
        )

        result = await planner.run(ctx)

        plan = result.state["turn_plan"]
        assert plan["complexity"] == ComplexityLevel.COMPLEX

    @pytest.mark.asyncio
    async def test_planner_emits_memory_only_requests_when_v2_enabled(self):
        """Planner emits only goal/session MemoryTool requests when v2 is enabled."""
        planner = PlannerNode()
        ctx = _make_context(
            "I want to improve my routines",
            session_goals=[{"id": "g1", "description": "improve routines", "status": "active"}],
            tool_ir={"requests": []},
        )
        ctx.metadata["tool_system_v2_enabled"] = True

        result = await planner.run(ctx)

        requests = list(result.state.get("tool_ir", {}).get("requests") or [])
        assert len(requests) == 2
        assert [r.get("intent") for r in requests] == ["goal_lookup", "session_memory_fetch"]
        assert all(str(r.get("tool_name") or "") == "memory_lookup" for r in requests)

    def test_turn_plan_trivial_factory(self):
        """TurnPlan.trivial() returns a valid, minimal plan dict."""
        plan = TurnPlan.trivial()
        d = plan.to_dict()
        assert d["intent_type"] == IntentType.STATEMENT
        assert d["estimated_turns"] == 1
        assert d["subgoals"] == []


# ---------------------------------------------------------------------------
# 3. Strict Tool Router/Executor
# ---------------------------------------------------------------------------


class TestToolRoutingPipeline:
    """Verify strict compilation and deterministic execution boundaries."""

    @pytest.mark.asyncio
    async def test_router_strict_validation_dedup_and_order(self):
        router = ToolRouterNode()
        ctx = _make_context(
            "plan tools",
            tool_ir={
                "requests": [
                    {
                        "tool_name": "memory_lookup",
                        "args": {"query": "q", "scope": "session"},
                        "intent": "session_memory_fetch",
                        "expected_output": "session_memory_context",
                        "priority": 20,
                    },
                    {
                        "tool_name": "memory_lookup",
                        "args": {"query": "q", "scope": "goals", "goal_ids": ["g1"]},
                        "intent": "goal_lookup",
                        "expected_output": "goal_context",
                        "priority": 10,
                    },
                    {
                        "tool_name": "memory_lookup",
                        "args": {"query": "q", "scope": "session"},
                        "intent": "session_memory_fetch",
                        "expected_output": "session_memory_context",
                        "priority": 20,
                    },
                    {
                        "tool_name": "echo",
                        "args": {"message": "bad"},
                        "intent": "session_memory_fetch",
                        "expected_output": "ignored",
                        "priority": 1,
                    },
                ]
            },
        )

        result = await router.run(ctx)

        compiler = dict(result.state.get("tool_ir", {}).get("compiler") or {})
        plan = list(result.state.get("tool_ir", {}).get("execution_plan") or [])
        assert compiler.get("strict") is True
        assert compiler.get("compiled_count") == 2
        assert compiler.get("rejected_count") == 2
        assert [p.get("intent") for p in plan] == ["goal_lookup", "session_memory_fetch"]
        assert [p.get("sequence") for p in plan] == [0, 1]

    @pytest.mark.asyncio
    async def test_executor_logs_all_planned_outputs(self):
        executor = ToolExecutorNode()
        ctx = _make_context(
            "execute tools",
            session_goals=[{"id": "g1", "description": "goal", "status": "active"}],
            memories=[{"content": "memory one"}],
            tool_ir={
                "execution_plan": [
                    {
                        "sequence": 0,
                        "tool_name": "memory_lookup",
                        "args": {"query": "q", "scope": "goals", "goal_ids": ["g1"]},
                        "intent": "goal_lookup",
                        "expected_output": "goal_context",
                        "priority": 10,
                        "deterministic_id": "",  # filled below
                    },
                    {
                        "sequence": 1,
                        "tool_name": "memory_lookup",
                        "args": {"query": "q", "scope": "session"},
                        "intent": "session_memory_fetch",
                        "expected_output": "session_memory_context",
                        "priority": 20,
                        "deterministic_id": "",  # filled below
                    },
                ]
            },
        )
        for item in ctx.state["tool_ir"]["execution_plan"]:
            from dadbot.core.tool_ir import deterministic_tool_id

            item["deterministic_id"] = deterministic_tool_id(item["tool_name"], item["args"])

        result = await executor.run(ctx)

        executions = list(result.state.get("tool_ir", {}).get("executions") or [])
        tool_results = list(result.state.get("tool_results") or [])
        assert len(executions) == 2
        assert len(tool_results) == 2
        assert all(str(item.get("status") or "") == "ok" for item in tool_results)
        assert bool(result.metadata.get("tool_execution_graph_hash")) is True


# ---------------------------------------------------------------------------
# 4. CritiqueEngine
# ---------------------------------------------------------------------------

class TestCritiqueEngine:
    """Verify CritiqueEngine pass/fail heuristics and revision hints."""

    def test_good_reply_passes(self):
        """A substantive, on-topic reply should pass with a high score."""
        engine = CritiqueEngine()
        plan = {"strategy": "direct_answer", "intent_type": "question", "complexity": "simple"}

        result = engine.critique(
            "Chess improves through deliberate practice, studying openings, and analysing your games.",
            "How do I get better at chess?",
            plan,
            iteration=0,
        )

        assert result.passed is True
        assert result.score >= PASS_THRESHOLD

    def test_empty_reply_fails(self):
        """An empty reply should always fail."""
        engine = CritiqueEngine()
        plan = {"strategy": "direct_answer", "intent_type": "question", "complexity": "simple"}

        result = engine.critique("", "Tell me something.", plan, iteration=0)

        assert result.passed is False
        assert "reply_empty" in result.issues

    def test_fallback_reply_fails(self):
        """A fallback/error string should always fail."""
        engine = CritiqueEngine()
        plan = {"strategy": "direct_answer", "intent_type": "statement", "complexity": "simple"}

        result = engine.critique(
            "Something went sideways. Try again in a moment.",
            "Hello",
            plan,
            iteration=0,
        )

        assert result.passed is False
        assert "fallback_detected" in result.issues

    def test_emotional_reply_without_empathy_penalised(self):
        """A reply to an emotional input that misses empathy tokens is penalised."""
        engine = CritiqueEngine()
        plan = {"strategy": "empathy_first", "intent_type": "emotional", "complexity": "moderate"}

        result = engine.critique(
            "You should exercise and eat vegetables.",
            "I've been feeling really sad and lonely lately.",
            plan,
            iteration=0,
        )

        assert "missing_empathy" in result.issues
        assert result.score < 1.0

    def test_empathetic_reply_passes_emotional_strategy(self):
        """A reply with empathy tokens satisfies the emotional strategy check."""
        engine = CritiqueEngine()
        plan = {"strategy": "empathy_first", "intent_type": "emotional", "complexity": "moderate"}

        result = engine.critique(
            "I hear you — that sounds really difficult. It makes sense that you're struggling. I'm here for you.",
            "I've been feeling really sad and lonely lately.",
            plan,
            iteration=0,
        )

        assert "missing_empathy" not in result.issues

    def test_revision_hint_provided_on_failure(self):
        """Failed critiques always include a non-empty revision hint."""
        engine = CritiqueEngine()
        plan = {"strategy": "direct_answer", "intent_type": "statement", "complexity": "simple"}

        result = engine.critique("", "Hello", plan, iteration=0)

        assert result.revision_hint != ""

    def test_needs_revision_respects_max_iterations(self):
        """needs_revision returns False on the last allowed iteration."""
        engine = CritiqueEngine(max_iterations=2)
        plan = {"strategy": "direct_answer", "intent_type": "statement", "complexity": "simple"}

        # Iteration 0 of 2 (0-indexed) — can still revise.
        r0 = engine.critique("", "test", plan, iteration=0)
        assert engine.needs_revision(r0) is True

        # Last iteration (index 1 of 2) — no more revision allowed.
        r1 = engine.critique("", "test", plan, iteration=1)
        assert engine.needs_revision(r1) is False

    def test_tool_omission_detected_for_question_without_tool_plan(self):
        engine = CritiqueEngine()
        plan = {"strategy": "direct_answer", "intent_type": "question", "complexity": "simple"}

        result = engine.critique(
            "You can improve by practicing every day and reviewing mistakes.",
            "How do I improve?",
            plan,
            iteration=0,
            tool_ir={"execution_plan": []},
            tool_results=[],
        )

        assert "tool_omission_detected" in result.issues
        assert result.tool_necessity_score < 1.0

    def test_tool_correctness_detected_on_error_results(self):
        engine = CritiqueEngine()
        plan = {"strategy": "goal_track", "intent_type": "goal_oriented", "complexity": "moderate"}

        result = engine.critique(
            "Let's make a plan with your goal context.",
            "I want to save money.",
            plan,
            iteration=0,
            tool_ir={"execution_plan": [{"tool_name": "memory_lookup"}], "executions": [{"tool_name": "memory_lookup"}]},
            tool_results=[{"tool_name": "memory_lookup", "status": "error", "output": "boom"}],
        )

        assert "tool_correctness_low" in result.issues
        assert result.tool_correctness_score < 1.0

    def test_custom_pass_threshold(self):
        """Custom pass_threshold is respected."""
        strict = CritiqueEngine(pass_threshold=0.99)
        plan = {"strategy": "direct_answer", "intent_type": "question", "complexity": "simple"}
        # A normally passing reply may fail under strict=0.99.
        result = strict.critique("Short reply.", "What is love?", plan, iteration=0)
        # Score will be ≤ 1.0 but likely < 0.99 due to question mismatch or brevity.
        # We only assert the threshold is consulted.
        assert result.passed == (result.score >= 0.99 and not any(
            i in ("reply_empty", "fallback_detected") for i in result.issues
        ))


# ---------------------------------------------------------------------------
# 5. GoalAwareRanker
# ---------------------------------------------------------------------------

class TestGoalAwareRanker:
    """Verify goal-aware memory re-ranking promotes relevant entries."""

    def _mem(self, content: str) -> dict:
        return {"content": content}

    def test_relevant_memory_promoted(self):
        """A memory that overlaps with an active goal is ranked first."""
        ranker = GoalAwareRanker()
        memories = [
            self._mem("I enjoyed the movie last night"),            # unrelated
            self._mem("I have been practicing Python programming"),  # goal-relevant
            self._mem("We had pizza for dinner"),                    # unrelated
        ]
        goals = [{"id": "g1", "description": "learn Python programming", "status": "active"}]

        ranked = ranker.rerank(memories, goals)

        assert ranked[0]["content"] == "I have been practicing Python programming"

    def test_no_goals_preserves_order(self):
        """With no active goals, the original order is preserved."""
        ranker = GoalAwareRanker()
        memories = [self._mem("A"), self._mem("B"), self._mem("C")]

        ranked = ranker.rerank(memories, active_goals=[])

        assert [m["content"] for m in ranked] == ["A", "B", "C"]

    def test_no_memories_returns_empty(self):
        """Empty memory list always returns empty."""
        ranker = GoalAwareRanker()
        goals = [{"id": "g1", "description": "learn guitar", "status": "active"}]

        ranked = ranker.rerank([], goals)

        assert ranked == []

    def test_multiple_goals_combined(self):
        """All active goal tokens are combined for ranking."""
        ranker = GoalAwareRanker()
        memories = [
            self._mem("I bought new guitar strings"),
            self._mem("I cooked a healthy pasta"),
            self._mem("Read a book about nutrition"),
        ]
        goals = [
            {"id": "g1", "description": "learn guitar", "status": "active"},
            {"id": "g2", "description": "eat healthier nutrition", "status": "active"},
        ]

        ranked = ranker.rerank(memories, goals)
        ranked_texts = [m["content"] for m in ranked]

        # Both goal-relevant items (guitar, healthy/nutrition) should outrank generic one.
        assert "I bought new guitar strings" in ranked_texts[:2]
        # The irrelevant memory should be last.
        assert ranked_texts[-1] == "I cooked a healthy pasta" or ranked_texts[-1] == "I bought new guitar strings"

    def test_goal_relevance_scores_length_matches(self):
        """goal_relevance_scores returns one float per memory entry."""
        ranker = GoalAwareRanker()
        memories = [self._mem("x"), self._mem("y"), self._mem("z")]
        goals = [{"id": "g1", "description": "learn piano", "status": "active"}]

        scores = ranker.goal_relevance_scores(memories, goals)

        assert len(scores) == 3
        assert all(0.0 <= s <= ranker.max_boost for s in scores)

    def test_boost_capped_at_max_boost(self):
        """No memory receives more than max_boost even with heavy overlap."""
        ranker = GoalAwareRanker(max_boost=0.3)
        # Craft memory with lots of goal-matching tokens.
        memories = [self._mem("learn guitar play guitar guitar chords guitar lessons guitar practice")]
        goals = [{"id": "g1", "description": "learn guitar play guitar well", "status": "active"}]

        scores = ranker.goal_relevance_scores(memories, goals)

        assert scores[0] <= 0.3 + 1e-9

    def test_string_memories_supported(self):
        """Plain string memories are handled without errors."""
        ranker = GoalAwareRanker()
        memories = ["cooking healthy food", "watching TV", "running daily"]
        goals = [{"id": "g1", "description": "eat healthier cooking", "status": "active"}]

        ranked = ranker.rerank(memories, goals)

        assert ranked[0] == "cooking healthy food"


# ---------------------------------------------------------------------------
# 5. Integration: pipeline wiring
# ---------------------------------------------------------------------------

@pytest.fixture
def orchestrator(bot) -> DadBotOrchestrator:
    return bot.turn_orchestrator


@pytest.fixture(autouse=True)
def fast_deterministic_agent(orchestrator: DadBotOrchestrator, monkeypatch):
    """Stub expensive services so cognition tests run in milliseconds."""
    service = orchestrator.registry.get("agent_service")

    async def _agent(context: TurnContext, _rich: dict) -> tuple[str, bool]:
        # Return a reasonable reply that contains some question-answer overlap.
        user = str(context.user_input or "")
        hint = str(context.state.get("_critique_revision_context") or "")
        reply = f"Here is my answer about {user[:40]}."
        if hint:
            reply = f"Revised: {reply}"
        return (reply, False)

    monkeypatch.setattr(service, "run_agent", _agent)

    bot_obj = orchestrator.bot
    mc = getattr(bot_obj, "memory_coordinator", None)
    if mc is not None:
        monkeypatch.setattr(mc, "consolidate_memories", lambda **kw: None)
        monkeypatch.setattr(mc, "apply_controlled_forgetting", lambda **kw: None)
    rm = getattr(bot_obj, "relationship_manager", None)
    if rm is not None:
        monkeypatch.setattr(rm, "materialize_projection", lambda **kw: None)
    mm = getattr(bot_obj, "memory_manager", None)
    gm = getattr(mm, "graph_manager", None) if mm is not None else None
    if gm is not None:
        monkeypatch.setattr(gm, "sync_graph_store", lambda **kw: None)
    if hasattr(bot_obj, "validate_reply"):
        monkeypatch.setattr(bot_obj, "validate_reply", lambda _u, r: r)
    if hasattr(bot_obj, "current_runtime_health_snapshot"):
        monkeypatch.setattr(bot_obj, "current_runtime_health_snapshot", lambda **kw: {})


async def _run(orch: DadBotOrchestrator, user_input: str, *, session_id: str = "default"):
    result = await orch.handle_turn(user_input, session_id=session_id)
    ctx = getattr(orch, "_last_turn_context", None)
    return result, ctx


class TestCognitionPipelineIntegration:
    """End-to-end integration tests through the full pipeline."""

    @pytest.mark.asyncio
    async def test_planner_ran_flag_set_in_pipeline(self, orchestrator: DadBotOrchestrator):
        """PlannerNode runs and stamps planner_ran in metadata."""
        _, ctx = await _run(orchestrator, "How do I become more productive?", session_id="cog-test-1")
        assert ctx is not None
        assert ctx.metadata.get("planner_ran") is True

    @pytest.mark.asyncio
    async def test_turn_plan_present_in_context_state(self, orchestrator: DadBotOrchestrator):
        """context.state contains a turn_plan dict after a turn."""
        _, ctx = await _run(orchestrator, "What is the meaning of life?", session_id="cog-test-2")
        assert ctx is not None
        plan = ctx.state.get("turn_plan")
        assert isinstance(plan, dict)
        assert "intent_type" in plan
        assert "strategy" in plan

    @pytest.mark.asyncio
    async def test_new_goal_persisted_to_session(self, orchestrator: DadBotOrchestrator):
        """A goal-bearing input causes a GoalRecord to be stored in session state."""
        sid = "cog-goals-" + uuid.uuid4().hex[:8]
        await _run(orchestrator, "I want to get better at managing my finances", session_id=sid)

        session = orchestrator.session_registry.get_or_create(sid)
        goals = session.get("state", {}).get("goals", [])
        assert len(goals) >= 1
        assert any("financ" in g.get("description", "").lower() for g in goals), \
            f"Expected finance-related goal, got: {goals}"

    @pytest.mark.asyncio
    async def test_persisted_goals_loaded_next_turn(self, orchestrator: DadBotOrchestrator):
        """Goals persisted in session state are loaded into context.state['session_goals'] on subsequent turns."""
        sid = "cog-load-goals-" + uuid.uuid4().hex[:8]
        # Turn 1: introduce a goal.
        await _run(orchestrator, "I want to learn how to meditate", session_id=sid)

        # Turn 2: goals should now be in context.state.
        _, ctx2 = await _run(orchestrator, "Any progress on that?", session_id=sid)
        assert ctx2 is not None
        session_goals = ctx2.state.get("session_goals", [])
        assert len(session_goals) >= 1

    @pytest.mark.asyncio
    async def test_critique_record_present_in_context(self, orchestrator: DadBotOrchestrator):
        """CritiqueEngine ran and stored a critique_record in context.state."""
        _, ctx = await _run(orchestrator, "Tell me something interesting.", session_id="cog-critique")
        assert ctx is not None
        # critique_record is set by InferenceNode when CritiqueEngine is wired.
        record = ctx.state.get("critique_record")
        assert isinstance(record, dict), "Expected critique_record in context state"
        assert "score" in record
        assert "passed" in record

    @pytest.mark.asyncio
    async def test_candidate_still_produced_after_critique_loop(self, orchestrator: DadBotOrchestrator):
        """InferenceNode always produces a candidate even after the critique loop."""
        _, ctx = await _run(orchestrator, "How are you doing today?", session_id="cog-candidate")
        assert ctx is not None
        candidate = ctx.state.get("candidate")
        assert candidate is not None
