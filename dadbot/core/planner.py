"""PlannerNode: structured intent decomposition before inference.

Sits in the pipeline after the preflight group (Health + ContextBuilder)
and before InferenceNode.  It performs lightweight heuristic analysis of
the user input and produces a ``TurnPlan`` stored in
``context.state["turn_plan"]``.  InferenceNode reads this plan to select
the right strategy; the CritiqueEngine uses it to evaluate candidates.

No LLM calls are made here — everything is pattern-matching + statistics.
This keeps PlannerNode latency near-zero while still adding cognitive
structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dadbot.core.graph import TurnContext


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------


class IntentType(StrEnum):
    STATEMENT = "statement"  # sharing a fact or observation
    QUESTION = "question"  # seeking information or clarification
    REQUEST = "request"  # asking Dad to do or advise something
    EMOTIONAL_SHARE = "emotional"  # expressing feelings, seeking support
    GOAL_ORIENTED = "goal_oriented"  # relates to a long-running objective
    MULTI_STEP = "multi_step"  # complex task requiring multiple steps
    CASUAL = "casual"  # small talk, greetings


class ComplexityLevel(StrEnum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class ReplyStrategy(StrEnum):
    DIRECT_ANSWER = "direct_answer"  # answer the question concisely
    EMPATHY_FIRST = "empathy_first"  # acknowledge feelings before advising
    TASK_PLAN = "task_plan"  # lay out actionable steps
    GOAL_TRACK = "goal_track"  # connect to active goals, show progress
    CLARIFY = "clarify"  # ask for more info before answering


# ---------------------------------------------------------------------------
# TurnPlan value object
# ---------------------------------------------------------------------------


@dataclass
class TurnPlan:
    """Structured decomposition of a turn's intent and required response strategy."""

    intent_type: str
    complexity: str
    subgoals: list[str]
    strategy: str
    active_goal_ids: list[str]
    estimated_turns: int
    new_goal_detected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_type": str(self.intent_type),
            "complexity": str(self.complexity),
            "subgoals": list(self.subgoals),
            "strategy": str(self.strategy),
            "active_goal_ids": list(self.active_goal_ids),
            "estimated_turns": int(self.estimated_turns),
            "new_goal_detected": bool(self.new_goal_detected),
        }

    @classmethod
    def trivial(cls) -> TurnPlan:
        """Return a minimal plan for turns where classification is not needed."""
        return cls(
            intent_type=IntentType.STATEMENT,
            complexity=ComplexityLevel.SIMPLE,
            subgoals=[],
            strategy=ReplyStrategy.DIRECT_ANSWER,
            active_goal_ids=[],
            estimated_turns=1,
        )


# ---------------------------------------------------------------------------
# Pattern banks (pre-compiled for speed)
# ---------------------------------------------------------------------------

_QUESTION_STARTS = re.compile(
    r"^(?:how|what|when|where|why|who|which|can you|could you|do you|"
    r"is there|are there|should i|would you|will you)\b",
    re.IGNORECASE,
)
_QUESTION_MARK = re.compile(r"\?")

_EMOTIONAL_TOKENS = frozenset(
    {
        "feel",
        "feeling",
        "feelings",
        "felt",
        "anxious",
        "anxiety",
        "stressed",
        "stress",
        "sad",
        "unhappy",
        "depressed",
        "upset",
        "angry",
        "frustrated",
        "scared",
        "afraid",
        "worried",
        "worry",
        "nervous",
        "lonely",
        "hurt",
        "excited",
        "happy",
        "proud",
        "overwhelmed",
        "exhausted",
        "tired",
        "hopeless",
        "hopeful",
        "confused",
        "lost",
        "broken",
        "struggling",
    },
)

_REQUEST_STARTS = re.compile(
    r"^(?:please|can you|could you|would you|help me|assist me|advise me|"
    r"tell me|show me|remind me|explain|teach me|guide me)\b",
    re.IGNORECASE,
)

_GOAL_MARKERS = re.compile(
    r"(?:i want|i'd like|i wish|my goal|my objective|i aim|i'm trying|"
    r"i need to|i have to|i plan to|i intend to|i hope to|i'm working on)",
    re.IGNORECASE,
)

_MULTI_STEP_MARKERS = re.compile(
    r"(?:first.*then|step by step|multiple|several|few things|"
    r"a few|also|additionally|furthermore|and then|after that)",
    re.IGNORECASE,
)

_CASUAL_TOKENS = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "howdy",
        "sup",
        "yo",
        "morning",
        "evening",
        "night",
        "bye",
        "goodbye",
        "later",
        "thanks",
        "thank",
        "cool",
        "awesome",
        "nice",
        "ok",
        "okay",
        "sure",
        "yep",
        "nope",
    },
)

_CLARIFY_TOKENS = frozenset(
    {
        "not sure",
        "don't know",
        "unclear",
        "confused about",
        "what do you mean",
        "can you clarify",
        "i'm not certain",
    },
)


# ---------------------------------------------------------------------------
# PlannerNode
# ---------------------------------------------------------------------------


class PlannerNode:
    """Decomposes user intent into a structured TurnPlan before inference.

    The plan is stored in ``context.state["turn_plan"]`` and used by:
    - InferenceNode: to build a strategy-aware prompt prefix
    - CritiqueEngine: to evaluate reply quality against the plan
    - GoalStore: to detect and register new user objectives
    """

    name = "planner"

    def __init__(self) -> None:
        pass

    async def run(self, context: TurnContext) -> TurnContext:
        from dadbot.core.goals import detect_goal_in_input

        user_input = str(context.user_input or "").strip()
        active_goals = list(context.state.get("session_goals") or [])

        intent_type = self._classify_intent(user_input)
        complexity = self._estimate_complexity(user_input, active_goals)
        subgoals = self._extract_subgoals(user_input, intent_type, complexity)
        strategy = self._select_strategy(intent_type, complexity)
        active_goal_ids = self._match_active_goals(user_input, active_goals)

        # Detect if this turn introduces a new user goal.
        new_goal = detect_goal_in_input(user_input)
        new_goal_detected = new_goal is not None
        if new_goal is not None:
            new_goal.source_turn = str(context.trace_id or "")
            existing_new = list(context.state.get("new_goals") or [])
            existing_new.append(new_goal.to_dict())
            context.state["new_goals"] = existing_new

        plan = TurnPlan(
            intent_type=intent_type,
            complexity=complexity,
            subgoals=subgoals,
            strategy=strategy,
            active_goal_ids=active_goal_ids,
            estimated_turns=2 if complexity == ComplexityLevel.COMPLEX else 1,
            new_goal_detected=new_goal_detected,
        )
        context.state["turn_plan"] = plan.to_dict()

        # Phase 3: Planner emits only MemoryTool requests behind the v2 switch.
        if bool(context.metadata.get("tool_system_v2_enabled", False)):
            tool_ir = dict(context.state.get("tool_ir") or {})
            tool_ir["requests"] = self._memory_tool_requests(
                user_input=user_input,
                active_goal_ids=active_goal_ids,
            )
            context.state["tool_ir"] = tool_ir

        context.metadata["planner_ran"] = True
        context.metadata["intent_type"] = str(intent_type)
        return context

    def _memory_tool_requests(
        self,
        *,
        user_input: str,
        active_goal_ids: list[str],
    ) -> list[dict[str, Any]]:
        goal_lookup = {
            "tool_name": "memory_lookup",
            "args": {
                "query": user_input,
                "scope": "goals",
                "goal_ids": list(active_goal_ids),
            },
            "intent": "goal_lookup",
            "expected_output": "goal_context",
            "priority": 10,
        }
        session_fetch = {
            "tool_name": "memory_lookup",
            "args": {
                "query": user_input,
                "scope": "session",
            },
            "intent": "session_memory_fetch",
            "expected_output": "session_memory_context",
            "priority": 20,
        }
        return [goal_lookup, session_fetch]

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    def _classify_intent(self, user_input: str) -> str:
        """Heuristic multi-class intent classifier. Order matters: most specific first."""
        text = user_input.lower().strip()
        tokens = set(re.split(r"\W+", text))

        # Emotional sharing: feelings vocabulary is high-signal.
        if tokens & _EMOTIONAL_TOKENS:
            return IntentType.EMOTIONAL_SHARE

        # Goal statement.
        if _GOAL_MARKERS.search(text):
            return IntentType.GOAL_ORIENTED

        # Explicit question.
        if _QUESTION_MARK.search(text) or _QUESTION_STARTS.match(text):
            return IntentType.QUESTION

        # Request / directive.
        if _REQUEST_STARTS.match(text):
            return IntentType.REQUEST

        # Multi-step complex task.
        if _MULTI_STEP_MARKERS.search(text) and len(text.split()) > 15:
            return IntentType.MULTI_STEP

        # Casual greeting / acknowledgement.
        if tokens & _CASUAL_TOKENS and len(tokens) <= 5:
            return IntentType.CASUAL

        return IntentType.STATEMENT

    def _estimate_complexity(self, user_input: str, active_goals: list) -> str:
        words = len(user_input.split())
        has_conjunction = bool(
            re.search(r"\band\b.*\band\b|\bfirst\b.*\bthen\b", user_input, re.IGNORECASE),
        )
        has_active_goals = len(active_goals) > 0

        if words > 40 or (has_conjunction and words > 20) or has_active_goals:
            return ComplexityLevel.COMPLEX
        if words > 12 or has_conjunction:
            return ComplexityLevel.MODERATE
        return ComplexityLevel.SIMPLE

    def _extract_subgoals(
        self,
        user_input: str,
        intent_type: str,
        complexity: str,
    ) -> list[str]:
        """Break a complex/multi-step input into atomic sub-goals."""
        if complexity == ComplexityLevel.SIMPLE:
            return []
        # Split on common coordinators.
        parts = re.split(
            r"(?:\band\b|\bthen\b|\bnext\b|\balso\b|\badditionally\b|\bafter that\b)",
            user_input,
            flags=re.IGNORECASE,
        )
        subgoals = [p.strip().strip(",.;") for p in parts if len(p.strip()) > 8]
        return subgoals[:6]  # cap to prevent noise

    def _select_strategy(self, intent_type: str, complexity: str) -> str:
        """Map intent + complexity to a reply strategy."""
        if intent_type == IntentType.EMOTIONAL_SHARE:
            return ReplyStrategy.EMPATHY_FIRST
        if intent_type == IntentType.GOAL_ORIENTED:
            return ReplyStrategy.GOAL_TRACK
        if intent_type == IntentType.MULTI_STEP or complexity == ComplexityLevel.COMPLEX:
            return ReplyStrategy.TASK_PLAN
        if intent_type == IntentType.QUESTION:
            return ReplyStrategy.DIRECT_ANSWER
        return ReplyStrategy.DIRECT_ANSWER

    def _match_active_goals(
        self,
        user_input: str,
        active_goals: list[dict],
    ) -> list[str]:
        """Return IDs of active goals whose description overlaps with the user input.

        Uses prefix matching (4-char prefix) so inflected forms like
        'cooking' match goal token 'cook'.
        """
        if not active_goals:
            return []
        _SKIP = {
            "i",
            "a",
            "the",
            "is",
            "to",
            "of",
            "in",
            "it",
            "my",
            "me",
            "been",
            "some",
            "new",
        }
        text_tokens = set(re.split(r"\W+", user_input.lower())) - _SKIP
        # Build a set of 4-char prefixes from user input tokens (length >= 4).
        text_prefixes = {t[:4] for t in text_tokens if len(t) >= 4}
        matched: list[str] = []
        for goal in active_goals:
            desc = str(
                goal.get("description", "") if isinstance(goal, dict) else "",
            ).lower()
            goal_tokens = set(re.split(r"\W+", desc)) - _SKIP
            goal_prefixes = {t[:4] for t in goal_tokens if len(t) >= 4}
            # Match on either exact token overlap OR 4-char prefix overlap.
            if (text_tokens & goal_tokens) or (text_prefixes & goal_prefixes):
                gid = goal.get("id") if isinstance(goal, dict) else None
                if gid:
                    matched.append(str(gid))
        return matched
