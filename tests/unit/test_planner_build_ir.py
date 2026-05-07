"""Unit tests for Planner.build_ir() IR adapter."""

import pytest

from dadbot.core.planner_ir import PlannerDecision
from dadbot.runtime.planner.planner import Planner


class TestPlannerBuildIR:
    """Test Planner.build_ir() method."""

    def test_planner_build_ir_returns_planner_decision(self):
        """Test Planner.build_ir() returns PlannerDecision IR."""
        planner = Planner()

        # Mock minimal turn_state
        turn_state = {
            "runtime": type("Runtime", (), {"plan_candidates": lambda **kw: []}),
            "event": type("Event", (), {"payload": {}}),
            "user_text": "test",
            "attachments": [],
            "thread_state": {},
        }

        # Mock execution_result
        execution_result = {
            "initial_result": {"intent_type": "respond"},
        }

        result = planner.build_ir(turn_state, execution_result)

        assert isinstance(result, PlannerDecision)

    def test_planner_build_ir_is_immutable(self):
        """Test Planner.build_ir() produces immutable IR."""
        planner = Planner()

        turn_state = {
            "runtime": type("Runtime", (), {"plan_candidates": lambda **kw: []}),
            "event": type("Event", (), {"payload": {}}),
            "user_text": "test",
            "attachments": [],
            "thread_state": {},
        }

        execution_result = {
            "initial_result": {"intent_type": "respond"},
        }

        decision = planner.build_ir(turn_state, execution_result)

        # Attempt to mutate should raise
        with pytest.raises(AttributeError):
            decision.primary_intent = "modified"

    def test_planner_build_ir_preserves_context(self):
        """Test Planner.build_ir() carries _runtime_context."""
        planner = Planner()

        turn_state = {
            "runtime": type("Runtime", (), {"plan_candidates": lambda **kw: []}),
            "event": type("Event", (), {"payload": {}}),
            "user_text": "test",
            "attachments": [],
            "thread_state": {},
            "turn_id": "t123",
        }

        execution_result = {
            "initial_result": {"intent_type": "respond"},
        }

        decision = planner.build_ir(turn_state, execution_result)

        # _runtime_context should be the turn_state
        assert decision._runtime_context == turn_state
