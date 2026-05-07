"""Unit tests for TurnIR Assembly (integrated IR routing)."""

import pytest

from dadbot.core.planner_ir import CandidateIR, PlannerDecision
from dadbot.core.turn_ir import ExecutionContext, TurnIntent
from dadbot.core.turn_ir_assembly import (
    TurnIRAssembly,
    assemble_turn_ir,
)


class TestTurnIRAssembly:
    """Test TurnIRAssembly structure and immutability."""

    def test_turn_ir_assembly_creation_minimal(self):
        """Test TurnIRAssembly creation with minimal fields."""
        intent = TurnIntent(intent_type="test", strategy="direct")
        assembly = TurnIRAssembly(turn_intent=intent)

        assert assembly.turn_intent == intent
        assert assembly.planner_decision is None
        assert assembly.policy_input is None
        assert assembly.policy_decision is None

    def test_turn_ir_assembly_immutable(self):
        """Test TurnIRAssembly is frozen."""
        intent = TurnIntent(intent_type="test", strategy="direct")
        assembly = TurnIRAssembly(turn_intent=intent)

        with pytest.raises(AttributeError):
            assembly.turn_intent = TurnIntent(intent_type="modified", strategy="direct")

    def test_turn_ir_assembly_with_planner_decision(self):
        """Test TurnIRAssembly carries PlannerDecision."""
        intent = TurnIntent(intent_type="test", strategy="direct")
        candidates = (CandidateIR(candidate_id="c1", candidate={}),)
        planner_decision = PlannerDecision(candidates=candidates)
        assembly = TurnIRAssembly(turn_intent=intent, planner_decision=planner_decision)

        assert assembly.has_planner_output() is True
        assert assembly.planner_decision == planner_decision

    def test_turn_ir_assembly_has_planner_output_false(self):
        """Test has_planner_output() returns False for None."""
        intent = TurnIntent(intent_type="test", strategy="direct")
        assembly = TurnIRAssembly(turn_intent=intent)

        assert assembly.has_planner_output() is False

    def test_turn_ir_assembly_has_policy_output_false(self):
        """Test has_policy_output() returns False for None."""
        intent = TurnIntent(intent_type="test", strategy="direct")
        assembly = TurnIRAssembly(turn_intent=intent)

        assert assembly.has_policy_output() is False

    def test_turn_ir_assembly_has_policy_output_true(self):
        """Test has_policy_output() returns True for non-None policy_decision."""
        intent = TurnIntent(intent_type="test", strategy="direct")
        # Mock policy decision (just needs to be not None)
        assembly = TurnIRAssembly(turn_intent=intent, policy_decision={"output": "safe"})

        assert assembly.has_policy_output() is True

    def test_turn_ir_assembly_runtime_context_opaque(self):
        """Test TurnIRAssembly carries opaque _runtime_context."""
        intent = TurnIntent(intent_type="test", strategy="direct")
        runtime_ctx = {"turn_id": "t123", "session": "s456"}
        assembly = TurnIRAssembly(turn_intent=intent, _runtime_context=runtime_ctx)

        assert assembly._runtime_context == runtime_ctx
        # Should not appear in repr (repr=False)
        assert "_runtime_context" not in repr(assembly)


class TestAssembleTurnIR:
    """Test assemble_turn_ir() boundary adapter."""

    def test_assemble_turn_ir_minimal(self):
        """Test assemble_turn_ir() with minimal arguments."""
        user_input = "test message"
        turn_context = {"turn_id": "t1"}

        assembly = assemble_turn_ir(user_input, turn_context)

        assert isinstance(assembly, TurnIRAssembly)
        assert assembly.turn_intent.intent_type in ("unspecified", "")
        assert assembly.turn_intent.strategy in ("default", "")
        assert assembly.planner_decision is None
        assert assembly.policy_input is None

    def test_assemble_turn_ir_with_planner_result(self):
        """Test assemble_turn_ir() with plan_result."""
        user_input = "test message"
        turn_context = {
            "turn_id": "t1",
            "execution_result": {
                "initial_result": {
                    "intent_type": "respond",
                    "strategy": "narrative",
                }
            },
        }

        # Mock PlanResult
        class MockPlanResult:
            def __init__(self):
                self.candidates = [{"id": "c1", "data": "test"}]
                self.metadata = {}
                self.evaluation_hint = {}

        plan_result = MockPlanResult()

        assembly = assemble_turn_ir(user_input, turn_context, plan_result=plan_result)

        assert assembly.has_planner_output() is True
        assert assembly.planner_decision is not None
        assert len(assembly.planner_decision.candidates) == 1

    def test_assemble_turn_ir_extracts_intent(self):
        """Test assemble_turn_ir() correctly extracts intent."""
        user_input = "user message"
        turn_context = {
            "execution_result": {
                "initial_result": {
                    "intent_type": "clarify",
                    "strategy": "questioning",
                }
            },
        }

        assembly = assemble_turn_ir(user_input, turn_context)

        assert assembly.turn_intent.intent_type == "clarify"
        assert assembly.turn_intent.strategy == "questioning"

    def test_assemble_turn_ir_with_dict_turn_context(self):
        """Test assemble_turn_ir() handles dict turn_context."""
        user_input = "test"
        turn_context = {
            "execution_result": {
                "initial_result": {
                    "intent_type": "respond",
                    "strategy": "default",
                }
            },
        }

        assembly = assemble_turn_ir(user_input, turn_context)

        assert assembly.turn_intent.intent_type == "respond"
        assert assembly.turn_intent.strategy == "default"

    def test_assemble_turn_ir_with_candidate(self):
        """Test assemble_turn_ir() creates PolicyInput from candidate."""
        user_input = "test"
        turn_context = {"turn_id": "t1"}
        candidate = {"text": "response text"}

        assembly = assemble_turn_ir(user_input, turn_context, candidate=candidate)

        assert assembly.policy_input is not None
        assert assembly.policy_input.candidate == candidate

    def test_assemble_turn_ir_preserves_runtime_context(self):
        """Test assemble_turn_ir() preserves _runtime_context."""
        user_input = "test"
        turn_context = {"turn_id": "t123", "session": "s456"}

        assembly = assemble_turn_ir(user_input, turn_context)

        assert assembly._runtime_context == turn_context

    def test_assemble_turn_ir_full_pipeline(self):
        """Test assemble_turn_ir() with all components (end-to-end)."""
        user_input = "test message"
        turn_context = {
            "turn_id": "t1",
            "execution_result": {
                "initial_result": {
                    "intent_type": "respond",
                    "strategy": "narrative",
                }
            },
        }

        class MockCandidate:
            candidate_id = "c1"
            intent_type = "respond"
            strategy = "narrative"

        class MockPlanResult:
            def __init__(self):
                self.candidates = [MockCandidate()]
                self.metadata = {}
                self.evaluation_hint = {}

        plan_result = MockPlanResult()
        candidate = {"text": "response"}

        assembly = assemble_turn_ir(
            user_input,
            turn_context,
            plan_result=plan_result,
            candidate=candidate,
        )

        # Verify all IR types assembled
        assert assembly.turn_intent is not None
        assert assembly.turn_intent.intent_type == "respond"
        assert assembly.has_planner_output() is True
        assert assembly.selected_candidate_ir is not None
        assert assembly.policy_input is not None


class TestTurnIRAssemblyIntegration:
    """Integration tests for TurnIRAssembly."""

    def test_turn_ir_assembly_immutability_prevents_mutation(self):
        """Test that frozen assembly prevents field mutations."""
        intent = TurnIntent(intent_type="test", strategy="direct")
        assembly = TurnIRAssembly(turn_intent=intent)

        # Multiple field mutations should all raise
        with pytest.raises(AttributeError):
            assembly.planner_decision = None

        with pytest.raises(AttributeError):
            assembly.policy_input = None

        with pytest.raises(AttributeError):
            assembly.policy_decision = {}

    def test_turn_ir_assembly_candidate_selection(self):
        """Test selecting candidate from assembly."""
        intent = TurnIntent(intent_type="test", strategy="direct")
        candidates = (
            CandidateIR(candidate_id="c1", candidate={"text": "response1"}),
            CandidateIR(candidate_id="c2", candidate={"text": "response2"}),
        )
        planner_decision = PlannerDecision(candidates=candidates)
        assembly = TurnIRAssembly(
            turn_intent=intent,
            planner_decision=planner_decision,
            selected_candidate_ir=candidates[0],
        )

        # Verify primary candidate
        assert assembly.selected_candidate_ir.candidate_id == "c1"
        assert assembly.selected_candidate_ir.candidate["text"] == "response1"
