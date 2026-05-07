"""Unit tests for PlannerIR (Intermediate Representation for planner output)."""

import pytest

from dadbot.core.planner_ir import (
    CandidateIR,
    PlannerDecision,
    build_planner_ir,
)


class TestCandidateIR:
    """Test CandidateIR immutability and structure."""

    def test_candidate_ir_creation(self):
        """Test CandidateIR can be created with required fields."""
        candidate_obj = {"data": "test"}
        cand_ir = CandidateIR(
            candidate_id="cand_123",
            candidate=candidate_obj,
            intent_type="clarify",
            strategy="direct",
        )
        assert cand_ir.candidate_id == "cand_123"
        assert cand_ir.candidate == candidate_obj
        assert cand_ir.intent_type == "clarify"
        assert cand_ir.strategy == "direct"

    def test_candidate_ir_immutable(self):
        """Test CandidateIR is frozen (immutable)."""
        cand_ir = CandidateIR(candidate_id="cand_1", candidate={})
        with pytest.raises(AttributeError):
            cand_ir.candidate_id = "cand_2"

    def test_candidate_ir_with_metadata(self):
        """Test CandidateIR carries metadata dict."""
        metadata = {"confidence": 0.95, "source": "inference"}
        cand_ir = CandidateIR(
            candidate_id="cand_1",
            candidate={},
            metadata=metadata,
        )
        assert cand_ir.metadata == metadata

    def test_candidate_ir_runtime_context_opaque(self):
        """Test CandidateIR carries opaque _runtime_context for handlers."""
        runtime_ctx = {"turn_id": "t123"}
        cand_ir = CandidateIR(
            candidate_id="cand_1",
            candidate={},
            _runtime_context=runtime_ctx,
        )
        assert cand_ir._runtime_context == runtime_ctx
        # _runtime_context should not appear in repr (due to repr=False)
        assert "_runtime_context" not in repr(cand_ir)


class TestPlannerDecision:
    """Test PlannerDecision immutability and structure."""

    def test_planner_decision_creation_empty(self):
        """Test PlannerDecision can be created with minimal args."""
        decision = PlannerDecision()
        assert decision.candidates == ()
        assert decision.primary_intent == ""
        assert decision.primary_strategy == ""
        assert decision.metadata == {}

    def test_planner_decision_with_candidates(self):
        """Test PlannerDecision holds tuple of CandidateIR."""
        cands = [
            CandidateIR(candidate_id="c1", candidate={}),
            CandidateIR(candidate_id="c2", candidate={}),
        ]
        decision = PlannerDecision(candidates=tuple(cands))
        assert len(decision.candidates) == 2
        assert decision.candidates[0].candidate_id == "c1"

    def test_planner_decision_immutable(self):
        """Test PlannerDecision is frozen."""
        decision = PlannerDecision()
        with pytest.raises(AttributeError):
            decision.primary_intent = "test"

    def test_planner_decision_get_primary_candidate(self):
        """Test get_primary_candidate() returns first candidate."""
        cands = [
            CandidateIR(candidate_id="c1", candidate={}),
            CandidateIR(candidate_id="c2", candidate={}),
        ]
        decision = PlannerDecision(candidates=tuple(cands))
        primary = decision.get_primary_candidate()
        assert primary.candidate_id == "c1"

    def test_planner_decision_get_primary_candidate_empty(self):
        """Test get_primary_candidate() returns None for empty candidates."""
        decision = PlannerDecision()
        assert decision.get_primary_candidate() is None

    def test_planner_decision_select_candidate_by_id(self):
        """Test select_candidate_by_id() finds candidate."""
        cands = [
            CandidateIR(candidate_id="c1", candidate={}),
            CandidateIR(candidate_id="c2", candidate={}),
        ]
        decision = PlannerDecision(candidates=tuple(cands))
        found = decision.select_candidate_by_id("c2")
        assert found.candidate_id == "c2"

    def test_planner_decision_select_candidate_by_id_not_found(self):
        """Test select_candidate_by_id() returns None if not found."""
        cands = [CandidateIR(candidate_id="c1", candidate={})]
        decision = PlannerDecision(candidates=tuple(cands))
        found = decision.select_candidate_by_id("c999")
        assert found is None

    def test_planner_decision_metadata_and_hints(self):
        """Test PlannerDecision carries metadata and evaluation hints."""
        metadata = {"substrate_mode": "multi_agent"}
        hints = {"rewards": [0.8, 0.6]}
        decision = PlannerDecision(
            metadata=metadata,
            evaluation_hint=hints,
        )
        assert decision.metadata == metadata
        assert decision.evaluation_hint == hints


class TestBuildPlannerIR:
    """Test build_planner_ir() boundary adapter."""

    def test_build_planner_ir_none_plan_result(self):
        """Test build_planner_ir() handles None plan_result."""
        decision = build_planner_ir(None, {})
        assert isinstance(decision, PlannerDecision)
        assert decision.candidates == ()

    def test_build_planner_ir_from_mock_plan_result(self):
        """Test build_planner_ir() converts PlanResult-like object to PlannerDecision."""
        # Mock PlanResult
        class MockPlanResult:
            def __init__(self):
                self.candidates = [{"id": "c1", "data": "test"}]
                self.metadata = {"substrate_mode": "single"}
                self.evaluation_hint = {"rewards": [0.5]}

        mock_result = MockPlanResult()
        turn_context = {"turn_id": "t123"}

        decision = build_planner_ir(mock_result, turn_context)

        assert isinstance(decision, PlannerDecision)
        assert len(decision.candidates) == 1
        assert decision.candidates[0].candidate_id == "c1"
        assert decision.metadata["substrate_mode"] == "single"
        assert decision.evaluation_hint["rewards"] == [0.5]

    def test_build_planner_ir_extracts_runtime_context(self):
        """Test build_planner_ir() preserves _runtime_context for handlers."""
        class MockPlanResult:
            def __init__(self):
                self.candidates = [{}]
                self.metadata = {}
                self.evaluation_hint = {}

        mock_result = MockPlanResult()
        turn_context = {"turn_id": "t123", "session": "sess_456"}

        decision = build_planner_ir(mock_result, turn_context)

        # Both PlannerDecision and CandidateIR should have _runtime_context
        assert decision._runtime_context == turn_context
        assert decision.candidates[0]._runtime_context == turn_context

    def test_build_planner_ir_empty_candidates_list(self):
        """Test build_planner_ir() handles empty candidate list."""
        class MockPlanResult:
            def __init__(self):
                self.candidates = []
                self.metadata = {}
                self.evaluation_hint = {}

        mock_result = MockPlanResult()
        decision = build_planner_ir(mock_result, {})

        assert decision.candidates == ()

    def test_build_planner_ir_candidate_extraction(self):
        """Test build_planner_ir() correctly extracts candidate metadata."""
        class MockCandidate:
            def __init__(self):
                self.candidate_id = "mock_c1"
                self.intent_type = "clarify"
                self.strategy = "direct"

        class MockPlanResult:
            def __init__(self):
                self.candidates = [MockCandidate()]
                self.metadata = {}
                self.evaluation_hint = {}

        mock_result = MockPlanResult()
        decision = build_planner_ir(mock_result, {})

        cand = decision.candidates[0]
        assert cand.candidate_id == "mock_c1"
        assert cand.intent_type == "clarify"
        assert cand.strategy == "direct"

    def test_build_planner_ir_candidate_dict_support(self):
        """Test build_planner_ir() supports dict candidates."""
        class MockPlanResult:
            def __init__(self):
                self.candidates = [
                    {
                        "id": "c_dict",
                        "intent_type": "respond",
                        "strategy": "narrative",
                        "confidence": 0.92,
                    }
                ]
                self.metadata = {}
                self.evaluation_hint = {}

        mock_result = MockPlanResult()
        decision = build_planner_ir(mock_result, {})

        cand = decision.candidates[0]
        assert cand.candidate_id == "c_dict"
        assert cand.intent_type == "respond"
        assert cand.strategy == "narrative"
        assert cand.metadata["confidence"] == 0.92

    def test_build_planner_ir_fallback_to_index_ids(self):
        """Test build_planner_ir() generates index-based IDs if not found."""
        class MockPlanResult:
            def __init__(self):
                self.candidates = [{}, {}, {}]
                self.metadata = {}
                self.evaluation_hint = {}

        mock_result = MockPlanResult()
        decision = build_planner_ir(mock_result, {})

        assert len(decision.candidates) == 3
        assert decision.candidates[0].candidate_id == "candidate_0"
        assert decision.candidates[1].candidate_id == "candidate_1"
        assert decision.candidates[2].candidate_id == "candidate_2"


class TestPlannerIRIntegration:
    """Integration tests for PlannerIR with related IR layers."""

    def test_planner_ir_is_immutable_and_hashable_safe(self):
        """Test PlannerDecision immutability preserves semantic safety."""
        decision = PlannerDecision(primary_intent="test")
        # Should not raise any exception on repeated reads
        _ = decision.primary_intent
        _ = decision.primary_intent
        assert decision.primary_intent == "test"

    def test_candidate_ir_with_all_fields(self):
        """Test CandidateIR with full field set (like from real planner)."""
        metadata = {
            "confidence": 0.87,
            "model": "gpt-4",
            "tokens": 150,
        }
        cand_ir = CandidateIR(
            candidate_id="full_test",
            candidate={"text": "response"},
            intent_type="narrative",
            strategy="storytelling",
            metadata=metadata,
            _runtime_context={"turn_id": "t1"},
        )
        # Verify all fields accessible
        assert cand_ir.candidate_id == "full_test"
        assert cand_ir.candidate["text"] == "response"
        assert cand_ir.intent_type == "narrative"
        assert cand_ir.strategy == "storytelling"
        assert cand_ir.metadata["confidence"] == 0.87
        assert cand_ir._runtime_context["turn_id"] == "t1"
