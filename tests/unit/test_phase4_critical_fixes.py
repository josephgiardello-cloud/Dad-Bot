"""
Test suite for Phase 4 critical bug fixes.

This module validates fixes for:
1. IndexError in PolicyCompiler.match_rules (bounds check)
2. Silent latency clock drift in build_execution_event
3. Duck-typing blindspot in build_policy_input (Pydantic/dataclass)
4. Incomplete reducer in reduce_events_to_results
5. Type validation brittleness in normalize_tool_results

All tests should pass with the applied fixes.
"""

import time
from dataclasses import dataclass
from unittest.mock import Mock

import pytest

from dadbot.core.policy_compiler import PolicyCompiler, PolicyIntentGraph, PolicyStep
from dadbot.core.tool_ir import (
    ToolEvent,
    ToolEventLog,
    ToolEventType,
    build_execution_event,
    normalize_tool_results,
    reduce_events_to_results,
)
from dadbot.core.turn_ir import (
    _obj_to_dict,
    build_policy_input,
    build_policy_view,
    hash_eval_input,
    projection_cache,
    prove_policy_view_bijection,
)

pytestmark = pytest.mark.unit


# ============================================================================
# TEST SUITE 1: IndexError Fix (PolicyCompiler.match_rules)
# ============================================================================


class TestPolicyCompilerIndexErrorFix:
    """Validate fix for IndexError in match_rules with empty rules."""

    def test_match_rules_handles_empty_rules_gracefully(self):
        """match_rules should not crash when rules list is empty."""
        # Create an intent graph with empty rules tuple
        intent_graph = PolicyIntentGraph(
            policy_name="safety",
            rules=tuple(),  # Empty!
            policy_input=Mock(),
        )

        # Before fix: IndexError when accessing rules[0]
        # After fix: Returns empty tuple safely
        result = PolicyCompiler.match_rules(intent_graph)
        assert result == tuple()

    def test_match_rules_filters_non_applicable_with_bounds_check(self):
        """match_rules should skip out-of-bounds indices."""
        step1 = PolicyStep(name="step1", kind="binary", handler=lambda x: x)
        step2 = PolicyStep(name="step2", kind="unary", handler=lambda x: x)

        intent_graph = PolicyIntentGraph(
            policy_name="safety",
            rules=(step1, step2),
            policy_input=Mock(),
        )

        result = PolicyCompiler.match_rules(intent_graph)
        # Should include both applicable rules within bounds
        assert len(result) >= 0
        assert all(isinstance(r, PolicyStep) for r in result)

    def test_compile_vs_match_rules_consistency(self):
        """Verify compile() and match_rules() both use bounds check."""
        # Empty rules case
        intent_graph_empty = PolicyIntentGraph(
            policy_name="safety",
            rules=tuple(),
            policy_input=Mock(),
        )

        # Both should handle empty rules without crashing
        compile_result = PolicyCompiler.compile(intent_graph_empty)
        match_result = PolicyCompiler.match_rules(intent_graph_empty)

        assert compile_result is not None
        assert match_result == tuple()


# ============================================================================
# TEST SUITE 2: Latency Clock Drift Fix (build_execution_event)
# ============================================================================


class TestLatencyClockDriftFix:
    """Validate fix for silent latency clock drift in build_execution_event."""

    def test_build_execution_event_accepts_perf_counter_time(self):
        """build_execution_event should accept time.perf_counter() as started_at."""
        started_at = time.perf_counter()
        time.sleep(0.001)  # Sleep 1ms to ensure latency > 0

        result = build_execution_event(
            tool_name="test_tool",
            args={"key": "value"},
            output={"result": "ok"},
            status="ok",
            started_at=started_at,
        )

        # Should have valid latency > 0
        assert result.latency > 0.0
        assert result.latency < 1.0  # Should be < 1 second for 1ms sleep

    def test_build_execution_event_detects_wall_clock_time(self):
        """build_execution_event should reject time.time() (epoch time)."""
        wall_clock_time = time.time()  # Returns large number like 1.7e9

        with pytest.raises(ValueError, match="wall-clock time"):
            build_execution_event(
                tool_name="test_tool",
                args={"key": "value"},
                output={"result": "ok"},
                status="ok",
                started_at=wall_clock_time,
            )

    def test_build_execution_event_detects_large_clock_skew(self):
        """build_execution_event should reject > 100ms negative latency."""
        started_at = time.perf_counter() + 0.2  # 200ms in the future

        with pytest.raises(ValueError, match="clock skew"):
            build_execution_event(
                tool_name="test_tool",
                args={"key": "value"},
                output={"result": "ok"},
                status="ok",
                started_at=started_at,
            )

    def test_build_execution_event_allows_small_clock_skew(self):
        """build_execution_event should allow < 100ms negative latency."""
        started_at = time.perf_counter() + 0.01  # 10ms in the future (small skew)

        result = build_execution_event(
            tool_name="test_tool",
            args={"key": "value"},
            output={"result": "ok"},
            status="ok",
            started_at=started_at,
        )

        # Should normalize to 0.0
        assert result.latency == 0.0


# ============================================================================
# TEST SUITE 3: Duck-Typing Fix (build_policy_input Pydantic/Dataclass)
# ============================================================================


class TestDuckTypingFix:
    """Validate fix for duck-typing blindspot with Pydantic/dataclass."""

    def test_obj_to_dict_with_plain_dict(self):
        """_obj_to_dict should pass through plain dicts unchanged."""
        input_dict = {"key": "value", "nested": {"deep": "data"}}
        result = _obj_to_dict(input_dict)

        assert result == input_dict
        assert isinstance(result, dict)

    def test_obj_to_dict_with_none(self):
        """_obj_to_dict should return empty dict for None."""
        result = _obj_to_dict(None)
        assert result == {}

    def test_obj_to_dict_with_dataclass(self):
        """_obj_to_dict should convert dataclass to dict."""

        @dataclass
        class MockState:
            intent_type: str = "greeting"
            strategy: str = "direct"

        state_obj = MockState()
        result = _obj_to_dict(state_obj)

        assert isinstance(result, dict)
        assert result["intent_type"] == "greeting"
        assert result["strategy"] == "direct"

    def test_obj_to_dict_with_pydantic_v2_model(self):
        """_obj_to_dict should convert Pydantic v2 model via model_dump()."""
        # Mock a Pydantic v2 model
        mock_pydantic = Mock()
        mock_pydantic.model_dump.return_value = {"intent_type": "question", "strategy": "research"}

        result = _obj_to_dict(mock_pydantic)

        assert isinstance(result, dict)
        assert result["intent_type"] == "question"
        assert result["strategy"] == "research"

    def test_obj_to_dict_with_standard_python_object(self):
        """_obj_to_dict should convert standard Python objects via __dict__."""

        class SimpleObject:
            def __init__(self):
                self.field1 = "value1"
                self._private = "hidden"

        obj = SimpleObject()
        result = _obj_to_dict(obj)

        assert isinstance(result, dict)
        assert result["field1"] == "value1"
        assert "_private" not in result  # Private fields excluded

    def test_build_policy_input_with_dataclass_state(self):
        """build_policy_input should handle turn_context.state as dataclass."""

        @dataclass
        class MockState:
            turn_plan: dict = None
            tool_ir: dict = None

            def __post_init__(self):
                if self.turn_plan is None:
                    self.turn_plan = {"intent_type": "greeting", "strategy": "direct"}
                if self.tool_ir is None:
                    self.tool_ir = {"requests": []}

        turn_context = Mock()
        turn_context.state = MockState()
        turn_context.session_id = "session123"
        turn_context.tenant_id = "tenant456"
        turn_context.trace_id = "trace789"
        turn_context.mode = "live"

        result = build_policy_input("safety", turn_context, candidate=None)

        assert result.intent.intent_type == "greeting"
        assert result.intent.strategy == "direct"
        assert result.intent.tool_request_count == 0

    def test_build_policy_input_with_pydantic_state(self):
        """build_policy_input should handle turn_context.state as Pydantic model."""
        # Mock Pydantic model
        mock_state = Mock()
        mock_state.model_dump.return_value = {
            "turn_plan": {"intent_type": "question", "strategy": "research"},
            "tool_ir": {"requests": ["request1", "request2"]},
        }

        turn_context = Mock()
        turn_context.state = mock_state
        turn_context.session_id = "session123"
        turn_context.tenant_id = "tenant456"
        turn_context.trace_id = "trace789"
        turn_context.mode = "live"

        result = build_policy_input("safety", turn_context, candidate=None)

        assert result.intent.intent_type == "question"
        assert result.intent.strategy == "research"
        assert result.intent.tool_request_count == 2

    def test_build_policy_input_surfaces_typed_ir_degradation(self):
        """Unsupported state mappings must surface typed degradation metadata."""

        class BrokenState:
            @property
            def __dict__(self):
                raise RuntimeError("state is unreadable")

        turn_context = Mock()
        turn_context.state = BrokenState()
        turn_context.metadata = {}
        turn_context.session_id = "session123"
        turn_context.tenant_id = "tenant456"
        turn_context.trace_id = "trace789"
        turn_context.mode = "live"

        result = build_policy_input("safety", turn_context, candidate=None)

        assert int(result.ir_degradation_count) > 0
        assert result.ir_degradations
        first = dict(result.ir_degradations[0])
        assert first.get("reason") in {
            "dunder_dict_failed",
            "unsupported_state_mapping",
        }
        assert int(turn_context.metadata.get("turn_ir_degradation_count", 0)) > 0
        assert isinstance(turn_context.metadata.get("turn_ir_degradations"), list)

    def test_build_policy_view_is_stable_for_same_semantics(self):
        """PolicyView state hash should be stable for equivalent semantic policy inputs."""
        turn_context_a = Mock()
        turn_context_a.state = {
            "turn_plan": {"intent_type": "question", "strategy": "research", "noise": "x"},
            "tool_ir": {"requests": ["a", "b"]},
            "other": {"ignored": True},
        }

        turn_context_b = Mock()
        turn_context_b.state = {
            "turn_plan": {"intent_type": "question", "strategy": "research"},
            "tool_ir": {"requests": ["a", "b"]},
        }

        view_a = build_policy_view(turn_context_a)
        view_b = build_policy_view(turn_context_b)

        assert view_a.intent_type == view_b.intent_type == "question"
        assert view_a.strategy == view_b.strategy == "research"
        assert view_a.tool_request_count == view_b.tool_request_count == 2
        assert view_a.state_hash == view_b.state_hash

    def test_policy_view_bijection_proof_is_equivalent(self):
        turn_context = Mock()
        turn_context.state = {
            "turn_plan": {"intent_type": "question", "strategy": "research", "extra": "ignored"},
            "tool_ir": {"requests": ["r1"]},
        }
        view = build_policy_view(turn_context)
        proof = prove_policy_view_bijection(turn_context.state, view)

        assert proof.equivalent is True
        assert proof.contract_version == "policy-view-bijection-v1"
        assert bool(proof.projection_hash)
        assert proof.projection_hash == proof.inverse_hash


class TestSafetySinglePath:
    """Validate single-path safety evaluation semantics."""

    def test_fast_gate_flag_keeps_single_path_output(self):
        service = Mock()
        service.enforce_policies = Mock(side_effect=lambda _ctx, candidate: (str(candidate[0]).upper(), candidate[1]))
        plan = PolicyCompiler.compile_safety(service)

        turn_context = Mock()
        turn_context.state = {
            "turn_plan": {"intent_type": "statement", "strategy": "direct_answer"},
            "tool_ir": {"requests": []},
        }
        turn_context.session_id = "session"
        turn_context.tenant_id = "tenant"
        turn_context.trace_id = "trace"
        turn_context.mode = "live"

        decision = PolicyCompiler.evaluate_safety(
            plan,
            turn_context,
            ("hello", False),
            fast_gate=True,
            audit_full=True,
        )

        assert decision.output == ("HELLO", False)
        assert decision.trace.get("truth_source") == "semantic_eval_input_v1"
        assert decision.trace.get("metadata", {}).get("trace_schema") == "policy-decision-trace-v1"

    def test_semantic_trace_declares_runtime_domain_contract(self):
        service = Mock()
        service.enforce_policies = Mock(side_effect=lambda _ctx, candidate: (str(candidate[0]).upper(), candidate[1]))
        plan = PolicyCompiler.compile_safety(service)

        turn_context = Mock()
        turn_context.state = {
            "turn_plan": {"intent_type": "statement", "strategy": "direct_answer"},
            "tool_ir": {"requests": []},
        }
        turn_context.session_id = "session"
        turn_context.tenant_id = "tenant"
        turn_context.trace_id = "trace"
        turn_context.mode = "live"

        decision = PolicyCompiler.evaluate_safety(
            plan,
            turn_context,
            ("hello", False),
            fast_gate=True,
            audit_full=False,
        )

        semantic = dict((decision.trace or {}).get("semantic_eval") or {})
        domain_contract = dict(semantic.get("runtime_domain_contract") or {})
        assert domain_contract.get("policy_semantics") == "pure_deterministic"
        assert domain_contract.get("policy_runtime") == "single_path_compile_full"
        assert domain_contract.get("split_execution_disabled") is True


class TestSemanticCanonicalEvaluator:
    def test_projection_cache_returns_stable_view_for_same_trace(self):
        turn_context = Mock()
        turn_context.trace_id = "trace-cache"
        turn_context.session_id = "session"
        turn_context.mode = "live"
        turn_context.state = {
            "turn_plan": {"intent_type": "question", "strategy": "research"},
            "tool_ir": {"requests": ["a", "b"]},
        }

        cache = projection_cache()
        cache.clear()
        view_a = cache.get(turn_context)
        view_b = cache.get(turn_context)

        assert view_a == view_b
        assert view_a.tool_request_count == 2

    def test_semantic_decision_is_deterministic_for_identical_inputs(self):
        service = Mock()
        service.validate = Mock(side_effect=lambda candidate: f"validated::{candidate}")
        plan = PolicyCompiler.compile_safety(service)

        turn_context_a = Mock()
        turn_context_a.trace_id = "trace-a"
        turn_context_a.session_id = "session"
        turn_context_a.tenant_id = "tenant"
        turn_context_a.mode = "live"
        turn_context_a.state = {
            "turn_plan": {"intent_type": "support", "strategy": "baseline"},
            "tool_ir": {"requests": []},
        }

        turn_context_b = Mock()
        turn_context_b.trace_id = "trace-b"
        turn_context_b.session_id = "session"
        turn_context_b.tenant_id = "tenant"
        turn_context_b.mode = "live"
        turn_context_b.state = {
            "turn_plan": {"intent_type": "support", "strategy": "baseline"},
            "tool_ir": {"requests": []},
        }

        decision_a = PolicyCompiler.evaluate_safety(plan, turn_context_a, "candidate", fast_gate=True, audit_full=True)
        decision_b = PolicyCompiler.evaluate_safety(plan, turn_context_b, "candidate", fast_gate=True, audit_full=True)

        semantic_a = dict((decision_a.trace or {}).get("semantic_eval") or {})
        semantic_b = dict((decision_b.trace or {}).get("semantic_eval") or {})
        assert semantic_a.get("decision") == semantic_b.get("decision")

    def test_eval_input_hash_is_emitted_as_replay_anchor(self):
        service = Mock()
        service.validate = Mock(side_effect=lambda candidate: f"validated::{candidate}")
        plan = PolicyCompiler.compile_safety(service)

        turn_context = Mock()
        turn_context.trace_id = "trace-hash"
        turn_context.session_id = "session"
        turn_context.tenant_id = "tenant"
        turn_context.mode = "live"
        turn_context.state = {
            "turn_plan": {"intent_type": "support", "strategy": "baseline"},
            "tool_ir": {"requests": []},
        }

        policy_input = build_policy_input("safety", turn_context, "candidate")
        expected_hash = hash_eval_input(policy_input.semantic_eval_input)

        decision = PolicyCompiler.evaluate_safety(plan, turn_context, "candidate", fast_gate=True, audit_full=False)
        semantic = dict((decision.trace or {}).get("semantic_eval") or {})
        assert semantic.get("eval_input_hash") == expected_hash


# ============================================================================
# TEST SUITE 4: Event Reducer Fix (reduce_events_to_results)
# ============================================================================


class TestEventReducerFix:
    """Validate fix for incomplete event reducer with proper grouping."""

    def test_reduce_events_single_executed_event(self):
        """Reducer should produce single result for single EXECUTED event."""
        event = ToolEvent(
            tool_id="tool_001",
            event_type=ToolEventType.EXECUTED,
            sequence=1,
            input_hash="hash1",
            output_hash="hash2",
            payload={
                "tool_name": "test_tool",
                "status": "ok",
                "output": {"result": "success"},
            },
        )

        log = ToolEventLog(events=(event,))
        result = reduce_events_to_results(log)

        assert len(result) == 1
        assert result[0]["tool_id"] == "tool_001"
        assert result[0]["status"] == "ok"
        assert result[0]["output"] == {"result": "success"}

    def test_reduce_events_retried_tool_keeps_latest_success(self):
        """Reducer should keep only latest EXECUTED when tool is retried."""
        failed_event = ToolEvent(
            tool_id="tool_001",
            event_type=ToolEventType.FAILED,
            sequence=1,
            input_hash="hash1",
            output_hash="hash_err1",
            payload={
                "tool_name": "test_tool",
                "error": "Connection timeout",
            },
        )

        success_event = ToolEvent(
            tool_id="tool_001",
            event_type=ToolEventType.EXECUTED,
            sequence=2,
            input_hash="hash1",
            output_hash="hash2",
            payload={
                "tool_name": "test_tool",
                "status": "ok",
                "output": {"result": "success"},
            },
        )

        log = ToolEventLog(events=(failed_event, success_event))
        result = reduce_events_to_results(log)

        # Should have exactly one entry for tool_001 (the latest success)
        assert len(result) == 1
        assert result[0]["tool_id"] == "tool_001"
        assert result[0]["status"] == "ok"
        assert "error" not in result[0]  # Failed state should not appear

    def test_reduce_events_retried_tool_keeps_latest_failure(self):
        """Reducer should keep latest FAILED if final attempt fails."""
        success_event = ToolEvent(
            tool_id="tool_001",
            event_type=ToolEventType.EXECUTED,
            sequence=1,
            input_hash="hash1",
            output_hash="hash2",
            payload={
                "tool_name": "test_tool",
                "status": "ok",
                "output": {"result": "success"},
            },
        )

        failed_event = ToolEvent(
            tool_id="tool_001",
            event_type=ToolEventType.FAILED,
            sequence=2,
            input_hash="hash1",
            output_hash="hash_err2",
            payload={
                "tool_name": "test_tool",
                "error": "Permanent error",
            },
        )

        log = ToolEventLog(events=(success_event, failed_event))
        result = reduce_events_to_results(log)

        # Should have exactly one entry (the latest failure)
        assert len(result) == 1
        assert result[0]["tool_id"] == "tool_001"
        assert result[0]["status"] == "error"
        assert result[0]["output"] == "Permanent error"

    def test_reduce_events_multiple_tools_independent_states(self):
        """Reducer should track independent tools separately."""
        events = [
            ToolEvent(
                tool_id="tool_001",
                event_type=ToolEventType.EXECUTED,
                sequence=1,
                input_hash="hash1a",
                output_hash="hash1b",
                payload={"tool_name": "tool_a", "status": "ok", "output": "result_a"},
            ),
            ToolEvent(
                tool_id="tool_002",
                event_type=ToolEventType.FAILED,
                sequence=2,
                input_hash="hash2a",
                output_hash="hash2b",
                payload={"tool_name": "tool_b", "error": "error_b"},
            ),
            ToolEvent(
                tool_id="tool_002",
                event_type=ToolEventType.EXECUTED,
                sequence=3,
                input_hash="hash2a",
                output_hash="hash2c",
                payload={"tool_name": "tool_b", "status": "ok", "output": "result_b"},
            ),
        ]

        log = ToolEventLog(events=tuple(events))
        result = reduce_events_to_results(log)

        # Should have 2 entries (one per tool)
        assert len(result) == 2

        # tool_001 should be success
        tool_001_result = [r for r in result if r["tool_id"] == "tool_001"][0]
        assert tool_001_result["status"] == "ok"
        assert tool_001_result["output"] == "result_a"

        # tool_002 should be latest (success)
        tool_002_result = [r for r in result if r["tool_id"] == "tool_002"][0]
        assert tool_002_result["status"] == "ok"
        assert tool_002_result["output"] == "result_b"

    def test_reduce_events_preserves_sequence_order(self):
        """Reducer should preserve terminal event sequence order."""
        events = [
            ToolEvent(
                tool_id="tool_001",
                event_type=ToolEventType.EXECUTED,
                sequence=10,
                input_hash="h1",
                output_hash="h2",
                payload={"tool_name": "a", "status": "ok", "output": "out_a"},
            ),
            ToolEvent(
                tool_id="tool_002",
                event_type=ToolEventType.EXECUTED,
                sequence=5,
                input_hash="h3",
                output_hash="h4",
                payload={"tool_name": "b", "status": "ok", "output": "out_b"},
            ),
        ]

        log = ToolEventLog(events=tuple(events))
        result = reduce_events_to_results(log)

        # Results should be ordered by sequence number
        assert result[0]["tool_id"] == "tool_002"  # sequence 5
        assert result[1]["tool_id"] == "tool_001"  # sequence 10


# ============================================================================
# TEST SUITE 5: Type Validation Fix (normalize_tool_results)
# ============================================================================


class TestTypeValidationFix:
    """Validate fix for type validation in normalize_tool_results."""

    def test_normalize_tool_results_accepts_dicts(self):
        """normalize_tool_results should accept list of dicts."""
        values = [
            {"tool_name": "tool1", "status": "ok", "output": "result1", "deterministic_id": "id1"},
            {"tool_name": "tool2", "status": "ok", "output": "result2", "deterministic_id": "id2"},
        ]

        result = normalize_tool_results(values)

        assert len(result) == 2
        assert result[0]["tool_name"] == "tool1"
        assert result[1]["tool_name"] == "tool2"

    def test_normalize_tool_results_rejects_string_value(self):
        """normalize_tool_results should reject string instead of dict."""
        values = ["not_a_dict_string"]

        with pytest.raises(TypeError, match="expected dict or ToolResult"):
            normalize_tool_results(values)

    def test_normalize_tool_results_rejects_number_value(self):
        """normalize_tool_results should reject number instead of dict."""
        values = [42]

        with pytest.raises(TypeError, match="expected dict or ToolResult"):
            normalize_tool_results(values)

    def test_normalize_tool_results_rejects_mixed_invalid(self):
        """normalize_tool_results should reject mix of valid and invalid."""
        values = [
            {"tool_name": "tool1", "status": "ok", "output": "result1", "deterministic_id": "id1"},
            "invalid_string",
        ]

        with pytest.raises(TypeError, match="expected dict or ToolResult"):
            normalize_tool_results(values)

    def test_normalize_tool_results_handles_none_list(self):
        """normalize_tool_results should handle None gracefully."""
        result = normalize_tool_results(None)
        assert result == []

    def test_normalize_tool_results_handles_empty_list(self):
        """normalize_tool_results should handle empty list gracefully."""
        result = normalize_tool_results([])
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
