"""tests/unit/test_response_engine_control_plane_integration.py — ResponseEngine + control-plane integration."""

from __future__ import annotations

import pytest
from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock

from dadbot.core.response_engine import ResponseEngine, ResponseCandidate
from dadbot.core.control_plane import ExecutionControlPlane


class TestResponseEngineControlPlaneIntegration:
    """Test ResponseEngine integration with control-plane execution flow."""

    def test_control_plane_initializes_response_engine(self):
        """ExecutionControlPlane initializes ResponseEngine on startup."""
        # Mock dependencies with proper values
        mock_registry = Mock()
        mock_executor = Mock()
        mock_options = Mock()
        mock_options.stream_sink = None
        mock_options.worker_id = "test-worker"
        mock_options.max_inflight_jobs = 10
        mock_options.enable_observability = False
        mock_options.ledger = None
        mock_options.scheduler = None
        mock_options.lease_ttl_seconds = 300.0
        mock_options.redelivery_retry_interval_seconds = 5.0

        with patch("dadbot.core.control_plane.Scheduler"):
            with patch("dadbot.core.control_plane.RecoveryManager"):
                cp = ExecutionControlPlane(
                    registry=mock_registry,
                    kernel_executor=mock_executor,
                    options=mock_options,
                )

                assert hasattr(cp, "_response_engine")
                assert isinstance(cp._response_engine, ResponseEngine)

    def test_build_response_engine_context_creates_valid_context(self):
        """_build_response_engine_context() creates context with required fields."""
        # Mock dependencies with proper values
        mock_registry = Mock()
        mock_executor = Mock()
        mock_options = Mock()
        mock_options.stream_sink = None
        mock_options.worker_id = "test-worker"
        mock_options.max_inflight_jobs = 10
        mock_options.enable_observability = False
        mock_options.ledger = None
        mock_options.scheduler = None
        mock_options.lease_ttl_seconds = 300.0
        mock_options.redelivery_retry_interval_seconds = 5.0
        mock_registry.get_or_create = Mock(
            return_value={
                "state": {"turn_trace": {"trace_id": "test-trace"}},
            }
        )

        with patch("dadbot.core.control_plane.Scheduler"):
            with patch("dadbot.core.control_plane.RecoveryManager"):
                cp = ExecutionControlPlane(
                    registry=mock_registry,
                    kernel_executor=mock_executor,
                    options=mock_options,
                )

                mock_job = Mock()
                mock_job.user_input = "Test input"
                mock_job.metadata = {"trace_id": "test-trace"}
                mock_job.trace_id = "test-trace"

                context = cp._build_response_engine_context(
                    job=mock_job,
                    session_key="test-session",
                    initial_response="Test response",
                )

                assert hasattr(context, "user_input")
                assert hasattr(context, "initial_response")
                assert hasattr(context, "session_state")
                assert hasattr(context, "job_metadata")
                assert hasattr(context, "trace_id")
                assert context.user_input == "Test input"
                assert context.initial_response == "Test response"

    def test_response_engine_ranking_preserves_execution_result(self):
        """ResponseEngine ranking preserves successful execution result format."""
        engine = ResponseEngine()

        class MockContext:
            user_input = "What is the meaning?"
            initial_response = "Life is meaning."
            session_state = {}
            job_metadata = {}
            trace_id = "test"

        context = MockContext()

        # run() should return a string
        result = engine.run(context)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_response_engine_run_on_empty_input_returns_fallback(self):
        """ResponseEngine.run() returns fallback on empty user input."""
        engine = ResponseEngine()

        class MockContext:
            user_input = ""
            initial_response = ""
            session_state = {}
            job_metadata = {}
            trace_id = "test"

        context = MockContext()
        result = engine.run(context)

        # Should return fallback
        assert isinstance(result, str)
        assert len(result) > 0

    def test_response_engine_uses_shadow_bus_for_decision_weighting(self):
        engine = ResponseEngine()

        candidate_a = ResponseCandidate(
            text="Candidate A stable response",
            source="direct",
            tone="neutral",
            response_goal="inform",
            risk_level=0.2,
        )
        candidate_b = ResponseCandidate(
            text="Candidate B tool-informed response",
            source="tool_registry",
            tone="neutral",
            response_goal="inform",
            risk_level=0.2,
        )

        engine.generate_candidates = Mock(return_value=[candidate_a, candidate_b])
        engine.filter_candidates = Mock(return_value=[candidate_a, candidate_b])

        # Keep base scores equivalent so shadow influences are the deciding factor.
        engine._score_coherence = Mock(return_value=0.7)
        engine._score_relevance = Mock(return_value=0.7)
        engine._score_persona_consistency = Mock(return_value=0.7)
        engine._score_novelty = Mock(return_value=0.7)
        engine._score_redundancy = Mock(return_value=0.2)
        engine._score_memory_relevance = Mock(return_value=0.7)
        engine._score_user_alignment = Mock(return_value=0.7)
        engine._score_trajectory_alignment = Mock(return_value=0.7)
        engine._simulate_user_reaction = Mock(return_value=0.7)
        engine.emotion_alignment = Mock(return_value=0.7)
        engine._distribution_collapse_penalty = Mock(return_value=0.0)
        engine._source_dominance_penalty = Mock(return_value=0.0)
        engine._interaction_bonus = Mock(return_value=0.0)
        engine._learned_preference_score = Mock(return_value=0.0)

        context = SimpleNamespace(
            user_input="Should I use the reminder tool?",
            trace_id="trace-shadow",
            session_state={},
            job_metadata={
                "shadow_decision_bus": [
                    {
                        "source": "safety_support",
                        "type": "veto",
                        "reason": "direct path has policy uncertainty",
                        "would_replace": True,
                        "priority": 1.0,
                        "timestamp": 1.0,
                    },
                    {
                        "source": "tool_registry",
                        "type": "suggestion",
                        "reason": "tool output increases groundedness",
                        "would_replace": True,
                        "priority": 1.0,
                        "timestamp": 2.0,
                    },
                ]
            },
            memory_context=[],
            persona_traits=[],
            persona_constraints={},
            user_preferences={},
            conversation_trajectory={},
        )

        _ = engine.run(context)
        selected = dict(getattr(context, "response_engine_telemetry", {}).get("selected") or {})
        assert selected
        assert selected.get("source") == "tool_registry"
        reasoning = dict(selected.get("reasoning") or {})
        assert "safety" in reasoning
        assert "tools" in reasoning

    def test_finalize_submit_success_persists_response_engine_telemetry(self):
        mock_registry = Mock()
        mock_executor = Mock()
        mock_options = Mock()
        mock_options.stream_sink = None
        mock_options.worker_id = "test-worker"
        mock_options.max_inflight_jobs = 10
        mock_options.enable_observability = False
        mock_options.ledger = None
        mock_options.scheduler = None
        mock_options.lease_ttl_seconds = 300.0
        mock_options.redelivery_retry_interval_seconds = 5.0

        mock_registry.get_or_create = Mock(return_value={"state": {}})

        with patch("dadbot.core.control_plane.Scheduler"):
            with patch("dadbot.core.control_plane.RecoveryManager"):
                cp = ExecutionControlPlane(
                    registry=mock_registry,
                    kernel_executor=mock_executor,
                    options=mock_options,
                )

        cp._post_execution_pre_commit_contract_gate = Mock()
        cp._validate_trace_invariant = Mock()
        cp._apply_turn_committed_core_state = Mock()
        cp._maybe_compact_ledger = Mock()

        # Return a deterministic ranked response and attach telemetry as side-effect.
        def _run_with_telemetry(context):
            context.response_engine_telemetry = {
                "selected": {"source": "direct", "components": {"base_score": 0.7}},
                "candidates": [{"source": "direct", "final_score": 0.7}],
            }
            return "ranked response"

        cp._response_engine.run = Mock(side_effect=_run_with_telemetry)

        job = Mock()
        job.user_input = "hello"
        job.trace_id = "trace-1"
        job.metadata = {"execution_result": {"status": "running"}}
        job.job_id = "job-1"

        result = cp._finalize_submit_success(
            job=job,
            result=("initial response", False),
            session_key="session-1",
            trace_token="trace-1",
            before_state_hash="hash-before",
            dedupe_future=None,
            loop_iterations=1,
        )

        assert result[0] == "ranked response"
        assert "response_engine_telemetry" in job.metadata
        assert job.metadata["response_engine_telemetry"]["selected"]["source"] == "direct"

    def test_finalize_submit_success_persists_structured_decision_reasoning(self):
        mock_registry = Mock()
        mock_executor = Mock()
        mock_options = Mock()
        mock_options.stream_sink = None
        mock_options.worker_id = "test-worker"
        mock_options.max_inflight_jobs = 10
        mock_options.enable_observability = False
        mock_options.ledger = None
        mock_options.scheduler = None
        mock_options.lease_ttl_seconds = 300.0
        mock_options.redelivery_retry_interval_seconds = 5.0

        mock_registry.get_or_create = Mock(return_value={"state": {}})

        with patch("dadbot.core.control_plane.Scheduler"):
            with patch("dadbot.core.control_plane.RecoveryManager"):
                cp = ExecutionControlPlane(
                    registry=mock_registry,
                    kernel_executor=mock_executor,
                    options=mock_options,
                )

        cp._post_execution_pre_commit_contract_gate = Mock()
        cp._validate_trace_invariant = Mock()
        cp._apply_turn_committed_core_state = Mock()
        cp._maybe_compact_ledger = Mock()

        def _run_with_telemetry(context):
            context.response_engine_telemetry = {
                "selected": {
                    "source": "tool_registry",
                    "final_score": 0.84,
                    "selected_score": 0.84,
                    "second_best_score": 0.74,
                    "decision_confidence": 0.10,
                    "reason": "response_engine_selection",
                    "reasoning": {
                        "safety": "safety_support: no veto on selected path",
                        "tools": "tool_registry: tool output increases groundedness",
                        "memory": "No memory-side shadow boost applied",
                        "coherence": "No generic shadow coherence adjustment",
                    },
                    "components": {
                        "safety_weight": -0.02,
                        "tool_weight": 0.14,
                        "memory_weight": 0.11,
                        "coherence_weight": 0.06,
                    },
                },
                "candidates": [
                    {"source": "tool_registry", "final_score": 0.84},
                    {"source": "direct", "final_score": 0.74},
                ],
                "decision_confidence": 0.10,
                "selected_score": 0.84,
                "second_best_score": 0.74,
                "shadow_decision_bus": [
                    {
                        "source": "safety_support",
                        "type": "veto",
                        "content_preview": "candidate direct",
                        "reason": "policy uncertainty",
                        "would_replace": True,
                        "priority": 1.0,
                        "timestamp": 1.0,
                    }
                ],
            }
            return "ranked response"

        cp._response_engine.run = Mock(side_effect=_run_with_telemetry)

        job = Mock()
        job.user_input = "hello"
        job.trace_id = "trace-1"
        job.metadata = {"execution_result": {"status": "running"}}
        job.job_id = "job-1"

        _ = cp._finalize_submit_success(
            job=job,
            result=("initial response", False),
            session_key="session-1",
            trace_token="trace-1",
            before_state_hash="hash-before",
            dedupe_future=None,
            loop_iterations=1,
        )

        report = dict(job.metadata.get("response_engine_decision_report") or {})
        assert report
        reasoning = dict(report.get("reasoning") or {})
        assert reasoning.get("tools")
        assert reasoning.get("safety")
        selected = dict(report.get("selected") or {})
        selected_reasoning = dict(selected.get("reasoning") or {})
        assert selected_reasoning.get("coherence")
        assert float(selected.get("decision_confidence", 0.0)) > 0.0
        influence_share = dict(selected.get("influence_share") or {})
        assert set(influence_share.keys()) == {"safety", "tools", "memory", "coherence"}

        drift = dict(job.metadata.get("response_engine_drift_monitor") or {})
        assert int(drift.get("window_size", 0)) >= 1
        rolling = dict(drift.get("rolling_averages") or {})
        assert set(rolling.keys()) == {"safety", "tools", "memory", "coherence"}

    def test_build_response_engine_context_synthesizes_pending_feedback_from_follow_up(self):
        mock_registry = Mock()
        mock_executor = Mock()
        mock_options = Mock()
        mock_options.stream_sink = None
        mock_options.worker_id = "test-worker"
        mock_options.max_inflight_jobs = 10
        mock_options.enable_observability = False
        mock_options.ledger = None
        mock_options.scheduler = None
        mock_options.lease_ttl_seconds = 300.0
        mock_options.redelivery_retry_interval_seconds = 5.0

        session = {
            "state": {
                "response_learning_pending": {
                    "trace_id": "trace-prev",
                    "selected": {
                        "risk_level": 0.2,
                        "components": {
                            "base_score": 0.7,
                            "emotion_bias": 0.1,
                            "memory_relevance": 0.6,
                            "user_alignment": 0.8,
                            "trajectory_alignment": 0.7,
                            "predicted_user_reaction": 0.75,
                        },
                    },
                    "feedback_attempts": 0,
                },
            }
        }
        mock_registry.get_or_create = Mock(return_value=session)

        with patch("dadbot.core.control_plane.Scheduler"):
            with patch("dadbot.core.control_plane.RecoveryManager"):
                cp = ExecutionControlPlane(
                    registry=mock_registry,
                    kernel_executor=mock_executor,
                    options=mock_options,
                )

        mock_job = Mock()
        mock_job.user_input = "Thanks, that helps a lot"
        mock_job.metadata = {}
        mock_job.trace_id = "trace-new"

        context = cp._build_response_engine_context(
            job=mock_job,
            session_key="session-1",
            initial_response="initial",
        )

        feedback = dict(context.job_metadata.get("reward_feedback") or {})
        assert feedback
        assert feedback["reward"] > 0.0
        assert feedback["confidence"] >= 0.35
        assert "response_learning_pending" not in session["state"]

    def test_finalize_submit_success_persists_pending_selection_for_next_turn_learning(self):
        mock_registry = Mock()
        mock_executor = Mock()
        mock_options = Mock()
        mock_options.stream_sink = None
        mock_options.worker_id = "test-worker"
        mock_options.max_inflight_jobs = 10
        mock_options.enable_observability = False
        mock_options.ledger = None
        mock_options.scheduler = None
        mock_options.lease_ttl_seconds = 300.0
        mock_options.redelivery_retry_interval_seconds = 5.0

        session = {"state": {}}
        mock_registry.get_or_create = Mock(return_value=session)

        with patch("dadbot.core.control_plane.Scheduler"):
            with patch("dadbot.core.control_plane.RecoveryManager"):
                cp = ExecutionControlPlane(
                    registry=mock_registry,
                    kernel_executor=mock_executor,
                    options=mock_options,
                )

        cp._post_execution_pre_commit_contract_gate = Mock()
        cp._validate_trace_invariant = Mock()
        cp._apply_turn_committed_core_state = Mock()
        cp._maybe_compact_ledger = Mock()

        def _run_with_telemetry(context):
            context.response_engine_telemetry = {
                "selected": {
                    "source": "direct",
                    "risk_level": 0.25,
                    "components": {
                        "base_score": 0.8,
                        "emotion_bias": 0.1,
                        "memory_relevance": 0.5,
                        "user_alignment": 0.7,
                        "trajectory_alignment": 0.6,
                        "predicted_user_reaction": 0.75,
                    },
                },
                "selected_text_preview": "preview",
                "candidates": [{"source": "direct", "final_score": 0.8}],
            }
            return "ranked response"

        cp._response_engine.run = Mock(side_effect=_run_with_telemetry)

        job = Mock()
        job.user_input = "hello"
        job.trace_id = "trace-1"
        job.metadata = {"execution_result": {"status": "running"}}
        job.job_id = "job-1"

        _ = cp._finalize_submit_success(
            job=job,
            result=("initial response", False),
            session_key="session-1",
            trace_token="trace-1",
            before_state_hash="hash-before",
            dedupe_future=None,
            loop_iterations=1,
        )

        pending = dict(session["state"].get("response_learning_pending") or {})
        assert pending
        assert pending["trace_id"] == "trace-1"
        assert dict(pending.get("selected") or {}).get("source") == "direct"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
