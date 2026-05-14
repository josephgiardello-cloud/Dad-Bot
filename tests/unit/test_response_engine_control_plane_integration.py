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

    def test_build_response_engine_context_includes_persistent_identity_state(self):
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
                "emotion_state": {
                    "valence": 0.6,
                    "arousal": 0.3,
                    "attachment": 0.4,
                    "confidence": 0.5,
                },
                "persona_state": {
                    "tone_baseline": "steady",
                    "conversational_style_anchor": "balanced",
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
        mock_job.user_input = "test"
        mock_job.metadata = {}
        mock_job.trace_id = "trace-identity"

        context = cp._build_response_engine_context(
            job=mock_job,
            session_key="session-identity",
            initial_response="hello",
        )

        assert dict(context.emotion_state)["valence"] == pytest.approx(0.6)
        assert dict(context.persona_state)["tone_baseline"] == "steady"

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

    def test_finalize_submit_success_updates_identity_state_and_continuity(self):
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

        session = {"state": {"persona_state": {"conversational_style_anchor": "concise"}}}
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
                "selected": {"source": "direct", "final_score": 0.8},
                "candidates": [{"source": "direct", "final_score": 0.8}],
            }
            return "Great to hear that!! We can do this next."

        cp._response_engine.run = Mock(side_effect=_run_with_telemetry)

        job = Mock()
        job.user_input = "I am worried but thanks"
        job.trace_id = "trace-identity-1"
        job.metadata = {
            "execution_result": {"status": "running"},
            "semantic_memory_context": [{"memory_id": "m1"}],
        }
        job.job_id = "job-identity-1"

        result = cp._finalize_submit_success(
            job=job,
            result=("initial response", False),
            session_key="session-identity",
            trace_token="trace-identity-1",
            before_state_hash="hash-before",
            dedupe_future=None,
            loop_iterations=1,
        )

        assert "!!" not in result[0]
        assert isinstance(dict(job.metadata.get("response_continuity") or {}), dict)

        state = dict(session.get("state") or {})
        assert dict(state.get("emotion_state_memory") or {}).get("state")
        assert dict(state.get("persona_state") or {}).get("tone_baseline") in {"steady", "engaged"}
        assert dict(job.metadata.get("identity_state") or {}).get("emotion_state")
        felt = dict(state.get("felt_persona_state") or {})
        assert str(felt.get("narrative_phase") or "").strip()
        assert str(felt.get("target_stance") or "").strip()
        assert -1.0 <= float(felt.get("emotional_momentum", 0.0) or 0.0) <= 1.0
        trajectory = dict(state.get("conversation_trajectory") or {})
        assert 0.0 <= float(trajectory.get("continuity_pressure", 0.0) or 0.0) <= 1.0
        assert str(trajectory.get("preferred_tone") or "").strip()
        assert isinstance(list(trajectory.get("continuity_markers") or []), list)
        assert str(trajectory.get("narrative_phase") or "").strip()

    def test_build_response_engine_context_strengthens_constraints_under_drift_guard(self):
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
                "persona_state": {
                    "tone_baseline": "calm",
                    "identity_stability": {
                        "score": 0.9,
                        "drift_guard_level": 0.9,
                    },
                },
                "persona_constraints": {},
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
        mock_job.user_input = "test"
        mock_job.metadata = {}
        mock_job.trace_id = "trace-constraints"

        context = cp._build_response_engine_context(
            job=mock_job,
            session_key="session-constraints",
            initial_response="hello",
        )

        constraints = dict(context.persona_constraints or {})
        assert constraints.get("required_tone") == "calm"
        assert float(constraints.get("max_risk", 1.0) or 1.0) <= 0.45
        assert isinstance(dict(context.felt_persona_state or {}), dict)

    def test_phase3_behavioral_stability_gate_30_turn_conflict_sequence(self):
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
                    "final_score": 0.82,
                    "risk_level": 0.22,
                    "components": {
                        "base_score": 0.78,
                        "emotion_bias": 0.08,
                        "memory_relevance": 0.62,
                        "user_alignment": 0.71,
                        "trajectory_alignment": 0.74,
                        "predicted_user_reaction": 0.77,
                    },
                },
                "candidates": [{"source": "direct", "final_score": 0.82}],
            }
            prompt = str(getattr(context, "user_input", "") or "").lower()
            if "anxious" in prompt or "worried" in prompt:
                return "I hear you. Next step plan together: breathe, then take the safest move."
            if "aggressive" in prompt or "push hard" in prompt:
                return "We can stay steady and still move forward. Next step plan together is safer."
            if "detailed" in prompt:
                return "Next step plan together: define the outcome, compare tradeoffs, and choose the smallest safe action."
            return "Next step plan together. We keep this steady and move one safe step at a time."

        cp._response_engine.run = Mock(side_effect=_run_with_telemetry)

        prompts = [
            "Be concise and direct.",
            "Give me a detailed answer now.",
            "I am anxious and overwhelmed.",
            "Actually push hard and be aggressive.",
            "Switch tone and be playful.",
            "No, be formal and technical.",
        ]

        arousal_trace: list[float] = []
        valence_trace: list[float] = []
        continuity_retained: list[bool] = []
        felt_causality_retained: list[bool] = []

        for turn in range(30):
            user_input = prompts[turn % len(prompts)]
            job = Mock()
            job.user_input = user_input
            job.trace_id = f"trace-phase3-{turn}"
            job.metadata = {
                "execution_result": {"status": "running"},
                "semantic_memory_context": [{"memory_id": f"m-{turn}", "text": "next step together"}],
            }
            job.job_id = f"job-phase3-{turn}"

            _ = cp._finalize_submit_success(
                job=job,
                result=("initial response", False),
                session_key="session-phase3-behavioral",
                trace_token=f"trace-phase3-{turn}",
                before_state_hash=f"hash-{turn}",
                dedupe_future=None,
                loop_iterations=1,
            )

            state = dict(session.get("state") or {})
            emotion = dict(state.get("emotion_state") or {})
            trajectory = dict(state.get("conversation_trajectory") or {})
            arousal_trace.append(float(emotion.get("arousal", 0.0) or 0.0))
            valence_trace.append(float(emotion.get("valence", 0.0) or 0.0))
            continuity_retained.append(
                bool(str(trajectory.get("preferred_tone") or "").strip())
                and float(trajectory.get("continuity_pressure", 0.0) or 0.0) >= 0.5
                and bool(list(trajectory.get("continuity_markers") or []))
            )
            felt_state = dict(state.get("felt_persona_state") or {})
            felt_causality_retained.append(
                bool(str(felt_state.get("narrative_phase") or "").strip())
                and bool(str(felt_state.get("target_stance") or "").strip())
                and float(felt_state.get("coherence_pressure", 0.0) or 0.0) >= 0.5
            )

        # Metric 1: Identity drift budget remains bounded under conflicting prompts.
        persona_state = dict(dict(session.get("state") or {}).get("persona_state") or {})
        drift_violations = int(persona_state.get("drift_violations", 0) or 0)
        stability = dict(persona_state.get("identity_stability") or {})
        assert drift_violations <= 10
        assert float(stability.get("score", 0.0) or 0.0) >= 0.55

        # Metric 2: Emotional arc remains smooth (bounded per-turn deltas).
        arousal_deltas = [abs(arousal_trace[i] - arousal_trace[i - 1]) for i in range(1, len(arousal_trace))]
        valence_deltas = [abs(valence_trace[i] - valence_trace[i - 1]) for i in range(1, len(valence_trace))]
        assert max(arousal_deltas) <= 0.12
        assert max(valence_deltas) <= 0.12
        assert (sum(arousal_deltas) / max(len(arousal_deltas), 1)) <= 0.05
        assert (sum(valence_deltas) / max(len(valence_deltas), 1)) <= 0.05

        # Metric 3: Continuity retention rate under conflict remains high.
        retention_rate = sum(1 for item in continuity_retained if item) / float(len(continuity_retained))
        assert retention_rate >= 0.85
        felt_retention_rate = sum(1 for item in felt_causality_retained if item) / float(len(felt_causality_retained))
        assert felt_retention_rate >= 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
