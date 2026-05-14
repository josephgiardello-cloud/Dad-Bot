"""tests/unit/test_response_engine.py — ResponseEngine generation + ranking validation."""

from __future__ import annotations

import pytest

from dadbot.core.response_engine import (
    ResponseCandidate,
    ResponseEngine,
    ScoringWeights,
)
from dadbot.core.emotion_state import EmotionState


class MockContext:
    """Mock ExecutionContext for testing."""

    def __init__(self, user_input: str = "", persona_traits: list[str] | None = None):
        self.user_input = user_input
        self.persona_traits = persona_traits or []
        self.persona_constraints: dict[str, object] = {}
        self.user_preferences: dict[str, object] = {}
        self.memory_context: list[object] = []
        self.conversation_trajectory: dict[str, object] = {}
        self.session_state: dict[str, object] = {}
        self.job_metadata: dict[str, object] = {}
        self.trace_id: str = "test-trace"


class TestResponseEngineGeneration:
    """Test candidate generation strategies."""

    def test_generate_candidates_returns_list(self):
        """generate_candidates() returns list of ResponseCandidate."""
        engine = ResponseEngine()
        context = MockContext(user_input="What is the meaning of life?")

        candidates = engine.generate_candidates(context, n=5)

        assert isinstance(candidates, list)
        assert len(candidates) > 0
        assert all(isinstance(c, ResponseCandidate) for c in candidates)

    def test_generate_candidates_respects_count(self):
        """generate_candidates(n=X) returns at most X candidates."""
        engine = ResponseEngine()
        context = MockContext(user_input="Tell me a story")

        for n in [1, 3, 5, 10]:
            candidates = engine.generate_candidates(context, n=n)
            assert len(candidates) <= n

    def test_generate_candidates_distinct_sources(self):
        """Candidates from different generation strategies have different sources."""
        engine = ResponseEngine()
        context = MockContext(user_input="How should I handle this?")

        candidates = engine.generate_candidates(context, n=5)

        sources = [c.source for c in candidates]
        # Should have multiple distinct generation approaches
        assert len(set(sources)) > 1

    def test_generate_candidates_includes_diversity_injection_strategies(self):
        """Generation includes intent reinterpretation and goal-shift variants."""
        engine = ResponseEngine()
        context = MockContext(user_input="I'm worried about this decision")

        candidates = engine.generate_candidates(context, n=8)
        sources = {c.source for c in candidates}
        goals = {c.response_goal for c in candidates}

        assert "reinterpret_intent" in sources
        assert "goal_shift" in sources
        assert "inform" in goals
        assert "engage" in goals or "clarify" in goals

    def test_generate_candidates_includes_structured_variation_axes(self):
        """Structured variants expose controlled tone/depth/risk/intensity axes."""
        engine = ResponseEngine()
        context = MockContext(user_input="Help me think through this decision")

        candidates = engine.generate_candidates(context, n=12)
        structured = [c for c in candidates if c.source.startswith("structured_axis_")]

        assert structured
        assert len({c.tone for c in structured}) >= 2
        assert len({c.depth for c in structured}) >= 2
        assert len({round(c.risk_level, 2) for c in structured}) >= 2
        assert len({round(c.intensity, 2) for c in structured}) >= 2

    def test_generate_candidates_non_empty_text(self):
        """All candidates have non-empty text."""
        engine = ResponseEngine()
        context = MockContext(user_input="What do you think?")

        candidates = engine.generate_candidates(context, n=5)

        assert all(c.text and len(c.text) > 0 for c in candidates)

    def test_generate_candidates_includes_stochastic_and_multimodel_paths(self):
        engine = ResponseEngine()
        context = MockContext(user_input="How can I improve this?")
        context.job_metadata = {
            "model_candidate_texts": [
                "External model says: compare options by downside first.",
            ]
        }

        candidates = engine.generate_candidates(context, n=20)
        sources = {c.source for c in candidates}

        assert any(source.startswith("stochastic_prompt_") for source in sources)
        assert any(source.startswith("multi_model_") for source in sources)

    def test_generate_candidates_empty_context_handles_gracefully(self):
        """generate_candidates() handles empty context without crashing."""
        engine = ResponseEngine()
        context = MockContext(user_input="")

        # Should not raise; may return fewer or fallback candidates
        candidates = engine.generate_candidates(context, n=5)
        assert isinstance(candidates, list)


class TestResponseEngineScoring:
    """Test candidate scoring logic."""

    def test_score_candidate_returns_float(self):
        """score_candidate() returns a float score."""
        engine = ResponseEngine()
        context = MockContext(user_input="What is success?")
        candidate = ResponseCandidate(
            text="Success is doing what matters to you.",
            source="direct",
        )

        score = engine.score_candidate(candidate, context)

        assert isinstance(score, float)

    def test_filter_candidates_applies_hard_coherence_relevance_gate(self):
        """Hard filter removes candidates failing coherence or relevance threshold."""
        engine = ResponseEngine()
        context = MockContext(user_input="How do I handle conflict?")
        candidates = [
            ResponseCandidate(text="How conflict is handled depends on communication.", source="good"),
            ResponseCandidate(text="x", source="too_short"),
            ResponseCandidate(text="Bananas in the sky are yellow.", source="irrelevant"),
        ]

        filtered = engine.filter_candidates(candidates, context)

        assert any(c.source == "good" for c in filtered)
        assert all(c.source != "too_short" for c in filtered)
        assert all(c.source != "irrelevant" for c in filtered)

    def test_score_candidates_normalization_stable_distribution(self):
        """score_candidates normalizes dimensions and returns finite stable scores."""
        engine = ResponseEngine()
        context = MockContext(user_input="Help me decide")
        candidates = [
            ResponseCandidate(text="Let's compare options and tradeoffs.", source="a"),
            ResponseCandidate(text="What outcome matters most to you first?", source="b"),
            ResponseCandidate(text="Try a quick 2-step plan.", source="c"),
        ]

        scored = engine.score_candidates(candidates, context)

        assert len(scored) == 3
        for _, score in scored:
            assert isinstance(score, float)
            assert score == score
            assert score != float("inf")
            assert score != float("-inf")

    def test_score_candidate_coherence_dimension(self):
        """Coherence scoring: longer, structured text scores higher."""
        engine = ResponseEngine()
        context = MockContext(user_input="Tell me")

        # Short, minimal text
        short = ResponseCandidate(text="X", source="test")
        short_score = engine.score_candidate(short, context)

        # Longer, structured text
        long = ResponseCandidate(
            text="This is a complete sentence with proper structure and depth.",
            source="test",
        )
        long_score = engine.score_candidate(long, context)

        # Longer text should score higher on coherence contribution
        assert long_score > short_score

    def test_score_candidate_relevance_dimension(self):
        """Relevance scoring: responses with keyword overlap score higher."""
        engine = ResponseEngine()
        context = MockContext(user_input="How do I handle conflict?")

        # Relevant: mentions key terms
        relevant = ResponseCandidate(
            text="Conflict requires communication and patience.",
            source="test",
        )
        relevant_score = engine.score_candidate(relevant, context)

        # Irrelevant: no connection
        irrelevant = ResponseCandidate(
            text="The weather is nice today.",
            source="test",
        )
        irrelevant_score = engine.score_candidate(irrelevant, context)

        assert relevant_score > irrelevant_score

    def test_score_candidate_memory_relevance_influences_selection(self):
        engine = ResponseEngine()
        context = MockContext(user_input="How should I handle this?")
        context.memory_context = ["You prefer step-by-step plans and concrete next actions."]

        memory_aligned = ResponseCandidate(
            text="Here is a concrete step-by-step plan with your next actions.",
            source="memory_aligned",
            tone="calm",
            depth="deep",
            response_goal="inform",
        )
        memory_agnostic = ResponseCandidate(
            text="Interesting question, maybe there are many possibilities.",
            source="memory_agnostic",
            tone="neutral",
            depth="short",
            response_goal="inform",
        )

        aligned_score = engine.score_candidate(memory_aligned, context)
        agnostic_score = engine.score_candidate(memory_agnostic, context)

        assert aligned_score > agnostic_score

    def test_score_candidate_user_alignment_respects_preferences(self):
        engine = ResponseEngine()
        context = MockContext(user_input="What should I do?")
        context.user_preferences = {
            "preferred_tone": "calm",
            "preferred_depth": "short",
            "style": "concise",
        }

        aligned = ResponseCandidate(
            text="Let's keep this simple and steady.",
            source="aligned",
            tone="calm",
            depth="short",
            response_goal="clarify",
        )
        misaligned = ResponseCandidate(
            text="Let's deeply unpack this with many layers and assumptions.",
            source="misaligned",
            tone="engaging",
            depth="deep",
            response_goal="inform",
        )

        assert engine.score_candidate(aligned, context) > engine.score_candidate(misaligned, context)

    def test_score_candidate_trajectory_continuity_pressure_prefers_structural_match(self):
        engine = ResponseEngine()
        context = MockContext(user_input="I need help with this next step")
        context.conversation_trajectory = {
            "desired_goal": "clarify",
            "preferred_response_goal": "clarify",
            "preferred_tone": "calm",
            "continuity_pressure": 0.9,
            "emotional_target": {"arousal": 0.3},
            "continuity_markers": ["next step", "plan"],
            "felt_state": {
                "narrative_phase": "stabilizing",
                "target_stance": "supportive",
                "emotional_momentum": -0.1,
            },
        }

        continuity_aligned = ResponseCandidate(
            text="Let's keep this steady. Here is the next step plan.",
            source="aligned",
            tone="calm",
            intensity=0.35,
            response_goal="clarify",
            risk_level=0.2,
        )
        continuity_misaligned = ResponseCandidate(
            text="Honestly, this is random and we should jump topics immediately.",
            source="misaligned",
            tone="engaging",
            intensity=0.9,
            response_goal="engage",
            risk_level=0.2,
        )

        aligned_score = engine.score_candidate(continuity_aligned, context)
        misaligned_score = engine.score_candidate(continuity_misaligned, context)

        assert aligned_score > misaligned_score

    def test_score_candidate_felt_state_guides_narrative_causality(self):
        engine = ResponseEngine()
        context = MockContext(user_input="I feel overwhelmed")
        context.felt_persona_state = {
            "narrative_phase": "stabilizing",
            "preferred_response_goal": "engage",
            "target_stance": "supportive",
            "emotional_momentum": -0.2,
        }
        context.conversation_trajectory = {
            "desired_goal": "engage",
            "preferred_response_goal": "engage",
            "preferred_tone": "warm",
            "continuity_pressure": 0.85,
            "emotional_target": {"arousal": 0.25},
            "continuity_markers": ["with you", "next step"],
        }

        causality_aligned = ResponseCandidate(
            text="I hear you. I'm with you. Let's take one next step together.",
            source="aligned",
            tone="warm",
            intensity=0.3,
            stance="supportive",
            response_goal="engage",
            risk_level=0.2,
        )
        causality_misaligned = ResponseCandidate(
            text="Let's jump to a risky immediate action with no context.",
            source="misaligned",
            tone="engaging",
            intensity=0.9,
            stance="forward",
            response_goal="inform",
            risk_level=0.7,
        )

        assert engine.score_candidate(causality_aligned, context) > engine.score_candidate(causality_misaligned, context)

    def test_score_candidate_redundancy_penalty(self):
        """Redundancy penalty: repeated responses score lower."""
        engine = ResponseEngine()
        context = MockContext(user_input="Test")

        candidate = ResponseCandidate(
            text="This is my response.",
            source="test",
        )

        # First time: no redundancy
        score_first = engine.score_candidate(candidate, context)

        # Add to recent responses manually
        engine._recent_responses = ["This is my response."]

        # Second time: should be penalized
        score_second = engine.score_candidate(candidate, context)

        assert score_second < score_first

    def test_score_candidate_novelty_dimension(self):
        """Novelty scoring: word diversity correlates with higher scores."""
        engine = ResponseEngine()
        context = MockContext(user_input="How should I approach this?")

        # Low diversity: repetitive words (but relevant)
        repetitive = ResponseCandidate(
            text="approach approach approach approach approach",
            source="test",
        )
        repetitive_score = engine.score_candidate(repetitive, context)

        # High diversity: varied vocabulary (and relevant)
        diverse = ResponseCandidate(
            text="Here's my approach: explore different perspectives thoughtfully and creatively.",
            source="test",
        )
        diverse_score = engine.score_candidate(diverse, context)

        # High diversity should score higher on novelty contribution
        assert diverse_score > repetitive_score


class TestResponseEngineSelection:
    """Test candidate selection logic."""

    def test_select_best_returns_highest_scored(self):
        """select_best() returns candidate with highest score."""
        engine = ResponseEngine()

        candidates_scores = [
            (ResponseCandidate(text="Response A", source="test"), 0.50),
            (ResponseCandidate(text="Response B", source="test"), 0.90),
            (ResponseCandidate(text="Response C", source="test"), 0.65),
        ]

        best = engine.select_best(candidates_scores)

        assert best.text == "Response B"

    def test_select_best_raises_on_empty(self):
        """select_best() raises ValueError on empty list."""
        engine = ResponseEngine()

        with pytest.raises(ValueError, match="No candidates"):
            engine.select_best([])

    def test_select_best_single_candidate(self):
        """select_best() handles single candidate."""
        engine = ResponseEngine()

        candidate = ResponseCandidate(text="Only one", source="test")
        candidates_scores = [(candidate, 0.75)]

        best = engine.select_best(candidates_scores)

        assert best.text == "Only one"


class TestResponseEngineOrchestration:
    """Test run() orchestrator method."""

    def test_run_returns_string(self):
        """run() returns a string response."""
        engine = ResponseEngine()
        context = MockContext(user_input="What is the answer?")

        result = engine.run(context)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_run_tracks_recent_responses(self):
        """run() tracks recent responses for redundancy checking."""
        engine = ResponseEngine()
        context = MockContext(user_input="Question")

        initial_count = len(engine._recent_responses)

        engine.run(context)

        assert len(engine._recent_responses) == initial_count + 1

    def test_run_caps_recent_responses_history(self):
        """run() maintains bounded history of recent responses."""
        engine = ResponseEngine()
        context = MockContext(user_input="Test")

        # Run multiple times
        for _ in range(20):
            engine.run(context)

        # History should be capped (default 10)
        assert len(engine._recent_responses) <= 10

    def test_run_flow_integration(self):
        """run() integrates generate → score → select → track."""
        engine = ResponseEngine()
        context = MockContext(user_input="How do I be a better person?")

        result = engine.run(context)

        # Should have selected something
        assert isinstance(result, str)
        # Should have tracked it
        assert result in engine._recent_responses

    def test_run_uses_filtered_candidates(self):
        """run() returns deterministic terminal fallback when all candidates are filtered."""
        engine = ResponseEngine()
        context = MockContext(user_input="Anything")

        engine.generate_candidates = lambda *args, **kwargs: [
            ResponseCandidate(text="x", source="bad1"),
            ResponseCandidate(text=".", source="bad2"),
        ]

        result = engine.run(context)

        assert result == "I hear you."
        assert isinstance(context.response_engine_telemetry, dict)
        assert context.response_engine_telemetry.get("status") == "no_valid_candidates"
        assert context.response_engine_telemetry.get("reason") == "all_filtered"


class TestResponseEngineWeights:
    """Test scoring weight customization."""

    def test_custom_weights_affect_scoring(self):
        """Custom ScoringWeights affect candidate scoring."""
        # Standard weights
        engine_default = ResponseEngine()
        context = MockContext(user_input="Test question")
        candidate = ResponseCandidate(
            text="Detailed response about the test.",
            source="test",
        )
        score_default = engine_default.score_candidate(candidate, context)

        # Custom weights: zero out coherence
        custom_weights = ScoringWeights(coherence=0.0)
        engine_custom = ResponseEngine(weights=custom_weights)
        score_custom = engine_custom.score_candidate(candidate, context)

        # Scores should differ
        assert score_default != score_custom


class TestResponseEngineEmotionAndRisk:
    """Phase 2 behavior: adaptive emotion weighting and risk alignment."""

    def test_emotion_weight_is_adaptive(self):
        engine = ResponseEngine()
        low = EmotionState(valence=0.0, arousal=0.1, attachment=0.2, confidence=0.3)
        high = EmotionState(valence=0.0, arousal=0.9, attachment=0.9, confidence=0.9)

        low_weight = engine.compute_emotion_weight(low)
        high_weight = engine.compute_emotion_weight(high)

        assert high_weight > low_weight

    def test_extract_emotional_features_classifies_sentiment_and_risk(self):
        engine = ResponseEngine()
        features = engine.extract_emotional_features("I love this, honestly we can do it together!")

        assert features["valence"] > 0
        assert features["arousal"] >= 0.8
        assert features["attachment"] >= 0.8
        assert features["risk"] >= 0.8

    def test_emotion_alignment_prefers_closer_emotional_distance(self):
        engine = ResponseEngine()
        state = EmotionState(valence=0.6, arousal=0.8, attachment=0.8, confidence=0.8)
        closer = "Great, we can definitely do this together!"
        farther = "Sorry, maybe this might not work."

        close_score = engine.emotion_alignment(closer, state)
        far_score = engine.emotion_alignment(farther, state)

        assert close_score > far_score

    def test_emotion_alignment_applies_attachment_based_risk_gating(self):
        engine = ResponseEngine()
        low_attachment = EmotionState(valence=0.2, arousal=0.4, attachment=0.0, confidence=0.7)
        high_attachment = EmotionState(valence=0.2, arousal=0.4, attachment=1.0, confidence=0.7)
        risky = "Honestly, I feel this deeply with you."

        low_score = engine.emotion_alignment(risky, low_attachment)
        high_score = engine.emotion_alignment(risky, high_attachment)

        assert high_score >= low_score


class TestResponseEngineConstraintsAndTrajectory:
    def test_filter_candidates_enforces_persona_constraints(self):
        engine = ResponseEngine()
        context = MockContext(user_input="Can you help me?")
        context.persona_constraints = {
            "disallow_words": ["damn"],
            "required_tone": "calm",
            "max_risk": 0.5,
        }

        blocked = ResponseCandidate(
            text="Damn, that's rough.",
            source="blocked",
            tone="engaging",
            risk_level=0.9,
        )
        allowed = ResponseCandidate(
            text="Let's keep this steady and solve it step by step to help you.",
            source="allowed",
            tone="calm",
            risk_level=0.3,
        )

        filtered = engine.filter_candidates([blocked, allowed], context)
        assert [c.source for c in filtered] == ["allowed"]

    def test_trajectory_alignment_rewards_desired_goal(self):
        engine = ResponseEngine()
        context = MockContext(user_input="What should I do first?")
        context.conversation_trajectory = {"desired_goal": "clarify"}

        clarify = ResponseCandidate(text="Quick clarifier: what's your priority?", source="c1", response_goal="clarify")
        engage = ResponseCandidate(text="You're not alone in this; we can handle it.", source="c2", response_goal="engage")

        assert engine.score_candidate(clarify, context) > engine.score_candidate(engage, context)

    def test_internal_simulation_penalizes_high_risk_when_attachment_low(self):
        engine = ResponseEngine()
        context = MockContext(user_input="I'm not sure what to do")
        context.session_state = {
            "emotion_state": {
                "valence": 0.0,
                "arousal": 0.2,
                "attachment": 0.1,
                "confidence": 0.3,
            }
        }

        low_risk = ResponseCandidate(
            text="Let's take one steady step now.",
            source="low",
            response_goal="clarify",
            risk_level=0.2,
        )
        high_risk = ResponseCandidate(
            text="Honestly, I feel this deeply and definitely know the only way.",
            source="high",
            response_goal="clarify",
            risk_level=0.9,
        )

        assert engine.score_candidate(low_risk, context) > engine.score_candidate(high_risk, context)

    def test_attachment_phase_guardrail_risk_tolerance_progression(self):
        """Across phases, higher attachment should increase risky-response tolerance."""
        engine = ResponseEngine()
        risky = ResponseCandidate(
            text="Honestly, I feel this deeply and definitely know the only way.",
            source="risky",
            response_goal="clarify",
            risk_level=0.9,
        )

        phase_states = [
            {"valence": 0.0, "arousal": 0.2, "attachment": 0.1, "confidence": 0.3},
            {"valence": 0.0, "arousal": 0.4, "attachment": 0.5, "confidence": 0.6},
            {"valence": 0.0, "arousal": 0.6, "attachment": 0.9, "confidence": 0.9},
        ]

        scores: list[float] = []
        for state in phase_states:
            context = MockContext(user_input="I need guidance")
            context.session_state = {"emotion_state": state}
            scores.append(engine.score_candidate(risky, context))

        assert scores[0] <= scores[1] <= scores[2]

    def test_attachment_phase_guardrail_safe_vs_risky_low_attachment(self):
        """At low attachment phase, high-risk candidate remains less preferred than safe candidate."""
        engine = ResponseEngine()
        context = MockContext(user_input="I need help deciding")
        context.session_state = {
            "emotion_state": {
                "valence": 0.0,
                "arousal": 0.2,
                "attachment": 0.1,
                "confidence": 0.3,
            }
        }

        safe = ResponseCandidate(
            text="Let's take one steady step now to help you decide.",
            source="safe",
            response_goal="clarify",
            risk_level=0.2,
        )
        risky = ResponseCandidate(
            text="Honestly, I feel this deeply and definitely know the only way.",
            source="risky",
            response_goal="clarify",
            risk_level=0.9,
        )

        assert engine.score_candidate(safe, context) > engine.score_candidate(risky, context)


class TestResponseEngineTelemetry:
    def test_run_emits_component_telemetry(self):
        engine = ResponseEngine()
        context = MockContext(user_input="How should I proceed?")

        _ = engine.run(context)

        telemetry = getattr(context, "response_engine_telemetry", None)
        assert isinstance(telemetry, dict)
        assert isinstance(telemetry.get("candidates"), list)
        assert isinstance(telemetry.get("selected"), dict)
        selected = dict(telemetry.get("selected") or {})
        components = dict(selected.get("components") or {})

        for key in [
            "base_score",
            "emotion_score",
            "emotion_weight",
            "emotion_bias",
            "persona_signal",
            "memory_relevance",
            "user_alignment",
            "trajectory_alignment",
            "predicted_user_reaction",
            "distribution_collapse_penalty",
            "interaction_bonus",
            "learned_preference",
        ]:
            assert key in components

        reward_model = dict(telemetry.get("reward_model") or {})
        assert isinstance(reward_model.get("weights"), dict)

        persona_calibration = dict(telemetry.get("persona_calibration") or {})
        assert 0.70 <= float(persona_calibration.get("factor", 0.0)) <= 1.10

    def test_persona_calibration_reflects_traits_and_constraints(self):
        engine = ResponseEngine()

        unconstrained = MockContext(user_input="Help me decide")
        constrained = MockContext(user_input="Help me decide", persona_traits=["supportive"])
        constrained.persona_constraints = {"required_tone": "warm"}

        _ = engine.run(unconstrained)
        unconstrained_factor = float(
            dict(getattr(unconstrained, "response_engine_telemetry", {}) or {})
            .get("persona_calibration", {})
            .get("factor", 0.0),
        )

        _ = engine.run(constrained)
        constrained_factor = float(
            dict(getattr(constrained, "response_engine_telemetry", {}) or {})
            .get("persona_calibration", {})
            .get("factor", 0.0),
        )

        assert constrained_factor > unconstrained_factor

    def test_feedback_update_adjusts_learned_reward_weights(self):
        engine = ResponseEngine()
        baseline = dict(engine._reward_weights)

        context = MockContext(user_input="Help me decide")
        context.job_metadata = {
            "reward_feedback": {
                "reward": 1.0,
                "features": {
                    "user_alignment": 1.0,
                    "memory_relevance": 0.8,
                    "trajectory_alignment": 0.7,
                    "predicted_user_reaction": 0.9,
                },
            }
        }

        _ = engine.run(context)

        assert engine._reward_weights["user_alignment"] > baseline["user_alignment"]
        assert engine._reward_weights["memory_relevance"] > baseline["memory_relevance"]

    def test_semantic_memory_relevance_signal_prefers_related_memory(self):
        engine = ResponseEngine()
        context = MockContext(user_input="What should I do next?")
        context.memory_context = [
            "Use a step by step plan and choose the smallest safe next action.",
        ]

        related = ResponseCandidate(
            text="Let's pick the smallest safe next action in a step by step plan.",
            source="related",
        )
        unrelated = ResponseCandidate(
            text="The color of the sunset is beautiful and abstract.",
            source="unrelated",
        )

        assert engine._score_memory_relevance(related, context) > engine._score_memory_relevance(unrelated, context)


class TestResponseEngineLearningStability:
    def test_low_confidence_feedback_is_ignored(self):
        engine = ResponseEngine()
        baseline = dict(engine._reward_weights)

        context = MockContext(user_input="test")
        context.job_metadata = {
            "reward_feedback": {
                "reward": 1.0,
                "confidence": 0.05,
                "features": {"user_alignment": 1.0},
            }
        }
        _ = engine.run(context)

        assert engine._reward_weights == baseline

    def test_attribution_limits_credit_assignment(self):
        engine = ResponseEngine()
        base_user = engine._reward_weights["user_alignment"]
        base_memory = engine._reward_weights["memory_relevance"]

        context = MockContext(user_input="test")
        context.job_metadata = {
            "reward_feedback": {
                "reward": 1.0,
                "confidence": 1.0,
                "features": {
                    "user_alignment": 1.0,
                    "memory_relevance": 1.0,
                },
                "attribution": {
                    "user_alignment": 1.0,
                    "memory_relevance": 0.0,
                },
            }
        }
        _ = engine.run(context)

        assert engine._reward_weights["user_alignment"] > base_user
        assert engine._reward_weights["memory_relevance"] == pytest.approx(base_memory)

    def test_reward_weights_are_bounded_under_repeated_feedback(self):
        engine = ResponseEngine()
        context = MockContext(user_input="stress")

        for _ in range(60):
            context.job_metadata = {
                "reward_feedback": {
                    "reward": 1.0,
                    "confidence": 1.0,
                    "features": {
                        "user_alignment": 2.0,
                        "memory_relevance": 2.0,
                        "trajectory_alignment": 2.0,
                    },
                }
            }
            _ = engine.run(context)

        assert all(abs(value) <= engine._reward_weight_bound for value in engine._reward_weights.values())

    def test_distribution_collapse_penalty_increases_for_repeated_style(self):
        engine = ResponseEngine()
        candidate = ResponseCandidate(
            text="steady",
            source="direct",
            tone="calm",
            depth="short",
            response_goal="inform",
        )

        initial = engine._distribution_collapse_penalty(candidate)
        for _ in range(12):
            engine._record_selection(candidate)
        later = engine._distribution_collapse_penalty(candidate)

        assert later >= initial


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
