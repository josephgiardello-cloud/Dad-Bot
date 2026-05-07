from dadbot.core.goal_resynthesis import GoalRecalibrationEngine


def test_no_trigger_returns_monitor_state():
    engine = GoalRecalibrationEngine()
    result = engine.synthesize(
        goals=[{"description": "Ship weekly summary"}],
        friction_analysis={
            "composite_score": 0.4,
            "should_trigger_re_synthesis": False,
            "confidence": 0.7,
        },
    )

    assert result.should_re_synthesize is False
    assert result.urgency == "monitor"
    assert "not sustained" in result.message.lower()


def test_trigger_builds_goal_adjustment_proposal():
    engine = GoalRecalibrationEngine()
    result = engine.synthesize(
        goals=[{"description": "Finish integration test hardening"}],
        friction_analysis={
            "composite_score": 0.78,
            "should_trigger_re_synthesis": True,
            "primary_friction_factor": "context_exhaustion",
            "recommended_intervention": "Session fatigue evident",
            "confidence": 0.82,
        },
        reflection_summary={"likely_trigger_category": "fatigue"},
    )

    assert result.should_re_synthesize is True
    assert result.proposal is not None
    assert result.urgency == "high"
    assert "smallest shippable slice" in result.proposal.revised_goal.lower()
    assert len(result.proposal.suggested_constraints) >= 2
    assert "fatigue" in result.proposal.rationale.lower()


def test_topic_drift_adaptation_is_focus_directed():
    engine = GoalRecalibrationEngine()
    result = engine.synthesize(
        goals=[{"goal": "Finalize release notes and publish"}],
        friction_analysis={
            "composite_score": 0.9,
            "should_trigger_re_synthesis": True,
            "primary_friction_factor": "topic_drift",
            "confidence": 0.9,
        },
    )

    assert result.urgency == "immediate"
    assert result.proposal is not None
    assert "core objective" in result.proposal.revised_goal.lower()
    assert any("relevance" in item.lower() for item in result.proposal.suggested_constraints)


def test_payload_serialization_is_stable():
    engine = GoalRecalibrationEngine()
    result = engine.synthesize(
        goals=[],
        friction_analysis={
            "composite_score": 0.72,
            "should_trigger_re_synthesis": True,
            "primary_friction_factor": "halt_streak",
            "confidence": 0.66,
        },
    )

    payload = engine.to_context_payload(result)
    assert payload["should_re_synthesize"] is True
    assert payload["urgency"] in {"moderate", "high", "immediate"}
    assert isinstance(payload.get("proposal"), dict)
    assert payload["proposal"]["confidence"] == result.proposal.confidence
