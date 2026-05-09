from dadbot.api_models import build_pulse_envelope, build_turn_envelope


def test_build_turn_envelope_maps_defaults_and_core_fields():
    envelope = build_turn_envelope(
        session_id="s1",
        request_id="r1",
        tenant_id="family-a",
        response_payload={"reply": "hello", "active_model": "llama3.2", "status": "completed"},
        session_state={},
        session_metadata={},
    )

    data = envelope.model_dump(mode="json")
    assert data["response_text"] == "hello"
    assert data["trust_meter"]["trust_level"] == 50
    assert data["trust_meter"]["openness_level"] == 50
    assert data["drift_alarm"]["active"] is False
    assert data["integrity_status"]["merkle_check_passed"] is True
    assert data["subconscious_metadata"] == []
    assert data["thinking_state"] == []


def test_build_pulse_envelope_sets_drift_active_and_integrity_failure():
    envelope = build_pulse_envelope(
        session_id="s2",
        tenant_id="family-a",
        session_state={
            "cognition_stream": [{"step_id": "t1", "thought_trace": "calibrate", "confidence_score": 0.8}],
            "last_reflection_summary": {
                "current_risk_level": "critical",
                "predicted_drift_probability": 0.87,
                "likely_trigger_category": "goal_conflict",
                "confidence_score": 0.76,
            },
            "last_integrity_status": {
                "merkle_check_passed": False,
                "reason": "state_hash_mismatch",
            },
            "subconscious_memory_fragments": [
                {
                    "memory_id": "mem-1",
                    "similarity_score": 0.9,
                    "summary": "high relevance memory",
                }
            ],
        },
        session_metadata={},
    )

    data = envelope.model_dump(mode="json")
    assert data["event_type"] == "pulse.snapshot"
    assert data["status"] == "thinking"
    assert data["inference_intent"] == "calibrate"
    assert data["drift_alarm"]["active"] is True
    assert data["drift_alarm"]["current_risk_level"] == "critical"
    assert data["drift_alarm"]["predicted_drift_probability"] == 0.87
    assert data["integrity_status"]["merkle_check_passed"] is False
    assert data["integrity_status"]["reason"] == "state_hash_mismatch"
    assert len(data["subconscious_metadata"]) == 1
