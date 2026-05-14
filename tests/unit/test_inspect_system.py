from __future__ import annotations

import json

import pytest

import tools.inspect_system as inspect_system

pytestmark = pytest.mark.unit


def test_compact_diagnostic_snapshot_hides_raw_details():
    payload = {
        "response_engine_diagnostics": {
            "confidence_avg": 0.31,
            "confidence_trend": -0.06,
            "dominant_signal": "safety",
            "anomalies": ["safety_overriding_too_often", "decision_confidence_low"],
            "last_turn": {"confidence": 0.09, "dominant_signal": "safety"},
            "stability": "drifting",
            "history": [{"decision_confidence": 0.09}],
        }
    }

    snapshot = inspect_system._compact_diagnostic_snapshot(payload)

    assert snapshot == {
        "confidence_avg": 0.31,
        "confidence_trend": -0.06,
        "dominant_signal": "safety",
        "anomalies": ["safety_overriding_too_often", "decision_confidence_low"],
        "last_turn": {"confidence": 0.09, "dominant_signal": "safety"},
        "stability": "drifting",
    }
    assert "history" not in snapshot
    assert "response_engine_drift_monitor" not in snapshot
    assert "shadow_decision_bus" not in snapshot


def test_cmd_health_prints_compact_json_only(monkeypatch, capsys):
    monkeypatch.setattr(
        inspect_system,
        "_live_health",
        lambda: {
            "status": "ok",
            "response_engine_diagnostics": {
                "confidence_avg": 0.42,
                "confidence_trend": 0.02,
                "dominant_signal": "coherence",
                "anomalies": [],
                "last_turn": {"confidence": 0.51, "dominant_signal": "coherence"},
                "stability": "stable",
            },
        },
    )

    exit_code = inspect_system.cmd_health(object())
    captured = capsys.readouterr().out.strip()

    assert exit_code == 0
    snapshot = json.loads(captured)
    assert snapshot["dominant_signal"] == "coherence"
    assert set(snapshot.keys()) == {
        "confidence_avg",
        "confidence_trend",
        "dominant_signal",
        "anomalies",
        "last_turn",
        "stability",
    }


def test_health_schema_frozen_v1_canonical():
    """Verify health schema is locked at v1_canonical with expected keys."""
    assert inspect_system.HEALTH_SCHEMA_VERSION == "v1_canonical"
    assert inspect_system.EXPECTED_HEALTH_KEYS == frozenset({
        "anomalies",
        "confidence_avg",
        "confidence_trend",
        "dominant_signal",
        "last_turn",
        "stability",
    })
    assert "fallback_rate" in inspect_system.FORBIDDEN_HEALTH_KEYS


def test_health_schema_rejects_fallback_rate_regression():
    """Schema validation should reject any fallback_rate field (regression test)."""
    invalid_snapshot = {
        "confidence_avg": 0.5,
        "confidence_trend": 0.0,
        "dominant_signal": "coherence",
        "anomalies": [],
        "last_turn": {"confidence": 0.5, "dominant_signal": "coherence"},
        "stability": "stable",
        "fallback_rate": 0.0,  # REGRESSION: forbidden field
    }
    
    with pytest.raises(AssertionError) as exc_info:
        inspect_system._validate_health_schema(invalid_snapshot)
    
    assert "Forbidden key surfaced post-schema-lock" in str(exc_info.value)
    assert "fallback_rate" in str(exc_info.value)


def test_health_schema_rejects_missing_fields():
    """Schema validation should reject incomplete snapshots."""
    incomplete_snapshot = {
        "confidence_avg": 0.5,
        "confidence_trend": 0.0,
        # missing: dominant_signal, anomalies, last_turn, stability
    }
    
    with pytest.raises(ValueError) as exc_info:
        inspect_system._validate_health_schema(incomplete_snapshot)
    
    assert "Missing expected keys" in str(exc_info.value)


def test_health_schema_rejects_unexpected_fields():
    """Schema validation should reject new fields without version bump."""
    extra_field_snapshot = {
        "confidence_avg": 0.5,
        "confidence_trend": 0.0,
        "dominant_signal": "coherence",
        "anomalies": [],
        "last_turn": {"confidence": 0.5, "dominant_signal": "coherence"},
        "stability": "stable",
        "new_experimental_field": "should_fail",  # Regression: new field without version bump
    }
    
    with pytest.raises(ValueError) as exc_info:
        inspect_system._validate_health_schema(extra_field_snapshot)
    
    assert "Unexpected keys detected" in str(exc_info.value)
    assert "version bump required" in str(exc_info.value).lower()


def test_health_determinism_identical_input_identical_output():
    """Same input must produce identical output across multiple runs (determinism)."""
    # Fixed payload: carefully constructed to test all code paths
    payload = {
        "response_engine_diagnostics": {
            "confidence_avg": 0.35,
            "confidence_trend": 0.02,
            "dominant_signal": "memory",
            "anomalies": ["low_coherence"],
            "last_turn": {"confidence": 0.40, "dominant_signal": "memory"},
            "stability": "stable",
            "history": [
                {"decision_confidence": 0.30},
                {"decision_confidence": 0.32},
                {"decision_confidence": 0.35},
                {"decision_confidence": 0.38},
            ],
        }
    }
    
    # Generate output multiple times with identical input
    outputs = [inspect_system._compact_diagnostic_snapshot(payload) for _ in range(5)]
    
    # All outputs must be identical
    assert all(out == outputs[0] for out in outputs), "Health output is non-deterministic!"
    
    # Verify schema is correct
    assert outputs[0] is not None
    inspect_system._validate_health_schema(outputs[0])
    
    # Verify content matches
    assert outputs[0]["confidence_avg"] == 0.35
    assert outputs[0]["dominant_signal"] == "memory"
    assert outputs[0]["anomalies"] == ["low_coherence"]
    assert outputs[0]["stability"] == "stable"


def test_health_determinism_multiple_decision_reports():
    """Determinism test with decision report fallback path (early return)."""
    # Payload with decision report instead of diagnostics
    payload = {
        "response_engine_decision_report": {
            "selected": {
                "decision_confidence": 0.25,
                "influence_share": {
                    "safety": 0.4,
                    "tools": 0.2,
                    "memory": 0.3,
                    "coherence": 0.1,
                },
            }
        }
    }
    
    # Generate multiple times
    outputs = [inspect_system._compact_diagnostic_snapshot(payload) for _ in range(3)]
    
    # All must be identical
    assert all(out == outputs[0] for out in outputs), "Early-return health output is non-deterministic!"
    
    # Verify schema and content
    inspect_system._validate_health_schema(outputs[0])
    assert outputs[0]["confidence_avg"] == 0.25
    assert outputs[0]["dominant_signal"] in {"safety", "tools", "memory", "coherence"}

