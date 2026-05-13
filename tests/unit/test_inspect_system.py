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
            "fallback_rate": 0.18,
            "dominant_signal": "safety",
            "anomalies": ["safety_overriding_too_often", "decision_confidence_low"],
            "last_turn": {"confidence": 0.09, "dominant_signal": "safety"},
            "stability": "drifting",
            "history": [{"decision_confidence": 0.09, "is_fallback": False}],
        }
    }

    snapshot = inspect_system._compact_diagnostic_snapshot(payload)

    assert snapshot == {
        "confidence_avg": 0.31,
        "confidence_trend": -0.06,
        "fallback_rate": 0.18,
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
                "fallback_rate": 0.0,
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
        "fallback_rate",
        "dominant_signal",
        "anomalies",
        "last_turn",
        "stability",
    }
