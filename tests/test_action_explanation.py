from __future__ import annotations

import hashlib

import pytest

from dadbot_system.action_explanation import explain_last_actions

pytestmark = pytest.mark.unit


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_explain_last_actions_summarizes_only_bounded_audit_metadata() -> None:
    record = {
        "tool_name": "set_reminder",
        "event_id": "event-1",
        "event_checksum": "evtchk-abc",
        "previous_checksum": "evtchk-previous",
        "status": "succeeded",
        "authorization_decision": "verified",
        "authority_mode": "delegated_task",
        "trace_id": "trace-1",
        "timestamp": "2026-11-09T12:00:00+00:00",
        "input_hash": _fingerprint("private input"),
        "output_hash": _fingerprint("private output"),
        "input_parameters": {"title": "private reminder title"},
        "result": "private tool response",
    }

    explanation = explain_last_actions([record])

    assert explanation["scope"] == "latest_turn"
    assert explanation["actions"] == [
        {
            "event_id": "event-1",
            "event_checksum": "evtchk-abc",
            "previous_checksum": "evtchk-previous",
            "tool_name": "set_reminder",
            "status": "succeeded",
            "authorization_decision": "verified",
            "authority_mode": "delegated_task",
            "trace_id": "trace-1",
            "timestamp": "2026-11-09T12:00:00+00:00",
            "input_fingerprint": record["input_hash"],
            "output_fingerprint": record["output_hash"],
            "explanation": "set_reminder completed after verified delegated authority.",
        }
    ]
    assert "private" not in str(explanation)


def test_explain_last_actions_bounds_results_and_handles_empty_records() -> None:
    records = [
        {
            "tool_name": f"tool-{index}",
            "status": "cached",
            "authorization_decision": "trusted_local",
            "authority_mode": "trusted_local",
        }
        for index in range(60)
    ]

    explanation = explain_last_actions(records, limit=500)

    assert len(explanation["actions"]) == 50
    assert explanation["actions"][0]["tool_name"] == "tool-10"
    assert "reused a previously recorded result" in explanation["summary"]
    assert explain_last_actions(None)["summary"] == "No tool actions were recorded for the latest turn."
