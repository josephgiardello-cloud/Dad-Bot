from __future__ import annotations

import json

import pytest

from dadbot.core.persistence_record_schema import (
    CHECKPOINT_RECORD_SCHEMA_VERSION,
    IO_CAPTURE_SCHEMA_VERSION,
    PersistenceSchemaError,
    REPLAY_RECORD_SCHEMA_VERSION,
    STATE_SNAPSHOT_SCHEMA_VERSION,
    TRACE_EVENT_SCHEMA_VERSION,
    assert_valid_checkpoint_record,
    assert_valid_replay_record,
    normalize_checkpoint_record,
    normalize_io_capture,
    normalize_replay_record,
    normalize_state_snapshot,
    normalize_trace_event,
)

pytestmark = pytest.mark.unit


def test_normalize_io_capture_emits_canonical_shape() -> None:
    payload = normalize_io_capture(
        {
            "tool_name": "Current_Time",
            "version": "v7",
            "status": "OK",
            "output": {"value": "2026-01-01T00:00:00"},
        },
    )

    assert payload == {
        "schema_version": IO_CAPTURE_SCHEMA_VERSION,
        "tool": "current_time",
        "version": "v7",
        "status": "ok",
        "response": {"value": "2026-01-01T00:00:00"},
    }


def test_normalize_trace_event_preserves_extras_but_canonicalizes_core_fields() -> None:
    payload = normalize_trace_event(
        {
            "trace_id": " trace-1 ",
            "event_id": "evt",
            "sequence": "4",
            "event_type": "GRAPH_CHECKPOINT",
            "stage": "Save",
            "status": "After",
            "occurred_at": "2026-01-01T00:00:00",
            "extra": {"x": 1},
        },
    )

    assert payload["schema_version"] == TRACE_EVENT_SCHEMA_VERSION
    assert payload["trace_id"] == "trace-1"
    assert payload["sequence"] == 4
    assert payload["event_type"] == "graph_checkpoint"
    assert payload["stage"] == "save"
    assert payload["status"] == "after"
    assert payload["extra"] == {"x": 1}


def test_normalize_state_snapshot_canonical_shape() -> None:
    payload = normalize_state_snapshot(
        {
            "trace_id": "trace-2",
            "created_at": "2026-01-01T00:00:00",
            "last_sequence": "12",
            "phase": "EXECUTE",
            "strict_sequence_hash": "abc",
            "event_count": "9",
            "determinism_lock": {"lock_hash": "lk", "consistent": "true"},
        },
    )

    assert payload == {
        "schema_version": STATE_SNAPSHOT_SCHEMA_VERSION,
        "trace_id": "trace-2",
        "created_at": "2026-01-01T00:00:00",
        "last_sequence": 12,
        "phase": "EXECUTE",
        "strict_sequence_hash": "abc",
        "event_count": 9,
        "determinism_lock": {"lock_hash": "lk", "consistent": True},
    }


def test_normalize_replay_record_contains_canonical_and_compat_fields() -> None:
    replay = normalize_replay_record(
        trace_id="trace-3",
        events=[
            {
                "trace_id": "trace-3",
                "event_id": "e1",
                "sequence": 1,
                "event_type": "tool_result",
                "stage": "execute",
                "status": "after",
                "occurred_at": "2026-01-01T00:00:00",
            },
        ],
        strict_sequence_hash="h",
        strict_sequence=[{"sequence": 1, "event_id": "e1"}],
        phase="EXECUTE",
        replayed_state={"answer": "ok"},
        replayed_metadata={"terminal_state": {}},
        determinism={"consistent": True},
    )

    assert replay["schema_version"] == REPLAY_RECORD_SCHEMA_VERSION
    assert replay["trace_id"] == "trace-3"
    assert replay["event_count"] == 1
    assert replay["events"][0]["schema_version"] == TRACE_EVENT_SCHEMA_VERSION
    assert replay["replay_output"]["phase"] == "EXECUTE"
    assert replay["phase"] == "EXECUTE"
    assert replay["replayed_state"] == {"answer": "ok"}


def test_replay_record_round_trip_is_serialization_stable() -> None:
    replay = normalize_replay_record(
        trace_id="trace-rt",
        events=[
            {
                "trace_id": "trace-rt",
                "event_id": "e1",
                "sequence": 1,
                "event_type": "tool_result",
                "stage": "execute",
                "status": "after",
                "occurred_at": "2026-01-01T00:00:00",
            },
        ],
        strict_sequence_hash="h",
        strict_sequence=[{"sequence": 1, "event_id": "e1"}],
        phase="EXECUTE",
        replayed_state={"answer": "ok"},
        replayed_metadata={"terminal_state": {}},
        determinism={"consistent": True},
    )

    encoded = json.dumps(replay, sort_keys=True, ensure_ascii=True)
    decoded = json.loads(encoded)

    assert decoded == replay
    assert_valid_replay_record(decoded)


def test_replay_record_strict_validation_fails_on_event_count_drift() -> None:
    replay = normalize_replay_record(
        trace_id="trace-drift",
        events=[
            {
                "trace_id": "trace-drift",
                "event_id": "e1",
                "sequence": 1,
                "event_type": "tool_result",
                "stage": "execute",
                "status": "after",
                "occurred_at": "2026-01-01T00:00:00",
            },
        ],
        strict_sequence_hash="h",
        strict_sequence=[{"sequence": 1, "event_id": "e1"}],
        phase="EXECUTE",
        replayed_state={"answer": "ok"},
        replayed_metadata={"terminal_state": {}},
        determinism={"consistent": True},
    )
    replay["event_count"] = 9

    with pytest.raises(PersistenceSchemaError, match="event_count mismatch"):
        assert_valid_replay_record(replay)


def test_normalize_checkpoint_record_canonical_shape() -> None:
    checkpoint = normalize_checkpoint_record(
        {
            "trace_id": " trace-cp ",
            "session_id": "default",
            "stage": "Save",
            "status": "After",
            "phase": "RESPOND",
            "event_sequence_id": "9",
            "occurred_at": "2026-01-01T00:00:00",
            "execution_mode": "RECOVERY",
            "determinism_lock": {"lock_hash": "lk", "enforced": "true"},
            "execution_state": {
                "state": {"safe_result": "ok"},
                "metadata": {"a": 1},
                "phase_history": [{"from": "plan", "to": "respond"}],
                "stage_traces": [{"stage": "save", "duration_ms": 1.0}],
                "event_sequence": 9,
            },
            "continuity": {
                "execution_fingerprint": "fp-1",
                "strict_sequence_hash": "seq-1",
                "event_count": "7",
            },
        },
    )

    assert checkpoint["schema_version"] == CHECKPOINT_RECORD_SCHEMA_VERSION
    assert checkpoint["trace_id"] == "trace-cp"
    assert checkpoint["stage"] == "save"
    assert checkpoint["status"] == "after"
    assert checkpoint["event_sequence_id"] == 9
    assert checkpoint["execution_mode"] == "recovery"
    assert checkpoint["determinism_lock"] == {"lock_hash": "lk", "enforced": True}
    assert checkpoint["execution_state"]["state"] == {"safe_result": "ok"}
    assert checkpoint["execution_state"]["metadata"] == {"a": 1}
    assert checkpoint["continuity"]["execution_fingerprint"] == "fp-1"
    assert checkpoint["continuity"]["event_count"] == 7


def test_checkpoint_record_strict_validation_rejects_missing_occurred_at() -> None:
    with pytest.raises(PersistenceSchemaError, match="occurred_at missing"):
        assert_valid_checkpoint_record(
            {
                "trace_id": "trace-cp",
                "stage": "save",
                "status": "after",
            },
        )
