from __future__ import annotations

import pytest

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.session_store import SessionStore
from dadbot.core.snapshot_engine import SnapshotConsistencyError, SnapshotEngine


class TailLedger:
    def __init__(self, events):
        self._events = events

    def read(self):
        return list(self._events)


def seal(snapshot):
    snapshot["snapshot_hash"] = SnapshotEngine._hash_snapshot(snapshot)
    return snapshot


def test_snapshot_tail_replay_preserves_checkpoint_state_and_applies_tail():
    snapshot = seal({
        "head_sequence": 10,
        "session_snapshot": {
            "sessions": {
                "existing": {"old_field": "preserved", "changed": "old"},
                "deleted": {"value": 1},
            },
        },
    })
    ledger = TailLedger([
        {
            "type": "SESSION_STATE_UPDATED",
            "session_id": "existing",
            "sequence": 11,
            "payload": {"state": {"changed": "new"}},
        },
        {"type": "SESSION_STATE_DELETED", "session_id": "deleted", "sequence": 12, "payload": {}},
        {"type": "SESSION_STATE_UPDATED", "session_id": "added", "sequence": 13, "payload": {"state": {"x": 1}}},
    ])
    store = SessionStore()

    report = SnapshotEngine().replay_tail(snapshot, ledger=ledger, session_store=store)

    assert store.get("existing") == {"old_field": "preserved", "changed": "new"}
    assert store.get("deleted") is None
    assert store.get("added") == {"x": 1}
    assert report["tail_events_applied"] == 3
    assert report["final_sequence"] == 13


def test_snapshot_tail_replay_without_tail_restores_checkpoint():
    snapshot = seal({"head_sequence": 10, "session_snapshot": {"sessions": {"existing": {"value": 1}}}})
    store = SessionStore()
    report = SnapshotEngine().replay_tail(snapshot, ledger=TailLedger([]), session_store=store)

    assert store.get("existing") == {"value": 1}
    assert report["tail_events_applied"] == 0
    assert report["final_sequence"] == 10


@pytest.mark.parametrize("snapshot", [{}, seal({"session_snapshot": {"sessions": {"bad": []}}})])
def test_invalid_snapshot_is_rejected_without_changing_projection(snapshot):
    store = SessionStore()
    store.set("existing", {"value": 1})
    with pytest.raises(SnapshotConsistencyError):
        SnapshotEngine().replay_tail(snapshot, ledger=TailLedger([]), session_store=store)
    assert store.get("existing") == {"value": 1}


def test_tampered_snapshot_is_rejected_before_restore():
    snapshot = seal({"session_snapshot": {"sessions": {"existing": {"value": 1}}}})
    snapshot["session_snapshot"]["sessions"]["existing"]["value"] = 2
    store = SessionStore()
    store.set("existing", {"value": 3})
    with pytest.raises(SnapshotConsistencyError, match="hash"):
        SnapshotEngine().restore_from_snapshot(snapshot, session_store=store)
    assert store.get("existing") == {"value": 3}


def test_snapshot_verification_detects_projection_drift_even_when_snapshot_is_rehashed():
    ledger = ExecutionLedger()
    ledger.write({
        "type": "SESSION_STATE_UPDATED",
        "session_id": "s1",
        "trace_id": "test",
        "timestamp": "2026-01-01",
        "kernel_step_id": "test",
        "payload": {"state": {"value": 1}},
    })
    store = SessionStore()
    store.rebuild_from_ledger(ledger.read())
    engine = SnapshotEngine()
    snapshot = engine.take_snapshot(ledger=ledger, session_store=store)
    snapshot["session_snapshot"]["sessions"]["s1"]["value"] = 2
    seal(snapshot)

    report = engine.verify_snapshot(snapshot, ledger=ledger)

    assert report["snapshot_hash_ok"]
    assert not report["session_projection_ok"]
    assert not report["ok"]


def test_snapshot_verification_rejects_missing_prefix():
    snapshot = seal({
        "head_sequence": 10,
        "session_snapshot": {"sessions": {}},
        "reduced_state": {"sessions": {}, "jobs": {}, "last_sequence": 0},
    })
    report = SnapshotEngine().verify_snapshot(snapshot, ledger=TailLedger([]))
    assert not report["prefix_present"]
    assert not report["ok"]


def test_snapshot_plus_tail_matches_full_session_replay():
    ledger = ExecutionLedger()

    def write(event_type, session_id, payload):
        ledger.write({
            "type": event_type,
            "session_id": session_id,
            "trace_id": "test",
            "timestamp": "2026-01-01",
            "kernel_step_id": "test",
            "payload": payload,
        })

    write("SESSION_STATE_UPDATED", "a", {"state": {"kept": 1, "changed": 0}})
    write("SESSION_STATE_UPDATED", "b", {"state": {"value": 2}})
    checkpoint_store = SessionStore()
    checkpoint_store.rebuild_from_ledger(ledger.read())
    engine = SnapshotEngine()
    snapshot = engine.take_snapshot(ledger=ledger, session_store=checkpoint_store)
    write("SESSION_STATE_UPDATED", "a", {"state": {"changed": 3}})
    write("SESSION_STATE_DELETED", "b", {})
    write("SESSION_STATE_UPDATED", "c", {"state": {"value": 4}})

    expected = SessionStore()
    expected.rebuild_from_ledger(ledger.read())
    recovered = SessionStore()
    engine.replay_tail(snapshot, ledger=ledger, session_store=recovered)

    assert recovered.snapshot()["sessions"] == expected.snapshot()["sessions"]
