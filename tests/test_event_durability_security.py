from __future__ import annotations

import sqlite3

from dadbot_system.event_durability import SQLiteEventDurabilityStore


def test_event_durability_encrypts_payloads_and_checkpoints(tmp_path):
    db_path = tmp_path / "durable-events.sqlite3"
    store = SQLiteEventDurabilityStore(db_path, encryption_key="durability-test-key")

    run_id = store.start_run(session_id="session-1", tenant_id="family-a")
    store.append_event(
        run_id=run_id,
        event_type="assistant.reply",
        payload={"secret": "top-secret", "message": "hello"},
    )
    store.append_checkpoint(run_id=run_id, state={"reply": "classified"})

    with sqlite3.connect(db_path) as conn:
        payload_json = str(conn.execute("SELECT payload_json FROM events").fetchone()[0])
        state_json = str(conn.execute("SELECT state_json FROM checkpoints").fetchone()[0])

    assert "top-secret" not in payload_json
    assert "classified" not in state_json

    events = store.list_events(run_id)
    checkpoint = store.latest_checkpoint(run_id)

    assert events[0].payload["secret"] == "top-secret"
    assert checkpoint is not None
    assert checkpoint["state"]["reply"] == "classified"