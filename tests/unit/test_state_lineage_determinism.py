from __future__ import annotations

from dadbot.core.state_lineage import build_state_snapshot_entry, canonical_state_hash


def test_identical_input_produces_identical_snapshot_hash() -> None:
    state = {
        "count": 3,
        "nested": {"alpha": "beta", "nums": [1, 2, 3]},
    }
    state_hash = canonical_state_hash(state)

    first = build_state_snapshot_entry(
        session_id="session-1",
        trace_id="trace-1",
        version=7,
        prev_snapshot_hash="prev-abc",
        state_hash=state_hash,
        reason="turn_commit",
    )
    second = build_state_snapshot_entry(
        session_id="session-1",
        trace_id="trace-1",
        version=7,
        prev_snapshot_hash="prev-abc",
        state_hash=state_hash,
        reason="turn_commit",
    )

    assert first["snapshot_hash"] == second["snapshot_hash"]
    assert first == second
