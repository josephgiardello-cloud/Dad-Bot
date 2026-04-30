from __future__ import annotations

from types import SimpleNamespace

from dadbot.core.merkle_anchor import append_leaf_and_anchor, verify_inclusion_proof
from dadbot.services.persistence import PersistenceService


def test_merkle_anchor_inclusion_proof_roundtrip() -> None:
    leaves: list[str] = []
    anchor_a = append_leaf_and_anchor(leaves, {"turn": 1, "text": "hello"})
    anchor_b = append_leaf_and_anchor(leaves, {"turn": 2, "text": "world"})
    assert anchor_b["leaf_count"] == 2
    assert verify_inclusion_proof(
        anchor_b["leaf_hash"],
        anchor_b["inclusion_proof"],
        anchor_b["merkle_root"],
    )
    assert anchor_a["merkle_root"] != anchor_b["merkle_root"]


def test_persistence_commit_transaction_writes_merkle_anchor_event() -> None:
    events: list[dict] = []

    class _PM:
        def persist_turn_event(self, event):
            events.append(dict(event))

    service = PersistenceService(
        _PM(), turn_service=SimpleNamespace(bot=SimpleNamespace(config=SimpleNamespace(merkle_anchor_enabled=True)))
    )

    context = SimpleNamespace(
        trace_id="trace-1",
        temporal=SimpleNamespace(wall_time="2026-04-27T00:00:00+00:00"),
        state={
            "_save_transaction": {"commit_id": "commit-1"},
            "memory_recent_buffer": [],
            "memory_rolling_summary": "",
            "memory_structured": {},
        },
        metadata={
            "control_plane": {"session_id": "session-1"},
        },
    )

    service.commit_transaction(context)
    assert "merkle_anchor" in context.state
    assert "merkle_anchor" in context.metadata
    assert any(str(item.get("event_type") or "") == "merkle_anchor_commit" for item in events)
