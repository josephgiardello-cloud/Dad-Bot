from __future__ import annotations

from dadbot.core.graph_mutation import MutationIntent, MutationQueue

import pytest
pytestmark = pytest.mark.unit


def _ledger_intent(*, op: str, session_id: str, sequence_id: int, note: str) -> MutationIntent:
    return MutationIntent(
        type="ledger",
        payload={
            "op": op,
            "session_id": session_id,
            "note": note,
            "temporal": {"wall_time": "10:00", "wall_date": "2026-05-07"},
        },
        priority=10,
        turn_index=1,
        sequence_id=sequence_id,
    )


def test_mutation_queue_uses_canonical_tiebreak_order_for_equal_ordering_keys() -> None:
    intent_a = _ledger_intent(
        op="append_history",
        session_id="s1",
        sequence_id=7,
        note="alpha",
    )
    intent_b = _ledger_intent(
        op="append_history",
        session_id="s1",
        sequence_id=7,
        note="beta",
    )

    left = MutationQueue()
    left.bind_owner("trace-left")
    left.queue(intent_b)
    left.queue(intent_a)

    right = MutationQueue()
    right.bind_owner("trace-right")
    right.queue(intent_a)
    right.queue(intent_b)

    left_order: list[str] = []
    right_order: list[str] = []

    def _left_executor(intent: MutationIntent) -> None:
        left_order.append(intent.payload_hash)

    def _right_executor(intent: MutationIntent) -> None:
        right_order.append(intent.payload_hash)

    left.drain(_left_executor)
    right.drain(_right_executor)

    assert left_order == right_order


def test_mutation_snapshot_pending_order_hash_stable_for_equal_keys() -> None:
    intent_a = _ledger_intent(
        op="append_history",
        session_id="s1",
        sequence_id=9,
        note="alpha",
    )
    intent_b = _ledger_intent(
        op="append_history",
        session_id="s1",
        sequence_id=9,
        note="beta",
    )

    queue_left = MutationQueue()
    queue_left.bind_owner("trace-left")
    queue_left.queue(intent_a)
    queue_left.queue(intent_b)

    queue_right = MutationQueue()
    queue_right.bind_owner("trace-right")
    queue_right.queue(intent_b)
    queue_right.queue(intent_a)

    left_hash = str(queue_left.snapshot().get("canonical_pending_order_hash") or "")
    right_hash = str(queue_right.snapshot().get("canonical_pending_order_hash") or "")

    assert bool(left_hash)
    assert left_hash == right_hash


def test_mutation_snapshot_semantic_multiset_hash_stable_for_equal_set() -> None:
    intent_a = _ledger_intent(
        op="append_history",
        session_id="s1",
        sequence_id=4,
        note="alpha",
    )
    intent_b = _ledger_intent(
        op="append_history",
        session_id="s1",
        sequence_id=8,
        note="beta",
    )

    queue_left = MutationQueue()
    queue_left.bind_owner("trace-left-sem")
    queue_left.queue(intent_a)
    queue_left.queue(intent_b)

    queue_right = MutationQueue()
    queue_right.bind_owner("trace-right-sem")
    queue_right.queue(intent_b)
    queue_right.queue(intent_a)

    left_hash = str(queue_left.snapshot().get("semantic_pending_multiset_hash") or "")
    right_hash = str(queue_right.snapshot().get("semantic_pending_multiset_hash") or "")

    assert bool(left_hash)
    assert left_hash == right_hash
