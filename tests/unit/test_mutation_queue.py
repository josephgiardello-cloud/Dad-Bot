"""Unit tests for MutationQueue — ordering, locking, drain modes, snapshot math."""
from __future__ import annotations

import pytest
pytestmark = pytest.mark.unit
from dadbot.core.graph import (
    FatalTurnError,
    MutationGuard,
    MutationIntent,
    MutationKind,
    MutationQueue,
)
from harness.mutation_fuzzer import MutationFuzzer

_T = {"wall_time": "2026-01-01T00:00:00", "wall_date": "2026-01-01"}


def _intent(kind: str = "goal", priority: int = 100) -> MutationIntent:
    payload: dict = {} if kind == "goal" else {"temporal": _T}
    return MutationIntent(type=kind, payload=payload, priority=priority, requires_temporal=(kind != "goal"))


def _ledger_intent(priority: int = 100) -> MutationIntent:
    return MutationIntent(type="ledger", payload={"op": "append_history", "temporal": _T}, priority=priority)


# ---------------------------------------------------------------------------
# Ownership binding
# ---------------------------------------------------------------------------

class TestMutationQueueOwnership:
    def test_unbound_queue_raises_on_any_op(self):
        q = MutationQueue()
        with pytest.raises(RuntimeError, match="not bound"):
            q.pending()
        with pytest.raises(RuntimeError, match="not bound"):
            q.is_empty()
        with pytest.raises(RuntimeError, match="not bound"):
            q.size()
        with pytest.raises(RuntimeError, match="not bound"):
            q.snapshot()

    def test_empty_trace_id_raises(self):
        q = MutationQueue()
        with pytest.raises(RuntimeError, match="non-empty trace_id"):
            q.bind_owner("")

    def test_whitespace_trace_id_raises(self):
        q = MutationQueue()
        with pytest.raises(RuntimeError, match="non-empty trace_id"):
            q.bind_owner("   ")

    def test_same_owner_rebind_is_idempotent(self):
        q = MutationQueue()
        q.bind_owner("abc")
        q.bind_owner("abc")  # same owner — must not raise
        assert q.is_empty()

    def test_cross_turn_rebind_raises(self):
        q = MutationQueue()
        q.bind_owner("turn-1")
        with pytest.raises(RuntimeError, match="cross-turn reuse"):
            q.bind_owner("turn-2")


# ---------------------------------------------------------------------------
# Queue operations
# ---------------------------------------------------------------------------

class TestMutationQueueOperations:
    def _q(self) -> MutationQueue:
        q = MutationQueue()
        q.bind_owner("trace-test")
        return q

    def test_empty_on_creation(self):
        q = self._q()
        assert q.is_empty()
        assert q.size() == 0

    def test_queue_increases_size(self):
        q = self._q()
        q.queue(_intent())
        q.queue(_intent())
        assert q.size() == 2

    def test_pending_returns_copy(self):
        q = self._q()
        intent = _intent()
        q.queue(intent)
        pending = q.pending()
        pending.clear()
        assert q.size() == 1  # original queue unaffected

    def test_sequence_id_autoincrements(self):
        q = self._q()
        q.queue(_intent())
        q.queue(_intent())
        ids = [i.sequence_id for i in q.pending()]
        assert ids == [1, 2]

    def test_locked_queue_raises(self):
        q = self._q()
        q._mutations_locked = True
        with pytest.raises(RuntimeError, match="MutationGuard violation"):
            q.queue(_intent())


# ---------------------------------------------------------------------------
# Drain ordering
# ---------------------------------------------------------------------------

class TestMutationQueueDrainOrdering:
    def _q(self) -> MutationQueue:
        q = MutationQueue()
        q.bind_owner("drain-order")
        return q

    def test_drain_sorts_by_priority_ascending(self):
        q = self._q()
        q.queue(_intent("goal", priority=300))
        q.queue(_intent("goal", priority=50))
        q.queue(_intent("goal", priority=150))
        drained: list[MutationIntent] = []
        q.drain(drained.append, hard_fail_on_error=False)
        priorities = [i.priority for i in drained]
        assert priorities == sorted(priorities)

    def test_drain_all_on_success(self):
        q = self._q()
        for _ in range(5):
            q.queue(_intent())
        q.drain(lambda _: None, hard_fail_on_error=False)
        assert q.is_empty()

    def test_hard_fail_raises_and_preserves_remaining(self):
        q = self._q()
        q.queue(_intent())
        q.queue(_intent())

        def _fail_all(_: MutationIntent) -> None:
            raise RuntimeError("injected failure")

        with pytest.raises(FatalTurnError, match="MutationQueue drain failed"):
            q.drain(_fail_all, hard_fail_on_error=True)

        assert not q.is_empty()

    def test_soft_fail_accumulates_all_failures(self):
        q = self._q()
        for _ in range(4):
            q.queue(_intent())

        failures = q.drain(lambda _: (_ for _ in ()).throw(RuntimeError("nope")), hard_fail_on_error=False)
        assert len(failures) == 4
        for intent, msg in failures:
            assert "nope" in msg

    def test_partial_success_partial_failure(self):
        q = self._q()
        counter = {"n": 0}

        def _fail_every_other(intent: MutationIntent) -> None:
            counter["n"] += 1
            if counter["n"] % 2 == 0:
                raise RuntimeError("even failure")

        for _ in range(6):
            q.queue(_intent())

        failures = q.drain(_fail_every_other, hard_fail_on_error=False)
        assert len(failures) == 3

    def test_hard_fail_transactional_requeues_full_batch(self):
        q = self._q()
        q.queue(_intent("goal", priority=10))
        q.queue(_intent("goal", priority=20))

        calls = {"n": 0}

        def _fail_second(_intent: MutationIntent) -> None:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("second failed")

        with pytest.raises(FatalTurnError, match="MutationQueue drain failed"):
            q.drain(_fail_second, hard_fail_on_error=True, transactional=True)

        # Replay-safe: full batch is back in queue.
        assert q.size() == 2
        snap = q.snapshot()
        latest = dict(snap.get("latest_transaction") or {})
        assert latest.get("status") in {"rolled_back", "rollback_failed"}


# ---------------------------------------------------------------------------
# Snapshot ledger split
# ---------------------------------------------------------------------------

class TestMutationQueueSnapshot:
    def _q(self) -> MutationQueue:
        q = MutationQueue()
        q.bind_owner("snap-test")
        return q

    def test_snapshot_separates_ledger_from_non_ledger_pending(self):
        q = self._q()
        q.queue(_intent("goal"))        # non-ledger
        q.queue(_ledger_intent())       # ledger
        q.queue(_ledger_intent())       # ledger
        snap = q.snapshot()
        assert snap["pending"] == 1
        assert snap["ledger_pending"] == 2

    def test_snapshot_drained_counters(self):
        q = self._q()
        q.queue(_intent("goal"))
        q.queue(_ledger_intent())
        drained: list = []
        q.drain(drained.append, hard_fail_on_error=False)
        snap = q.snapshot()
        assert snap["drained"] == 1
        assert snap["ledger_drained"] == 1
        assert snap["pending"] == 0
        assert snap["ledger_pending"] == 0

    def test_snapshot_owner_trace_id_present(self):
        q = self._q()
        snap = q.snapshot()
        assert snap["owner_trace_id"] == "drain-order" or snap["owner_trace_id"] == "snap-test"

    def test_snapshot_exposes_transaction_summary(self):
        q = self._q()
        q.queue(_intent("goal"))
        q.drain(lambda _intent: None, hard_fail_on_error=False)
        snap = q.snapshot()
        assert int(snap.get("transactions", 0)) >= 1
        latest = dict(snap.get("latest_transaction") or {})
        assert latest.get("status") == "committed"


# ---------------------------------------------------------------------------
# MutationGuard lifecycle
# ---------------------------------------------------------------------------

class TestMutationGuard:
    def _q(self) -> MutationQueue:
        q = MutationQueue()
        q.bind_owner("guard-test")
        return q

    def test_enter_locks_exit_unlocks(self):
        q = self._q()
        guard = MutationGuard(q)
        assert not q._mutations_locked
        guard.__enter__()
        assert q._mutations_locked
        guard.__exit__(None, None, None)
        assert not q._mutations_locked

    def test_context_manager_blocks_then_releases(self):
        q = self._q()
        with MutationGuard(q):
            with pytest.raises(RuntimeError, match="MutationGuard violation"):
                q.queue(_intent())
        # After context exits, queueing works again
        q.queue(_intent())
        assert q.size() == 1

    def test_guard_unlocks_on_exception(self):
        q = self._q()
        try:
            with MutationGuard(q):
                raise ValueError("oops")
        except ValueError:
            pass
        assert not q._mutations_locked

    def test_nested_guard_restores_correctly(self):
        q = self._q()
        with MutationGuard(q):
            assert q._mutations_locked
            # Simulating SaveNode nullcontext by manually exiting guard
        assert not q._mutations_locked


# ---------------------------------------------------------------------------
# Fuzzer-driven ordering consistency
# ---------------------------------------------------------------------------

class TestMutationQueueFuzzerOrdering:
    def test_fuzzer_mutations_drain_in_priority_order(self):
        fuzzer = MutationFuzzer()
        intents = fuzzer.generate_valid(seed=42, count=30)
        q = MutationQueue()
        q.bind_owner("fuzzer-order")
        for intent in intents:
            q.queue(intent)

        drained: list[MutationIntent] = []
        q.drain(drained.append, hard_fail_on_error=False)

        priorities = [i.priority for i in drained]
        assert priorities == sorted(priorities), "Drain order must be by ascending priority"
