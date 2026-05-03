"""Tests for: DurableCheckpoint, ExecutionLease, and boot_reconcile startup gate."""

from __future__ import annotations

import asyncio
import time

import pytest

from dadbot.core.durable_checkpoint import CheckpointIntegrityError, DurableCheckpoint
from dadbot.core.execution_lease import ExecutionLease, LeaseConflictError
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.recovery_manager import RecoveryManager, StartupReconciliationError
from dadbot.core.session_store import SessionStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ledger_with_complete_job() -> ExecutionLedger:
    """Return a ledger that has a complete job so session projection is non-empty."""
    from dadbot.core.control_plane import ExecutionJob
    from dadbot.core.ledger_writer import LedgerWriter

    ledger = ExecutionLedger()
    writer = LedgerWriter(ledger)
    job = ExecutionJob(session_id="s1", user_input="hi")
    writer.append_job_submitted(job)
    writer.append_session_bound("s1", job.job_id)
    writer.append_job_queued(job)
    writer.append_job_started(job)
    writer.append_job_completed(job, {"reply": "hello"})
    return ledger


# ===========================================================================
# DurableCheckpoint
# ===========================================================================


class TestDurableCheckpoint:
    def test_clean_start_passes_without_checkpoints(self):
        ledger = ExecutionLedger()
        cp = DurableCheckpoint(ledger=ledger)
        report = cp.assert_resume_at_head()
        assert report["ok"] is True
        assert report["checkpoint_count"] == 0

    def test_save_and_resume_at_matching_head(self):
        ledger = _make_ledger_with_complete_job()
        cp = DurableCheckpoint(ledger=ledger)
        saved = cp.save(label="post-boot")
        assert saved["checkpoint_hash"]
        assert saved["replay_hash"] == ledger.replay_hash()
        assert saved["chain_hash"] == ledger.chain_hash()

        report = cp.assert_resume_at_head()
        assert report["ok"] is True

    def test_resume_fails_when_replay_hash_diverges(self):
        ledger = _make_ledger_with_complete_job()
        cp = DurableCheckpoint(ledger=ledger)
        cp.save(label="pre-corruption")

        # Manually corrupt the saved checkpoint hash to simulate divergence.
        with cp._lock:
            cp._checkpoints[-1]["replay_hash"] = "bad-hash-00000"

        with pytest.raises(CheckpointIntegrityError, match="mismatch"):
            cp.assert_resume_at_head()

    def test_resume_fails_when_chain_hash_diverges(self):
        ledger = _make_ledger_with_complete_job()
        cp = DurableCheckpoint(ledger=ledger)
        cp.save(label="pre-chain-corruption")

        with cp._lock:
            cp._checkpoints[-1]["chain_hash"] = "bad-chain-00000"

        with pytest.raises(CheckpointIntegrityError, match="chain hash mismatch"):
            cp.assert_resume_at_head()

    def test_resume_fails_when_ledger_truncated(self):
        ledger = _make_ledger_with_complete_job()
        cp = DurableCheckpoint(ledger=ledger)
        cp.save(label="full")

        # Manually inflate the recorded event count to simulate truncation.
        with cp._lock:
            cp._checkpoints[-1]["ledger_event_count"] = 999_999

        with pytest.raises(CheckpointIntegrityError, match="truncation"):
            cp.assert_resume_at_head()

    def test_chain_integrity_passes_for_valid_chain(self):
        ledger = _make_ledger_with_complete_job()
        cp = DurableCheckpoint(ledger=ledger)
        cp.save(label="first")
        cp.save(label="second")
        report = cp.verify_chain_integrity()
        assert report["ok"] is True
        assert report["chain_length"] == 2

    def test_chain_integrity_detects_tampering(self):
        ledger = _make_ledger_with_complete_job()
        cp = DurableCheckpoint(ledger=ledger)
        cp.save(label="first")
        cp.save(label="second")

        # Tamper with the first checkpoint.
        with cp._lock:
            cp._checkpoints[0]["label"] = "tampered"

        report = cp.verify_chain_integrity()
        assert report["ok"] is False
        assert report["violations"]

    def test_history_returns_all_checkpoints(self):
        ledger = _make_ledger_with_complete_job()
        cp = DurableCheckpoint(ledger=ledger)
        cp.save(label="a")
        cp.save(label="b")
        history = cp.history()
        assert len(history) == 2
        assert history[0]["label"] == "a"
        assert history[1]["label"] == "b"


# ===========================================================================
# ExecutionLease
# ===========================================================================


class TestExecutionLease:
    def test_acquire_returns_lease_with_expected_fields(self):
        lease_mgr = ExecutionLease()
        lease = lease_mgr.acquire(session_id="sess-1", owner_id="worker-A")
        assert lease["session_id"] == "sess-1"
        assert lease["owner_id"] == "worker-A"
        assert lease["ttl_seconds"] == ExecutionLease.DEFAULT_TTL_SECONDS
        assert lease["lease_id"]

    def test_same_owner_can_reacquire_own_lease(self):
        lease_mgr = ExecutionLease()
        lease_mgr.acquire(session_id="sess-1", owner_id="worker-A")
        # Should not raise.
        renewed = lease_mgr.acquire(session_id="sess-1", owner_id="worker-A")
        assert renewed["owner_id"] == "worker-A"

    def test_different_owner_raises_lease_conflict(self):
        lease_mgr = ExecutionLease()
        lease_mgr.acquire(session_id="sess-1", owner_id="worker-A")
        with pytest.raises(LeaseConflictError):
            lease_mgr.acquire(session_id="sess-1", owner_id="worker-B")

    def test_release_allows_new_owner_to_acquire(self):
        lease_mgr = ExecutionLease()
        lease_mgr.acquire(session_id="sess-1", owner_id="worker-A")
        released = lease_mgr.release(session_id="sess-1", owner_id="worker-A")
        assert released is True
        # Now worker-B can acquire.
        lease = lease_mgr.acquire(session_id="sess-1", owner_id="worker-B")
        assert lease["owner_id"] == "worker-B"

    def test_release_by_non_owner_returns_false(self):
        lease_mgr = ExecutionLease()
        lease_mgr.acquire(session_id="sess-1", owner_id="worker-A")
        released = lease_mgr.release(session_id="sess-1", owner_id="worker-B")
        assert released is False
        assert lease_mgr.owner_of(session_id="sess-1") == "worker-A"

    def test_expired_lease_allows_new_owner(self):
        lease_mgr = ExecutionLease(default_ttl_seconds=0.05)
        lease_mgr.acquire(session_id="sess-1", owner_id="worker-A", ttl_seconds=0.01)
        time.sleep(0.05)
        # Expired — worker-B can now acquire without conflict.
        lease = lease_mgr.acquire(session_id="sess-1", owner_id="worker-B")
        assert lease["owner_id"] == "worker-B"

    def test_require_session_lease_passes_when_no_lease_held(self):
        lease_mgr = ExecutionLease()
        # No lease — should not raise.
        lease_mgr.require_session_lease(session_id="sess-x", owner_id="worker-A")

    def test_require_session_lease_passes_for_correct_owner(self):
        lease_mgr = ExecutionLease()
        lease_mgr.acquire(session_id="sess-1", owner_id="worker-A")
        lease_mgr.require_session_lease(session_id="sess-1", owner_id="worker-A")

    def test_require_session_lease_raises_for_wrong_owner(self):
        lease_mgr = ExecutionLease()
        lease_mgr.acquire(session_id="sess-1", owner_id="worker-A")
        with pytest.raises(LeaseConflictError):
            lease_mgr.require_session_lease(session_id="sess-1", owner_id="worker-B")

    def test_evict_expired_cleans_stale_leases(self):
        lease_mgr = ExecutionLease()
        lease_mgr.acquire(session_id="a", owner_id="w", ttl_seconds=0.01)
        lease_mgr.acquire(session_id="b", owner_id="w", ttl_seconds=60)
        time.sleep(0.05)
        evicted = lease_mgr.evict_expired()
        assert "a" in evicted
        assert "b" not in evicted
        assert lease_mgr.is_held(session_id="b")
        assert not lease_mgr.is_held(session_id="a")

    def test_snapshot_shows_active_leases_only(self):
        lease_mgr = ExecutionLease()
        lease_mgr.acquire(session_id="active", owner_id="w", ttl_seconds=60)
        snap = lease_mgr.snapshot()
        assert snap["active_lease_count"] == 1
        assert snap["leases"][0]["session_id"] == "active"

    def test_empty_session_id_raises(self):
        lease_mgr = ExecutionLease()
        with pytest.raises(ValueError):
            lease_mgr.acquire(session_id="", owner_id="w")


# ===========================================================================
# boot_reconcile / StartupReconciliation
# ===========================================================================


class TestBootReconcile:
    def test_clean_boot_with_empty_ledger_passes(self):
        ledger = ExecutionLedger()
        store = SessionStore(ledger=ledger)
        rm = RecoveryManager(ledger=ledger)
        result = rm.boot_reconcile(session_store=store)
        assert result["ok"] is True
        assert result["boot_complete"] is True
        assert rm.boot_complete is True

    def test_clean_boot_with_complete_ledger_passes(self):
        ledger = _make_ledger_with_complete_job()
        store = SessionStore(ledger=ledger)
        rm = RecoveryManager(ledger=ledger)
        result = rm.boot_reconcile(session_store=store)
        assert result["ok"] is True
        assert int(result.get("ledger_events") or 0) > 0

    def test_boot_reconcile_saves_checkpoint(self):
        ledger = _make_ledger_with_complete_job()
        store = SessionStore(ledger=ledger)
        cp = DurableCheckpoint(ledger=ledger)
        rm = RecoveryManager(ledger=ledger)
        rm.boot_reconcile(session_store=store, checkpoint=cp)
        assert cp.latest() is not None
        assert cp.latest()["label"] == "boot_reconcile"

    def test_second_call_is_idempotent(self):
        ledger = ExecutionLedger()
        store = SessionStore(ledger=ledger)
        cp = DurableCheckpoint(ledger=ledger)
        rm = RecoveryManager(ledger=ledger)
        rm.boot_reconcile(session_store=store, checkpoint=cp)
        # Second call: already reconciled, should return fast without saving another checkpoint.
        result = rm.boot_reconcile(session_store=store, checkpoint=cp)
        assert result["ok"] is True
        # Checkpoint was only saved once (first call).
        assert len(cp.history()) == 1

    def test_corrupted_checkpoint_triggers_startup_reconciliation_error(self):
        ledger = _make_ledger_with_complete_job()
        store = SessionStore(ledger=ledger)
        cp = DurableCheckpoint(ledger=ledger)
        # Save a checkpoint with a wrong replay hash to simulate corruption.
        cp.save(label="corrupted")
        with cp._lock:
            cp._checkpoints[-1]["replay_hash"] = "corrupted-hash"

        rm = RecoveryManager(ledger=ledger)
        with pytest.raises(StartupReconciliationError):
            rm.boot_reconcile(session_store=store, checkpoint=cp)

    def test_boot_reconcile_not_complete_before_called(self):
        ledger = ExecutionLedger()
        rm = RecoveryManager(ledger=ledger)
        assert rm.boot_complete is False


# ===========================================================================
# Lease integration with Scheduler / ExecutionControlPlane
# ===========================================================================


class TestLeaseSchedulerIntegration:
    def test_single_worker_acquires_and_releases_lease_per_job(self):
        """Worker acquires lease before execution and releases after."""
        from dadbot.core.control_plane import (
            ExecutionControlPlane,
            SessionRegistry,
        )

        execution_log: list[str] = []

        async def mock_executor(session, job):
            execution_log.append(f"executing:{job.session_id}")
            return {"reply": "ok"}

        async def _run():
            registry = SessionRegistry()
            lease_mgr = ExecutionLease()
            plane = ExecutionControlPlane(
                registry=registry,
                kernel_executor=mock_executor,
                execution_lease=lease_mgr,
                worker_id="worker-test",
            )
            result = await plane.submit_turn(session_id="s1", user_input="hello")
            return result, lease_mgr

        result, lease_mgr = asyncio.run(_run())
        assert result["reply"] == "ok"
        assert "executing:s1" in execution_log
        # Lease released after job completes.
        assert not lease_mgr.is_held(session_id="s1")

    def test_lease_conflict_requeues_and_eventually_executes(self):
        """If lease is held by another worker, job is re-queued and retried."""
        from dadbot.core.control_plane import (
            ExecutionControlPlane,
            SessionRegistry,
        )

        call_count = {"n": 0}

        async def mock_executor(session, job):
            call_count["n"] += 1
            return {"reply": "done"}

        async def _run():
            registry = SessionRegistry()
            lease_mgr = ExecutionLease()
            # Pre-hold the lease as a different worker with very short TTL.
            lease_mgr.acquire(session_id="s1", owner_id="other-worker", ttl_seconds=0.05)
            plane = ExecutionControlPlane(
                registry=registry,
                kernel_executor=mock_executor,
                execution_lease=lease_mgr,
                worker_id="this-worker",
            )
            # Job must eventually execute once the competing lease expires.
            return await plane.submit_turn(session_id="s1", user_input="hello", timeout_seconds=5.0)

        result = asyncio.run(_run())
        assert result["reply"] == "done"
        assert call_count["n"] == 1
