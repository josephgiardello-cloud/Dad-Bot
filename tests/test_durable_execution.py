"""Tests for Durable Execution (crash-safe resume) feature.

Covers:
 1. ResumePoint saved after each successful stage.
 2. Recovery: completed stages are skipped on re-entry.
 3. Idempotent node guarantee: re-running a stage does not corrupt state.
 4. Resume point cleared on successful pipeline completion.
 5. TurnResumeStore: atomic write, load, clear, list_pending, purge_expired.
 6. ExecutionRecovery: startup discovery, purge helpers.
 7. ResumabilityPolicy: disabled policy suppresses all I/O.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.execution_policy import ResumabilityPolicy
from dadbot.core.turn_resume_store import ResumePoint, TurnResumeStore


def _new_resume_store() -> TurnResumeStore:
    return TurnResumeStore(ledger=ExecutionLedger())

# ---------------------------------------------------------------------------
# TurnResumeStore unit tests
# ---------------------------------------------------------------------------


class TestTurnResumeStore:
    def test_save_and_load_round_trips(self, tmp_path: Path) -> None:
        store = _new_resume_store()
        store.save(
            turn_id="abc123",
            last_completed_stage="health",
            next_stage="context_builder",
            checkpoint_hash="deadbeef" * 4,
            completed_stages=["temporal", "health"],
        )
        point = store.load("abc123")
        assert point is not None
        assert point.turn_id == "abc123"
        assert point.last_completed_stage == "health"
        assert point.next_stage == "context_builder"
        assert point.completed_stages == ("temporal", "health")

    def test_load_returns_none_for_missing_turn(self, tmp_path: Path) -> None:
        store = _new_resume_store()
        assert store.load("nonexistent") is None

    def test_clear_removes_record(self, tmp_path: Path) -> None:
        store = _new_resume_store()
        store.save(
            turn_id="del_me",
            last_completed_stage="save",
            next_stage="",
            checkpoint_hash="aa" * 16,
            completed_stages=["temporal", "save"],
        )
        store.clear("del_me")
        assert store.load("del_me") is None

    def test_clear_is_idempotent_for_missing_id(self, tmp_path: Path) -> None:
        store = _new_resume_store()
        # Should not raise even if record never existed.
        store.clear("ghost_turn")

    def test_subsequent_save_preserves_created_at(self, tmp_path: Path) -> None:
        store = _new_resume_store()
        store.save(
            turn_id="t1",
            last_completed_stage="temporal",
            next_stage="health",
            checkpoint_hash="00" * 16,
            completed_stages=["temporal"],
        )
        first = store.load("t1")
        assert first is not None
        original_created_at = first.created_at

        store.save(
            turn_id="t1",
            last_completed_stage="health",
            next_stage="context_builder",
            checkpoint_hash="11" * 16,
            completed_stages=["temporal", "health"],
        )
        second = store.load("t1")
        assert second is not None
        assert second.created_at == original_created_at
        assert second.last_completed_stage == "health"

    def test_list_pending_returns_all_records(self, tmp_path: Path) -> None:
        store = _new_resume_store()
        for i in range(3):
            store.save(
                turn_id=f"turn_{i}",
                last_completed_stage="temporal",
                next_stage="health",
                checkpoint_hash="aa" * 16,
                completed_stages=["temporal"],
            )
        pending = store.list_pending()
        assert len(pending) == 3
        ids = {p.turn_id for p in pending}
        assert ids == {"turn_0", "turn_1", "turn_2"}

    def test_purge_expired_removes_old_records(self, tmp_path: Path) -> None:
        store = _new_resume_store()
        store.save(
            turn_id="old_turn",
            last_completed_stage="temporal",
            next_stage="health",
            checkpoint_hash="aa" * 16,
            completed_stages=["temporal"],
        )
        point = store.load("old_turn")
        assert point is not None
        stale = point.to_dict()
        stale["updated_at"] = time.time() - 7200
        stale["cleared"] = False
        store._append_ledger_payload(stale)  # type: ignore[attr-defined]

        removed = store.purge_expired(max_age_seconds=3600)
        assert removed == 1
        assert store.load("old_turn") is None

    def test_load_ignores_schema_mismatch_payload(self, tmp_path: Path) -> None:
        store = _new_resume_store()
        store._append_ledger_payload(  # type: ignore[attr-defined]
            {
                "schema_version": "99",
                "turn_id": "mismatch",
                "completed_stages": [],
                "created_at": time.time(),
                "updated_at": time.time(),
            }
        )
        assert store.load("mismatch") is None

    def test_schema_version_mismatch_returns_none(self, tmp_path: Path) -> None:
        store = _new_resume_store()
        store._append_ledger_payload(  # type: ignore[attr-defined]
            {
                "schema_version": "99",
                "turn_id": "v99",
                "completed_stages": [],
                "created_at": time.time(),
                "updated_at": time.time(),
            }
        )
        assert store.load("v99") is None

    def test_init_requires_ledger(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            TurnResumeStore(ledger=None)


# ---------------------------------------------------------------------------
# ResumePoint unit tests
# ---------------------------------------------------------------------------


class TestResumePoint:
    def test_is_expired_false_when_fresh(self) -> None:
        point = ResumePoint(
            turn_id="x",
            last_completed_stage="temporal",
            next_stage="health",
            checkpoint_hash="aa" * 16,
            completed_stages=("temporal",),
            created_at=time.time(),
            updated_at=time.time(),
        )
        assert not point.is_expired(max_age_seconds=3600)

    def test_is_expired_true_when_old(self) -> None:
        old_time = time.time() - 7200
        point = ResumePoint(
            turn_id="x",
            last_completed_stage="temporal",
            next_stage="health",
            checkpoint_hash="aa" * 16,
            completed_stages=("temporal",),
            created_at=old_time,
            updated_at=old_time,
        )
        assert point.is_expired(max_age_seconds=3600)

    def test_round_trip_via_dict(self) -> None:
        point = ResumePoint(
            turn_id="abc",
            last_completed_stage="inference",
            next_stage="safety",
            checkpoint_hash="ff" * 16,
            completed_stages=("temporal", "health", "context_builder", "inference"),
            created_at=1234567890.0,
            updated_at=1234567900.0,
        )
        reconstructed = ResumePoint.from_dict(point.to_dict())
        assert reconstructed == point


# ---------------------------------------------------------------------------
# ResumabilityPolicy unit tests
# ---------------------------------------------------------------------------


class TestResumabilityPolicy:
    def test_default_values(self) -> None:
        policy = ResumabilityPolicy()
        assert policy.enabled is True
        assert policy.max_age_seconds == 3600.0
        assert policy.skip_completed_stages is True

    def test_custom_values(self) -> None:
        policy = ResumabilityPolicy(enabled=False, max_age_seconds=1800.0, skip_completed_stages=False)
        assert policy.enabled is False
        assert policy.max_age_seconds == 1800.0
        assert policy.skip_completed_stages is False

    def test_is_frozen(self) -> None:
        policy = ResumabilityPolicy()
        with pytest.raises((AttributeError, TypeError)):
            policy.enabled = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TurnGraph integration: configure_resume() wires recovery properly
# ---------------------------------------------------------------------------


class TestTurnGraphConfigureResume:
    """Integration-level checks for resume wiring in ledger-only mode."""

    def test_configure_resume_rejects_store_dir_mode(self, tmp_path: Path) -> None:
        from dadbot.core.graph import TurnGraph

        graph = TurnGraph()
        with pytest.raises(RuntimeError, match=r"configure_resume\(store_dir"):
            graph.configure_resume(tmp_path / "resume")

    def test_configure_resume_store_is_noop(self, tmp_path: Path) -> None:
        from dadbot.core.graph import TurnGraph

        graph = TurnGraph()
        policy = ResumabilityPolicy(enabled=False)
        # Phase 3: configure_resume_store is a no-op (recovery is ledger-only).
        graph.configure_resume_store(_new_resume_store(), policy=policy)
        # No assertion needed; just verify it doesn't crash.


# ---------------------------------------------------------------------------
# Idempotent node guarantee: re-running a stage does not corrupt state
# ---------------------------------------------------------------------------


class TestIdempotentNodeGuarantee:
    """The idempotency guard must skip already-completed stages cleanly."""

    def test_is_already_completed_returns_false_for_fresh_context(self, tmp_path: Path) -> None:
        recovery, _ = _make_recovery(tmp_path)
        ctx = MagicMock()
        ctx.state = {}
        assert recovery.is_already_completed("temporal", ctx) is False

    def test_is_already_completed_does_not_modify_state(self, tmp_path: Path) -> None:
        recovery, _ = _make_recovery(tmp_path)
        ctx = MagicMock()
        ctx.state = {"_graph_executed_stages": {"temporal"}}
        _ = recovery.is_already_completed("temporal", ctx)
        # State must not change after a pure read.
        assert ctx.state["_graph_executed_stages"] == {"temporal"}

    def test_restore_is_additive_never_removes_stages(self, tmp_path: Path) -> None:
        """Already-present stages must survive restore."""
        recovery, store = _make_recovery(tmp_path)
        store.save(
            turn_id="t",
            last_completed_stage="health",
            next_stage="context_builder",
            checkpoint_hash="aa" * 16,
            completed_stages=["temporal", "health"],
        )
        point = recovery.check_resume("t")
        ctx = MagicMock()
        ctx.state = {"_graph_executed_stages": {"preflight"}}
        recovery.restore_executed_stages(point, ctx)
        executed = ctx.state["_graph_executed_stages"]
        # Original stage must still be present.
        assert "preflight" in executed
        # Restored stages added.
        assert "temporal" in executed
        assert "health" in executed
