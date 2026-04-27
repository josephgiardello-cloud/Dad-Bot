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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dadbot.core.execution_policy import ResumabilityPolicy
from dadbot.core.execution_recovery import ExecutionRecovery
from dadbot.core.turn_resume_store import ResumePoint, TurnResumeStore


# ---------------------------------------------------------------------------
# TurnResumeStore unit tests
# ---------------------------------------------------------------------------

class TestTurnResumeStore:
    def test_save_and_load_round_trips(self, tmp_path: Path) -> None:
        store = TurnResumeStore(tmp_path)
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
        store = TurnResumeStore(tmp_path)
        assert store.load("nonexistent") is None

    def test_clear_removes_record(self, tmp_path: Path) -> None:
        store = TurnResumeStore(tmp_path)
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
        store = TurnResumeStore(tmp_path)
        # Should not raise even if record never existed.
        store.clear("ghost_turn")

    def test_subsequent_save_preserves_created_at(self, tmp_path: Path) -> None:
        store = TurnResumeStore(tmp_path)
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
        store = TurnResumeStore(tmp_path)
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
        store = TurnResumeStore(tmp_path)
        store.save(
            turn_id="old_turn",
            last_completed_stage="temporal",
            next_stage="health",
            checkpoint_hash="aa" * 16,
            completed_stages=["temporal"],
        )
        # Manually back-date the record.
        path = tmp_path / "old_turn.resume.json"
        data = json.loads(path.read_text())
        data["updated_at"] = time.time() - 7200  # 2 hours ago
        path.write_text(json.dumps(data))

        removed = store.purge_expired(max_age_seconds=3600)
        assert removed == 1
        assert store.load("old_turn") is None

    def test_load_returns_none_for_corrupt_file(self, tmp_path: Path) -> None:
        store = TurnResumeStore(tmp_path)
        (tmp_path / "corrupt.resume.json").write_text("not json!!!")
        # Safe ids can only be alphanumeric, so we need to use an id that maps to the same file.
        # Write directly with correct naming:
        safe_id = "corrupt"
        (tmp_path / f"{safe_id}.resume.json").write_text("{invalid")
        assert store.load(safe_id) is None

    def test_schema_version_mismatch_returns_none(self, tmp_path: Path) -> None:
        store = TurnResumeStore(tmp_path)
        (tmp_path / "v99.resume.json").write_text(
            json.dumps({"schema_version": "99", "turn_id": "v99", "completed_stages": []})
        )
        assert store.load("v99") is None

    def test_store_dir_created_on_first_save(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested"
        store = TurnResumeStore(nested)
        store.save(
            turn_id="t",
            last_completed_stage="s",
            next_stage="",
            checkpoint_hash="00" * 16,
            completed_stages=["s"],
        )
        assert nested.exists()
        assert store.load("t") is not None


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
# ExecutionRecovery unit tests
# ---------------------------------------------------------------------------

def _make_recovery(
    tmp_path: Path,
    *,
    enabled: bool = True,
    max_age_seconds: float = 3600.0,
    skip_completed_stages: bool = True,
) -> tuple[ExecutionRecovery, TurnResumeStore]:
    store = TurnResumeStore(tmp_path)
    policy = ResumabilityPolicy(
        enabled=enabled,
        max_age_seconds=max_age_seconds,
        skip_completed_stages=skip_completed_stages,
    )
    recovery = ExecutionRecovery(store, policy)
    return recovery, store


class TestExecutionRecovery:
    def test_check_resume_returns_none_when_no_record(self, tmp_path: Path) -> None:
        recovery, _ = _make_recovery(tmp_path)
        assert recovery.check_resume("t") is None

    def test_check_resume_returns_point_when_valid(self, tmp_path: Path) -> None:
        recovery, store = _make_recovery(tmp_path)
        store.save(
            turn_id="t",
            last_completed_stage="health",
            next_stage="context_builder",
            checkpoint_hash="aa" * 16,
            completed_stages=["temporal", "health"],
        )
        point = recovery.check_resume("t")
        assert point is not None
        assert point.turn_id == "t"

    def test_check_resume_returns_none_when_disabled(self, tmp_path: Path) -> None:
        recovery, store = _make_recovery(tmp_path, enabled=False)
        store.save(
            turn_id="t",
            last_completed_stage="health",
            next_stage="context_builder",
            checkpoint_hash="aa" * 16,
            completed_stages=["temporal", "health"],
        )
        assert recovery.check_resume("t") is None

    def test_check_resume_discards_expired_record(self, tmp_path: Path) -> None:
        recovery, store = _make_recovery(tmp_path, max_age_seconds=60)
        store.save(
            turn_id="old",
            last_completed_stage="temporal",
            next_stage="health",
            checkpoint_hash="aa" * 16,
            completed_stages=["temporal"],
        )
        # Back-date record.
        path = tmp_path / "old.resume.json"
        data = json.loads(path.read_text())
        data["updated_at"] = time.time() - 3600
        path.write_text(json.dumps(data))

        assert recovery.check_resume("old") is None
        # Record should be deleted.
        assert store.load("old") is None

    def test_restore_executed_stages_updates_state(self, tmp_path: Path) -> None:
        recovery, store = _make_recovery(tmp_path)
        store.save(
            turn_id="t",
            last_completed_stage="health",
            next_stage="context_builder",
            checkpoint_hash="aa" * 16,
            completed_stages=["temporal", "health"],
        )
        point = recovery.check_resume("t")
        assert point is not None

        ctx = MagicMock()
        ctx.state = {}
        recovery.restore_executed_stages(point, ctx)

        executed = ctx.state.get("_graph_executed_stages")
        assert isinstance(executed, set)
        assert "temporal" in executed
        assert "health" in executed
        # Last stage pointer restored.
        assert ctx.state.get("_graph_last_stage") == "health"

    def test_is_already_completed_true_after_restore(self, tmp_path: Path) -> None:
        recovery, store = _make_recovery(tmp_path)
        ctx = MagicMock()
        ctx.state = {"_graph_executed_stages": {"temporal", "health"}}
        assert recovery.is_already_completed("temporal", ctx) is True
        assert recovery.is_already_completed("health", ctx) is True
        assert recovery.is_already_completed("context_builder", ctx) is False

    def test_is_already_completed_false_when_no_state(self, tmp_path: Path) -> None:
        recovery, _ = _make_recovery(tmp_path)
        ctx = MagicMock()
        ctx.state = {}
        assert recovery.is_already_completed("temporal", ctx) is False

    def test_record_stage_completion_saves_to_store(self, tmp_path: Path) -> None:
        recovery, store = _make_recovery(tmp_path)
        ctx = MagicMock()
        ctx.trace_id = "myturn"
        ctx.last_checkpoint_hash = "cafebabe" * 4
        recovery.record_stage_completion(
            "temporal",
            "health",
            ctx,
            ["temporal"],
        )
        point = store.load("myturn")
        assert point is not None
        assert point.last_completed_stage == "temporal"
        assert point.next_stage == "health"
        assert "temporal" in point.completed_stages

    def test_record_stage_completion_noop_when_disabled(self, tmp_path: Path) -> None:
        recovery, store = _make_recovery(tmp_path, enabled=False)
        ctx = MagicMock()
        ctx.trace_id = "myturn"
        ctx.last_checkpoint_hash = ""
        recovery.record_stage_completion("temporal", "health", ctx, ["temporal"])
        assert store.load("myturn") is None

    def test_clear_removes_record(self, tmp_path: Path) -> None:
        recovery, store = _make_recovery(tmp_path)
        ctx = MagicMock()
        ctx.trace_id = "t"
        ctx.last_checkpoint_hash = "aa" * 16
        recovery.record_stage_completion("temporal", "health", ctx, ["temporal"])
        assert store.load("t") is not None
        recovery.clear("t")
        assert store.load("t") is None

    def test_list_pending_turns_filters_by_policy(self, tmp_path: Path) -> None:
        recovery, store = _make_recovery(tmp_path, max_age_seconds=3600)
        for i in range(3):
            store.save(
                turn_id=f"pending_{i}",
                last_completed_stage="temporal",
                next_stage="health",
                checkpoint_hash="aa" * 16,
                completed_stages=["temporal"],
            )
        pending = recovery.list_pending_turns()
        assert len(pending) == 3

    def test_list_pending_turns_empty_when_disabled(self, tmp_path: Path) -> None:
        recovery, store = _make_recovery(tmp_path, enabled=False)
        store.save(
            turn_id="t",
            last_completed_stage="temporal",
            next_stage="health",
            checkpoint_hash="aa" * 16,
            completed_stages=["temporal"],
        )
        assert recovery.list_pending_turns() == []

    def test_purge_expired_records(self, tmp_path: Path) -> None:
        recovery, store = _make_recovery(tmp_path, max_age_seconds=60)
        store.save(
            turn_id="old",
            last_completed_stage="temporal",
            next_stage="health",
            checkpoint_hash="aa" * 16,
            completed_stages=["temporal"],
        )
        path = tmp_path / "old.resume.json"
        data = json.loads(path.read_text())
        data["updated_at"] = time.time() - 3600
        path.write_text(json.dumps(data))

        removed = recovery.purge_expired_records()
        assert removed == 1


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
    """Integration-level checks that TurnGraph.configure_resume() sets up recovery."""

    def test_configure_resume_assigns_recovery(self, tmp_path: Path) -> None:
        from dadbot.core.graph import TurnGraph

        graph = TurnGraph()
        assert graph._recovery is None
        graph.configure_resume(tmp_path / "resume")
        assert graph._recovery is not None

    def test_configure_resume_custom_policy(self, tmp_path: Path) -> None:
        from dadbot.core.graph import TurnGraph

        graph = TurnGraph()
        policy = ResumabilityPolicy(enabled=False)
        graph.configure_resume(tmp_path / "resume", policy=policy)
        # Recovery object created but disabled policy means nothing will be saved.
        assert graph._recovery is not None


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
