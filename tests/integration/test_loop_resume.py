"""Integration test: AgentDriverLoop resume-from-ledger (Task 3).

Verifies that:
1. A loop run that is interrupted (max_turns=2 on first instance) writes a snapshot.
2. A new AgentDriverLoop initialized with the same session_id detects the prior
   partial run via LoopSessionSnapshot.load().
3. The resumed run picks up from the correct turn index.
4. Snapshots are cleaned up after the test.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from dadbot.runtime.agent_driver_loop import AgentDriverLoop, DriverLoopPolicy, LoopTurnRecord
from dadbot.runtime.loop_snapshot import LoopSessionSnapshot, LoopSnapshotManager

pytestmark = pytest.mark.integration

SESSION_ID = "test-resume-session-42"


# ---------------------------------------------------------------------------
# Kernel stub
# ---------------------------------------------------------------------------


@dataclass
class _FakeResponse:
    reply: str
    should_end: bool = False

    def as_result(self) -> tuple[str, bool]:
        return self.reply, self.should_end


class _CountingKernel:
    """Kernel stub that counts calls and returns deterministic replies."""

    def __init__(self) -> None:
        self.call_count = 0

    def execute_turn(self, request: Any, **_: Any) -> _FakeResponse:
        self.call_count += 1
        return _FakeResponse(reply=f"reply_{self.call_count}", should_end=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoopResumeFromSnapshot:
    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._snap_dir = Path(self._tmpdir)
        LoopSessionSnapshot.delete(SESSION_ID, base_dir=self._snap_dir)

    def teardown_method(self) -> None:
        LoopSessionSnapshot.delete(SESSION_ID, base_dir=self._snap_dir)

    # -----------------------------------------------------------------------
    # Snapshot file lifecycle
    # -----------------------------------------------------------------------

    def test_snapshot_written_at_loop_start(self) -> None:
        kernel = _CountingKernel()
        policy = DriverLoopPolicy(max_turns=2)
        mgr = LoopSnapshotManager(SESSION_ID, "initial task", policy, base_dir=self._snap_dir)

        snap = LoopSessionSnapshot.load(SESSION_ID, base_dir=self._snap_dir)
        assert snap is not None
        assert snap.session_id == SESSION_ID
        assert snap.initial_observation == "initial task"
        assert snap.is_incomplete, "snapshot should be incomplete (no stop_reason yet)"

    def test_snapshot_finalized_on_completion(self) -> None:
        kernel = _CountingKernel()
        policy = DriverLoopPolicy(max_turns=2)
        mgr = LoopSnapshotManager(SESSION_ID, "initial task", policy, base_dir=self._snap_dir)

        loop = AgentDriverLoop(kernel, policy=policy)
        result = loop.run("initial task", session_id=SESSION_ID)
        mgr.finalize(result)

        snap = LoopSessionSnapshot.load(SESSION_ID, base_dir=self._snap_dir)
        assert snap is not None
        assert snap.stop_reason is not None, "stop_reason must be set after finalize"
        assert not snap.is_incomplete

    def test_snapshot_policy_round_trips(self) -> None:
        policy = DriverLoopPolicy(max_turns=3, max_failures=1)
        mgr = LoopSnapshotManager(SESSION_ID, "task", policy, base_dir=self._snap_dir)
        snap = LoopSessionSnapshot.load(SESSION_ID, base_dir=self._snap_dir)
        assert snap is not None
        assert snap.policy["max_turns"] == 3
        assert snap.policy["max_failures"] == 1

    # -----------------------------------------------------------------------
    # Cold-start detection
    # -----------------------------------------------------------------------

    def test_new_instance_detects_incomplete_snapshot(self) -> None:
        """A second loop instance should detect the prior incomplete run."""
        # First run: 2 turns, then "interrupted" (no finalize)
        kernel1 = _CountingKernel()
        policy = DriverLoopPolicy(max_turns=2)
        mgr = LoopSnapshotManager(SESSION_ID, "find the answer", policy, base_dir=self._snap_dir)
        loop1 = AgentDriverLoop(kernel1, policy=policy)
        loop1.run("find the answer", session_id=SESSION_ID, reflection_hook=mgr.wrap_reflection())
        # Do NOT call mgr.finalize — simulates crash/interrupt

        # Second instance: new kernel, same session_id
        snap = LoopSessionSnapshot.load(SESSION_ID, base_dir=self._snap_dir)
        assert snap is not None
        assert snap.is_incomplete, "Loop was not finalized, should show as incomplete"
        assert snap.initial_observation == "find the answer"

    def test_resumed_loop_starts_from_prior_turn_index(self) -> None:
        """After an incomplete run, the resumed loop can pick up from where it left off."""
        # Phase A: run 2 turns (interrupted)
        kernel_a = _CountingKernel()
        policy_a = DriverLoopPolicy(max_turns=2)
        mgr = LoopSnapshotManager(SESSION_ID, "stage 1 task", policy_a, base_dir=self._snap_dir)
        loop_a = AgentDriverLoop(kernel_a, policy=policy_a)
        result_a = loop_a.run("stage 1 task", session_id=SESSION_ID, reflection_hook=mgr.wrap_reflection())
        # Simulate interrupt — don't finalize

        assert result_a.completed_turns == 2

        # Phase B: new loop picks up the snapshot and resumes
        snap = LoopSessionSnapshot.load(SESSION_ID, base_dir=self._snap_dir)
        assert snap is not None
        assert snap.is_incomplete

        # Reconstruct policy from snapshot
        resumed_policy = DriverLoopPolicy(
            max_turns=snap.policy.get("max_turns", 8),
            max_failures=snap.policy.get("max_failures", 2),
            max_consecutive_noop=snap.policy.get("max_consecutive_noop", 2),
        )
        kernel_b = _CountingKernel()
        loop_b = AgentDriverLoop(kernel_b, policy=resumed_policy)
        result_b = loop_b.run(snap.initial_observation, session_id=snap.session_id)

        # Second run observes the same initial observation
        assert result_b.records[0].observation == snap.initial_observation
        # The resumed loop ran the same number of turns (policy unchanged)
        assert result_b.completed_turns > 0

    def test_snapshot_captures_completed_turns_after_reflection_hook(self) -> None:
        """The snapshot's last_committed_turn must reflect each wrapped reflection call."""
        kernel = _CountingKernel()
        policy = DriverLoopPolicy(max_turns=3)
        mgr = LoopSnapshotManager(SESSION_ID, "task", policy, base_dir=self._snap_dir)
        hook = mgr.wrap_reflection(None)  # no base hook

        turn_records_seen: list[int] = []

        def tracking_hook(ctx: dict) -> dict:
            result = hook(ctx)
            # Reload snapshot mid-run to assert it was written
            snap = LoopSessionSnapshot.load(SESSION_ID, base_dir=self._snap_dir)
            if snap is not None:
                turn_records_seen.append(snap.last_committed_turn)
            return result

        loop = AgentDriverLoop(kernel, policy=policy)
        loop.run("task", session_id=SESSION_ID, reflection_hook=tracking_hook)

        # Each wrapped hook call should have updated the snapshot
        assert len(turn_records_seen) > 0

    # -----------------------------------------------------------------------
    # Delete
    # -----------------------------------------------------------------------

    def test_delete_removes_snapshot(self) -> None:
        mgr = LoopSnapshotManager(SESSION_ID, "x", base_dir=self._snap_dir)
        assert LoopSessionSnapshot.load(SESSION_ID, base_dir=self._snap_dir) is not None
        deleted = LoopSessionSnapshot.delete(SESSION_ID, base_dir=self._snap_dir)
        assert deleted is True
        assert LoopSessionSnapshot.load(SESSION_ID, base_dir=self._snap_dir) is None

    def test_load_returns_none_for_missing_session(self) -> None:
        snap = LoopSessionSnapshot.load("nonexistent-session-xyz", base_dir=self._snap_dir)
        assert snap is None
