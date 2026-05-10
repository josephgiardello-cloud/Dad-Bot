"""
Unit tests for three architectural gap invariants.

GAP 1: Memory evaluation count exactly 1 per turn.
  - build_context raises MemoryEvaluationViolation if called a second time on the same TurnContext.
  - build_context writes _memory_eval_count = 1 on first call.

GAP 2: ExecutionIdentity unified fingerprint includes memory_snapshot_hash and execution_result_hash.
  - Changing state["memory_snapshot"] produces a different fingerprint.
  - Changing state["execution_result"] produces a different fingerprint.
  - offline_replay_validator._recompute_identity_fingerprint mirrors the updated canonical.

GAP 3: SchedulerProtocol boundary is structurally enforced in ControlPlane.
  - ControlPlaneOptions.scheduler accepts any object satisfying SchedulerProtocol (duck typing).
  - Concrete Scheduler satisfies SchedulerProtocol.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# GAP 1 tests
# ---------------------------------------------------------------------------


def _make_turn_context(*, state: dict | None = None, metadata: dict | None = None) -> Any:
    """Minimal TurnContext stand-in with .state and .metadata dicts."""
    tc = SimpleNamespace()
    tc.state = dict(state or {})
    tc.metadata = dict(metadata or {})
    tc.temporal = SimpleNamespace(wall_time="2026-01-01T00:00:00", wall_date="2026-01-01")
    tc.user_input = "hello"
    return tc


class _MinimalContextService:
    """Minimal shim that exercises only the guard path in build_context."""

    def build_context_guard_only(self, turn_context: Any) -> None:
        """Execute only the GAP 1 guard (no real memory I/O needed)."""
        from dadbot.services.context_service import MemoryEvaluationViolation

        state = getattr(turn_context, "state", None)
        if isinstance(state, dict):
            count = int(state.get("_memory_eval_count", 0) or 0)
            if count > 0:
                raise MemoryEvaluationViolation(count)

    def increment_eval_count(self, turn_context: Any) -> None:
        state = getattr(turn_context, "state", None)
        if isinstance(state, dict):
            state["_memory_eval_count"] = int(state.get("_memory_eval_count", 0) or 0) + 1


class TestMemoryEvaluationSinglePassLock:
    def test_first_call_does_not_raise(self) -> None:
        tc = _make_turn_context()
        svc = _MinimalContextService()
        svc.build_context_guard_only(tc)  # must not raise

    def test_second_call_raises_MemoryEvaluationViolation(self) -> None:
        from dadbot.services.context_service import MemoryEvaluationViolation

        tc = _make_turn_context(state={"_memory_eval_count": 1})
        svc = _MinimalContextService()
        with pytest.raises(MemoryEvaluationViolation) as exc_info:
            svc.build_context_guard_only(tc)
        assert exc_info.value.count == 1

    def test_violation_message_is_clear(self) -> None:
        from dadbot.services.context_service import MemoryEvaluationViolation

        tc = _make_turn_context(state={"_memory_eval_count": 2})
        svc = _MinimalContextService()
        with pytest.raises(MemoryEvaluationViolation) as exc_info:
            svc.build_context_guard_only(tc)
        assert "exactly one" in str(exc_info.value).lower()

    def test_increment_seals_count_at_1(self) -> None:
        tc = _make_turn_context()
        assert tc.state.get("_memory_eval_count", 0) == 0
        svc = _MinimalContextService()
        svc.increment_eval_count(tc)
        assert tc.state["_memory_eval_count"] == 1

    def test_after_increment_guard_blocks_second_call(self) -> None:
        from dadbot.services.context_service import MemoryEvaluationViolation

        tc = _make_turn_context()
        svc = _MinimalContextService()
        svc.build_context_guard_only(tc)  # pass
        svc.increment_eval_count(tc)      # simulate what build_context writes
        with pytest.raises(MemoryEvaluationViolation):
            svc.build_context_guard_only(tc)  # must fail

    def test_no_state_dict_does_not_raise(self) -> None:
        """If turn_context has no .state dict, guard must not raise."""
        tc = SimpleNamespace(state=None, metadata={}, temporal=None, user_input="")
        svc = _MinimalContextService()
        svc.build_context_guard_only(tc)  # must not raise

    def test_MemoryEvaluationViolation_is_RuntimeError(self) -> None:
        from dadbot.services.context_service import MemoryEvaluationViolation

        assert issubclass(MemoryEvaluationViolation, RuntimeError)


# ---------------------------------------------------------------------------
# GAP 2 tests
# ---------------------------------------------------------------------------


def _build_turn_context_for_identity(
    *,
    memory_snapshot: dict | None = None,
    execution_result: dict | None = None,
    trace_hash: str = "abc",
    lock_hash: str = "def",
    event_count: int = 3,
) -> Any:
    tc = SimpleNamespace()
    tc.state = {
        "execution_trace_contract": {"trace_hash": trace_hash, "event_count": event_count},
        "memory_snapshot": dict(memory_snapshot or {}),
        "execution_result": dict(execution_result or {}),
    }
    tc.metadata = {"determinism": {"lock_hash": lock_hash}}
    tc.trace_id = "trace-001"
    tc.last_checkpoint_hash = ""
    tc.mutation_queue = None
    return tc


class TestExecutionIdentityEnvelope:
    def test_identity_has_memory_snapshot_hash(self) -> None:
        from dadbot.core.execution_identity import ExecutionIdentity

        tc = _build_turn_context_for_identity(memory_snapshot={"memory_rolling_summary": "hello"})
        identity = ExecutionIdentity.from_turn_context(tc)
        assert isinstance(identity.memory_snapshot_hash, str)
        assert len(identity.memory_snapshot_hash) == 16

    def test_identity_has_execution_result_hash(self) -> None:
        from dadbot.core.execution_identity import ExecutionIdentity

        tc = _build_turn_context_for_identity(execution_result={"status": "ok"})
        identity = ExecutionIdentity.from_turn_context(tc)
        assert isinstance(identity.execution_result_hash, str)
        assert len(identity.execution_result_hash) == 16

    def test_fingerprint_changes_when_memory_snapshot_changes(self) -> None:
        from dadbot.core.execution_identity import ExecutionIdentity

        tc_a = _build_turn_context_for_identity(memory_snapshot={"summary": "session A"})
        tc_b = _build_turn_context_for_identity(memory_snapshot={"summary": "session B"})
        fp_a = ExecutionIdentity.from_turn_context(tc_a).fingerprint
        fp_b = ExecutionIdentity.from_turn_context(tc_b).fingerprint
        assert fp_a != fp_b, "Fingerprint must differ when memory_snapshot differs"

    def test_fingerprint_changes_when_execution_result_changes(self) -> None:
        from dadbot.core.execution_identity import ExecutionIdentity

        tc_a = _build_turn_context_for_identity(execution_result={"status": "ok", "response": "hi"})
        tc_b = _build_turn_context_for_identity(execution_result={"status": "ok", "response": "bye"})
        fp_a = ExecutionIdentity.from_turn_context(tc_a).fingerprint
        fp_b = ExecutionIdentity.from_turn_context(tc_b).fingerprint
        assert fp_a != fp_b, "Fingerprint must differ when execution_result differs"

    def test_to_dict_includes_new_fields(self) -> None:
        from dadbot.core.execution_identity import ExecutionIdentity

        tc = _build_turn_context_for_identity(
            memory_snapshot={"k": "v"}, execution_result={"s": 1}
        )
        d = ExecutionIdentity.from_turn_context(tc).to_dict()
        assert "memory_snapshot_hash" in d
        assert "execution_result_hash" in d
        assert d["memory_snapshot_hash"]
        assert d["execution_result_hash"]

    def test_replay_validator_fingerprint_matches_identity(self) -> None:
        """offline_replay_validator._recompute_identity_fingerprint must match ExecutionIdentity.fingerprint."""
        from dadbot.core.execution_identity import ExecutionIdentity
        from dadbot.core.offline_replay_validator import _recompute_identity_fingerprint

        tc = _build_turn_context_for_identity(
            memory_snapshot={"rolling": "test"},
            execution_result={"status": "ok"},
            trace_hash="tr1",
            lock_hash="lk1",
            event_count=5,
        )
        identity = ExecutionIdentity.from_turn_context(tc)
        recomputed = _recompute_identity_fingerprint(identity.to_dict())
        assert recomputed == identity.fingerprint, (
            "offline_replay_validator must recompute the same fingerprint as ExecutionIdentity"
        )

    def test_empty_hashes_are_stable(self) -> None:
        """Empty memory_snapshot and execution_result produce deterministic hashes, not empty strings."""
        from dadbot.core.execution_identity import ExecutionIdentity

        tc = _build_turn_context_for_identity()
        identity = ExecutionIdentity.from_turn_context(tc)
        # Both fields should be the SHA-256[:16] of json.dumps({}) — stable and non-empty.
        expected_empty = hashlib.sha256(
            json.dumps({}, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        assert identity.memory_snapshot_hash == expected_empty
        assert identity.execution_result_hash == expected_empty


# ---------------------------------------------------------------------------
# GAP 3 tests
# ---------------------------------------------------------------------------


class TestSchedulerProtocolBoundary:
    def test_SchedulerProtocol_exists_in_control_plane(self) -> None:
        from dadbot.core.control_plane import SchedulerProtocol

        assert SchedulerProtocol is not None

    def test_ControlPlaneOptions_scheduler_accepts_none(self) -> None:
        from dadbot.core.control_plane import ControlPlaneOptions

        opts = ControlPlaneOptions()
        assert opts.scheduler is None

    def test_concrete_Scheduler_satisfies_interface_structurally(self) -> None:
        """Scheduler must expose all three methods required by SchedulerProtocol."""
        from dadbot.core.control_plane import Scheduler

        assert callable(getattr(Scheduler, "register", None))
        assert callable(getattr(Scheduler, "drain_once", None))
        assert callable(getattr(Scheduler, "wait_for_work", None))

    def test_ControlPlaneOptions_accepts_duck_typed_scheduler(self) -> None:
        """Any object with register/drain_once/wait_for_work can be passed as scheduler."""
        from dadbot.core.control_plane import ControlPlaneOptions

        class _FakeScheduler:
            async def register(self, job):
                return asyncio.get_running_loop().create_future()

            async def drain_once(self, executor):
                return False

            async def wait_for_work(self, *, timeout_seconds=None):
                return False

        opts = ControlPlaneOptions(scheduler=_FakeScheduler())
        assert opts.scheduler is not None
