import pytest

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.replay_verifier import ReplayVerifier
from dadbot.core.session_store import SessionMutationError, SessionStore
from dadbot.core.system_health_checker import SystemHealthChecker


def test_execution_ledger_enforces_strict_session_causal_chain():
    ledger = ExecutionLedger()
    first = ledger.write(
        {
            "type": "JOB_QUEUED",
            "session_id": "s1",
            "trace_id": "t1",
            "timestamp": 1.0,
            "kernel_step_id": "control_plane.enqueue",
            "payload": {"job_id": "j1"},
        }
    )

    # Wrong parent (not current session head) must fail hard.
    with pytest.raises(Exception):
        ledger.write(
            {
                "type": "JOB_STARTED",
                "session_id": "s1",
                "trace_id": "t1",
                "timestamp": 2.0,
                "kernel_step_id": "scheduler.execute.start",
                "parent_event_id": "nonexistent-parent",
                "payload": {"job_id": "j1"},
            }
        )

    second = ledger.write(
        {
            "type": "JOB_STARTED",
            "session_id": "s1",
            "trace_id": "t1",
            "timestamp": 3.0,
            "kernel_step_id": "scheduler.execute.start",
            "payload": {"job_id": "j1"},
        }
    )
    assert second["parent_event_id"] == first["event_id"]
    assert second["session_index"] == 2


def test_replay_verifier_proves_equivalence_for_identical_trace():
    ledger = ExecutionLedger()
    ledger.write(
        {
            "type": "JOB_QUEUED",
            "session_id": "sx",
            "trace_id": "tx",
            "timestamp": 1.0,
            "kernel_step_id": "control_plane.enqueue",
            "payload": {"job_id": "jx"},
        }
    )
    ledger.write(
        {
            "type": "JOB_COMPLETED",
            "session_id": "sx",
            "trace_id": "tx",
            "timestamp": 2.0,
            "kernel_step_id": "scheduler.execute.complete",
            "payload": {"job_id": "jx", "result": ("ok", False)},
        }
    )

    events = ledger.read()
    verifier = ReplayVerifier()
    report = verifier.verify_equivalence(events, list(events))
    assert report["ok"] is True
    assert report["original_trace_hash"] == report["replayed_trace_hash"]
    assert report["original_state_hash"] == report["replayed_state_hash"]


def test_session_store_projection_only_blocks_direct_mutation():
    store = SessionStore(projection_only=True)
    with pytest.raises(SessionMutationError):
        store.set("s1", {"x": 1})
    with pytest.raises(SessionMutationError):
        store.delete("s1")


def test_system_health_checker_reports_causal_partitioning_ok():
    ledger = ExecutionLedger()
    store = SessionStore()
    first = ledger.write(
        {
            "type": "JOB_QUEUED",
            "session_id": "s-health",
            "trace_id": "th",
            "timestamp": 1.0,
            "kernel_step_id": "control_plane.enqueue",
            "payload": {"job_id": "jh"},
        }
    )
    ledger.write(
        {
            "type": "JOB_COMPLETED",
            "session_id": "s-health",
            "trace_id": "th",
            "timestamp": 2.0,
            "kernel_step_id": "scheduler.execute.complete",
            "parent_event_id": first["event_id"],
            "payload": {"job_id": "jh", "result": ("ok", False)},
        }
    )

    checker = SystemHealthChecker(base_path=".")
    report = checker.run_all(ledger=ledger, session_store=store)
    assert report["session_causal_partitioning"]["ok"] is True
    assert report["replay_equivalence"]["ok"] is True
