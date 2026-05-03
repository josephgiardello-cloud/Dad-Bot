import pytest

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger.enforcement import LedgerEnforcementError


def test_execution_ledger_rejects_event_without_kernel_lineage():
    ledger = ExecutionLedger()

    with pytest.raises(LedgerEnforcementError):
        ledger.write(
            {
                "type": "JOB_QUEUED",
                "session_id": "s1",
                "trace_id": "t1",
                "timestamp": 1.0,
                "kernel_step_id": "",
                "payload": {"job_id": "j1"},
            }
        )


def test_execution_ledger_accepts_valid_event_and_assigns_sequence_fields():
    ledger = ExecutionLedger()

    event = ledger.write(
        {
            "type": "JOB_QUEUED",
            "session_id": "s1",
            "trace_id": "t1",
            "timestamp": 1.0,
            "kernel_step_id": "control_plane.enqueue",
            "payload": {"job_id": "j1"},
        }
    )

    assert event["_seq"] == 0
    assert event["sequence"] == 1
    assert str(event.get("event_sha256") or "")
    assert str(event.get("chain_hash") or "")
    assert str(event.get("prev_chain_hash") or "") == ""


def test_execution_ledger_chain_verification_passes_for_valid_sequence():
    ledger = ExecutionLedger()

    ledger.write(
        {
            "type": "JOB_QUEUED",
            "session_id": "s1",
            "trace_id": "t1",
            "timestamp": 1.0,
            "kernel_step_id": "control_plane.enqueue",
            "payload": {"job_id": "j1"},
        }
    )
    ledger.write(
        {
            "type": "JOB_STARTED",
            "session_id": "s1",
            "trace_id": "t1",
            "timestamp": 2.0,
            "kernel_step_id": "scheduler.execute.start",
            "payload": {"job_id": "j1"},
        }
    )

    report = ledger.verify_replay(mode="chain")
    assert report["ok"] is True
    assert report["event_count"] == 2
    assert isinstance(report["chain_hash"], str)
    assert report["chain_hash"]


def test_execution_ledger_chain_verification_detects_tampering():
    ledger = ExecutionLedger()

    ledger.write(
        {
            "type": "JOB_QUEUED",
            "session_id": "s1",
            "trace_id": "t1",
            "timestamp": 1.0,
            "kernel_step_id": "control_plane.enqueue",
            "payload": {"job_id": "j1"},
        }
    )
    ledger.write(
        {
            "type": "JOB_STARTED",
            "session_id": "s1",
            "trace_id": "t1",
            "timestamp": 2.0,
            "kernel_step_id": "scheduler.execute.start",
            "payload": {"job_id": "j1"},
        }
    )

    # Simulate in-memory corruption after persistence.
    ledger._events[1]["kernel_step_id"] = "tampered.step"

    report = ledger.verify_replay(mode="chain")
    assert report["ok"] is False
    assert any("mismatch" in item for item in report["violations"])
