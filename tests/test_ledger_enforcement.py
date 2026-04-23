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
