from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from dadbot.core.contracts.lifecycle_events import Claimed, Completed, Redelivered, Submitted
from dadbot.core.control_plane import ControlPlaneOptions, ExecutionControlPlane, ExecutionJob, SessionRegistry
from dadbot.core.control_plane_projection import ExecutionProjection
from dadbot.core.control_plane_reducer import ExecutionStatus, reduce_execution_lifecycle
from dadbot.core.control_plane_submission import _register_submit_job_impl
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.kernel_gateway import KernelGateway
from dadbot.core.ledger_writer import LedgerWriter


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize("already_submitted", [False, True])
async def test_submission_rebuilds_complete_lifecycle_without_copying_other_payloads(monkeypatch, already_submitted):
    from dadbot.core import execution_ledger

    async def executor(_session, _job):
        return "ok", False

    ledger = ExecutionLedger()
    writer = LedgerWriter(ledger)
    plane = ExecutionControlPlane(
        registry=SessionRegistry(), kernel_executor=executor, options=ControlPlaneOptions(ledger=ledger),
    )
    ledger.write({
        "type": "SESSION_STATE_UPDATED", "session_id": "unrelated", "trace_id": "other",
        "kernel_step_id": "test", "timestamp": 1.0,
        "payload": {"state": {"history": ["large-unrelated-payload"] * 100}},
    })
    now = datetime.now()
    for event in (
        Submitted(execution_id="prior-tenant-a", occurred_at=now),
        Claimed(
            execution_id="prior-tenant-a", occurred_at=now, worker_id="worker-a",
            lease_expiry=now + timedelta(seconds=30),
        ),
        Completed(execution_id="prior-tenant-a", occurred_at=now, result_ref="prior-result"),
        Submitted(execution_id="prior-tenant-b", occurred_at=now),
    ):
        writer.append_execution_lifecycle(
            event, session_id="shared", trace_id="prior-trace", kernel_step_id="test",
        )
    metadata = {"tenant_id": "tenant-a", "request_id": "new-request"}
    candidate = ExecutionJob(session_id="shared", user_input="hello", metadata=metadata, trace_id="new-trace")
    if already_submitted:
        writer.append_execution_lifecycle(
            Submitted(execution_id=candidate.job_id, occurred_at=now),
            session_id="shared", trace_id="new-trace", kernel_step_id="test",
        )
    plane.lifecycle_projection.rebuild_from_ledger(ledger.read())
    copied = []
    original_copy = execution_ledger.deepcopy

    def capture_copy(value):
        if isinstance(value, list):
            assert all(event["type"] == "EXECUTION_LIFECYCLE" for event in value)
            copied.append(original_copy(value))
        return original_copy(value)

    with monkeypatch.context() as patch, KernelGateway.open_scope():
        patch.setattr(execution_ledger, "deepcopy", capture_copy)
        job, future, timestamp = await _register_submit_job_impl(
            plane, execution_job_type=ExecutionJob, session_key="shared", user_input="hello",
            attachments=None, metadata=metadata, trace_token="new-trace",
        )
    assert len(copied) == 1
    baseline = ExecutionProjection()
    baseline.rebuild_from_ledger(ledger.read())
    assert plane.lifecycle_projection.snapshot() == baseline.snapshot()
    assert len(baseline._events[candidate.job_id]) == 1
    assert job.job_id == candidate.job_id
    assert job.metadata["execution_state"]["lifecycle_state"] == "submitted"
    assert timestamp > 0 and not future.done()
    future.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["corrupt_lifecycle", "uncertain"])
async def test_submission_selective_projection_still_rejects_invalid_ledger(monkeypatch, invalid):
    async def executor(_session, _job):
        return "ok", False

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=executor)
    register = AsyncMock(wraps=plane._scheduler.register)
    monkeypatch.setattr(plane._scheduler, "register", register)
    if invalid == "corrupt_lifecycle":
        plane.ledger.write({
            "type": "EXECUTION_LIFECYCLE", "session_id": "other", "trace_id": "other",
            "kernel_step_id": "test", "timestamp": 1.0, "payload": {"schema_version": "invalid"},
        })
    else:
        plane.ledger._backend_uncertain = True
    with pytest.raises((RuntimeError, ValueError)), KernelGateway.open_scope():
        await _register_submit_job_impl(
            plane, execution_job_type=ExecutionJob, session_key="shared", user_input="hello",
            attachments=None, metadata={"tenant_id": "tenant-a"}, trace_token="new-trace",
        )
    register.assert_not_awaited()


@pytest.mark.asyncio
async def test_submission_preserves_complete_input_for_custom_lifecycle_projections():
    class CustomProjection(ExecutionProjection):
        def rebuild_from_ledger(self, events):
            self.received = list(events)
            super().rebuild_from_ledger(self.received)

    async def executor(_session, _job):
        return "ok", False

    plane = ExecutionControlPlane(registry=SessionRegistry(), kernel_executor=executor)
    plane.ledger.write({
        "type": "SESSION_STATE_UPDATED", "session_id": "other", "trace_id": "other",
        "kernel_step_id": "test", "timestamp": 1.0, "payload": {"state": {"retained": True}},
    })
    projection = CustomProjection()
    plane.lifecycle_projection = projection
    with KernelGateway.open_scope():
        _job, future, _stamp = await _register_submit_job_impl(
            plane, execution_job_type=ExecutionJob, session_key="shared", user_input="hello",
            attachments=None, metadata={"tenant_id": "tenant-a"}, trace_token="new-trace",
        )
    future.cancel()
    assert {event["type"] for event in projection.received} == {
        "SESSION_STATE_UPDATED", "EXECUTION_LIFECYCLE",
    }


def test_reduce_execution_lifecycle_counts_claim_attempts() -> None:
    now = datetime.now()
    state = reduce_execution_lifecycle(
        [
            Submitted(execution_id="job-1", occurred_at=now),
            Claimed(
                execution_id="job-1",
                occurred_at=now,
                worker_id="worker-a",
                lease_expiry=now + timedelta(seconds=30),
            ),
            Redelivered(
                execution_id="job-1",
                occurred_at=now + timedelta(seconds=31),
                previous_worker_id="worker-a",
                new_worker_id="worker-b",
            ),
            Claimed(
                execution_id="job-1",
                occurred_at=now + timedelta(seconds=31),
                worker_id="worker-b",
                lease_expiry=now + timedelta(seconds=60),
            ),
        ]
    )

    assert state.status == ExecutionStatus.CLAIMED
    assert state.owner == "worker-b"
    assert state.attempt_count == 2


def test_projection_marks_submitted_job_runnable() -> None:
    now = datetime.now()
    projection = ExecutionProjection()
    projection.apply(Submitted(execution_id="job-1", occurred_at=now))

    assert projection.get_runnable(now=now, execution_ids=["job-1"]) == ["job-1"]


def test_projection_reduces_terminal_state() -> None:
    now = datetime.now()
    projection = ExecutionProjection()
    projection.apply(Submitted(execution_id="job-1", occurred_at=now))
    projection.apply(
        Claimed(
            execution_id="job-1",
            occurred_at=now,
            worker_id="worker-a",
            lease_expiry=now + timedelta(seconds=30),
        )
    )
    projection.apply(
        Completed(
            execution_id="job-1",
            occurred_at=now + timedelta(seconds=1),
            result_ref="job:job-1:result",
        )
    )

    reduced = projection.get("job-1")
    assert reduced is not None
    assert reduced.status == ExecutionStatus.COMPLETED
    assert reduced.owner is None
