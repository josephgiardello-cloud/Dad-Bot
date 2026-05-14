from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from dadbot.core.contracts.lifecycle_events import Claimed, Completed, Redelivered, Submitted
from dadbot.core.control_plane_projection import ExecutionProjection
from dadbot.core.control_plane_reducer import ExecutionStatus, reduce_execution_lifecycle


pytestmark = pytest.mark.unit


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
