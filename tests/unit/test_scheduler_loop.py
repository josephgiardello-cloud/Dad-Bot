from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from dadbot.core.contracts.lifecycle_events import Claimed, Submitted
from dadbot.core.control_plane_projection import ExecutionProjection
from dadbot.core.scheduler.loop import SchedulerLoop


pytestmark = pytest.mark.unit


def test_scheduler_loop_dispatches_submitted_candidate() -> None:
    projection = ExecutionProjection()
    now = datetime.now()
    projection.apply(Submitted(execution_id="job-1", occurred_at=now))
    loop = SchedulerLoop(projection=projection)

    dispatched: list[str] = []
    result = loop.dispatch_once(
        execution_ids=["job-1"],
        try_claim=lambda execution_id: execution_id == "job-1",
        dispatch_to_worker=dispatched.append,
        now=now,
    )

    assert result == ["job-1"]
    assert dispatched == ["job-1"]


def test_scheduler_loop_skips_claimed_job_before_expiry() -> None:
    projection = ExecutionProjection()
    now = datetime.now()
    projection.apply(Submitted(execution_id="job-1", occurred_at=now))
    projection.apply(
        Claimed(
            execution_id="job-1",
            occurred_at=now,
            worker_id="worker-a",
            lease_expiry=now + timedelta(seconds=30),
        )
    )
    loop = SchedulerLoop(projection=projection)

    result = loop.dispatch_once(
        execution_ids=["job-1"],
        try_claim=lambda _execution_id: True,
        dispatch_to_worker=lambda _execution_id: None,
        now=now,
    )

    assert result == []
