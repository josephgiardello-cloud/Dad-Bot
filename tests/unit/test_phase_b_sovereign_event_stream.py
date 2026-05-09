from __future__ import annotations

from copy import deepcopy

import pytest

from dadbot.contracts import (
    PlannerDecisionPayload,
    SovereignEvent,
    ToolExecutionPayload,
)
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.trace_utils import build_trace_from_events


class _StaticBackend:
    def __init__(self, events: list[dict]) -> None:
        self._events = list(events)

    def append(self, event: dict, *, committed: bool = False) -> None:
        _ = (event, committed)

    def load(self) -> list[dict]:
        return list(self._events)

    def close(self) -> None:
        return


def test_sovereign_event_checksum_chain_verifies() -> None:
    first = SovereignEvent(
        turn_id="turn-1",
        event_type="PLANNER_DECISION",
        payload=PlannerDecisionPayload(
            planner_node="planner",
            selected_branch="default",
            rationale="initial branch",
        ),
    )
    second = SovereignEvent(
        turn_id="turn-1",
        event_type="TOOL_EXECUTION",
        previous_checksum=first.checksum,
        payload=ToolExecutionPayload(
            tool_name="lookup_web",
            status="success",
            input_hash="in-hash",
            output_hash="out-hash",
        ),
    )

    assert first.verify_checksum("")
    assert second.verify_checksum(first.checksum)


def test_build_trace_from_events_reconstructs_projection_state() -> None:
    first = SovereignEvent(
        turn_id="turn-2",
        event_type="PLANNER_DECISION",
        payload=PlannerDecisionPayload(
            planner_node="planner",
            selected_branch="safe_path",
            rationale="policy-compliant",
        ),
    )
    second = SovereignEvent(
        turn_id="turn-2",
        event_type="TOOL_EXECUTION",
        previous_checksum=first.checksum,
        payload=ToolExecutionPayload(
            tool_name="memory_lookup",
            status="success",
            input_hash="abc",
            output_hash="def",
        ),
    )

    trace = build_trace_from_events([first, second])

    assert trace.turn_id == "turn-2"
    assert len(trace.trace_events) == 2
    projected = trace.get_current_state()
    assert int(projected.get("event_count") or 0) == 2
    assert str(projected.get("last_event_type") or "") == "TOOL_EXECUTION"


def test_execution_ledger_refuses_corrupted_persisted_chain() -> None:
    source = ExecutionLedger()
    source.write(
        {
            "type": "JOB_SUBMITTED",
            "session_id": "s-1",
            "trace_id": "tr-phaseb",
            "kernel_step_id": "k-step-1",
            "timestamp": "2026-05-08T00:00:00Z",
            "payload": {"step": 1},
        },
    )
    source.write(
        {
            "type": "JOB_STARTED",
            "session_id": "s-1",
            "trace_id": "tr-phaseb",
            "kernel_step_id": "k-step-2",
            "timestamp": "2026-05-08T00:00:01Z",
            "payload": {"step": 2},
        },
    )

    tampered = deepcopy(source.read())
    tampered[1]["payload"] = {"step": "tampered"}

    candidate = ExecutionLedger(backend=_StaticBackend(tampered))
    with pytest.raises(RuntimeError, match="checksum chain verification failed"):
        candidate.load_from_backend()
