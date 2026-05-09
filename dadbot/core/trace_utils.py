from __future__ import annotations

from typing import Any

from dadbot.contracts import SovereignEvent
from dadbot.core.turn_trace import TurnTrace


def reduce_sovereign_event_state(state: dict[str, Any], event: SovereignEvent) -> dict[str, Any]:
    next_state = dict(state or {})
    next_state["turn_id"] = str(event.turn_id or next_state.get("turn_id") or "")
    next_state["last_event_type"] = str(event.event_type or "")
    next_state["last_event_checksum"] = str(event.checksum or "")
    next_state["event_count"] = int(next_state.get("event_count") or 0) + 1
    next_state["updated_at"] = event.timestamp.isoformat()

    payload = event.payload.model_dump(mode="json")
    if event.event_type == "PLANNER_DECISION":
        next_state["planner"] = payload
    elif event.event_type == "TOOL_EXECUTION":
        next_state["last_tool_execution"] = payload
    elif event.event_type == "POLICY_VETO":
        next_state["last_policy_veto"] = payload
    elif event.event_type == "LOGIC_BRANCH":
        next_state["last_logic_branch"] = payload
    else:
        generic_events = list(next_state.get("generic_events") or [])
        generic_events.append(payload)
        next_state["generic_events"] = generic_events
    return next_state


def build_trace_from_events(events: list[SovereignEvent]) -> TurnTrace:
    if not events:
        raise ValueError("cannot build trace from empty sovereign event stream")

    ordered = sorted(events, key=lambda item: item.timestamp)
    first = ordered[0]
    trace = TurnTrace(
        trace_id=str(first.turn_id or ""),
        turn_id=str(first.turn_id or ""),
        session_id="default",
        start_time=first.timestamp.timestamp(),
    )

    projection_state: dict[str, Any] = {}
    previous_checksum = ""
    for event in ordered:
        if not event.verify_checksum(previous_checksum):
            raise ValueError(
                f"corrupted sovereign event stream at event_id={event.event_id}: checksum chain mismatch",
            )
        trace.record_event(
            {
                "event_id": str(event.event_id),
                "timestamp": event.timestamp.isoformat(),
                "turn_id": str(event.turn_id),
                "event_type": str(event.event_type),
                "payload": event.payload.model_dump(mode="json"),
                "previous_checksum": str(event.previous_checksum or ""),
                "checksum": str(event.checksum or ""),
            },
        )
        projection_state = reduce_sovereign_event_state(projection_state, event)
        previous_checksum = str(event.checksum or "")

    trace.metadata["projection_state"] = projection_state
    return trace
