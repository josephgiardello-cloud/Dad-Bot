from __future__ import annotations

from copy import deepcopy
from typing import Any


class CanonicalEventReducer:
    """Single system-wide reducer for ledger event -> state evolution."""

    def reduce(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        ordered = sorted(
            [dict(event) for event in list(events or []) if isinstance(event, dict)],
            key=lambda event: int(event.get("sequence") or 0),
        )

        state: dict[str, Any] = {
            "sessions": {},
            "jobs": {},
            "last_sequence": 0,
        }

        for event in ordered:
            event_type = str(event.get("type") or "")
            session_id = str(event.get("session_id") or "").strip()
            payload = dict(event.get("payload") or {})
            sequence = int(event.get("sequence") or 0)
            if sequence > state["last_sequence"]:
                state["last_sequence"] = sequence

            if not session_id:
                continue

            session_state = dict(state["sessions"].get(session_id) or {})
            job_id = str(payload.get("job_id") or "").strip()

            if event_type == "SESSION_STATE_UPDATED":
                patch = payload.get("state")
                if isinstance(patch, dict):
                    session_state.update(deepcopy(patch))
            elif event_type == "SESSION_STATE_DELETED":
                state["sessions"].pop(session_id, None)
                continue
            elif event_type == "JOB_STARTED" and job_id:
                state["jobs"][job_id] = {
                    "session_id": session_id,
                    "status": "started",
                }
            elif event_type == "JOB_COMPLETED":
                session_state["last_result"] = deepcopy(payload.get("result"))
                if job_id:
                    state["jobs"][job_id] = {
                        "session_id": session_id,
                        "status": "completed",
                    }
            elif event_type == "JOB_FAILED" and job_id:
                state["jobs"][job_id] = {
                    "session_id": session_id,
                    "status": "failed",
                    "error": str(payload.get("error") or ""),
                }

            state["sessions"][session_id] = session_state

        return state