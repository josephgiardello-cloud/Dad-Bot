from __future__ import annotations

import logging
from typing import Any

from dadbot.core.runtime_errors import InvariantViolation


def _trace_event_invariant_counts_impl(trace_events: list[dict[str, Any]]) -> tuple[int, bool]:
    commit_count = 0
    has_node_events = False
    for event in trace_events:
        payload = dict(event.get("payload") or {}) if isinstance(event, dict) else {}
        event_type = str(event.get("event_type") or payload.get("event_type") or "").strip().lower()
        stage = str(
            event.get("stage")
            or event.get("node_type")
            or payload.get("stage")
            or payload.get("node_type")
            or ""
        ).strip().lower()
        if event_type in {
            "node_start",
            "node_complete",
            "node_completed",
            "turn_start",
            "turn_complete",
            "turn_failed",
        }:
            has_node_events = True
        if event_type in {"node_complete", "node_completed"} and stage == "save":
            commit_count += 1
    return commit_count, has_node_events


def _fallback_commit_count_from_session_impl(
    *,
    session: dict[str, Any] | None,
    trace_token: str,
) -> int:
    if not isinstance(session, dict):
        return 0
    session_state = dict(session.get("state") or {})
    turn_trace = dict(session_state.get("turn_trace") or {})
    if str(turn_trace.get("trace_id") or "").strip() != str(trace_token).strip():
        return 0
    fallback_commit_count = int(turn_trace.get("commit_boundary_count") or 0)
    return fallback_commit_count if fallback_commit_count > 0 else 0


def _validate_trace_invariant_impl(
    control_plane: Any,
    *,
    job: Any,
    result: Any,
    session: dict[str, Any] | None,
    state_before_hash: str,
    logger: logging.Logger,
) -> None:
    trace_id = job.metadata.get("trace_id") or ""
    if not trace_id.strip():
        logger.warning("Trace invariant violation: missing trace_id in job %s", job.job_id)
        return

    trace_events = control_plane._job_trace_events(job)
    if not trace_events:
        logger.warning("Trace invariant violation: no events recorded for trace %s", trace_id)
        return

    commit_count, has_node_events = _trace_event_invariant_counts_impl(trace_events)
    if commit_count == 0:
        fallback_commit_count = _fallback_commit_count_from_session_impl(
            session=session,
            trace_token=trace_id,
        )
        if fallback_commit_count > 0:
            commit_count = fallback_commit_count

    if not has_node_events and commit_count == 0:
        control_plane._record_trace_composition_contract(
            session=session,
            job=job,
            result=result,
            state_before_hash=state_before_hash,
        )
        return

    if commit_count != 1:
        detail = (
            "Trace invariant violation: expected exactly 1 commit boundary "
            f"(found {int(commit_count)})"
        )
        strict = bool(dict(job.metadata or {}).get("strict_trace_invariant", False))
        if strict:
            raise InvariantViolation(
                "Trace invariant violation: expected exactly 1 commit boundary",
                context={
                    "trace_id": str(trace_id or ""),
                    "job_id": str(job.job_id or ""),
                    "commit_boundary_count": int(commit_count),
                },
            )
        logger.warning(detail)
        return

    control_plane._record_trace_composition_contract(
        session=session,
        job=job,
        result=result,
        state_before_hash=state_before_hash,
    )