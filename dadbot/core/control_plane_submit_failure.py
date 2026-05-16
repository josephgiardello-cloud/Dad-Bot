from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from dadbot.core.control_plane_reducer import ExecutionStatus
from dadbot.core.execution_result_unified import build_unified_execution_result, mark_unified_execution_failure


def _record_submit_exception_impl(
    control_plane: Any,
    *,
    job: Any,
    future: asyncio.Future[Any],
    dedupe_future: asyncio.Future[Any] | None,
    session_key: str,
    trace_token: str,
    exc: BaseException,
    classify_execution_failure: Any,
    set_terminal_turn_state: Any,
    scheduler_exception_mapper: Any,
    mutate_runtime_plan: Any,
) -> None:
    classified = classify_execution_failure(exc)
    current_execution_result = mark_unified_execution_failure(
        dict(build_unified_execution_result()),
        failure_class=str(classified.get("failure_class") or "runtime_exception"),
        failure_source=str(classified.get("failure_source") or "execution"),
        retryable=bool(classified.get("retryable", False)),
        exception_type=str(classified.get("exception_type") or type(exc).__name__),
        message=str(exc),
    )
    job.metadata["execution_result"] = current_execution_result
    execution_state = dict(job.metadata.get("execution_state") or {})
    execution_state["failure_type"] = str(classified.get("failure_type") or "")
    execution_state["failure_action"] = str(classified.get("failure_action") or "")
    execution_state["auto_retry"] = bool(classified.get("auto_retry", False))
    execution_state["last_transition_reason"] = (
        f"control_plane.submit.failed:{str(classified.get('failure_action') or 'unknown')}"
    )
    job.metadata["execution_state"] = execution_state
    set_terminal_turn_state(
        job,
        terminal_state=scheduler_exception_mapper.from_exception(exc),
        reason=f"control_plane.submit.failed:{type(exc).__name__}",
        strict=False,
    )
    mutate_runtime_plan(
        metadata=job.metadata,
        reason=f"submit_failed:{type(exc).__name__}",
        status="failed",
        note=str(exc),
    )
    if future.done() and not future.cancelled():
        with contextlib.suppress(Exception):
            future.exception()
    if dedupe_future is not None and not dedupe_future.done():
        dedupe_future.set_exception(exc)
    projected = control_plane.lifecycle_projection.get(job.job_id)
    projection_terminal = bool(
        projected is not None and projected.status in {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED}
    )
    control_plane._emit_progress_snapshot(
        phase="during_scheduler_drain",
        session_id=session_key,
        trace_token=trace_token,
        job_id=job.job_id,
        future_done=future.done(),
        completion_expectations=control_plane._completion_expectations(
            job_id=job.job_id,
            future_done=future.done(),
            projection_terminal=projection_terminal,
        ),
        note="submit_turn exception",
        extra={"exception_type": type(exc).__name__, "exception": str(exc)},
    )
    control_plane._emit_runtime_stream_event(
        event_type="turn.failed",
        session_id=session_key,
        trace_id=trace_token,
        payload={
            "job_id": str(job.job_id or ""),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "runtime_plan": dict(job.metadata.get("runtime_plan") or {}),
        },
    )
    session_state = dict(control_plane.registry.get_or_create(session_key).get("state") or {})
    runtime_plan = dict(job.metadata.get("runtime_plan") or {})
    session_state.setdefault("authority_modes", {})
    if isinstance(session_state["authority_modes"], dict):
        session_state["authority_modes"]["response_selection"] = "response_engine"
        session_state["authority_modes"]["learning_update"] = "response_engine"
        session_state["authority_modes"]["memory_write"] = "memory_manager"
    failure_learning_telemetry = {
        "trace_id": str(trace_token or ""),
        "status": "failure",
        "strategy": str(runtime_plan.get("strategy") or "direct_answer"),
        "uncertainty_score": float(dict(runtime_plan.get("uncertainty") or {}).get("score") or 1.0),
        "failure_type": str(classified.get("failure_type") or "runtime_failure"),
        "non_canonical_loops": [
            "adaptation_engine",
            "belief_state_engine",
            "planning_optimizer",
            "alignment_trainer",
            "memory_hierarchy_manager",
            "tool_self_model",
            "multi_agent_swarm",
            "autonomous_goal_daemon",
        ],
    }
    session_state["learning_signal_telemetry"] = failure_learning_telemetry
    job.metadata["learning_signal_telemetry"] = dict(failure_learning_telemetry)
    control_plane._interactive_cognition_ui.emit_thought(
        state=session_state,
        trace_id=str(trace_token or ""),
        content=f"Turn failed: {type(exc).__name__}",
        confidence=0.9,
        category="outcome",
    )
    control_plane._merge_session_state(session_id=session_key, state_patch=session_state)


__all__ = ["_record_submit_exception_impl"]