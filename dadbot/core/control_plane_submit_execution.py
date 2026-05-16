from __future__ import annotations

import asyncio
from typing import Any

from dadbot.contracts import FinalizedTurnResult
from dadbot.core.control_plane_submit_phase import (
    _append_submit_turn_phase,
    _assert_submit_turn_phase_boundary,
    _assert_submit_turn_phase_trace_complete,
    submit_turn_phase_order,
)


async def _run_submit_execution_phase_impl(
    control_plane: Any,
    *,
    phase_trace: list[tuple[str, float]],
    future: asyncio.Future[FinalizedTurnResult],
    job: Any,
    session_key: str,
    trace_id: str,
    deadline: float,
    before_state_hash: str,
    dedupe_future: asyncio.Future[FinalizedTurnResult] | None,
) -> FinalizedTurnResult:
    try:
        _append_submit_turn_phase(phase_trace, "drain")
        _assert_submit_turn_phase_boundary(
            phase_trace=phase_trace,
            expected_phase="drain",
            operation="_drain_scheduler_until_resolved",
        )
        loop_iterations = await control_plane._drain_scheduler_until_resolved(
            future=future,
            job=job,
            session_key=session_key,
            trace_token=trace_id,
            deadline=deadline,
        )
        result = await future
        _append_submit_turn_phase(phase_trace, "finalize")
        _assert_submit_turn_phase_boundary(
            phase_trace=phase_trace,
            expected_phase="finalize",
            operation="_finalize_submit_success",
        )
        finalized = control_plane._finalize_submit_success(
            job=job,
            result=result,
            session_key=session_key,
            trace_token=trace_id,
            before_state_hash=before_state_hash,
            dedupe_future=dedupe_future,
            loop_iterations=loop_iterations,
        )
        _assert_submit_turn_phase_trace_complete(phase_trace)
        return finalized
    except Exception as exc:
        if len(phase_trace) < len(submit_turn_phase_order()):
            _append_submit_turn_phase(phase_trace, "finalize")
        _assert_submit_turn_phase_boundary(
            phase_trace=phase_trace,
            expected_phase="finalize",
            operation="_record_submit_exception",
        )
        control_plane._record_submit_exception(
            job=job,
            future=future,
            dedupe_future=dedupe_future,
            session_key=session_key,
            trace_token=trace_id,
            exc=exc,
        )
        _assert_submit_turn_phase_trace_complete(phase_trace)
        raise


__all__ = ["_run_submit_execution_phase_impl"]