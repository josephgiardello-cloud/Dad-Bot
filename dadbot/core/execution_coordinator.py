from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from typing import Any

from dadbot.contracts import FinalizedTurnResult
from dadbot.core.control_plane_reducer import ExecutionStatus


class ExecutionCoordinator:
    """Coordinates scheduler drain loops and completion expectation checks."""

    @staticmethod
    def completion_expectations(
        *,
        job_id: str,
        future_done: bool,
        projection_terminal: bool,
        scheduler: Any,
    ) -> dict[str, bool]:
        return {
            "future_done": bool(future_done),
            "projection_terminal": bool(projection_terminal),
            "job_removed_from_scheduler": bool(job_id not in getattr(scheduler, "_jobs", {})),
        }

    async def drain_scheduler_until_resolved(
        self,
        *,
        future: asyncio.Future[FinalizedTurnResult],
        job: Any,
        session_key: str,
        trace_token: str,
        deadline: float,
        scheduler: Any,
        kernel_executor: Callable[[dict[str, Any], Any], Awaitable[FinalizedTurnResult]],
        lifecycle_projection: Any,
        mutate_runtime_plan: Callable[..., dict[str, Any]],
        emit_runtime_stream_event: Callable[..., None],
        emit_progress_snapshot: Callable[..., None],
    ) -> int:
        loop_iterations = 0
        consecutive_idle_drains = 0
        last_progress_emit = time.monotonic()
        while not future.done():
            loop_iterations += 1
            if time.time() > deadline:
                projected_at_deadline = lifecycle_projection.get(job.job_id)
                projection_terminal_at_deadline = bool(
                    projected_at_deadline is not None
                    and projected_at_deadline.status in {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED}
                )
                emit_progress_snapshot(
                    phase="during_scheduler_drain",
                    session_id=session_key,
                    trace_token=trace_token,
                    job_id=job.job_id,
                    future_done=future.done(),
                    completion_expectations=self.completion_expectations(
                        job_id=job.job_id,
                        future_done=future.done(),
                        projection_terminal=projection_terminal_at_deadline,
                        scheduler=scheduler,
                    ),
                    note="deadline exceeded",
                    extra={"loop_iterations": loop_iterations, "consecutive_idle_drains": consecutive_idle_drains},
                )
                raise TimeoutError("submit_turn exceeded timeout")

            drained = await scheduler.drain_once(kernel_executor)
            if drained:
                consecutive_idle_drains = 0
                continue

            consecutive_idle_drains += 1
            remaining = max(0.0, deadline - time.time())
            if remaining <= 0.0:
                raise TimeoutError("submit_turn exceeded timeout")

            wait_task = asyncio.create_task(
                scheduler.wait_for_work(timeout_seconds=remaining),
            )
            done, _ = await asyncio.wait(
                {future, wait_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            wait_ready = False
            if wait_task in done:
                with contextlib.suppress(Exception):
                    wait_ready = bool(wait_task.result())

            if (time.monotonic() - last_progress_emit) >= 1.0 or consecutive_idle_drains >= 5:
                projected = lifecycle_projection.get(job.job_id)
                projection_terminal = bool(
                    projected is not None and projected.status in {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED}
                )
                if consecutive_idle_drains >= 5:
                    mutate_runtime_plan(
                        metadata=job.metadata,
                        reason="scheduler_stall_replan",
                        status="active",
                        strategy="clarify",
                        note="idle_drain_threshold_reached",
                    )
                    emit_runtime_stream_event(
                        event_type="plan.updated",
                        session_id=session_key,
                        trace_id=trace_token,
                        payload={
                            "job_id": str(job.job_id or ""),
                            "runtime_plan": dict(job.metadata.get("runtime_plan") or {}),
                            "trigger": "scheduler_stall_replan",
                        },
                    )
                emit_progress_snapshot(
                    phase="during_scheduler_drain",
                    session_id=session_key,
                    trace_token=trace_token,
                    job_id=job.job_id,
                    future_done=future.done(),
                    completion_expectations=self.completion_expectations(
                        job_id=job.job_id,
                        future_done=future.done(),
                        projection_terminal=projection_terminal,
                        scheduler=scheduler,
                    ),
                    note="idle drain iteration",
                    extra={
                        "loop_iterations": loop_iterations,
                        "consecutive_idle_drains": consecutive_idle_drains,
                        "wait_ready": bool(wait_ready),
                        "remaining_seconds": float(remaining),
                        "stall_phase_candidate": "during_scheduler_drain",
                        "potential_scheduler_ledger_cycle": bool(
                            consecutive_idle_drains >= 5
                            and int(len(getattr(scheduler, "_pending_job_ids", []) or [])) > 0
                            and not wait_ready
                        ),
                    },
                )
                last_progress_emit = time.monotonic()

            if wait_task not in done:
                wait_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await wait_task
        return loop_iterations
