from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from dadbot.contracts import AttachmentList, FinalizedTurnResult
from dadbot.core.contracts.lifecycle_events import Submitted
from dadbot.core.core_state import CoreState
from dadbot.core.execution_context import open_core_state_scope, push_core_state_event
from dadbot.core.control_plane_lifecycle import (
    _apply_projection_execution_state,
    _assert_lifecycle_emission_transition,
    _resolved_execution_mode,
)
from dadbot.core.planning_utils import build_runtime_plan as _build_runtime_plan


async def _register_submit_job_impl(
    control_plane: Any,
    *,
    execution_job_type: type[Any],
    session_key: str,
    user_input: str,
    attachments: AttachmentList | None,
    metadata: dict[str, Any],
    trace_token: str,
) -> tuple[Any, Any, float]:
    job = execution_job_type(
        session_id=session_key,
        user_input=str(user_input or ""),
        attachments=attachments,
        metadata=metadata,
        trace_id=trace_token,
    )
    current_lifecycle_state = control_plane._coerce_projection_lifecycle_state(job.job_id)
    if current_lifecycle_state is None:
        _assert_lifecycle_emission_transition(
            execution_id=str(job.job_id or ""),
            event=Submitted(execution_id=job.job_id, occurred_at=datetime.now()),
            current_state=current_lifecycle_state,
        )
        control_plane.ledger_writer.append_execution_lifecycle(
            Submitted(execution_id=job.job_id, occurred_at=datetime.now()),
            session_id=session_key,
            trace_id=job.trace_id,
            kernel_step_id="control_plane.submit_turn",
            committed=False,
        )
    control_plane.lifecycle_projection.rebuild_from_ledger(control_plane.ledger.read())
    _apply_projection_execution_state(job, control_plane.lifecycle_projection.get(job.job_id))
    job.metadata["execution_mode"] = _resolved_execution_mode(job)

    submitted_event = control_plane.ledger_writer.append_job_submitted(job)
    submitted_ts = float(submitted_event.get("timestamp") or 0.0)
    job.metadata["submitted_timestamp"] = submitted_ts
    job.metadata.setdefault("claim_order", {})
    job.metadata["claim_order"]["timestamp"] = submitted_ts
    job.metadata["claim_order"]["worker_id"] = str(
        getattr(control_plane._scheduler, "worker_id", "worker-1") or "worker-1",
    )
    job.metadata["claim_order"]["lease_epoch"] = int(
        dict(job.metadata.get("execution_state") or {}).get("redelivery_count") or 0,
    )
    control_plane.ledger_writer.append_session_bound(
        session_key,
        job.job_id,
        trace_id=job.trace_id,
        kernel_step_id="control_plane.bind_session",
    )
    future = await control_plane.scheduler.register(job)
    return job, future, submitted_ts


def _initialize_submit_scope_impl(
    control_plane: Any,
    *,
    session_key: str,
    trace_token: str,
    job_id: str,
    resolved_timeout_seconds: float,
) -> tuple[str, float, object]:
    session_before = control_plane.registry.get_or_create(session_key)
    before_state_hash = control_plane._stable_hash(dict(session_before.get("state") or {}))
    # Initialize CoreState from persisted session state; all turn mutations will
    # flow through the event bus (push_core_state_event) and update this binding.
    initial_core_state = CoreState.from_dict(
        dict(session_before.get("state") or {}).get("core_state"),
    )
    deadline = time.time() + resolved_timeout_seconds
    cs_token = open_core_state_scope(initial_core_state)
    push_core_state_event(
        "job_submitted",
        {
            "session_id": session_key,
            "trace_id": trace_token,
            "job_id": job_id,
        },
    )
    return before_state_hash, deadline, cs_token


def _prepare_submit_register_phase_impl(
    control_plane: Any,
    *,
    session_key: str,
    user_input: str,
    attachments: AttachmentList | None,
    metadata: dict[str, Any],
    request_id: str,
    effect_id: str,
) -> tuple[str, str]:
    trace_id, effect_id = control_plane._resolve_submit_trace_and_effect(
        session_key=session_key,
        user_input=user_input,
        attachments=attachments,
        metadata=metadata,
        request_id=request_id,
        effect_id=effect_id,
    )
    metadata["runtime_plan"] = _build_runtime_plan(
        session_id=session_key,
        trace_id=trace_id,
        user_input=str(user_input or ""),
        attachments=attachments,
        metadata=metadata,
    )
    metadata["effect_id"] = effect_id
    control_plane._emit_runtime_stream_event(
        event_type="turn.started",
        session_id=session_key,
        trace_id=trace_id,
        payload={
            "request_id": request_id,
            "effect_id": effect_id,
        },
    )
    control_plane._emit_runtime_stream_event(
        event_type="plan.created",
        session_id=session_key,
        trace_id=trace_id,
        payload={
            "runtime_plan": dict(metadata.get("runtime_plan") or {}),
            "semantic_memory_context_size": int(len(metadata.get("semantic_memory_context") or [])),
        },
    )
    control_plane._emit_runtime_stream_event(
        event_type="swarm.plan",
        session_id=session_key,
        trace_id=trace_id,
        payload={"swarm_plan": dict(metadata.get("swarm_plan") or {})},
    )
    return trace_id, effect_id


__all__ = [
    "_initialize_submit_scope_impl",
    "_prepare_submit_register_phase_impl",
    "_register_submit_job_impl",
]