from __future__ import annotations

import contextlib
from typing import Any


def _apply_submit_success_postprocessing_impl(
    control_plane: Any,
    *,
    job: Any,
    result: tuple[Any, Any],
    session_key: str,
    trace_token: str,
    loop_iterations: int,
) -> tuple[Any, Any]:
    response_text = str(result[0] if isinstance(result, tuple) and len(result) >= 1 else "")
    control_plane._promote_semantic_memory(
        session=control_plane.registry.get_or_create(session_key),
        job=job,
        response_text=response_text,
    )
    session_state = dict(control_plane.registry.get_or_create(session_key).get("state") or {})
    runtime_plan = dict(job.metadata.get("runtime_plan") or {})
    control_plane._semantic_memory_graph.update_from_turn(
        state=session_state,
        session_id=session_key,
        trace_id=str(trace_token or ""),
        user_input=str(job.user_input or ""),
        response_text=response_text,
    )
    control_plane._adaptation_engine.record_outcome(
        state=session_state,
        trace_id=str(trace_token or ""),
        strategy=str(runtime_plan.get("strategy") or "direct_answer"),
        success=True,
        uncertainty_score=float(dict(runtime_plan.get("uncertainty") or {}).get("score") or 0.0),
        explicit_feedback=None,
    )
    control_plane._belief_state_engine.update_from_turn(
        state=session_state,
        trace_id=str(trace_token or ""),
        user_input=str(job.user_input or ""),
        runtime_plan=runtime_plan,
        success=True,
    )
    semantic_items = [
        dict(item)
        for item in list(job.metadata.get("semantic_memory_context") or [])
        if isinstance(item, dict)
    ]

    identity_state = control_plane._update_identity_state(
        session_state=session_state,
        user_input=str(job.user_input or ""),
        response_text=response_text,
        semantic_items=semantic_items,
    )
    job.metadata["identity_state"] = dict(identity_state)
    felt_state = control_plane._update_felt_persona_stream(
        session_state=session_state,
        user_input=str(job.user_input or ""),
        response_text=response_text,
        identity_state=identity_state,
        semantic_items=semantic_items,
    )
    job.metadata["felt_persona_state"] = dict(felt_state)
    trajectory_state = control_plane._update_conversation_trajectory(
        session_state=session_state,
        user_input=str(job.user_input or ""),
        response_text=response_text,
        identity_state=identity_state,
        felt_state=felt_state,
    )
    job.metadata["conversation_trajectory"] = dict(trajectory_state)

    control_plane._memory_hierarchy_manager.promote_turn(
        state=session_state,
        trace_id=str(trace_token or ""),
        user_input=str(job.user_input or ""),
        response_text=response_text,
        semantic_items=semantic_items,
    )
    control_plane._memory_hierarchy_manager.lifecycle_maintenance(state=session_state)
    control_plane._planning_optimizer.record_plan_outcome(
        state=session_state,
        plan=runtime_plan,
        success=True,
    )
    session_state.setdefault("authority_modes", {})
    if isinstance(session_state["authority_modes"], dict):
        session_state["authority_modes"]["response_selection"] = "response_engine"
        session_state["authority_modes"]["learning_update"] = "response_engine"
        session_state["authority_modes"]["memory_write"] = "memory_manager"
    learning_telemetry = {
        "trace_id": str(trace_token or ""),
        "status": "success",
        "strategy": str(runtime_plan.get("strategy") or "direct_answer"),
        "uncertainty_score": float(dict(runtime_plan.get("uncertainty") or {}).get("score") or 0.0),
        "response_length": int(len(response_text or "")),
        "identity_state": dict(identity_state),
        "felt_persona_state": dict(felt_state),
        "conversation_trajectory": dict(trajectory_state),
        "response_continuity": dict(job.metadata.get("response_continuity") or {}),
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
    session_state["learning_signal_telemetry"] = learning_telemetry
    job.metadata["learning_signal_telemetry"] = dict(learning_telemetry)
    control_plane._interactive_cognition_ui.emit_thought(
        state=session_state,
        trace_id=str(trace_token or ""),
        content="Turn completed successfully",
        confidence=0.95,
        category="outcome",
    )
    control_plane._merge_session_state(session_id=session_key, state_patch=session_state)
    control_plane._emit_runtime_stream_event(
        event_type="partial.output",
        session_id=session_key,
        trace_id=trace_token,
        payload={
            "chunk": response_text,
            "is_terminal_chunk": True,
            "job_id": str(job.job_id or ""),
        },
    )
    control_plane._emit_runtime_stream_event(
        event_type="turn.completed",
        session_id=session_key,
        trace_id=trace_token,
        payload={
            "job_id": str(job.job_id or ""),
            "loop_iterations": int(loop_iterations),
            "should_end": bool(result[1] if isinstance(result, tuple) and len(result) >= 2 else False),
            "runtime_plan": dict(job.metadata.get("runtime_plan") or {}),
        },
    )
    with contextlib.suppress(Exception):
        control_plane._topology_record_node(
            node_id="persistence.finalize_turn",
            metadata={"trace_id": str(trace_token or "")},
        )
    job.metadata["submit_finalization"] = {
        "done": True,
        "response": str(result[0] if isinstance(result, tuple) and len(result) >= 1 else ""),
        "should_end": bool(result[1] if isinstance(result, tuple) and len(result) >= 2 else False),
    }
    return result


__all__ = ["_apply_submit_success_postprocessing_impl"]