from __future__ import annotations

from typing import Any

from dadbot.core.execution_result_unified import (
    build_unified_execution_result,
    ensure_unified_execution_result,
)
from dadbot.core.control_plane_lifecycle import ExecutionLifecycleState


def _prepare_submit_metadata_impl(
    control_plane: Any,
    *,
    metadata: dict[str, Any] | None,
    user_input: str,
    attachments: Any,
    timeout_seconds: float | None,
    normalize_tool_runtime_contract: Any,
    extract_execution_degradations: Any,
) -> tuple[dict[str, Any], float, str, str]:
    md = dict(metadata or {})
    request_id = str(md.get("request_id") or "").strip()
    effect_id = str(md.get("effect_id") or "").strip()
    control_plane._prepare_global_confluence_law(
        user_input=str(user_input or ""),
        attachments=attachments,
        metadata=md,
    )
    resolved_timeout_seconds = 30.0 if timeout_seconds is None else max(0.0, float(timeout_seconds))
    raw_execution_result = dict(
        md.get("execution_result")
        or build_unified_execution_result(
            timeout_seconds=resolved_timeout_seconds,
            degradation_items=extract_execution_degradations(md),
        ),
    )
    # Stamp the current resolved timeout before normalization so the canonical
    # module owns the final sanitized form.
    raw_execution_result.setdefault("timeout", {})["seconds"] = float(resolved_timeout_seconds)
    md["execution_result"] = ensure_unified_execution_result(raw_execution_result)
    md["execution_state"] = {
        "lifecycle_state": ExecutionLifecycleState.SUBMITTED.value,
        "redelivery_count": int(dict(md.get("execution_state") or {}).get("redelivery_count") or 0),
        "lease_conflict_count": int(dict(md.get("execution_state") or {}).get("lease_conflict_count") or 0),
        "last_worker_id": str(dict(md.get("execution_state") or {}).get("last_worker_id") or ""),
        "last_transition_reason": "control_plane.submit_turn",
        "retry_not_before_monotonic": 0.0,
    }
    md.setdefault("strict_trace_invariant", True)
    tool_specs = control_plane._discover_tool_specs()
    prior_plan = dict(md.get("runtime_plan") or {})
    md["runtime_plan"] = control_plane._cognitive_policy_engine.build_plan(
        session_id=str(md.get("session_id") or "default"),
        trace_id=str(md.get("trace_id") or ""),
        user_input=str(user_input or ""),
        existing_plan=prior_plan,
        memory_hits=int(len(list(md.get("semantic_memory_context") or []))),
        tool_candidates=int(len(tool_specs)),
    )
    md["tool_runtime_contract"] = normalize_tool_runtime_contract(md)
    md["tool_routing_plan"] = control_plane._tool_routing_engine.build_routing_plan(
        tool_request=dict(md.get("tool_runtime_contract") or {}),
        available_specs=tool_specs,
        uncertainty_score=float(dict(md.get("runtime_plan") or {}).get("uncertainty", {}).get("score") or 0.0),
    )
    md["compositional_tool_plan"] = control_plane._compositional_tool_planner.build_plan(
        user_input=str(user_input or ""),
        routing_plan=dict(md.get("tool_routing_plan") or {}),
        available_specs=tool_specs,
    )
    md["reasoning_hypotheses"] = control_plane._hypothesis_engine.infer_hypotheses(
        user_input=str(user_input or ""),
        runtime_plan=dict(md.get("runtime_plan") or {}),
        tool_routing_plan=dict(md.get("tool_routing_plan") or {}),
        memory_context=[
            dict(item)
            for item in list(md.get("semantic_memory_context") or [])
            if isinstance(item, dict)
        ],
    )
    md["safety_state"] = control_plane._semantic_safety_engine.classify(
        user_input=str(user_input or ""),
        runtime_plan=dict(md.get("runtime_plan") or {}),
        memory_context=[
            dict(item)
            for item in list(md.get("semantic_memory_context") or [])
            if isinstance(item, dict)
        ],
    )
    return md, resolved_timeout_seconds, request_id, effect_id


__all__ = ["_prepare_submit_metadata_impl"]