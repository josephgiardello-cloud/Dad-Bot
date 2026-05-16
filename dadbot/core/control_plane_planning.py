from __future__ import annotations

from typing import Any

from dadbot.core.planning_utils import mutate_runtime_plan as _mutate_runtime_plan


def _prepare_submit_runtime_planning_impl(
    control_plane: Any,
    *,
    session_key: str,
    user_input: str,
    metadata: dict[str, Any],
) -> None:
    control_plane._inject_semantic_memory_context(
        session_id=session_key,
        user_input=str(user_input or ""),
        metadata=metadata,
        limit=5,
    )
    session_state = dict(control_plane.registry.get_or_create(session_key).get("state") or {})
    for spec in control_plane._discover_tool_specs():
        control_plane._tool_ecosystem_hub.register_connector(
            state=session_state,
            name=str(spec.name or ""),
            capabilities=[str(item) for item in list(spec.capabilities or [])],
            endpoint="",
            health=1.0,
        )
    runtime_plan = dict(metadata.get("runtime_plan") or {})
    if runtime_plan:
        intent_type = str(runtime_plan.get("intent_type") or "statement")
        optimizer_strategy = control_plane._planning_optimizer.suggest(state=session_state, intent_type=intent_type)
        belief_strategy = control_plane._belief_state_engine.next_best_strategy(
            state=session_state,
            intent_type=intent_type,
        )
        aligned_strategy = control_plane._alignment_trainer.recommend_strategy(
            state=session_state,
            intent_type=intent_type,
            default_strategy=str(runtime_plan.get("strategy") or "direct_answer"),
        )
        if optimizer_strategy and optimizer_strategy != str(runtime_plan.get("strategy") or ""):
            runtime_plan["strategy"] = optimizer_strategy
            runtime_plan["optimizer_override"] = True
        if aligned_strategy and aligned_strategy != str(runtime_plan.get("strategy") or ""):
            runtime_plan["strategy"] = aligned_strategy
            runtime_plan["alignment_override"] = True
        if belief_strategy:
            runtime_plan["belief_suggested_strategy"] = belief_strategy
        metadata["runtime_plan"] = runtime_plan

    metadata["tool_routing_plan"] = control_plane._tool_self_model.apply_routing_feedback(
        state=session_state,
        routing_plan=dict(metadata.get("tool_routing_plan") or {}),
    )
    metadata["compositional_tool_plan"] = control_plane._compositional_tool_planner.build_plan(
        user_input=str(user_input or ""),
        routing_plan=dict(metadata.get("tool_routing_plan") or {}),
        available_specs=control_plane._discover_tool_specs(),
    )
    metadata["reasoning_hypotheses"] = control_plane._hypothesis_engine.infer_hypotheses(
        user_input=str(user_input or ""),
        runtime_plan=dict(metadata.get("runtime_plan") or {}),
        tool_routing_plan=dict(metadata.get("tool_routing_plan") or {}),
        memory_context=[
            dict(item)
            for item in list(metadata.get("semantic_memory_context") or [])
            if isinstance(item, dict)
        ],
    )

    pending_steering = control_plane._consume_pending_turn_steering(session_id=session_key)
    if pending_steering:
        desired_strategy = str(pending_steering.get("strategy") or "").strip()
        if desired_strategy:
            _mutate_runtime_plan(
                metadata=metadata,
                reason="user_steering",
                status="active",
                strategy=desired_strategy,
                note=str(pending_steering.get("note") or ""),
            )
            control_plane._interactive_cognition_ui.apply_plan_edit(
                state=session_state,
                trace_id=str(metadata.get("trace_id") or ""),
                edits={"strategy": desired_strategy, "status": "active"},
                actor="steering",
            )
        metadata["applied_steering"] = dict(pending_steering)

    runtime_plan = dict(metadata.get("runtime_plan") or {})
    control_plane._interactive_cognition_ui.register_plan(
        state=session_state,
        trace_id=str(metadata.get("trace_id") or ""),
        runtime_plan=runtime_plan,
        source="submit_turn",
    )
    control_plane._interactive_cognition_ui.emit_thought(
        state=session_state,
        trace_id=str(metadata.get("trace_id") or ""),
        content=f"Planning strategy: {str(runtime_plan.get('strategy') or 'direct_answer')}",
        confidence=1.0 - float(dict(runtime_plan.get("uncertainty") or {}).get("score") or 0.0),
        category="plan",
    )
    needed_capabilities = [
        str(item.get("capability") or "")
        for item in list(dict(metadata.get("tool_routing_plan") or {}).get("alternatives") or [])
        if isinstance(item, dict)
    ]
    metadata["external_tool_candidates"] = control_plane._tool_ecosystem_hub.rank_connectors(
        state=session_state,
        needed_capabilities=[item for item in needed_capabilities if item],
        limit=5,
    )
    metadata["swarm_plan"] = control_plane._multi_agent_swarm.build_plan(
        state=session_state,
        trace_id=str(metadata.get("trace_id") or ""),
        user_input=str(user_input or ""),
        runtime_plan=runtime_plan,
        compositional_tool_plan=dict(metadata.get("compositional_tool_plan") or {}),
        max_agents=4,
    )

    control_plane._hypothesis_engine.persist(
        state=session_state,
        trace_id=str(metadata.get("trace_id") or ""),
        hypotheses=[
            dict(item)
            for item in list(metadata.get("reasoning_hypotheses") or [])
            if isinstance(item, dict)
        ],
    )
    control_plane._post_planning_pre_tool_contract_gate(
        session_id=session_key,
        metadata=metadata,
    )
    control_plane._merge_session_state(session_id=session_key, state_patch=session_state)


__all__ = ["_prepare_submit_runtime_planning_impl"]