from __future__ import annotations

from typing import Any


# ------------------------------------------------------------------
# Step 1 helpers â€” extracted from _turn_phase_build_payload without
# any logic changes. Each owns a single concern.
# ------------------------------------------------------------------


def _build_pipeline_base(frame, selected_candidate, result) -> dict:
    """Merge active_pipeline, result.pipeline, and runtime annotations."""
    active_pipeline = dict(frame.state.get("active_pipeline") or selected_candidate.pipeline or {})
    pipeline: dict = dict(active_pipeline) if isinstance(active_pipeline, dict) else {}
    pipeline.update(dict(result.pipeline or {}))
    if isinstance(active_pipeline, dict):
        if "steps" in active_pipeline:
            pipeline["steps"] = list(active_pipeline.get("steps") or [])
        for key in ("reuse_verified_nodes",):
            if key in active_pipeline:
                pipeline[key] = active_pipeline[key]
    if frame.interaction_chain.get("critic_verdicts"):
        pipeline["critic_runtime"] = list(frame.interaction_chain.get("critic_verdicts") or [])
    if frame.state.get("execution_halt_reason"):
        pipeline["execution_halt_reason"] = str(frame.state.get("execution_halt_reason") or "")
    return pipeline


def _build_reasoning_dag_payload(pipeline, user_text, attachments, frame, stable_hash, build_reasoning_dag, planner_cache, event):
    """Build reasoning_steps, dag_signature, and annotated reasoning_dag."""
    reasoning_steps = list(pipeline.get("steps") or [])
    if not reasoning_steps:
        reasoning_steps = [dict(node) for node in list(frame.dag.get("nodes") or [])]
    dag_signature = stable_hash({"input": user_text, "steps": reasoning_steps})

    thread_cache = planner_cache.setdefault(str(event.thread_id or "default"), {})
    reasoning_dag = build_reasoning_dag(
        steps=reasoning_steps, user_text=user_text, attachments=attachments, thread_cache=thread_cache,
    )
    if frame.dag.get("nodes"):
        executed_by_id = {str(n.get("id") or ""): dict(n) for n in list(frame.dag.get("nodes") or [])}
        for dag_node in list(reasoning_dag.get("nodes") or []):
            source = executed_by_id.get(str(dag_node.get("id") or ""))
            if source:
                dag_node["status"] = str(source.get("status") or "pending")
                dag_node["result"] = str(source.get("result") or "")
                dag_node["output_hash"] = str(source.get("output_hash") or "")
    return reasoning_steps, dag_signature, reasoning_dag


def _build_tool_calls(reasoning_dag, plan_id, stable_hash, validate_interaction_tool) -> list:
    """Extract validated tool calls from reasoning_dag tool nodes."""
    tool_calls = []
    for idx, node in enumerate(list(reasoning_dag.get("nodes") or []), start=1):
        if str(node.get("kind") or "") != "tool":
            continue
        raw_tool_name = str(node.get("tool_name") or node.get("name") or "tool")
        tool_name = validate_interaction_tool(
            tool_name=raw_tool_name, node_id=str(node.get("id") or ""), plan_id=plan_id,
        )
        tool_call_id = f"tool_{stable_hash({'plan_id': plan_id, 'node_id': node.get('id'), 'idx': idx})[:16]}"
        tool_calls.append({"tool_call_id": tool_call_id, "tool_name": tool_name,
                            "depends_on": list(node.get("depends_on") or [])})
    return tool_calls


def _build_interaction_chain(event, plan_id, dag_signature, frame, thread_state, tool_calls, reply_id) -> dict:
    """Compose the interaction_chain tracking dict."""
    return {
        "user_input_id": str(event.id or ""),
        "plan_id": plan_id,
        "dag_signature": dag_signature,
        "steps": list(frame.interaction_chain.get("steps") or []),
        "tool_calls": tool_calls,
        "tool_results": list(frame.interaction_chain.get("tool_results") or []),
        "reply_id": reply_id,
        "iterations": int(frame.iteration),
        "continuity": {
            "thread_turn_count": int(thread_state.dag_state.get("turn_count") or 0),
            "fragments_tracked": len(dict(thread_state.dag_state.get("fragments") or {})),
        },
    }


def _build_replay_integrity(event, user_text, attachments, plan_id, tool_outputs, final_reply, frame, stable_hash) -> dict:
    """Build replay_integrity hash block, including optional multi-agent section."""
    replay_integrity = {
        "input_context_hash": stable_hash({
            "thread_id": str(event.thread_id or "default"),
            "text": user_text,
            "attachments": attachments,
            "correlation_id": str(event.correlation_id or ""),
            "plan_id": plan_id,
        }),
        "tool_outputs_hash": stable_hash(tool_outputs),
        "final_reply_hash": stable_hash(final_reply),
        "deterministic": True,
    }
    substrate_meta = dict(frame.state.get("multi_agent_substrate") or {})
    if bool(substrate_meta.get("enabled")):
        per_agent_contexts = dict(substrate_meta.get("agent_execution_contexts") or {})
        arbitration_payload = dict(substrate_meta.get("arbitration") or {})
        replay_integrity["multi_agent"] = {
            "per_agent_state_reconstruction": {
                str(agent_id): {
                    "seed": str(d.get("seed") or ""),
                    "memory_hash": str(d.get("memory_hash") or ""),
                    "planner_hash": str(d.get("planner_hash") or ""),
                    "critic_hash": str(d.get("critic_hash") or ""),
                    "reward_hash": str(d.get("reward_hash") or ""),
                }
                for agent_id, d in sorted(per_agent_contexts.items(), key=lambda x: str(x[0]))
            },
            "per_agent_dag_reproduction": {
                str(agent_id): str(d.get("dag_hash") or "")
                for agent_id, d in sorted(per_agent_contexts.items(), key=lambda x: str(x[0]))
            },
            "per_agent_execution_trace": {
                str(agent_id): str(d.get("execution_trace_hash") or "")
                for agent_id, d in sorted(per_agent_contexts.items(), key=lambda x: str(x[0]))
            },
            "deterministic_arbitration_result_hash": stable_hash(arbitration_payload),
        }
    return replay_integrity


def _assemble_pipeline(
    pipeline, reasoning_dag, plan_id, intent_label, frame, selected_candidate,
    critic_verdict, contract_report, last_reward_drift, event, export_execution_state,
    interaction_chain, replay_integrity,
) -> None:
    """Inject all sub-dicts into pipeline in place."""
    pipeline["reasoning_dag"] = dict(reasoning_dag)
    pipeline["plan_id"] = plan_id
    pipeline["planner"] = {
        "intent_label": intent_label,
        "candidates": list(frame.state.get("plan_candidates") or []),
        "selected_candidate_id": selected_candidate.candidate_id,
        "reward_scores": list(frame.state.get("plan_rewards") or []),
        "policy_bias": dict(frame.state.get("policy_bias") or {}),
        "multi_agent_substrate": dict(frame.state.get("multi_agent_substrate") or {}),
    }
    pipeline["critic"] = {
        "accepted_candidate_id": critic_verdict.accepted_candidate_id or selected_candidate.candidate_id,
        "accepted": bool(critic_verdict.accepted),
        "reason": critic_verdict.reason,
        "rejected_candidate_ids": list(critic_verdict.rejected_candidate_ids),
        "revised": bool(critic_verdict.revised_pipeline),
    }
    if frame.state.get("branch_results"):
        pipeline["branch_execution"] = dict(frame.state.get("branch_results") or {})
    pipeline["interaction_chain"] = dict(interaction_chain)
    pipeline["replay_integrity"] = dict(replay_integrity)
    pipeline["reply_contract"] = dict(contract_report)
    pipeline["observability"] = {
        "dag_mutation_log": list(frame.state.get("dag_mutation_log") or []),
        "dag_mutation_count": len(list(frame.state.get("dag_mutation_log") or [])),
        "execution_iterations": int(frame.iteration),
        "reward_drift": dict(last_reward_drift.get(str(event.thread_id or "default")) or {}),
    }
    pipeline["execution_state"] = export_execution_state(event.thread_id)


# ------------------------------------------------------------------
# Orchestrator â€” thin coordinator, no inline logic
# ------------------------------------------------------------------


def _turn_phase_build_payload(*, turn_state: dict[str, Any], execution_result: dict[str, Any]) -> tuple:
    event = turn_state["event"]
    user_text = turn_state["user_text"]
    attachments = list(turn_state["attachments"])
    thread_state = turn_state["thread_state"]
    frame = turn_state["frame"]
    final_reply = turn_state["final_reply"]
    contract_report = turn_state["contract_report"]
    plan_id = turn_state["plan_id"]
    selected_candidate = turn_state["selected_candidate"]
    critic_verdict = turn_state["critic_verdict"]
    intent_label = turn_state["intent_label"]

    result = execution_result["result"]
    candidates = list(execution_result["candidates"])  # noqa: F841

    stable_hash = turn_state["stable_hash"]
    build_reasoning_dag = turn_state["build_reasoning_dag"]
    validate_interaction_tool = turn_state["validate_interaction_tool"]
    export_execution_state = turn_state["export_execution_state"]
    planner_cache = turn_state["planner_cache"]
    last_reward_drift = turn_state["last_reward_drift"]

    pipeline = _build_pipeline_base(frame, selected_candidate, result)

    _reasoning_steps, dag_signature, reasoning_dag = _build_reasoning_dag_payload(
        pipeline, user_text, attachments, frame, stable_hash, build_reasoning_dag, planner_cache, event,
    )

    reply_id = f"reply_{stable_hash({'plan_id': plan_id, 'reply': final_reply})[:16]}"
    tool_outputs = list(pipeline.get("tool_outputs") or pipeline.get("tool_results") or [])

    tool_calls = _build_tool_calls(reasoning_dag, plan_id, stable_hash, validate_interaction_tool)
    interaction_chain = _build_interaction_chain(event, plan_id, dag_signature, frame, thread_state, tool_calls, reply_id)
    replay_integrity = _build_replay_integrity(event, user_text, attachments, plan_id, tool_outputs, final_reply, frame, stable_hash)

    _assemble_pipeline(
        pipeline, reasoning_dag, plan_id, intent_label, frame, selected_candidate,
        critic_verdict, contract_report, last_reward_drift, event, export_execution_state,
        interaction_chain, replay_integrity,
    )

    thread_state.dag_state["last_dag_signature"] = dag_signature
    return pipeline, interaction_chain, replay_integrity


class PayloadBuilder:
    def build(self, turn_state: dict[str, Any], execution_result: dict[str, Any]) -> tuple:
        return _turn_phase_build_payload(turn_state=turn_state, execution_result=execution_result)
