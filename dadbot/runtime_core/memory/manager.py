"""Thin memory manager stub."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from ..goal_reward_manager import GoalAndRewardManager
from ..runtime_dag_helpers import node_signature
from ..runtime_policy_helpers import intent_label


def _stable_hash(payload: Any) -> str:
    try:
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        raw = repr(payload).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class MemoryManager:
    """Memory manager stub."""

    def consolidate_turn_outcome(self, *args: Any, **kwargs: Any) -> None:
        runtime = kwargs.get("runtime")
        thread_state = kwargs.get("thread_state")
        memory_state = getattr(thread_state, "memory_state", None)
        forwarded = {
            key: value
            for key, value in kwargs.items()
            if key not in {"runtime", "thread_state", "memory_state"}
        }
        self.consolidate_turn_outcome_owned(
            runtime=runtime,
            thread_state=thread_state,
            memory_state=memory_state,
            **forwarded,
        )

    def consolidate_turn_outcome_owned(self, *args: Any, **kwargs: Any) -> None:
        runtime = kwargs.get("runtime")
        thread_state = kwargs.get("thread_state")
        memory_state = kwargs.get("memory_state")
        if not isinstance(memory_state, dict):
            return None

        thread_id = str(kwargs.get("thread_id") or "default")
        user_input = str(kwargs.get("user_input") or "")
        dag = dict(kwargs.get("dag") or {})
        branch_results = dict(kwargs.get("branch_results") or {})
        final_reply = str(kwargs.get("final_reply") or "")
        now = int(time.time())

        nodes = list(dag.get("nodes") or [])
        edges = list(dag.get("edges") or [])
        dag_shape_signature = _stable_hash(
            {
                "nodes": [
                    {
                        "name": str(node.get("name") or node.get("step") or ""),
                        "kind": str(node.get("kind") or "reasoning"),
                        "tool_name": str(node.get("tool_name") or ""),
                    }
                    for node in nodes
                ],
                "edges": [
                    {
                        "from": str(edge.get("from") or edge.get("source") or ""),
                        "to": str(edge.get("to") or edge.get("target") or ""),
                    }
                    for edge in edges
                ],
            }
        )

        intent_key = intent_label(user_input)
        strategy_name = str(branch_results.get("selected_candidate_id") or intent_key or "default_strategy")

        self._write_episodic_entry(
            memory_state,
            thread_id=thread_id,
            user_input=user_input,
            dag_shape_signature=dag_shape_signature,
            nodes=nodes,
            edges=edges,
            branch_results=branch_results,
            final_reply=final_reply,
            now=now,
        )
        self._update_procedural_memory(
            memory_state,
            intent_key=intent_key,
            strategy_name=strategy_name,
            branch_results=branch_results,
            nodes=nodes,
        )
        self._update_structural_memory(memory_state, dag_shape_signature=dag_shape_signature, branch_results=branch_results)
        self._update_semantic_memory(
            memory_state,
            thread_id=thread_id,
            dag_shape_signature=dag_shape_signature,
            intent_key=intent_key,
            now=now,
        )
        self._update_runtime_fragments_and_goals(
            runtime,
            thread_state,
            memory_state,
            nodes=nodes,
            final_reply=final_reply,
            dag_shape_signature=dag_shape_signature,
            user_input=user_input,
            branch_results=branch_results,
        )
        self._persist_turn_summary(
            runtime,
            thread_id=thread_id,
            final_reply=final_reply,
            dag_shape_signature=dag_shape_signature,
            strategy_name=strategy_name,
        )
        return None

    # -- Private helpers for consolidate_turn_outcome_owned ----------------------

    def _write_episodic_entry(
        self,
        memory_state: dict,
        *,
        thread_id: str,
        user_input: str,
        dag_shape_signature: str,
        nodes: list,
        edges: list,
        branch_results: dict,
        final_reply: str,
        now: int,
    ) -> None:
        episodic = list(memory_state.get("episodic") or [])
        episodic.append(
            {
                "turn_id": thread_id,
                "user_input": user_input,
                "dag_outcome": {
                    "dag_shape_signature": dag_shape_signature,
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                    "reply_hash": _stable_hash(final_reply),
                },
                "branch_results": dict(branch_results),
                "execution_summary": final_reply[:200],
                "timestamp": now,
            }
        )
        memory_state["episodic"] = episodic

    def _update_procedural_memory(
        self,
        memory_state: dict,
        *,
        intent_key: str,
        strategy_name: str,
        branch_results: dict,
        nodes: list,
    ) -> None:
        procedural = list(memory_state.get("procedural") or [])
        existing_proc = next((item for item in procedural if str(item.get("strategy_name") or "") == strategy_name), None)
        if existing_proc is None:
            existing_proc = {"strategy_name": strategy_name, "success_count": 0, "tool_patterns": {}}
            procedural.append(existing_proc)
        existing_proc["success_count"] = int(existing_proc.get("success_count") or 0) + 1
        tool_patterns = dict(existing_proc.get("tool_patterns") or {})
        for node in nodes:
            tool_name = str(node.get("tool_name") or "").strip()
            if tool_name:
                tool_patterns[tool_name] = max(0.8, float(tool_patterns.get(tool_name, 0.0)))
        existing_proc["tool_patterns"] = tool_patterns
        memory_state["procedural"] = procedural

    def _update_structural_memory(self, memory_state: dict, *, dag_shape_signature: str, branch_results: dict) -> None:
        structural = list(memory_state.get("structural") or [])
        existing_structural = next(
            (item for item in structural if str(item.get("dag_shape_signature") or "") == dag_shape_signature), None
        )
        if existing_structural is None:
            existing_structural = {
                "dag_shape_signature": dag_shape_signature,
                "success_correlation": 0.0,
                "branching_behavior": "high_success" if branch_results else "single_path",
                "turn_count": 0,
            }
            structural.append(existing_structural)
        existing_structural["turn_count"] = int(existing_structural.get("turn_count") or 0) + 1
        existing_structural["success_correlation"] = min(1.0, 0.5 + 0.1 * float(existing_structural.get("turn_count") or 0))
        memory_state["structural"] = structural

    def _update_semantic_memory(
        self,
        memory_state: dict,
        *,
        thread_id: str,
        dag_shape_signature: str,
        intent_key: str,
        now: int,
    ) -> None:
        semantic_memory = dict(memory_state.get("semantic_memory") or {})
        plan_history = list(semantic_memory.get("plan_history") or [])
        plan_history.append({"thread_id": thread_id, "dag_shape_signature": dag_shape_signature, "timestamp": now})
        semantic_memory["plan_history"] = plan_history
        strategy_bias = dict(semantic_memory.get("strategy_bias") or {})
        intent_bias = dict(strategy_bias.get(intent_key) or {})
        intent_bias["reuse_verified_nodes"] = True
        strategy_bias[intent_key] = intent_bias
        semantic_memory["strategy_bias"] = strategy_bias
        memory_state["semantic_memory"] = semantic_memory

    def _update_runtime_fragments_and_goals(
        self,
        runtime: Any,
        thread_state: Any,
        memory_state: dict,
        *,
        nodes: list,
        final_reply: str,
        dag_shape_signature: str,
        user_input: str,
        branch_results: dict,
    ) -> None:
        if runtime is None or thread_state is None:
            return
        fragments = dict(thread_state.dag_state.get("fragments") or {})
        for node in nodes:
            signature = str(node.get("signature") or node_signature(node))
            fragments[signature] = {
                "status": "done",
                "result": str(node.get("result") or final_reply),
                "output_hash": str(node.get("output_hash") or _stable_hash(str(node.get("result") or final_reply))),
            }
        thread_state.dag_state["fragments"] = fragments
        thread_state.dag_state["last_dag_signature"] = dag_shape_signature
        thread_state.dag_state["turn_count"] = int(thread_state.dag_state.get("turn_count") or 0) + 1

        goal_manager = runtime._goal_reward_manager
        goal_economy = goal_manager._load_goal_economy(thread_state=thread_state)
        if not list(goal_economy.goals or []):
            generated_goals = goal_manager._generate_goals_from_memory(user_text=user_input, thread_state=thread_state)
            goal_economy.goals = list(generated_goals)
            goal_economy.active_goal_ids = [str(item.get("goal_id") or "") for item in generated_goals if str(item.get("goal_id") or "")]
        selected_goal_ids = list(goal_economy.active_goal_ids[:1] or [])
        if branch_results.get("selected_candidate_id"):
            selected_goal_ids = selected_goal_ids or [str(branch_results.get("selected_candidate_id") or "")]
        goal_economy = goal_manager._retire_or_promote_goals(goal_economy=goal_economy, selected_goal_ids=selected_goal_ids)
        memory_state["goal_economy"] = GoalAndRewardManager._serialize_goal_economy(goal_economy)

    def _persist_turn_summary(
        self,
        runtime: Any,
        *,
        thread_id: str,
        final_reply: str,
        dag_shape_signature: str,
        strategy_name: str,
    ) -> None:
        if runtime is not None and hasattr(runtime.services, "write_memory"):
            runtime.services.write_memory(
                thread_id=thread_id,
                payload={
                    "summary": final_reply[:200],
                    "dag_shape_signature": dag_shape_signature,
                    "strategy_name": strategy_name,
                },
            )
