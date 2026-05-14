from __future__ import annotations

import time
from typing import Any


class MultiAgentSwarm:
    """Distributed swarm planning and coordination state for large-scale execution."""

    def build_plan(
        self,
        *,
        state: dict[str, Any],
        trace_id: str,
        user_input: str,
        runtime_plan: dict[str, Any],
        compositional_tool_plan: dict[str, Any],
        max_agents: int = 4,
    ) -> dict[str, Any]:
        steps = [dict(item) for item in list(compositional_tool_plan.get("steps") or []) if isinstance(item, dict)]
        capacity = max(1, int(max_agents))
        assignments: list[dict[str, Any]] = []
        for index, step in enumerate(steps[:capacity]):
            assignments.append(
                {
                    "agent_id": f"agent-{index + 1}",
                    "role": str(step.get("role") or "specialist"),
                    "tool_name": str(step.get("tool_name") or ""),
                    "task": str(step.get("task") or step.get("intent") or "execute_step"),
                    "status": "planned",
                },
            )

        if not assignments:
            assignments.append(
                {
                    "agent_id": "agent-1",
                    "role": "coordinator",
                    "tool_name": "",
                    "task": f"reason_over:{str(user_input or '')[:80]}",
                    "status": "planned",
                },
            )

        plan = {
            "trace_id": str(trace_id or ""),
            "intent_type": str(runtime_plan.get("intent_type") or "statement"),
            "coordinator": "agent-0",
            "assignments": assignments,
            "created_at": float(time.time()),
            "status": "planned",
        }
        store = dict(state.get("swarm") or {})
        history = [dict(item) for item in list(store.get("plans") or []) if isinstance(item, dict)]
        history.append(plan)
        store["plans"] = history[-256:]
        store["active_plan"] = plan
        store["updated_at"] = float(time.time())
        state["swarm"] = store
        return plan

    def record_outcome(
        self,
        *,
        state: dict[str, Any],
        trace_id: str,
        success: bool,
        latency_ms: float,
    ) -> dict[str, Any]:
        store = dict(state.get("swarm") or {})
        outcomes = [dict(item) for item in list(store.get("outcomes") or []) if isinstance(item, dict)]
        event = {
            "trace_id": str(trace_id or ""),
            "success": bool(success),
            "latency_ms": max(0.0, float(latency_ms)),
            "timestamp": float(time.time()),
        }
        outcomes.append(event)
        store["outcomes"] = outcomes[-512:]
        store["updated_at"] = float(time.time())
        state["swarm"] = store
        return event

    def health_snapshot(self, *, state: dict[str, Any]) -> dict[str, Any]:
        store = dict(state.get("swarm") or {})
        outcomes = [dict(item) for item in list(store.get("outcomes") or []) if isinstance(item, dict)]
        plans = [dict(item) for item in list(store.get("plans") or []) if isinstance(item, dict)]
        # A submitted turn can build a swarm plan before an outcome is persisted.
        # Treat that as an observed run for real-time diagnostics.
        total = int(max(len(outcomes), len(plans)))
        successes = int(sum(1 for item in outcomes if bool(item.get("success", False))))
        success_rate = float(successes) / float(max(1, total))
        return {
            "total_runs": total,
            "success_rate": round(success_rate, 6),
            "active_plan": dict(store.get("active_plan") or {}),
            "updated_at": float(store.get("updated_at") or 0.0),
        }
