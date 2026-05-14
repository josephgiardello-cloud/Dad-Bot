from __future__ import annotations

import time
from typing import Any


class SessionPlanningOptimizer:
    """Cross-session long-horizon plan optimizer."""

    def record_plan_outcome(
        self,
        *,
        state: dict[str, Any],
        plan: dict[str, Any],
        success: bool,
    ) -> dict[str, Any]:
        store = dict(state.get("session_planning_optimizer") or {})
        history = list(store.get("history") or [])
        history.append(
            {
                "plan_id": str(plan.get("plan_id") or ""),
                "strategy": str(plan.get("strategy") or "direct_answer"),
                "intent_type": str(plan.get("intent_type") or "statement"),
                "success": bool(success),
                "revision": int(plan.get("revision") or 1),
                "timestamp": float(time.time()),
            },
        )
        history = history[-512:]

        by_strategy: dict[str, list[dict[str, Any]]] = {}
        for item in history:
            key = str(item.get("strategy") or "direct_answer")
            by_strategy.setdefault(key, []).append(item)

        strategy_scores: dict[str, float] = {}
        for key, rows in by_strategy.items():
            score = float(sum(1 for row in rows if bool(row.get("success")))) / float(max(1, len(rows)))
            strategy_scores[key] = round(float(score), 6)

        preferred = "direct_answer"
        if strategy_scores:
            preferred = max(strategy_scores.items(), key=lambda pair: pair[1])[0]

        store["history"] = history
        store["strategy_scores"] = strategy_scores
        store["preferred_strategy"] = preferred
        store["updated_at"] = float(time.time())
        state["session_planning_optimizer"] = store
        return store

    def suggest(self, *, state: dict[str, Any], intent_type: str) -> str:
        store = dict(state.get("session_planning_optimizer") or {})
        preferred = str(store.get("preferred_strategy") or "").strip()
        if preferred:
            return preferred
        intent = str(intent_type or "").strip().lower()
        if intent == "request":
            return "task_execution"
        if intent == "question":
            return "grounded_answer"
        return "direct_answer"
