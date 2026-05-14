from __future__ import annotations

import time
from typing import Any


class AutonomousGoalDaemon:
    """Persistent autonomous goal loop planner for background execution."""

    def next_actions(self, *, state: dict[str, Any], max_items: int = 3) -> list[dict[str, Any]]:
        goals = [dict(item) for item in list(state.get("goals") or []) if isinstance(item, dict)]
        pending: list[dict[str, Any]] = []
        now = float(time.time())
        for goal in goals:
            status = str(goal.get("status") or "active")
            if status not in {"active", "pending"}:
                continue
            desc = str(goal.get("description") or "").strip()
            if not desc:
                continue
            pending.append(
                {
                    "goal_id": str(goal.get("id") or ""),
                    "description": desc,
                    "action": f"advance:{desc[:80]}",
                    "priority": str(goal.get("priority") or "medium"),
                    "scheduled_at": now,
                },
            )
        pending.sort(key=lambda item: {"high": 0, "medium": 1, "low": 2}.get(str(item.get("priority") or "medium"), 3))
        return pending[: max(1, int(max_items))]

    def persist_cycle(
        self,
        *,
        state: dict[str, Any],
        actions: list[dict[str, Any]],
        source: str,
    ) -> dict[str, Any]:
        store = dict(state.get("autonomous_goal_loop") or {})
        cycles = list(store.get("cycles") or [])
        cycles.append(
            {
                "timestamp": float(time.time()),
                "source": str(source or "control_plane"),
                "actions": [dict(item) for item in list(actions or [])],
            },
        )
        store["cycles"] = cycles[-256:]
        store["last_actions"] = [dict(item) for item in list(actions or [])]
        store["updated_at"] = float(time.time())
        state["autonomous_goal_loop"] = store
        return store
