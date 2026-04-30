"""Goal and reward management for the runtime memory subsystem."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GoalEconomy:
    goals: list[dict[str, Any]] = field(default_factory=list)
    active_goal_ids: list[str] = field(default_factory=list)
    reward_signals: list[dict[str, Any]] = field(default_factory=list)


class GoalAndRewardManager:
    """Manages goal lifecycle and reward accounting for a runtime session."""

    def _load_goal_economy(self, *, thread_state: Any) -> GoalEconomy:
        raw = getattr(thread_state, "memory_state", {}) or {}
        economy = raw.get("goal_economy") or {}
        return GoalEconomy(
            goals=list(economy.get("goals") or []),
            active_goal_ids=list(economy.get("active_goal_ids") or []),
            reward_signals=list(economy.get("reward_signals") or []),
        )

    def _generate_goals_from_memory(
        self,
        *,
        user_text: str,
        thread_state: Any,
    ) -> list[dict[str, Any]]:
        return [{"goal_id": f"inferred_goal_0", "description": (user_text or "")[:80]}]

    def _retire_or_promote_goals(
        self,
        *,
        goal_economy: GoalEconomy,
        selected_goal_ids: list[str],
    ) -> GoalEconomy:
        promoted = [
            g
            for g in goal_economy.goals
            if str(g.get("goal_id") or "") in selected_goal_ids
        ]
        goal_economy.active_goal_ids = [
            str(g.get("goal_id") or "") for g in promoted
        ] or goal_economy.active_goal_ids
        return goal_economy

    @classmethod
    def _serialize_goal_economy(cls, goal_economy: GoalEconomy) -> dict[str, Any]:
        return {
            "goals": list(goal_economy.goals),
            "active_goal_ids": list(goal_economy.active_goal_ids),
            "reward_signals": list(goal_economy.reward_signals),
        }


__all__ = ["GoalAndRewardManager", "GoalEconomy"]
