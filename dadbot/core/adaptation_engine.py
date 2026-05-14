from __future__ import annotations

import time
from typing import Any


class AdaptationEngine:
    """Online adaptation from execution outcomes and user feedback."""

    def record_outcome(
        self,
        *,
        state: dict[str, Any],
        trace_id: str,
        strategy: str,
        success: bool,
        uncertainty_score: float,
        explicit_feedback: float | None = None,
    ) -> dict[str, Any]:
        learning = dict(state.get("learning_profile") or {})
        policy = dict(learning.get("policy") or {})
        outcomes = list(learning.get("outcomes") or [])

        outcomes.append(
            {
                "trace_id": str(trace_id or ""),
                "strategy": str(strategy or "unknown"),
                "success": bool(success),
                "uncertainty_score": float(max(0.0, min(1.0, uncertainty_score))),
                "explicit_feedback": None if explicit_feedback is None else float(max(-1.0, min(1.0, explicit_feedback))),
                "timestamp": float(time.time()),
            },
        )

        recent = outcomes[-256:]
        success_rate = (
            float(sum(1 for row in recent if bool(row.get("success")))) / float(max(1, len(recent)))
        )

        by_strategy: dict[str, list[dict[str, Any]]] = {}
        for row in recent:
            key = str(row.get("strategy") or "unknown")
            by_strategy.setdefault(key, []).append(row)

        strategy_scores: dict[str, float] = {}
        for key, rows in by_strategy.items():
            strat_success = float(sum(1 for row in rows if bool(row.get("success")))) / float(max(1, len(rows)))
            feedback_values = [float(row.get("explicit_feedback")) for row in rows if row.get("explicit_feedback") is not None]
            feedback_bonus = (sum(feedback_values) / len(feedback_values)) if feedback_values else 0.0
            strategy_scores[key] = float(max(0.0, min(1.0, (0.85 * strat_success) + (0.15 * ((feedback_bonus + 1.0) / 2.0)))))

        preferred = max(strategy_scores.items(), key=lambda item: item[1])[0] if strategy_scores else "direct_answer"

        policy["success_rate"] = float(round(success_rate, 6))
        policy["strategy_scores"] = {k: round(float(v), 6) for k, v in strategy_scores.items()}
        policy["preferred_strategy"] = str(preferred)
        policy["updated_at"] = float(time.time())

        learning["policy"] = policy
        learning["outcomes"] = recent
        state["learning_profile"] = learning
        return learning

    def suggest_adjustments(self, *, state: dict[str, Any]) -> dict[str, Any]:
        learning = dict(state.get("learning_profile") or {})
        policy = dict(learning.get("policy") or {})
        success_rate = float(policy.get("success_rate") or 0.0)
        preferred_strategy = str(policy.get("preferred_strategy") or "direct_answer")
        action = "keep"
        if success_rate < 0.55:
            action = "increase_clarification"
        elif success_rate < 0.7:
            action = "increase_tool_grounding"
        return {
            "action": action,
            "success_rate": success_rate,
            "preferred_strategy": preferred_strategy,
            "updated_at": float(policy.get("updated_at") or 0.0),
        }
