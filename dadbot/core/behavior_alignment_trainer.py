from __future__ import annotations

import time
from typing import Any


class BehaviorAlignmentTrainer:
    """Large-scale alignment trainer using reward-style policy shaping."""

    def record_signal(
        self,
        *,
        state: dict[str, Any],
        trace_id: str,
        runtime_plan: dict[str, Any],
        success: bool,
        response_text: str,
        explicit_feedback: dict[str, Any] | None,
    ) -> dict[str, Any]:
        store = dict(state.get("alignment_trainer") or {})
        events = [dict(item) for item in list(store.get("signals") or []) if isinstance(item, dict)]
        strategy = str(runtime_plan.get("strategy") or "direct_answer")
        feedback = dict(explicit_feedback or {})
        reward = self._reward(
            success=bool(success),
            response_text=str(response_text or ""),
            feedback=feedback,
        )
        event = {
            "trace_id": str(trace_id or ""),
            "strategy": strategy,
            "intent_type": str(runtime_plan.get("intent_type") or "statement"),
            "reward": float(reward),
            "success": bool(success),
            "feedback": feedback,
            "timestamp": float(time.time()),
        }
        events.append(event)
        store["signals"] = events[-4096:]
        store["updated_at"] = float(time.time())
        state["alignment_trainer"] = store
        return event

    def update_policy(self, *, state: dict[str, Any]) -> dict[str, Any]:
        store = dict(state.get("alignment_trainer") or {})
        events = [dict(item) for item in list(store.get("signals") or []) if isinstance(item, dict)]
        strategy_rewards: dict[str, list[float]] = {}
        for event in events:
            strategy = str(event.get("strategy") or "direct_answer")
            strategy_rewards.setdefault(strategy, []).append(float(event.get("reward") or 0.0))

        policy: dict[str, dict[str, Any]] = {}
        for strategy, rewards in strategy_rewards.items():
            if not rewards:
                continue
            mean_reward = float(sum(rewards) / max(1, len(rewards)))
            policy[strategy] = {
                "mean_reward": round(mean_reward, 6),
                "samples": int(len(rewards)),
            }

        store["policy"] = policy
        store["updated_at"] = float(time.time())
        state["alignment_trainer"] = store
        return policy

    def recommend_strategy(
        self,
        *,
        state: dict[str, Any],
        intent_type: str,
        default_strategy: str,
    ) -> str:
        store = dict(state.get("alignment_trainer") or {})
        policy = dict(store.get("policy") or {})
        if not policy:
            return str(default_strategy or "direct_answer")

        baseline = str(default_strategy or "direct_answer")
        best_strategy = baseline
        best_reward = float(dict(policy.get(baseline) or {}).get("mean_reward") or -1.0)
        for strategy, row in policy.items():
            mean_reward = float(dict(row or {}).get("mean_reward") or -1.0)
            samples = int(dict(row or {}).get("samples") or 0)
            # Conservative policy updates require enough history.
            if samples < 5:
                continue
            if mean_reward > best_reward + 0.05:
                best_strategy = str(strategy)
                best_reward = mean_reward
        return best_strategy

    @staticmethod
    def _reward(*, success: bool, response_text: str, feedback: dict[str, Any]) -> float:
        reward = 1.0 if success else -1.0
        if len(str(response_text or "").strip()) > 24:
            reward += 0.15
        rating = float(feedback.get("rating") or 0.0)
        reward += max(-1.0, min(1.0, rating)) * 0.5
        if bool(feedback.get("unsafe", False)):
            reward -= 1.0
        return max(-2.0, min(2.0, reward))
