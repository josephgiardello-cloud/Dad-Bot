from __future__ import annotations

import time
from typing import Any

from dadbot.core.write_plane import get_write_plane


class BeliefStateEngine:
    """Persistent probabilistic belief state across turns."""

    def update_from_turn(
        self,
        *,
        state: dict[str, Any],
        trace_id: str,
        user_input: str,
        runtime_plan: dict[str, Any],
        success: bool,
    ) -> dict[str, Any]:
        belief = dict(state.get("belief_state") or {})
        priors = dict(belief.get("priors") or {})
        outcomes = list(belief.get("outcomes") or [])

        uncertainty = float(dict(runtime_plan.get("uncertainty") or {}).get("score") or 0.0)
        confidence = max(0.0, min(1.0, 1.0 - uncertainty))

        intent = str(runtime_plan.get("intent_type") or "statement")
        strategy = str(runtime_plan.get("strategy") or "direct_answer")

        key = f"{intent}:{strategy}"
        prior = float(priors.get(key) or 0.5)
        evidence = 0.08 if success else -0.12
        calibrated = max(0.0, min(1.0, prior + evidence + (0.04 * (confidence - 0.5))))
        priors[key] = round(float(calibrated), 6)

        outcomes.append(
            {
                "trace_id": str(trace_id or ""),
                "intent": intent,
                "strategy": strategy,
                "success": bool(success),
                "uncertainty": float(uncertainty),
                "confidence": float(confidence),
                "input_length": int(len(str(user_input or ""))),
                "timestamp": float(time.time()),
            },
        )

        recent = outcomes[-512:]
        success_rate = float(sum(1 for row in recent if bool(row.get("success")))) / float(max(1, len(recent)))

        belief["priors"] = priors
        belief["outcomes"] = recent
        belief["calibration"] = {
            "success_rate": round(float(success_rate), 6),
            "sample_size": int(len(recent)),
            "updated_at": float(time.time()),
        }
        get_write_plane().write("BeliefStateEngine", "memory.belief_state", belief)
        state["belief_state"] = belief
        return belief

    def next_best_strategy(self, *, state: dict[str, Any], intent_type: str) -> str:
        belief = dict(state.get("belief_state") or {})
        priors = dict(belief.get("priors") or {})
        intent = str(intent_type or "statement")
        candidates = [
            (f"{intent}:grounded_answer", "grounded_answer"),
            (f"{intent}:task_execution", "task_execution"),
            (f"{intent}:clarify_before_action", "clarify_before_action"),
            (f"{intent}:direct_answer", "direct_answer"),
        ]
        ranked = sorted(candidates, key=lambda item: float(priors.get(item[0]) or 0.5), reverse=True)
        return str(ranked[0][1] if ranked else "direct_answer")
