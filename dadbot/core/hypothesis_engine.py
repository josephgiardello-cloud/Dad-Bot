from __future__ import annotations

import time
from typing import Any


class MultiHypothesisEngine:
    """Maintains alternative hypotheses and confidence for a turn/session."""

    def infer_hypotheses(
        self,
        *,
        user_input: str,
        runtime_plan: dict[str, Any],
        tool_routing_plan: dict[str, Any],
        memory_context: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        text = str(user_input or "").strip().lower()
        intent = str(runtime_plan.get("intent_type") or "statement")
        uncertainty = float(dict(runtime_plan.get("uncertainty") or {}).get("score") or 0.0)
        memory_hits = int(len(memory_context or []))
        has_tools = bool(list(tool_routing_plan.get("candidates") or []))

        hypotheses: list[dict[str, Any]] = [
            {
                "id": "h1",
                "label": "direct_intent_execution",
                "confidence": round(float(max(0.05, 0.8 - uncertainty)), 6),
                "assumptions": ["intent_detected", f"intent={intent}"],
                "strategy": str(runtime_plan.get("strategy") or "direct_answer"),
            },
            {
                "id": "h2",
                "label": "clarification_required",
                "confidence": round(float(min(0.95, 0.2 + uncertainty + (0.1 if len(text) < 20 else 0.0))), 6),
                "assumptions": ["ambiguity_possible", "missing_context"],
                "strategy": "clarify_before_action",
            },
        ]

        if memory_hits > 0:
            hypotheses.append(
                {
                    "id": "h3",
                    "label": "memory_grounded_response",
                    "confidence": round(float(min(0.95, 0.35 + (0.12 * min(4, memory_hits)))), 6),
                    "assumptions": ["relevant_memory_available"],
                    "strategy": "grounded_answer",
                },
            )

        if has_tools:
            hypotheses.append(
                {
                    "id": "h4",
                    "label": "tool_grounded_execution",
                    "confidence": round(float(max(0.1, 0.65 - (0.4 * uncertainty))), 6),
                    "assumptions": ["tool_capability_exists"],
                    "strategy": "task_execution",
                },
            )

        hypotheses.sort(key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
        return hypotheses[:5]

    def persist(
        self,
        *,
        state: dict[str, Any],
        trace_id: str,
        hypotheses: list[dict[str, Any]],
    ) -> dict[str, Any]:
        store = dict(state.get("hypothesis_store") or {})
        entries = list(store.get("entries") or [])
        active = dict(hypotheses[0] if hypotheses else {})
        entries.append(
            {
                "trace_id": str(trace_id or ""),
                "active": active,
                "alternatives": [dict(item) for item in list(hypotheses or [])],
                "timestamp": float(time.time()),
            },
        )
        store["entries"] = entries[-256:]
        store["last_active"] = active
        store["updated_at"] = float(time.time())
        state["hypothesis_store"] = store
        return store
