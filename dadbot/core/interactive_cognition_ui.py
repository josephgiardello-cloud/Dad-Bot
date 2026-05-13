from __future__ import annotations

import time
from typing import Any


class InteractiveCognitionUI:
    """Real-time cognition stream with editable plans and live control directives."""

    def emit_thought(
        self,
        *,
        state: dict[str, Any],
        trace_id: str,
        content: str,
        confidence: float = 0.5,
        category: str = "reasoning",
    ) -> dict[str, Any]:
        store = dict(state.get("interactive_cognition_ui") or {})
        thoughts = [dict(item) for item in list(store.get("thought_stream") or []) if isinstance(item, dict)]
        event = {
            "trace_id": str(trace_id or ""),
            "content": str(content or "").strip(),
            "confidence": max(0.0, min(1.0, float(confidence))),
            "category": str(category or "reasoning"),
            "timestamp": float(time.time()),
        }
        if event["content"]:
            thoughts.append(event)
        store["thought_stream"] = thoughts[-512:]
        store["updated_at"] = float(time.time())
        state["interactive_cognition_ui"] = store
        return event

    def register_plan(
        self,
        *,
        state: dict[str, Any],
        trace_id: str,
        runtime_plan: dict[str, Any],
        source: str,
    ) -> dict[str, Any]:
        store = dict(state.get("interactive_cognition_ui") or {})
        plans = [dict(item) for item in list(store.get("plan_history") or []) if isinstance(item, dict)]
        plan_event = {
            "trace_id": str(trace_id or ""),
            "source": str(source or "runtime"),
            "plan": dict(runtime_plan or {}),
            "timestamp": float(time.time()),
        }
        plans.append(plan_event)
        store["plan_history"] = plans[-256:]
        store["active_plan"] = dict(runtime_plan or {})
        store["updated_at"] = float(time.time())
        state["interactive_cognition_ui"] = store
        return plan_event

    def apply_plan_edit(
        self,
        *,
        state: dict[str, Any],
        trace_id: str,
        edits: dict[str, Any],
        actor: str,
    ) -> dict[str, Any]:
        store = dict(state.get("interactive_cognition_ui") or {})
        active_plan = dict(store.get("active_plan") or {})
        if not active_plan:
            active_plan = {"status": "active", "revision": 1, "subgoals": []}

        for key in ("strategy", "status", "intent_type"):
            if key in edits:
                active_plan[key] = edits[key]
        if "subgoals" in edits and isinstance(edits.get("subgoals"), list):
            active_plan["subgoals"] = [item for item in list(edits.get("subgoals") or [])]
        active_plan["revision"] = int(active_plan.get("revision") or 1) + 1
        active_plan["updated_at"] = float(time.time())

        edit_events = [dict(item) for item in list(store.get("plan_edits") or []) if isinstance(item, dict)]
        edit_event = {
            "trace_id": str(trace_id or ""),
            "actor": str(actor or "operator"),
            "edits": dict(edits or {}),
            "timestamp": float(time.time()),
            "revision": int(active_plan.get("revision") or 1),
        }
        edit_events.append(edit_event)

        store["active_plan"] = active_plan
        store["plan_edits"] = edit_events[-256:]
        store["updated_at"] = float(time.time())
        state["interactive_cognition_ui"] = store
        return active_plan

    def apply_live_control(
        self,
        *,
        state: dict[str, Any],
        control: dict[str, Any],
        source: str,
    ) -> dict[str, Any]:
        store = dict(state.get("interactive_cognition_ui") or {})
        controls = [dict(item) for item in list(store.get("live_controls") or []) if isinstance(item, dict)]
        event = {
            "source": str(source or "operator"),
            "control": dict(control or {}),
            "timestamp": float(time.time()),
        }
        controls.append(event)
        store["live_controls"] = controls[-512:]
        store["pause_requested"] = bool(dict(control or {}).get("pause_requested", False))
        store["updated_at"] = float(time.time())
        state["interactive_cognition_ui"] = store
        return event

    def snapshot(self, *, state: dict[str, Any], limit: int = 20) -> dict[str, Any]:
        store = dict(state.get("interactive_cognition_ui") or {})
        thoughts = [dict(item) for item in list(store.get("thought_stream") or []) if isinstance(item, dict)]
        edits = [dict(item) for item in list(store.get("plan_edits") or []) if isinstance(item, dict)]
        controls = [dict(item) for item in list(store.get("live_controls") or []) if isinstance(item, dict)]
        max_items = max(1, int(limit))
        return {
            "active_plan": dict(store.get("active_plan") or {}),
            "pause_requested": bool(store.get("pause_requested", False)),
            "recent_thoughts": thoughts[-max_items:],
            "recent_plan_edits": edits[-max_items:],
            "recent_live_controls": controls[-max_items:],
            "updated_at": float(store.get("updated_at") or 0.0),
        }
