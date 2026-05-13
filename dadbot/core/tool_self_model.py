from __future__ import annotations

import time
from typing import Any


class ToolSelfModel:
    """Persistent tool reliability/failure self-model used by routing."""

    def record_observation(
        self,
        *,
        state: dict[str, Any],
        tool_name: str,
        success: bool,
        latency_ms: float = 0.0,
        failure_type: str = "",
    ) -> dict[str, Any]:
        model = dict(state.get("tool_self_model") or {})
        tools = dict(model.get("tools") or {})
        name = str(tool_name or "").strip().lower()
        if not name:
            return model

        row = dict(tools.get(name) or {})
        calls = int(row.get("calls") or 0) + 1
        ok = int(row.get("ok") or 0) + (1 if success else 0)
        fail = int(row.get("fail") or 0) + (0 if success else 1)
        prev_latency = float(row.get("avg_latency_ms") or 0.0)
        avg_latency = ((prev_latency * float(calls - 1)) + float(max(0.0, latency_ms))) / float(max(1, calls))
        reliability = float(ok) / float(max(1, calls))

        failures = dict(row.get("failure_types") or {})
        if not success and str(failure_type or "").strip():
            key = str(failure_type or "unknown")
            failures[key] = int(failures.get(key) or 0) + 1

        row.update(
            {
                "calls": int(calls),
                "ok": int(ok),
                "fail": int(fail),
                "reliability": round(float(reliability), 6),
                "avg_latency_ms": round(float(avg_latency), 4),
                "failure_types": failures,
                "updated_at": float(time.time()),
            },
        )
        tools[name] = row
        model["tools"] = tools
        model["updated_at"] = float(time.time())
        state["tool_self_model"] = model
        return row

    def reliability_for_tool(self, *, state: dict[str, Any], tool_name: str) -> float:
        model = dict(state.get("tool_self_model") or {})
        tools = dict(model.get("tools") or {})
        row = dict(tools.get(str(tool_name or "").strip().lower()) or {})
        return float(row.get("reliability") or 0.5)

    def apply_routing_feedback(
        self,
        *,
        state: dict[str, Any],
        routing_plan: dict[str, Any],
    ) -> dict[str, Any]:
        adjusted = dict(routing_plan or {})
        candidates = [dict(item) for item in list(adjusted.get("candidates") or []) if isinstance(item, dict)]
        for item in candidates:
            tool_name = str(item.get("tool_name") or "")
            base = float(item.get("score") or 0.0)
            reliability = self.reliability_for_tool(state=state, tool_name=tool_name)
            item["historical_reliability"] = round(float(reliability), 6)
            item["score"] = round(float((0.75 * base) + (0.25 * reliability)), 6)
        candidates.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
        adjusted["candidates"] = candidates
        adjusted["primary"] = dict(candidates[0]) if candidates else None
        adjusted["fallback"] = dict(candidates[1]) if len(candidates) > 1 else None
        return adjusted
