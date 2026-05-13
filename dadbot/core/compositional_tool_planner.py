from __future__ import annotations

from typing import Any

from dadbot.core.runtime_types import ToolSpec


class CompositionalToolPlanner:
    """Builds multi-tool execution strategy from available tool specs."""

    def build_plan(
        self,
        *,
        user_input: str,
        routing_plan: dict[str, Any],
        available_specs: list[ToolSpec],
    ) -> dict[str, Any]:
        text = str(user_input or "").lower()
        candidates = [dict(item) for item in list(routing_plan.get("candidates") or []) if isinstance(item, dict)]
        index = {str(spec.name or ""): spec for spec in list(available_specs or []) if isinstance(spec, ToolSpec)}

        selected: list[dict[str, Any]] = []
        for item in candidates[:4]:
            tool_name = str(item.get("tool_name") or "")
            spec = index.get(tool_name)
            if spec is None:
                continue
            selected.append(
                {
                    "tool_name": tool_name,
                    "step": len(selected) + 1,
                    "intent": self._infer_step_intent(tool_name=tool_name, user_input=text),
                    "requires_approval": bool(spec.has_side_effects()),
                    "idempotent": bool(spec.is_idempotent()),
                },
            )

        mode = "single" if len(selected) <= 1 else "composed"
        return {
            "mode": mode,
            "steps": selected,
            "estimated_steps": int(len(selected)),
            "requires_tool_execution": bool(len(selected) > 0),
        }

    def _infer_step_intent(self, *, tool_name: str, user_input: str) -> str:
        name = str(tool_name or "").lower()
        if any(token in name for token in ("search", "lookup", "memory")):
            return "retrieve_context"
        if any(token in name for token in ("write", "save", "commit", "update")):
            return "apply_mutation"
        if "plan" in user_input or "then" in user_input:
            return "multi_step_support"
        return "direct_support"
