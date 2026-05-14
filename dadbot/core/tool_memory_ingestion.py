from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _safe_preview(value: Any, *, max_chars: int = 500) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


@dataclass
class ToolResultMemorySink:
    """Bridge tool execution events into long-term memory for future retrieval."""

    memory_service: Any

    def __call__(self, event: dict[str, Any]) -> None:
        memory = self.memory_service
        if memory is None:
            return
        tool_name = str(event.get("tool_name") or "")
        status = str(event.get("status") or "")
        captured_at_ms = int(event.get("captured_at_ms") or 0)

        payload = {
            "type": "TOOL_EXECUTION_RESULT",
            "event_id": f"tool:{tool_name}:{captured_at_ms}",
            "tool_name": tool_name,
            "status": status,
            "attempts": int(event.get("attempts") or 0),
            "latency_ms": float(event.get("latency_ms") or 0.0),
            "degraded_reason": str(event.get("degraded_reason") or ""),
            "error": str(event.get("error") or ""),
            "output_preview": _safe_preview(event.get("output_preview")),
            "metadata": dict(event.get("metadata") or {}),
            "captured_at_ms": captured_at_ms,
        }

        if hasattr(memory, "commit_to_long_term"):
            memory.commit_to_long_term(
                turn_id=f"tool:{tool_name}:{captured_at_ms}",
                event_payload=payload,
                metadata={
                    "event_type": "TOOL_EXECUTION_RESULT",
                    "tool_name": tool_name,
                    "status": status,
                },
            )


def build_default_tool_memory_sink() -> ToolResultMemorySink | None:
    """Create a sink bound to global SovereignMemory when available."""
    try:
        from dadbot.services.vector_memory import get_global_memory
    except Exception:
        return None

    memory = get_global_memory()
    if memory is None:
        return None
    return ToolResultMemorySink(memory_service=memory)
