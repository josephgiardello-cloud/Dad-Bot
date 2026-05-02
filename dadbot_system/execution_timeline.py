from __future__ import annotations

from typing import Any


class ExecutionTimelineBuilder:
    """Derived timeline view built only from persisted event history."""

    @staticmethod
    def build(events: list[dict[str, Any]]) -> dict[str, Any]:
        ordered = sorted(
            [dict(item or {}) for item in list(events or [])],
            key=lambda item: (
                int(item.get("sequence_id") or 0),
                int(dict(item.get("payload") or {}).get("ordering_index") or 0),
                str(item.get("event_time") or ""),
                str(item.get("event_id") or ""),
            ),
        )

        node_trace: list[dict[str, Any]] = []
        tool_timeline: list[dict[str, Any]] = []
        memory_history: list[dict[str, Any]] = []
        turn_lifecycle: list[dict[str, Any]] = []

        for event in ordered:
            event_type = str(event.get("type") or event.get("event_type") or "").strip().upper()
            payload = dict(event.get("payload") or {})
            entry = {
                "sequence_id": int(event.get("sequence_id") or 0),
                "event_id": str(event.get("event_id") or ""),
                "type": event_type,
                "event_time": str(event.get("event_time") or ""),
                "payload": payload,
            }
            if event_type in {"NODE_ENTER", "NODE_EXIT"}:
                node_trace.append(entry)
            elif event_type in {"TOOL_CALL_START", "TOOL_CALL_END"}:
                tool_timeline.append(entry)
            elif event_type in {"MEMORY_READ", "MEMORY_WRITE"}:
                memory_history.append(entry)
            elif event_type in {"TURN_START", "TURN_END", "RECOVERY_REPLAY_START", "RECOVERY_REPLAY_END"}:
                turn_lifecycle.append(entry)

        return {
            "event_count": len(ordered),
            "turn_lifecycle": turn_lifecycle,
            "node_trace": node_trace,
            "tool_timeline": tool_timeline,
            "memory_access_history": memory_history,
        }
