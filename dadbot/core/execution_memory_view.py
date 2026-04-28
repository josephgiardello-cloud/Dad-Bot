from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any



def _stable_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ExecutionMemoryView:
    """Execution-scoped memory snapshot used for deterministic replay."""

    state_id: str
    memory_structured: dict[str, Any]
    memory_full_history_id: str
    memory_retrieval_set: list[dict[str, Any]]

    @classmethod
    def from_context(cls, context: Any) -> "ExecutionMemoryView":
        state = dict(getattr(context, "state", {}) or {})
        structured = dict(state.get("memory_structured") or {})
        history_id = str(state.get("memory_full_history_id") or "")
        retrieval = [
            item if isinstance(item, dict) else {"value": item}
            for item in list(state.get("memory_retrieval_set") or [])
        ]
        state_id = _stable_sha256(
            {
                "memory_structured": structured,
                "memory_full_history_id": history_id,
                "memory_retrieval_set": retrieval,
            }
        )
        return cls(
            state_id=f"memv-{state_id[:16]}",
            memory_structured=structured,
            memory_full_history_id=history_id,
            memory_retrieval_set=retrieval,
        )

    @classmethod
    def from_trace(
        cls,
        execution_trace_context: dict[str, Any],
        *,
        fallback_memory_view: dict[str, Any] | None = None,
    ) -> "ExecutionMemoryView":
        fallback = dict(fallback_memory_view or {})
        memory_snapshot = dict(execution_trace_context.get("memory_snapshot_used") or {})

        structured = dict(
            memory_snapshot.get("memory_structured")
            or fallback.get("memory_structured")
            or {}
        )
        history_id = str(
            memory_snapshot.get("memory_full_history_id")
            or fallback.get("memory_full_history_id")
            or ""
        )
        retrieval = [
            item if isinstance(item, dict) else {"value": item}
            for item in list(
                execution_trace_context.get("memory_retrieval_set")
                or fallback.get("memory_retrieval_set")
                or []
            )
        ]

        state_id = _stable_sha256(
            {
                "memory_structured": structured,
                "memory_full_history_id": history_id,
                "memory_retrieval_set": retrieval,
            }
        )
        return cls(
            state_id=f"memv-{state_id[:16]}",
            memory_structured=structured,
            memory_full_history_id=history_id,
            memory_retrieval_set=retrieval,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "memory_structured": dict(self.memory_structured),
            "memory_full_history_id": self.memory_full_history_id,
            "memory_retrieval_set": list(self.memory_retrieval_set),
        }
