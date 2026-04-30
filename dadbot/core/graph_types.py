"""Execution trace types and mutation op enums for the TurnGraph pipeline.

Extracted from graph.py to keep that module below 1800 lines.
All types remain re-exported from ``dadbot.core.graph`` for backward compatibility.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


# ---------------------------------------------------------------------------
# Execution trace types
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


@dataclass(frozen=True)
class ExecutionTraceEvent:
    """Deterministic execution event used for replay equivalence checks."""

    sequence: int
    event_type: str
    stage: str
    phase: str
    trace_id: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": int(self.sequence),
            "event_type": str(self.event_type or ""),
            "stage": str(self.stage or ""),
            "phase": str(self.phase or ""),
            "trace_id": str(self.trace_id or ""),
            "detail": _json_safe(dict(self.detail or {})),
        }


@dataclass
class StageTrace:
    """Execution record for a single pipeline stage stored inside TurnContext."""

    stage: str
    duration_ms: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Mutation operation enums
# ---------------------------------------------------------------------------


class MutationKind(StrEnum):
    MEMORY = "memory"
    RELATIONSHIP = "relationship"
    GRAPH = "graph"
    LEDGER = "ledger"
    GOAL = "goal"


class MemoryMutationOp(StrEnum):
    SAVE_MOOD_STATE = "save_mood_state"


class RelationshipMutationOp(StrEnum):
    UPDATE = "update"


class LedgerMutationOp(StrEnum):
    APPEND_HISTORY = "append_history"
    RECORD_TURN_STATE = "record_turn_state"
    SYNC_THREAD_SNAPSHOT = "sync_thread_snapshot"
    CLEAR_TURN_CONTEXT = "clear_turn_context"
    SCHEDULE_MAINTENANCE = "schedule_maintenance"
    HEALTH_SNAPSHOT = "health_snapshot"
    CAPABILITY_AUDIT_EVENT = "capability_audit_event"


class GoalMutationOp(StrEnum):
    UPSERT_GOAL = "upsert_goal"
    COMPLETE_GOAL = "complete_goal"
    ABANDON_GOAL = "abandon_goal"


class NodeType(StrEnum):
    STANDARD = "standard"
    COMMIT = "commit"


class MutationTransactionStatus(StrEnum):
    COMMITTED = "committed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    ROLLBACK_FAILED = "rollback_failed"


@dataclass
class MutationTransactionRecord:
    transaction_id: str
    status: MutationTransactionStatus
    applied_count: int
    failed_count: int
    rollback_count: int
    rollback_failures: int
    trace_id: str
    error: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": str(self.transaction_id or ""),
            "status": str(self.status.value),
            "applied_count": int(self.applied_count),
            "failed_count": int(self.failed_count),
            "rollback_count": int(self.rollback_count),
            "rollback_failures": int(self.rollback_failures),
            "trace_id": str(self.trace_id or ""),
            "error": str(self.error or ""),
            "created_at": float(self.created_at),
        }
