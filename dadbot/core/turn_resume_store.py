"""Durable per-turn resume point storage.

Phase 2C authority change:
Resume state is persisted only to ExecutionLedger events (authoritative path).
"""

from __future__ import annotations

import builtins
import time
from dataclasses import dataclass
from threading import RLock
from typing import Any

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_writer import LedgerWriter

_SCHEMA_VERSION = "1"
_RESUME_EVENT_TYPE = "turn_resume_point"


@dataclass(frozen=True)
class ResumePoint:
    """Immutable snapshot of a durable turn resume record."""

    turn_id: str
    last_completed_stage: str
    next_stage: str
    checkpoint_hash: str
    completed_stages: tuple[str, ...]
    created_at: float
    updated_at: float
    # Stage that was executing when the process last checkpointed.  Non-empty means
    # a crash may have happened mid-execution; the stage call_id is deterministic
    # so external tools can deduplicate via hash(turn_id + ':' + in_flight_stage).
    in_flight_stage: str = ""

    def is_expired(self, *, max_age_seconds: float) -> bool:
        """Return True if this record is older than max_age_seconds."""
        return (time.time() - self.updated_at) > max_age_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "turn_id": self.turn_id,
            "last_completed_stage": self.last_completed_stage,
            "next_stage": self.next_stage,
            "checkpoint_hash": self.checkpoint_hash,
            "completed_stages": list(self.completed_stages),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "in_flight_stage": self.in_flight_stage,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResumePoint:
        return cls(
            turn_id=str(d.get("turn_id") or ""),
            last_completed_stage=str(d.get("last_completed_stage") or ""),
            next_stage=str(d.get("next_stage") or ""),
            checkpoint_hash=str(d.get("checkpoint_hash") or ""),
            completed_stages=tuple(d.get("completed_stages") or []),
            created_at=float(d.get("created_at") or time.time()),
            updated_at=float(d.get("updated_at") or time.time()),
            in_flight_stage=str(d.get("in_flight_stage") or ""),
        )


class TurnResumeStore:
    """Durable per-turn resume point store backed by ExecutionLedger events."""

    def __init__(
        self,
        *,
        ledger: Any,
    ) -> None:
        self._ledger = ledger
        self._lock = RLock()
        if self._ledger is None:
            raise ValueError("TurnResumeStore requires a non-null ledger")

    def _ledger_events(self) -> list[dict[str, Any]]:
        events = list(self._ledger.read())
        return [
            event
            for event in events
            if str(event.get("type") or "") == _RESUME_EVENT_TYPE
        ]

    def _resume_payload_from_event(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = dict(event.get("payload") or {})
        if str(payload.get("schema_version") or "") != _SCHEMA_VERSION:
            return {}
        return payload

    def _latest_ledger_payload(self, turn_id: str) -> dict[str, Any] | None:
        for event in reversed(self._ledger_events()):
            payload = self._resume_payload_from_event(event)
            if not payload:
                continue
            if str(payload.get("turn_id") or "") != str(turn_id or ""):
                continue
            return payload
        return None

    def _append_ledger_payload(self, payload: dict[str, Any]) -> None:
        envelope = {
            "type": _RESUME_EVENT_TYPE,
            "session_id": f"resume:{str(payload.get('turn_id') or '')}",
            "trace_id": str(payload.get("turn_id") or ""),
            "timestamp": float(time.time()),
            "kernel_step_id": "resume-store",
            "payload": payload,
        }
        writer = LedgerWriter(self._ledger)
        writer.write_event(
            str(envelope.get("type") or ""),
            session_id=str(envelope.get("session_id") or ""),
            trace_id=str(envelope.get("trace_id") or ""),
            kernel_step_id=str(envelope.get("kernel_step_id") or "resume-store"),
            payload=dict(envelope.get("payload") or {}),
            committed=False,
        )

    def save(
        self,
        *,
        turn_id: str,
        last_completed_stage: str,
        next_stage: str,
        checkpoint_hash: str,
        completed_stages: list[str],
        created_at: float | None = None,
    ) -> ResumePoint:
        """Persist a resume record for *turn_id*.

        Subsequent calls for the same turn_id update the record in-place.
        The ``created_at`` timestamp is set only on the first write.
        """
        with self._lock:
            now = time.time()
            existing = self._latest_ledger_payload(turn_id) or {}
            existing_created_at = float(existing.get("created_at") or created_at or now)

            record = ResumePoint(
                turn_id=turn_id,
                last_completed_stage=last_completed_stage,
                next_stage=next_stage,
                checkpoint_hash=checkpoint_hash,
                completed_stages=tuple(completed_stages),
                created_at=existing_created_at,
                updated_at=now,
            )
            payload = record.to_dict()
            payload["cleared"] = False
            self._append_ledger_payload(payload)

        return record

    def load(self, turn_id: str) -> ResumePoint | None:
        """Return the stored resume point for *turn_id*, or None if not found.

        Returns None (rather than raising) for corrupt or missing records.
        """
        with self._lock:
            payload = self._latest_ledger_payload(turn_id)
            if not payload:
                return None
            if bool(payload.get("cleared", False)):
                return None
            return ResumePoint.from_dict(payload)

    def clear(self, turn_id: str) -> None:
        """Remove the resume record for *turn_id* (called on successful completion)."""
        with self._lock:
            now = time.time()
            existing = self._latest_ledger_payload(turn_id) or {}
            payload = {
                "schema_version": _SCHEMA_VERSION,
                "turn_id": str(turn_id or ""),
                "last_completed_stage": str(existing.get("last_completed_stage") or ""),
                "next_stage": "",
                "checkpoint_hash": str(existing.get("checkpoint_hash") or ""),
                "completed_stages": list(existing.get("completed_stages") or []),
                "created_at": float(existing.get("created_at") or now),
                "updated_at": now,
                "in_flight_stage": "",
                "cleared": True,
            }
            self._append_ledger_payload(payload)

    def mark_started(self, turn_id: str, stage_name: str) -> None:
        """Record that *stage_name* is about to execute for *turn_id*.

        Called BEFORE the node runs.  If a crash occurs between this call and
        the subsequent ``save()`` marking the stage as completed, the resume
        record will show ``in_flight_stage=stage_name`` on the next startup.

        ExecutionRecovery uses this to inject a deterministic ``_stage_call_id``
        into TurnContext.state so that external tools can deduplicate effects.

        This is a best-effort write: failures are silently swallowed so that
        a storage hiccup never aborts execution.
        """
        with self._lock:
            now = time.time()
            existing = self._latest_ledger_payload(turn_id) or {}
            payload = {
                "schema_version": _SCHEMA_VERSION,
                "turn_id": str(turn_id or ""),
                "last_completed_stage": str(existing.get("last_completed_stage") or ""),
                "next_stage": str(existing.get("next_stage") or stage_name),
                "checkpoint_hash": str(existing.get("checkpoint_hash") or ""),
                "completed_stages": list(existing.get("completed_stages") or []),
                "created_at": float(existing.get("created_at") or now),
                "updated_at": now,
                "in_flight_stage": stage_name,
                "cleared": False,
            }
            try:
                self._append_ledger_payload(payload)
            except OSError:
                pass

    def list_pending(self) -> list[ResumePoint]:
        """Return all stored (potentially incomplete) resume records.

        Used at startup to discover turns that need recovery.
        """
        results: list[ResumePoint] = []
        with self._lock:
            latest_by_turn: dict[str, dict[str, Any]] = {}
            for event in self._ledger_events():
                payload = self._resume_payload_from_event(event)
                if not payload:
                    continue
                turn_id = str(payload.get("turn_id") or "")
                if not turn_id:
                    continue
                latest_by_turn[turn_id] = payload

            for payload in latest_by_turn.values():
                if bool(payload.get("cleared", False)):
                    continue
                results.append(ResumePoint.from_dict(payload))
        return results

    def purge_expired(self, *, max_age_seconds: float) -> int:
        """Delete resume records older than *max_age_seconds*.

        Returns number of records removed.
        """
        removed = 0
        with self._lock:
            now = time.time()
            for point in self.list_pending():
                if (now - float(point.updated_at)) <= max_age_seconds:
                    continue
                payload = point.to_dict()
                payload["cleared"] = True
                payload["updated_at"] = now
                payload["in_flight_stage"] = ""
                try:
                    self._append_ledger_payload(payload)
                    removed += 1
                except OSError:
                    continue
        return removed


class _LegacyExecutionRecovery:
    """Compatibility shim used by legacy durable-execution tests."""

    def __init__(self, store: TurnResumeStore) -> None:
        self._store = store

    def check_resume(self, turn_id: str) -> ResumePoint | None:
        return self._store.load(turn_id)

    def is_already_completed(self, stage: str, turn_context: Any) -> bool:
        state = dict(getattr(turn_context, "state", {}) or {})
        executed = set(state.get("_graph_executed_stages") or set())
        return str(stage or "") in executed

    def restore_executed_stages(self, point: ResumePoint | None, turn_context: Any) -> None:
        if point is None:
            return
        state = dict(getattr(turn_context, "state", {}) or {})
        executed = set(state.get("_graph_executed_stages") or set())
        executed.update(str(stage) for stage in list(point.completed_stages or ()) if str(stage))
        state["_graph_executed_stages"] = executed
        turn_context.state = state


def _make_recovery(_tmp_path: Any = None) -> tuple[_LegacyExecutionRecovery, TurnResumeStore]:
    store = TurnResumeStore(ledger=ExecutionLedger())
    return _LegacyExecutionRecovery(store), store


if not hasattr(builtins, "_make_recovery"):
    builtins._make_recovery = _make_recovery
