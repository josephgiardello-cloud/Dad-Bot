from __future__ import annotations

import hashlib
import json
import threading
import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager
from copy import deepcopy
from typing import Any

from dadbot.core.canonical_event import (
    CANONICAL_EVENT_FIELDS,
    NON_REPLAY_EVENT_TYPES,
    canonicalize_event_payload,
)
from dadbot.core.event_schema import get_migrator, stamp_schema_version
from dadbot.core.ledger_backend import InMemoryLedgerBackend, SequenceValidator


class WriteBoundaryViolationError(RuntimeError):
    """Raised when strict ledger mode rejects a write outside the boundary guard."""


class WriteBoundaryGuard(AbstractContextManager["WriteBoundaryGuard"]):
    """Temporarily allow writes to a strict-mode ledger."""

    def __init__(self, ledger: ExecutionLedger) -> None:
        self._ledger = ledger

    def __enter__(self) -> WriteBoundaryGuard:
        self._ledger._write_guard_depth += 1
        return self

    def __exit__(self, exc_type: Any, exc: Any, _tb: Any) -> None:
        self._ledger._write_guard_depth = max(0, self._ledger._write_guard_depth - 1)


def _canonical_trace_payload(event_type: str, payload: Any) -> dict[str, Any]:
    if str(event_type or "") in NON_REPLAY_EVENT_TYPES:
        return {}
    return canonicalize_event_payload(payload)


def _deterministic_event_id(payload: dict[str, Any]) -> str:
    seed = {
        "type": str(payload.get("type") or ""),
        "session_id": str(payload.get("session_id") or ""),
        "session_index": int(payload.get("session_index") or 0),
        "kernel_step_id": str(payload.get("kernel_step_id") or ""),
        "payload": _canonical_trace_payload(
            str(payload.get("type") or ""),
            payload.get("payload"),
        ),
    }
    return (
        "evt-"
        + hashlib.sha256(
            json.dumps(seed, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()[:24]
    )


class ExecutionLedger:
    """In-memory append-only execution ledger with replay-hash support."""

    def __init__(
        self,
        backend: Any | None = None,
        *,
        strict_writes: bool = False,
    ) -> None:
        self._backend = backend or InMemoryLedgerBackend()
        self._events: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._session_heads: dict[str, str] = {}
        self._session_indices: dict[str, int] = {}
        self._write_guard_depth = 0
        self._strict_writes = bool(strict_writes)
        self._replay_filters: list[Callable[[list[dict[str, Any]]], list[dict[str, Any]]]] = []

    @property
    def sealed_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.read())

    def add_replay_filter(
        self,
        filter_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    ) -> None:
        self._replay_filters.append(filter_fn)

    def _ensure_write_allowed(self) -> None:
        if self._strict_writes and self._write_guard_depth <= 0:
            raise WriteBoundaryViolationError(
                "ExecutionLedger strict mode requires WriteBoundaryGuard",
            )

    def write(self, event: dict[str, Any]) -> dict[str, Any]:
        from dadbot.core.ledger.enforcement import LedgerEnforcer

        self._ensure_write_allowed()
        LedgerEnforcer().validate(dict(event or {}))
        with self._lock:
            payload = deepcopy(dict(event or {}))
            session_id = str(payload.get("session_id") or "")
            parent_event_id = str(payload.get("parent_event_id") or "")
            current_head = str(self._session_heads.get(session_id) or "")
            if session_id:
                if parent_event_id:
                    if current_head and parent_event_id != current_head:
                        raise RuntimeError(
                            f"causal chain violation: session_id={session_id!r} parent={parent_event_id!r} head={current_head!r}",
                        )
                else:
                    payload["parent_event_id"] = current_head
                payload["session_index"] = int(self._session_indices.get(session_id, 0)) + 1
            else:
                payload.setdefault("parent_event_id", "")
                payload.setdefault("session_index", 0)

            payload.setdefault("payload", {})
            payload["_seq"] = len(self._events)
            payload.setdefault("sequence", len(self._events) + 1)
            payload.setdefault("event_id", _deterministic_event_id(payload))
            stamp_schema_version(payload)

            event_id = str(payload.get("event_id") or "")
            if session_id:
                self._session_heads[session_id] = event_id
                self._session_indices[session_id] = int(
                    payload.get("session_index") or 0,
                )

            self._events.append(deepcopy(payload))
            self._backend.append(
                deepcopy(payload),
                committed=bool(payload.get("committed", False)),
            )
            return deepcopy(payload)

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        return self.write(event)

    def read(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._events)

    def replay_hash(self) -> str:
        events = [event for event in self.read() if str(event.get("type") or "") not in NON_REPLAY_EVENT_TYPES]
        canonical = [
            {field: event.get(field) for field in CANONICAL_EVENT_FIELDS}
            | {
                "payload": _canonical_trace_payload(
                    str(event.get("type") or ""),
                    event.get("payload"),
                ),
            }
            for event in sorted(events, key=lambda item: int(item.get("sequence") or 0))
        ]
        return hashlib.sha256(
            json.dumps(canonical, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()

    def load_from_backend(self) -> int:
        with self._lock:
            events = list(get_migrator().migrate_all(list(self._backend.load())))
            report = SequenceValidator.validate(events)
            if not bool(report.get("ok")):
                warnings.warn(
                    f"sequence anomaly: {report.get('violations') or []}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            for replay_filter in self._replay_filters:
                events = list(replay_filter(list(events)))
            self._events = deepcopy(events)
            self._session_heads.clear()
            self._session_indices.clear()
            for event in self._events:
                session_id = str(event.get("session_id") or "")
                event_id = str(event.get("event_id") or "")
                if session_id and event_id:
                    self._session_heads[session_id] = event_id
                    self._session_indices[session_id] = int(
                        event.get("session_index") or 0,
                    )
            return len(self._events)

InMemoryExecutionLedger = ExecutionLedger
