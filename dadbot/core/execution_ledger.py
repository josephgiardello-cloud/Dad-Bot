from __future__ import annotations

import hashlib
import json
import time
from contextvars import ContextVar
from copy import deepcopy
from threading import RLock
from typing import Any

from dadbot.core.canonical_event import canonicalize_event_payload
from dadbot.core.ledger.enforcement import LedgerEnforcer, LedgerEnforcementError
from dadbot.core.ledger_backend import LedgerBackend, InMemoryLedgerBackend
from dadbot.core.event_schema import stamp_schema_version


# ContextVar used by WriteBoundaryGuard (Step 7) to authorise writes.
_ledger_write_token: ContextVar[str] = ContextVar("_ledger_write_token", default="")


def _canonical_trace_payload(event_type: str, payload: Any) -> dict[str, Any]:  # noqa: ARG001
    """Strip all non-canonical fields from *payload*.

    Delegates to the system-wide ``canonicalize_event_payload`` function so
    that the canonical boundary is defined in exactly one place.
    The *event_type* parameter is retained for call-site compatibility but is
    no longer needed — canonicalization is unconditional and field-based, not
    event-type-based.
    """
    return canonicalize_event_payload(payload)


class ExecutionLedger:
    """Authoritative append-only execution history.

    All runtime event systems should derive from this ledger, not vice versa.
    Required event envelope keys:
      - type
      - session_id
      - trace_id
      - timestamp
      - kernel_step_id

    Args:
        backend: pluggable durability tier (default: InMemoryLedgerBackend).
        strict_writes: when True, only writes via LedgerWriter (carrying the
            correct _ledger_write_token) are accepted.  Direct ledger.write()
            calls from other code raise WriteBoundaryViolationError.
    """

    _REQUIRED_KEYS = ("type", "session_id", "trace_id", "timestamp", "kernel_step_id")

    def __init__(
        self,
        *,
        backend: LedgerBackend | None = None,
        strict_writes: bool = False,
    ) -> None:
        self._backend: LedgerBackend = backend or InMemoryLedgerBackend()
        self._strict_writes = bool(strict_writes)
        self._write_token: str = ""  # set by WriteBoundaryGuard
        self._events: list[dict[str, Any]] = []
        self._sequence = 0
        self._session_indexes: dict[str, int] = {}
        self._session_heads: dict[str, str] = {}
        self._event_session: dict[str, str] = {}
        self._lock = RLock()
        self.enforcer = LedgerEnforcer()        # Optional replay filters applied during load_from_backend().
        # Each filter is a callable(list[dict]) -> list[dict].
        self._replay_filters: list[Any] = []
    def write(self, event: dict[str, Any], *, committed: bool = False) -> dict[str, Any]:
        # Step 7: strict-write boundary check.
        if self._strict_writes:
            token = _ledger_write_token.get()
            if token != self._write_token or not self._write_token:
                raise WriteBoundaryViolationError(
                    "Direct ledger.write() is blocked in strict mode. "
                    "Use LedgerWriter â€” the authorised write boundary."
                )

        normalized = self._normalize_event(event)
        self.enforcer.validate(normalized)
        with self._lock:
            self._enforce_session_causality(normalized)

            normalized["_seq"] = self._sequence
            self._sequence += 1
            normalized["sequence"] = self._sequence

            session_id = str(normalized.get("session_id") or "default")
            next_session_index = int(self._session_indexes.get(session_id) or 0) + 1
            normalized["session_index"] = next_session_index

            if not str(normalized.get("event_id") or "").strip():
                normalized["event_id"] = f"evt-{normalized['sequence']}"

            event_id = str(normalized.get("event_id") or "").strip()
            if not event_id:
                raise LedgerEnforcementError("event_id cannot be empty")

            if event_id in self._event_session:
                raise LedgerEnforcementError(f"Duplicate event_id detected: {event_id}")

            self._session_indexes[session_id] = next_session_index
            self._session_heads[session_id] = event_id
            self._event_session[event_id] = session_id
            self._events.append(normalized)

        # Persist to backend outside the in-memory lock to avoid blocking reads.
        self._backend.append(normalized, committed=committed)

        return deepcopy(normalized)

    def append(self, event: dict[str, Any], *, committed: bool = False) -> dict[str, Any]:
        return self.write(event, committed=committed)

    @property
    def sealed_events(self) -> tuple[dict[str, Any], ...]:
        """Read-only tuple snapshot of the current event list.  Callers cannot
        mutate the ledger through this view."""
        with self._lock:
            return tuple(deepcopy(self._events))

    def add_replay_filter(self, fn) -> None:
        """Register a replay filter applied during load_from_backend().

        Example â€” discard uncommitted AtomicWriteUnit events::

            from dadbot.core.durability import AtomicWriteUnit
            ledger.add_replay_filter(AtomicWriteUnit.filter_committed)
        """
        self._replay_filters.append(fn)

    def load_from_backend(self) -> int:
        """Reload events from the backend into memory.

        Applies registered replay filters (e.g. AtomicWriteUnit.filter_committed)
        and runs SequenceValidator to detect ordering anomalies.
        Returns the number of events loaded.
        """
        from dadbot.core.ledger_backend import SequenceValidator
        from dadbot.core.event_schema import get_migrator
        events = self._backend.load()

        # Upgrade schema versions before applying events.
        migrator = get_migrator()
        events = migrator.migrate_all(events)

        # Apply replay filters (e.g. discard uncommitted write units).
        for fn in self._replay_filters:
            events = fn(events)

        # Validate backend ordering.
        report = SequenceValidator.validate(events)
        if not report["ok"]:
            import warnings
            for v in report["violations"]:
                warnings.warn(
                    f"ExecutionLedger.load_from_backend: sequence anomaly: {v}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        with self._lock:
            self._events.clear()
            self._sequence = 0
            self._session_indexes.clear()
            self._session_heads.clear()
            self._event_session.clear()

        for event in events:
            # Replay through the normalisation + causal chain without re-persisting.
            normalized = self._normalize_event(event)
            with self._lock:
                self._enforce_session_causality(normalized)
                normalized["_seq"] = self._sequence
                self._sequence += 1
                normalized["sequence"] = int(event.get("sequence") or self._sequence)
                session_id = str(normalized.get("session_id") or "default")
                session_index = int(event.get("session_index") or 0) or (
                    int(self._session_indexes.get(session_id) or 0) + 1
                )
                normalized["session_index"] = session_index
                if not str(normalized.get("event_id") or "").strip():
                    normalized["event_id"] = str(event.get("event_id") or f"evt-{normalized['sequence']}")
                event_id = str(normalized.get("event_id") or "").strip()
                self._session_indexes[session_id] = session_index
                self._session_heads[session_id] = event_id
                if event_id:
                    self._event_session[event_id] = session_id
                self._events.append(normalized)

        return len(self._events)

    def read(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._events)

    def snapshot(self) -> list[dict[str, Any]]:
        return self.read()

    def events_since(self, cursor: int) -> tuple[list[dict[str, Any]], int]:
        with self._lock:
            safe = max(0, int(cursor or 0))
            tail = deepcopy(self._events[safe:])
            return tail, len(self._events)

    def filter(self, *, event_type: str = "", session_id: str = "") -> list[dict[str, Any]]:
        events = self.read()
        if event_type:
            events = [event for event in events if str(event.get("type") or "") == str(event_type)]
        if session_id:
            events = [event for event in events if str(event.get("session_id") or "") == str(session_id)]
        return events

    def replay_hash(self, *, session_id: str = "") -> str:
        events = self.filter(session_id=session_id) if session_id else self.read()
        canonical = [
            {
                "type": str(event.get("type") or ""),
                "session_id": str(event.get("session_id") or ""),
                "session_index": int(event.get("session_index") or 0),
                "event_id": str(event.get("event_id") or ""),
                "parent_event_id": str(event.get("parent_event_id") or ""),
                "kernel_step_id": str(event.get("kernel_step_id") or ""),
                "payload": _canonical_trace_payload(
                    str(event.get("type") or ""),
                    event.get("payload"),
                ),
            }
            for event in events
        ]
        digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return digest

    def _normalize_event(self, event: dict[str, Any] | None) -> dict[str, Any]:
        payload = dict(event or {})
        payload.setdefault("type", "UNKNOWN")
        payload.setdefault("session_id", "default")
        payload.setdefault("trace_id", "")
        payload.setdefault("timestamp", time.time())
        payload.setdefault("kernel_step_id", "")
        payload.setdefault("event_id", "")
        payload.setdefault("parent_event_id", "")
        # Stamp schema version (no-op if already present).
        stamp_schema_version(payload)

        missing = [key for key in self._REQUIRED_KEYS if key not in payload]
        if missing:
            raise ValueError(f"ExecutionLedger event missing required keys: {missing}")
        return payload

    def _enforce_session_causality(self, event: dict[str, Any]) -> None:
        session_id = str(event.get("session_id") or "default")
        parent_event_id = str(event.get("parent_event_id") or "").strip()
        current_head = str(self._session_heads.get(session_id) or "").strip()

        if parent_event_id:
            parent_session = str(self._event_session.get(parent_event_id) or "").strip()
            if not parent_session:
                raise LedgerEnforcementError(f"Unknown parent_event_id: {parent_event_id}")
            if parent_session != session_id:
                raise LedgerEnforcementError(
                    f"Cross-session leakage: parent {parent_event_id!r} belongs to {parent_session!r}, event is {session_id!r}"
                )

        # Enforce a strict single-chain causal head per session.
        if current_head:
            if not parent_event_id:
                event["parent_event_id"] = current_head
            elif parent_event_id != current_head:
                raise LedgerEnforcementError(
                    f"Ambiguous ancestry for session {session_id!r}: expected parent {current_head!r}, got {parent_event_id!r}"
                )
        elif parent_event_id:
            raise LedgerEnforcementError(
                f"Session {session_id!r} has no head yet but parent_event_id {parent_event_id!r} was provided"
            )


# ---------------------------------------------------------------------------
# Step 7 â€” Strict write boundary
# ---------------------------------------------------------------------------

class WriteBoundaryViolationError(RuntimeError):
    """Raised when code bypasses LedgerWriter to write directly to the ledger
    while strict_writes=True is active."""


class WriteBoundaryGuard:
    """Context manager that authorises writes to a strict-mode ExecutionLedger.

    Only LedgerWriter (and code explicitly wrapped in this guard) may call
    ledger.write() when the ledger is in strict_writes mode.

    Usage:
        with WriteBoundaryGuard(ledger) as guard:
            ledger.write(event)  # allowed
    """

    def __init__(self, ledger: ExecutionLedger) -> None:
        self._ledger = ledger
        self._token = None

    def __enter__(self) -> "WriteBoundaryGuard":
        import uuid
        write_token = uuid.uuid4().hex
        self._ledger._write_token = write_token
        self._token = _ledger_write_token.set(write_token)
        return self

    def __exit__(self, *_) -> None:
        self._ledger._write_token = ""
        if self._token is not None:
            _ledger_write_token.reset(self._token)
