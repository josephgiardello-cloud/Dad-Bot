"""L4-P1 — External Event Authority Layer
L4-P2 — Execution Reconstruction Primitive

These two phases share a single module because they address the same
conceptual shift: state is not stored — it is derived from an immutable
event truth layer.

Architecture:
  EventAuthority  ← canonical source of truth (append-only)
      │
      └── derive_state()        ← CanonicalEventReducer (pure function)
      └── rebuild_state_from_events()  ← stateless reconstruction primitive

Invariant:
    "If event log is missing → system is undefined."

TurnContext and all runtime state are *projections* of the event log,
never authoritative sources.  Any runtime state that cannot be derived
from the event log is by definition undefined.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from threading import RLock
from typing import Any

from dadbot.core.event_reducer import CanonicalEventReducer

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class EventAuthorityError(RuntimeError):
    """Raised when the event authority invariant is violated."""


class UndefinedSystemStateError(EventAuthorityError):
    """Raised when the event log is missing or empty and state is requested.

    This is the formal enforcement of:
        "If event log is missing → system is undefined."
    """


# ---------------------------------------------------------------------------
# L4-P1 — EventAuthority
# ---------------------------------------------------------------------------


class EventAuthority:
    """Canonical source of truth for execution state.

    Contract:
    - Append-only: events are never deleted or mutated.
    - Derive-only: all state is derived from the event log via reducer.
    - Defined-only: state derivation raises if the log is empty (bootstrapping
      required before querying derived state).

    TurnContext and runtime objects are projections, not authorities.
    This object is the single authority for what has happened.
    """

    def __init__(
        self,
        reducer: CanonicalEventReducer | None = None,
        *,
        session_id: str = "",
    ) -> None:
        self._reducer = reducer or CanonicalEventReducer()
        self._session_id = str(session_id or "")
        self._events: list[dict[str, Any]] = []
        self._sequence: int = 0
        self._lock = RLock()

    # ------------------------------------------------------------------
    # Append-only write API
    # ------------------------------------------------------------------

    def append(self, event: dict[str, Any]) -> int:
        """Append a single event. Returns the assigned sequence number."""
        with self._lock:
            stamped = dict(event)
            stamped.setdefault("sequence", self._sequence)
            stamped.setdefault("session_id", self._session_id)
            self._events.append(stamped)
            self._sequence += 1
            return stamped["sequence"]

    def append_batch(self, events: list[dict[str, Any]]) -> list[int]:
        """Append multiple events atomically. Returns assigned sequence numbers."""
        with self._lock:
            seqs: list[int] = []
            for event in events or []:
                stamped = dict(event)
                stamped.setdefault("sequence", self._sequence)
                stamped.setdefault("session_id", self._session_id)
                self._events.append(stamped)
                seqs.append(self._sequence)
                self._sequence += 1
            return seqs

    # ------------------------------------------------------------------
    # Authority introspection
    # ------------------------------------------------------------------

    def is_defined(self) -> bool:
        """True iff the event log has at least one event (system is defined)."""
        with self._lock:
            return len(self._events) > 0

    def assert_defined(self) -> None:
        """Raise UndefinedSystemStateError if the log is empty.

        Invariant: if event log is missing → system is undefined.
        Callers must bootstrap at least one event before querying state.
        """
        if not self.is_defined():
            raise UndefinedSystemStateError(
                "EventAuthority: system state is undefined — event log is empty. "
                "Bootstrap with at least one SESSION_STATE_UPDATED event before "
                "deriving state.",
            )

    def event_count(self) -> int:
        with self._lock:
            return len(self._events)

    def head_sequence(self) -> int:
        """Sequence number of the last event (or -1 if empty)."""
        with self._lock:
            return self._sequence - 1 if self._events else -1

    def read_all(self) -> list[dict[str, Any]]:
        """Return a deep copy of all events (safe for external use)."""
        with self._lock:
            return deepcopy(self._events)

    def read_from(self, sequence: int) -> list[dict[str, Any]]:
        """Return events with sequence >= the given value."""
        with self._lock:
            return deepcopy(
                [e for e in self._events if int(e.get("sequence") or 0) >= sequence],
            )

    # ------------------------------------------------------------------
    # L4-P1: State derivation (authority → projection)
    # ------------------------------------------------------------------

    def derive_state(self) -> dict[str, Any]:
        """Derive full execution state from the event log.

        This is the ONLY legitimate way to read state.  Runtime objects
        (TurnContext, session_store) are projections of this.

        Raises UndefinedSystemStateError if log is empty.
        """
        self.assert_defined()
        with self._lock:
            events = deepcopy(self._events)
        return self._reducer.reduce(events)

    def derive_state_at(self, sequence: int) -> dict[str, Any]:
        """Derive state as of a specific sequence number (point-in-time query)."""
        self.assert_defined()
        with self._lock:
            events = deepcopy(
                [e for e in self._events if int(e.get("sequence") or 0) <= sequence],
            )
        return self._reducer.reduce(events)

    def derive_session_state(self, session_id: str | None = None) -> dict[str, Any]:
        """Derive state for a specific session."""
        full = self.derive_state()
        sid = session_id or self._session_id
        return dict(full.get("sessions", {}).get(sid) or {})

    # ------------------------------------------------------------------
    # L4-P2: Execution reconstruction primitive
    # ------------------------------------------------------------------

    def rebuild_state_from_events(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Pure function: reconstruct system state from an arbitrary event list.

        This is the reconstruction primitive: given only an event log,
        deterministically derive the full execution state.

        - Pure: no side effects, no writes to self.
        - Deterministic: same events → same state, always.
        - Sufficient: all state is derivable from events alone.

        Usage::

            # Replay a stored event log from scratch:
            state = authority.rebuild_state_from_events(stored_events)
        """
        return self._reducer.reduce(list(events or []))

    # ------------------------------------------------------------------
    # Identity / fingerprint
    # ------------------------------------------------------------------

    def authority_hash(self) -> str:
        """Content-addressed hash of the full event log.

        Same event log → same hash (deterministic fingerprint).
        Changes whenever a new event is appended.
        """
        with self._lock:
            payload = [{k: v for k, v in e.items() if k != "sequence"} for e in self._events]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self._session_id,
            "event_count": self.event_count(),
            "head_sequence": self.head_sequence(),
            "is_defined": self.is_defined(),
            "authority_hash": self.authority_hash(),
        }


# ---------------------------------------------------------------------------
# Reconstruction convenience function (standalone)
# ---------------------------------------------------------------------------


def rebuild_state_from_events(
    events: list[dict[str, Any]],
    *,
    reducer: CanonicalEventReducer | None = None,
) -> dict[str, Any]:
    """Standalone reconstruction primitive.

    Pure function — no side effects.  Identical to EventAuthority's method
    but usable without instantiating an authority object.

    This is the core L4-P2 primitive: events → reducer → state.
    """
    r = reducer or CanonicalEventReducer()
    return r.reduce(list(events or []))


__all__ = [
    "EventAuthority",
    "EventAuthorityError",
    "UndefinedSystemStateError",
    "rebuild_state_from_events",
]
