from collections import deque
from threading import RLock
from typing import Any


class EventStore:
    """
    Lightweight in-memory event buffer aligned with ledger-style execution.

    This is NOT the source of truth (ledger is).
    This is a fast-access cache for recent event inspection.
    """

    def __init__(self, maxlen: int = 1000):
        self._events = deque(maxlen=maxlen)
        self._lock = RLock()

    def append(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def filter(self, *, event_type: str | None = None, session_id: str | None = None):
        with self._lock:
            events = list(self._events)

        if event_type:
            events = [e for e in events if e.get("type") == event_type]

        if session_id:
            events = [e for e in events if e.get("session_id") == session_id]

        return events

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
