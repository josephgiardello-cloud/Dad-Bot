from __future__ import annotations

import threading
from copy import deepcopy
from typing import Any


class ExecutionLedger:
    """In-memory append-only execution ledger for control-plane lifecycle events."""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            payload = dict(event)
            payload.setdefault("sequence", len(self._events) + 1)
            self._events.append(payload)
            return dict(payload)

    def read(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._events)
