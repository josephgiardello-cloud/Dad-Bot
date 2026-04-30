from __future__ import annotations

import asyncio
from threading import RLock
from typing import Any


class IdempotencyBoundary:
    """Stores completed turn results keyed by (session_id, request_id)."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._completed: dict[tuple[str, str], Any] = {}
        self._inflight: dict[tuple[str, str], asyncio.Future] = {}

    def _key(self, *, session_id: str, request_id: str) -> tuple[str, str]:
        return (str(session_id or "").strip(), str(request_id or "").strip())

    def get_cached(self, *, session_id: str, request_id: str) -> Any | None:
        key = self._key(session_id=session_id, request_id=request_id)
        if not key[0] or not key[1]:
            return None
        with self._lock:
            return self._completed.get(key)

    def acquire_or_get(
        self,
        *,
        session_id: str,
        request_id: str,
        loop: asyncio.AbstractEventLoop,
    ) -> tuple[str, Any | None, asyncio.Future | None]:
        key = self._key(session_id=session_id, request_id=request_id)
        if not key[0] or not key[1]:
            return ("no-key", None, None)

        with self._lock:
            if key in self._completed:
                return ("cached", self._completed[key], None)
            if key in self._inflight:
                return ("inflight", None, self._inflight[key])
            shared_future: asyncio.Future = loop.create_future()
            self._inflight[key] = shared_future
            return ("acquired", None, shared_future)

    def store_result(self, *, session_id: str, request_id: str, result: Any) -> None:
        key = self._key(session_id=session_id, request_id=request_id)
        if not key[0] or not key[1]:
            return
        with self._lock:
            self._completed[key] = result
            inflight = self._inflight.pop(key, None)
        if inflight is not None and not inflight.done():
            inflight.set_result(result)

    def store_error(
        self,
        *,
        session_id: str,
        request_id: str,
        error: Exception,
    ) -> None:
        key = self._key(session_id=session_id, request_id=request_id)
        if not key[0] or not key[1]:
            return
        with self._lock:
            inflight = self._inflight.pop(key, None)
        if inflight is not None and not inflight.done():
            inflight.set_exception(error)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "completed": {
                    f"{session_id}:{request_id}": value for (session_id, request_id), value in self._completed.items()
                },
                "inflight_count": len(self._inflight),
            }
