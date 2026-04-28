from __future__ import annotations

from typing import Any

from dadbot.runtime_core.models import Event, new_event


def _attach_correlation_id(*, correlation_id: str | None) -> str | None:
    value = str(correlation_id or "").strip()
    return value or None


def _emit_event(
    event_type: str,
    *,
    thread_id: str,
    payload: dict[str, Any],
    parent_event_id: str | None,
    correlation_id: str | None,
) -> Event:
    return new_event(
        str(event_type or "").strip() or "event",
        thread_id=str(thread_id or "default"),
        payload=dict(payload or {}),
        parent_event_id=parent_event_id,
        correlation_id=_attach_correlation_id(correlation_id=correlation_id),
    )


class EventEmitter:
    def attach_correlation_id(self, *, correlation_id: str | None) -> str | None:
        return _attach_correlation_id(correlation_id=correlation_id)

    def emit_event(
        self,
        event_type: str,
        *,
        thread_id: str,
        payload: dict[str, Any],
        parent_event_id: str | None,
        correlation_id: str | None,
    ) -> Event:
        return _emit_event(
            event_type,
            thread_id=thread_id,
            payload=payload,
            parent_event_id=parent_event_id,
            correlation_id=correlation_id,
        )
