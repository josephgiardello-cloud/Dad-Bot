from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Literal

EventType = Literal[
    "user_message",
    "execution_region_started",
    "execution_region_completed",
    "assistant_reply",
    "assistant_attachment_added",
    "thinking_update",
    "tool_call",
    "tool_result",
    "memory_write",
    "mood_update",
    "thread_switch",
    "photo_request",
    "tts_request",
]


@dataclass(slots=True)
class Event:
    id: str
    type: EventType
    thread_id: str
    timestamp: float
    payload: dict


def new_event(event_type: EventType, *, thread_id: str, payload: dict | None = None) -> Event:
    return Event(
        id=uuid.uuid4().hex,
        type=event_type,
        thread_id=str(thread_id or "default"),
        timestamp=time.time(),
        payload=dict(payload or {}),
    )
