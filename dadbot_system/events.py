from __future__ import annotations

from collections import deque
from queue import Empty, Full, Queue
from threading import Lock
from typing import Any, Protocol

from .contracts import EventEnvelope


class EventBus(Protocol):
    def publish(self, event: EventEnvelope) -> None: ...


class InMemoryEventBus:
    def __init__(self, max_events: int = 500):
        self._events: deque[EventEnvelope] = deque(maxlen=max_events)
        self._subscribers: list[Queue] = []
        self._lock = Lock()

    def publish(self, event: EventEnvelope) -> None:
        self._events.append(event)
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except Full:
                continue

    def publish_ephemeral(self, event: Any) -> None:
        self.publish(event)

    def events(self) -> list[EventEnvelope]:
        return list(self._events)

    def subscribe(self, max_queue_size: int = 200):
        subscriber = Queue(maxsize=max_queue_size)
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber) -> None:
        with self._lock:
            self._subscribers = [candidate for candidate in self._subscribers if candidate is not subscriber]


class DurableEventBus(InMemoryEventBus):
    """Persist events to a state store before notifying process-local subscribers."""

    persists_events = True

    def __init__(self, state_store, max_events: int = 500):
        super().__init__(max_events=max_events)
        self._state_store = state_store

    def publish(self, event: EventEnvelope) -> None:
        self._state_store.append_event(event.session_id, event.to_dict())
        super().publish(event)

    def publish_ephemeral(self, event: Any) -> None:
        """Deliver transient process-local payloads without durable serialization."""
        super().publish(event)


class QueueEventBus:
    def __init__(self, queue):
        self._queue = queue

    def publish(self, event: EventEnvelope) -> None:
        self._queue.put(event)

    def drain(self, limit: int = 200) -> list[EventEnvelope]:
        events: list[EventEnvelope] = []
        for _ in range(limit):
            try:
                events.append(self._queue.get_nowait())
            except Empty:
                break
        return events
