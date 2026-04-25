from __future__ import annotations

from queue import Empty, Queue

from .models import Event


class EventBus:
    """Single ingestion queue for runtime events."""

    def __init__(self) -> None:
        self.queue: Queue[Event] = Queue()

    def emit(self, event: Event) -> None:
        self.queue.put(event)

    def next(self, *, timeout: float | None = None) -> Event:
        if timeout is None:
            return self.queue.get_nowait()
        return self.queue.get(timeout=timeout)

    def empty(self) -> bool:
        return self.queue.empty()

    def drain(self, *, limit: int = 200) -> list[Event]:
        events: list[Event] = []
        for _ in range(max(0, int(limit or 0))):
            try:
                events.append(self.next())
            except Empty:
                break
        return events
