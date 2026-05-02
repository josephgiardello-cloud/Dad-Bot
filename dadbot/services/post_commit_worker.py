from __future__ import annotations

import logging
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any

from dadbot.core.post_commit_events import POST_COMMIT_READY, PostCommitEvent

logger = logging.getLogger(__name__)


class _PostCommitCapability:
    """Minimal capability surface granted to the post-commit background worker.

    Holds only the two callables needed for post-commit memory operations.
    The ``bot`` reference is **not retained** after ``__init__`` completes.
    This severs the worker's access to the full service graph and prevents any
    path from reaching back into the execution kernel via the worker thread.

    This achieves *object-graph isolation* within the same process — the
    strongest available boundary short of a separate subprocess.
    """

    __slots__ = ("_consolidate", "_forget")

    def __init__(self, bot: Any) -> None:
        memory_coordinator = getattr(bot, "memory_coordinator", None)
        self._consolidate = getattr(memory_coordinator, "consolidate_memories", None)
        self._forget = getattr(memory_coordinator, "apply_controlled_forgetting", None)
        # bot reference is intentionally NOT stored

    def consolidate(self, *, turn_context: Any) -> None:
        if callable(self._consolidate):
            self._consolidate(turn_context=turn_context)

    def forget(self, *, turn_context: Any) -> None:
        if callable(self._forget):
            self._forget(turn_context=turn_context)


class PostCommitWorker:
    """Background worker for post-commit reasoning and maintenance work.

    Holds a ``_PostCommitCapability`` (two callables only) rather than the full
    ``bot`` reference.  The event bus reference is stored separately.  Neither
    object allows navigation back into the live execution graph.
    """

    def __init__(self, bot: Any, *, max_queue_size: int = 200) -> None:
        self._capability = _PostCommitCapability(bot)
        self._max_queue_size = max_queue_size
        self._stop_event = Event()
        self._subscriber: Queue[Any] | None = None
        self._thread: Thread | None = None
        # Store only the event bus — not bot itself
        self._event_bus = getattr(bot, "_runtime_event_bus", None)
        self.start()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        subscribe = getattr(self._event_bus, "subscribe", None)
        if not callable(subscribe):
            logger.warning("PostCommitWorker unavailable: runtime event bus missing subscribe()")
            return
        self._subscriber = subscribe(max_queue_size=self._max_queue_size)
        self._thread = Thread(
            target=self._run,
            name="dadbot-post-commit-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if self._subscriber is None:
                return
            try:
                event = self._subscriber.get(timeout=0.2)
            except Empty:
                continue
            if getattr(event, "event_type", "") != POST_COMMIT_READY:
                continue
            self._handle_post_commit_event(event)

    def _handle_post_commit_event(self, event: PostCommitEvent) -> None:
        turn_context = getattr(event, "payload", {}).get("turn_context")
        try:
            self._capability.consolidate(turn_context=turn_context)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Post-commit consolidate_memories failed (non-fatal): %s", exc)
        try:
            self._capability.forget(turn_context=turn_context)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Post-commit apply_controlled_forgetting failed (non-fatal): %s", exc)
