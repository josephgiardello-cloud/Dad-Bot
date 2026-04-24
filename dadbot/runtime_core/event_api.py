from __future__ import annotations

import logging

from .bus import EventBus
from .journal import EventJournal
from .models import Event, new_event
from .runtime import AgentRuntime
from .store import ConversationStore


logger = logging.getLogger(__name__)


class RuntimeEventAPI:
    """Thin API layer between UI emitters and the deterministic runtime loop.

    Observer-style consumers must read projections from this API or the store.
    They must not influence execution-time control flow.
    Returned views are detached derived data only: no runtime references, no
    mutation paths back into execution state, and no execution-time callbacks.
    Versioned view access is explicit so projection evolution stays additive.
    """

    def __init__(self, *, runtime: AgentRuntime, store: ConversationStore, bus: EventBus, journal: EventJournal | None = None) -> None:
        self.runtime = runtime
        self.store = store
        self.bus = bus
        self.journal = journal
        if self.journal is not None:
            for event in self.journal.replay():
                self.store.apply_event(event)

    def seed_thread(self, thread_id: str, messages: list[dict] | None) -> None:
        self.store.seed_thread_messages(str(thread_id or "default"), list(messages or []))

    def emit_user_message(self, *, thread_id: str, text: str, attachments: list[dict] | None = None) -> None:
        self.emit_event(
            new_event(
                "user_message",
                thread_id=str(thread_id or "default"),
                payload={
                    "text": str(text or ""),
                    "attachments": list(attachments or []),
                },
            )
        )

    def emit_event(self, event: Event) -> None:
        if self.journal is not None:
            self.journal.append(event)
        self.bus.emit(event)

    def emit_assistant_attachment(self, *, thread_id: str, attachment: dict) -> None:
        self.emit_event(
            new_event(
                "assistant_attachment_added",
                thread_id=str(thread_id or "default"),
                payload={"attachment": dict(attachment or {})},
            )
        )

    def process_until_idle(self, *, max_events: int = 256) -> list[Event]:
        processed: list[Event] = []
        budget = max(1, int(max_events or 1))
        while budget > 0 and not self.bus.empty():
            budget -= 1
            event = self.bus.next()
            processed.append(event)
            follow_up = self.runtime.handle_event(event)
            for created in follow_up:
                if self.journal is not None:
                    self.journal.append(created)
                self.bus.emit(created)
        return processed

    def thread_messages(self, thread_id: str) -> list[dict]:
        return self.store.thread_messages(str(thread_id or "default"))

    def thread_thinking(self, thread_id: str) -> dict:
        return self.store.thread_thinking(str(thread_id or "default"))

    def thread_execution_boundaries(self, thread_id: str) -> list[dict]:
        return self.store.thread_execution_boundaries(str(thread_id or "default"))

    def view_schema_policy(self) -> dict:
        return self.store.thread_view_schema_policy()

    def get_view(self, thread_id: str, *, version: str = ConversationStore.THREAD_VIEW_DEFAULT_VERSION) -> dict:
        """Return the canonical thread projection.

        The default view is the live projection contract (`v2`). Replay and audit
        callers must request `version="v1"` explicitly.
        """
        if version not in ConversationStore.THREAD_VIEW_SUPPORTED_VERSIONS:
            logger.warning("Unsupported projection version requested: %s", version)
        return self.store.thread_view(str(thread_id or "default"), version=version)

    @staticmethod
    def result_for_thread(thread_id: str, processed: list[Event]) -> dict:
        assistant_event = next(
            (item for item in processed if item.type == "assistant_reply" and item.thread_id == str(thread_id or "default")),
            None,
        )
        photo_requested = any(
            item.type == "photo_request" and item.thread_id == str(thread_id or "default") for item in processed
        )
        tts_requested = any(
            item.type == "tts_request" and item.thread_id == str(thread_id or "default") for item in processed
        )
        payload = dict(assistant_event.payload or {}) if assistant_event is not None else {}
        return {
            "reply": str(payload.get("text") or ""),
            "should_end": bool(payload.get("should_end", False)),
            "mood": str(payload.get("mood") or "neutral"),
            "pipeline": dict(payload.get("pipeline") or {}),
            "photo_requested": bool(photo_requested),
            "tts_requested": bool(tts_requested),
        }
