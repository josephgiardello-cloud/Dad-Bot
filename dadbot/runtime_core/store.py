from __future__ import annotations

from copy import deepcopy

from .models import Event


class ConversationStore:
    """Deterministic thread state for replay-friendly chat history."""

    def __init__(self) -> None:
        self.threads: dict[str, list[dict]] = {}
        self.thread_state: dict[str, dict] = {}

    def get_thread(self, thread_id: str) -> list[dict]:
        key = str(thread_id or "default")
        return self.threads.setdefault(key, [])

    def get_thread_state(self, thread_id: str) -> dict:
        key = str(thread_id or "default")
        return self.thread_state.setdefault(key, {"last_turn_thinking": {}})

    def seed_thread_messages(self, thread_id: str, messages: list[dict] | None) -> None:
        if self.get_thread(thread_id):
            return
        if not messages:
            return
        self.threads[str(thread_id or "default")] = [
            {
                "role": str(item.get("role") or "assistant"),
                "content": str(item.get("content") or ""),
                "attachments": list(item.get("attachments") or []),
            }
            for item in list(messages)
            if isinstance(item, dict)
        ]

    def append_user(self, thread_id: str, text: str, *, attachments: list[dict] | None = None) -> None:
        self.get_thread(thread_id).append(
            {
                "role": "user",
                "content": str(text or ""),
                "attachments": list(attachments or []),
            }
        )

    def append_assistant(self, thread_id: str, text: str, *, attachments: list[dict] | None = None) -> None:
        self.get_thread(thread_id).append(
            {
                "role": "assistant",
                "content": str(text or ""),
                "attachments": list(attachments or []),
            }
        )

    def append_to_last_assistant(self, thread_id: str, *, attachment: dict) -> None:
        thread = self.get_thread(thread_id)
        for message in reversed(thread):
            if str(message.get("role") or "") == "assistant":
                attachments = list(message.get("attachments") or [])
                attachments.append(dict(attachment or {}))
                message["attachments"] = attachments
                return

    def update_thinking(self, thread_id: str, thinking: dict) -> None:
        state = self.get_thread_state(thread_id)
        state["last_turn_thinking"] = dict(thinking or {})

    def apply_event(self, event: Event) -> None:
        if event.type == "user_message":
            self.append_user(
                event.thread_id,
                str(event.payload.get("text") or ""),
                attachments=list(event.payload.get("attachments") or []),
            )
            return
        if event.type == "assistant_reply":
            self.append_assistant(
                event.thread_id,
                str(event.payload.get("text") or ""),
                attachments=list(event.payload.get("attachments") or []),
            )
            return
        if event.type == "assistant_attachment_added":
            self.append_to_last_assistant(
                event.thread_id,
                attachment=dict(event.payload.get("attachment") or {}),
            )
            return
        if event.type == "thinking_update":
            self.update_thinking(event.thread_id, dict(event.payload or {}))

    def thread_messages(self, thread_id: str) -> list[dict]:
        return deepcopy(self.get_thread(thread_id))

    def thread_thinking(self, thread_id: str) -> dict:
        state = self.get_thread_state(thread_id)
        return deepcopy(state.get("last_turn_thinking") or {})
