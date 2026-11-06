from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


def classify_profile_question(user_input: str) -> str:
    text = str(user_input or "").strip().lower()
    if not text:
        return ""

    unrelated_context = (
        "in this code",
        "in the code",
        "in this repo",
        "in the repo",
        "in this project",
        "in the project",
        "in this file",
        "in the file",
        "in this app",
        "in this system",
        "in this model",
        "in this example",
        "in the example",
        "in this scenario",
        "in the scenario",
        "in this story",
        "in the story",
        "in this conversation",
    )
    if any(marker in text for marker in unrelated_context):
        return ""

    identity_markers = (
        "who are you",
        "what's your story",
        "whats your story",
        "tell me your story",
        "your story",
        "about yourself",
        "tell me about yourself",
    )
    if any(marker in text for marker in identity_markers):
        return "identity"
    age_markers = (
        "how old are you",
        "your age",
        "dad age",
        "what year were you born",
        "when were you born",
        "your birthday",
    )
    if any(marker in text for marker in age_markers):
        return "age"
    sports_markers = (
        "baseball",
        "football",
        "soccer",
        "basketball",
        "sports team",
        "favorite team",
        "favourite team",
        "which team do you like",
        "what team do you like",
    )
    if any(marker in text for marker in sports_markers):
        return "preferences"
    return ""


def set_runtime_thread_id(bot: Any, thread_id: str) -> None:
    if bot is None:
        return
    thread_id = str(thread_id or "default")
    manager = bot.runtime_state_manager
    with bot._session_lock:
        manager.ensure_chat_thread_state(preserve_active_runtime=True)
        if manager.active_thread_id == thread_id:
            return
        if thread_id not in manager.thread_snapshots:
            manager.thread_snapshots = {
                **manager.thread_snapshots,
                thread_id: manager.initial_thread_snapshot(),
            }
            manager.chat_threads = [*manager.chat_threads, {"thread_id": thread_id}]
        manager.switch_chat_thread(thread_id)


def serialize_thread_messages(
    thread_messages: list[dict[str, Any]],
    current_input: str = "",
    *,
    limit: int = 24,
) -> list[dict[str, str]]:
    messages = []
    for item in list(thread_messages or []):
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("text") or item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    current_text = str(current_input or "").strip()
    if current_text and not (
        messages
        and messages[-1]["role"] == "user"
        and messages[-1]["content"] == current_text
    ):
        messages.append({"role": "user", "content": current_text})

    return messages[-max(1, int(limit or 24)) :]


class StreamlitContextService:
    def __init__(self, bot: Any = None) -> None:
        self.bot = bot

    def build_context(self, turn_context: Any) -> dict[str, Any]:
        state = getattr(turn_context, "state", {})
        rich_context = state.get("rich_context") if isinstance(state, dict) else None
        messages = list(rich_context.get("messages") or []) if isinstance(rich_context, dict) else []
        if not messages:
            messages = list(getattr(self.bot, "_streamlit_context_messages", []) or [])
        if not messages:
            history = getattr(self.bot, "conversation_history", None)
            if callable(history):
                try:
                    messages = list(history() or [])
                except Exception:
                    messages = []
        temporal = dict(state.get("temporal") or {}) if isinstance(state, dict) else {}
        return {"messages": messages, "temporal": temporal}


class SQLiteStreamlitThreadStore:
    """Persist visible Streamlit threads locally without storing generated audio blobs."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        configured_path = str(db_path or os.environ.get("DADBOT_STREAMLIT_STATE_DB") or "session_logs/streamlit_threads.sqlite3")
        self.db_path = Path(configured_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS chat_threads ("
                "thread_id TEXT PRIMARY KEY, messages_json TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def load_threads(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT thread_id, messages_json FROM chat_threads ORDER BY thread_id").fetchall()
        threads: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            try:
                messages = json.loads(str(row["messages_json"] or "[]"))
            except (TypeError, ValueError):
                messages = []
            threads[str(row["thread_id"])] = [item for item in messages if isinstance(item, dict)]
        return threads

    def save_thread(self, thread_id: str, messages: list[dict[str, Any]]) -> None:
        cleaned = []
        for item in list(messages or []):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            text = str(item.get("text") or item.get("content") or "")
            if role in {"user", "assistant"}:
                cleaned.append(
                    {
                        "role": role,
                        "text": text,
                        "time": str(item.get("time") or ""),
                    }
                )

        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO chat_threads(thread_id, messages_json, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(thread_id) DO UPDATE SET messages_json=excluded.messages_json, updated_at=excluded.updated_at",
                (
                    str(thread_id or "default"),
                    json.dumps(cleaned, ensure_ascii=True, separators=(",", ":")),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )


class StreamlitReflectionService:
    def __init__(self, bot: Any, internal_state_manager: Any = None) -> None:
        self.bot = bot
        self.internal_state_manager = internal_state_manager

    def reflect_after_turn(self, turn_text: str, current_mood: str, reply_text: str = "") -> dict[str, Any]:
        reflect = getattr(self.internal_state_manager, "reflect_after_turn", None)
        if not callable(reflect):
            return {"status": "unavailable"}
        return dict(
            reflect(
                user_input=str(turn_text or ""),
                current_mood=str(current_mood or "neutral"),
                dad_reply=str(reply_text or ""),
            )
            or {}
        )

    def reflect(self, turn_context: Any, result: Any) -> dict[str, Any]:
        state = getattr(turn_context, "state", {})
        state = state if isinstance(state, dict) else {}
        if isinstance(result, tuple):
            reply = result[0] if result else ""
        else:
            reply = result
        mood = str(state.get("mood") or self.bot.last_saved_mood() or "neutral")
        return self.reflect_after_turn(
            str(getattr(turn_context, "user_input", "") or ""),
            mood,
            str(reply or ""),
        )