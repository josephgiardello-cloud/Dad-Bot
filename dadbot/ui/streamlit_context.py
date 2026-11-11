from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dadbot.core.dadbot import DadBot
    from dadbot.services.context_service import ContextService


def streamlit_turn_lock(bot: Any) -> Any:
    # The canonical sync path bridges into a worker that also needs _session_lock.
    with bot._session_lock:
        lock = getattr(bot, "_streamlit_turn_lock", None)
        if lock is None:
            lock = threading.RLock()
            bot._streamlit_turn_lock = lock
        return lock


class StreamlitBotPool:
    """Own one runtime per configured tenant across browser reconnects."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._bots: dict[str, tuple[str, DadBot]] = {}

    def get(self, tenant_id: str, build: str, factory: Callable[[str], DadBot]) -> DadBot:
        from dadbot_system.execution_identity import require_scoped_tenant_id

        tenant_id = require_scoped_tenant_id(tenant_id)
        with self._lock:
            current = self._bots.get(tenant_id)
            if current is not None:
                previous_build, previous_bot = current
                if previous_build == build and not getattr(previous_bot, "_shutdown_completed", False):
                    return previous_bot
                with streamlit_turn_lock(previous_bot):
                    previous_bot.shutdown()
                del self._bots[tenant_id]
            bot = factory(tenant_id)
            if bot.config.tenant_id != tenant_id:
                bot.shutdown()
                raise ValueError("Streamlit runtime tenant does not match the requested owner")
            bot._streamlit_managed_shared = True
            self._bots[tenant_id] = (build, bot)
            return bot

    def close(self) -> None:
        with self._lock:
            for _, bot in self._bots.values():
                with streamlit_turn_lock(bot):
                    bot.shutdown()
            self._bots.clear()


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
    def __init__(self, bot: Any = None, *, context_service: ContextService | None = None) -> None:
        self.bot = bot
        self.context_service = context_service

    def build_context(self, turn_context: Any) -> dict[str, Any]:
        context = self.context_service.build_context(turn_context) if self.context_service is not None else {}
        state = getattr(turn_context, "state", {})
        rich_context = state.get("rich_context") if isinstance(state, dict) else None
        messages = list(rich_context.get("messages") or []) if isinstance(rich_context, dict) else []
        if not messages:
            messages = list(getattr(self.bot, "_streamlit_context_messages", []) or [])
        if not messages:
            history = getattr(self.bot, "conversation_history", None)
            if callable(history):
                messages = list(history() or [])
        temporal = dict(state.get("temporal") or {}) if isinstance(state, dict) else {}
        return {**context, "messages": messages, "temporal": context.get("temporal", temporal)}


class SQLiteStreamlitThreadStore:
    """Persist tenant-owned display history; unqualified legacy rows are never adopted."""

    def __init__(self, db_path: str | Path | None = None, *, tenant_id: str) -> None:
        from dadbot_system.execution_identity import require_scoped_tenant_id

        self.tenant_id = require_scoped_tenant_id(tenant_id)
        configured_path = str(
            db_path or os.environ.get("DADBOT_STREAMLIT_STATE_DB") or "session_logs/streamlit_threads.sqlite3"
        )
        self.db_path = Path(configured_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS owned_chat_threads ("
                "tenant_id TEXT NOT NULL, thread_id TEXT NOT NULL, "
                "messages_json TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY (tenant_id, thread_id))"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        from dadbot_system.delegation import require_tenant_authority_if_active, tenant_mutation_fence

        with tenant_mutation_fence(self.tenant_id, required=False):
            require_tenant_authority_if_active(self.tenant_id)
            connection = sqlite3.connect(self.db_path, timeout=10.0)
            connection.row_factory = sqlite3.Row
            try:
                with connection:
                    yield connection
            finally:
                connection.close()

    def load_threads(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT thread_id, messages_json FROM owned_chat_threads WHERE tenant_id=? ORDER BY thread_id",
                (self.tenant_id,),
            ).fetchall()
        threads: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            try:
                messages = self._clean_messages(json.loads(row["messages_json"]))
            except ValueError as exc:
                raise ValueError(f"Invalid display history for thread {row['thread_id']}") from exc
            threads[str(row["thread_id"])] = messages
        return threads

    @staticmethod
    def _clean_messages(messages: Any) -> list[dict[str, str]]:
        if not isinstance(messages, list):
            raise ValueError("Display history must be a list of messages")
        cleaned = []
        for item in messages:
            if (
                not isinstance(item, dict)
                or item.get("role") not in ("user", "assistant")
                or not isinstance(item.get("text"), str)
                or not isinstance(item.get("time", ""), str)
            ):
                raise ValueError("Invalid display history message")
            cleaned.append({"role": item["role"], "text": item["text"], "time": item.get("time", "")})
        return cleaned

    def save_thread(self, thread_id: str, messages: list[dict[str, Any]]) -> None:
        if not isinstance(thread_id, str) or not thread_id:
            raise ValueError("Display history requires a nonempty thread ID")
        cleaned = self._clean_messages(messages)
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO owned_chat_threads(tenant_id, thread_id, messages_json, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(tenant_id, thread_id) DO UPDATE SET "
                "messages_json=excluded.messages_json, updated_at=excluded.updated_at",
                (
                    self.tenant_id,
                    thread_id,
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
