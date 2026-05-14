"""Thread state management mixin for RuntimeStateManager.

Extracted from dadbot/state.py to reduce the size and Halstead complexity of
that module.  RuntimeStateManager inherits from this mixin so all public
interfaces remain identical.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime

from pydantic import ValidationError

from dadbot.models import ChatThreadState, ThreadRuntimeSnapshot


class _StateThreadMixin:
    """Thread lifecycle and snapshot normalisation methods.

    Requires that the concrete class provides:
    - self.bot, self._container
    - self._normalize_planner_debug_state(value)
    - properties: history, session_moods, session_summary,
      session_summary_updated_at, session_summary_covered_messages,
      last_relationship_reflection_turn, pending_daily_checkin_context,
      active_tool_observation_context, planner_debug, chat_threads,
      thread_snapshots, active_thread_id (and their setters)
    """

    # ------------------------------------------------------------------
    # Low-level coercion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_non_negative_int(value):
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    # ------------------------------------------------------------------
    # Thread snapshot normalisation
    # ------------------------------------------------------------------

    def _normalize_thread_runtime_snapshot(self, snapshot):
        payload = dict(snapshot or {})
        normalized_payload = {
            "history": [dict(message) for message in payload.get("history", []) if isinstance(message, dict)]
            or self.bot.new_chat_session(),
            "session_moods": [str(mood) for mood in payload.get("session_moods", [])],
            "session_summary": str(payload.get("session_summary") or ""),
            "session_summary_updated_at": str(
                payload.get("session_summary_updated_at") or "",
            ).strip()
            or None,
            "session_summary_covered_messages": self._coerce_non_negative_int(
                payload.get("session_summary_covered_messages"),
            ),
            "last_relationship_reflection_turn": self._coerce_non_negative_int(
                payload.get("last_relationship_reflection_turn"),
            ),
            "pending_daily_checkin_context": bool(
                payload.get("pending_daily_checkin_context"),
            ),
            "active_tool_observation_context": str(
                payload.get("active_tool_observation_context") or "",
            ).strip()
            or None,
            "planner_debug": self._normalize_planner_debug_state(
                payload.get("planner_debug"),
            ),
            "closed": bool(payload.get("closed")),
        }
        try:
            validated = ThreadRuntimeSnapshot.model_validate(normalized_payload)
        except ValidationError:
            validated = ThreadRuntimeSnapshot.model_validate(
                {
                    "history": self.bot.new_chat_session(),
                    "session_moods": [],
                    "session_summary": "",
                    "session_summary_updated_at": None,
                    "session_summary_covered_messages": 0,
                    "last_relationship_reflection_turn": 0,
                    "pending_daily_checkin_context": False,
                    "active_tool_observation_context": None,
                    "planner_debug": self.bot.default_planner_debug_state(),
                    "closed": False,
                },
            )
        return validated.model_dump(mode="python")

    def thread_timestamp(self):
        return datetime.now().isoformat(timespec="seconds")

    def initial_thread_snapshot(self):
        return self._normalize_thread_runtime_snapshot(
            {
                "history": self.bot.new_chat_session(),
                "session_moods": [],
                "session_summary": "",
                "session_summary_updated_at": None,
                "session_summary_covered_messages": 0,
                "last_relationship_reflection_turn": 0,
                "pending_daily_checkin_context": False,
                "active_tool_observation_context": None,
                "planner_debug": self.bot.default_planner_debug_state(),
                "closed": False,
            },
        )

    def normalize_thread_snapshot(self, snapshot):
        return self._normalize_thread_runtime_snapshot(snapshot)

    def current_thread_runtime_snapshot(self):
        active_thread_id = self._container.state.active_thread_id
        closed = False
        for thread in self._container.state.chat_threads:
            if thread.get("thread_id") == active_thread_id:
                closed = bool(thread.get("closed"))
                break
        return self.normalize_thread_snapshot(
            {
                "history": self.history,
                "session_moods": self.session_moods,
                "session_summary": self.session_summary,
                "session_summary_updated_at": self.session_summary_updated_at,
                "session_summary_covered_messages": self.session_summary_covered_messages,
                "last_relationship_reflection_turn": self.last_relationship_reflection_turn,
                "pending_daily_checkin_context": self.pending_daily_checkin_context,
                "active_tool_observation_context": self.active_tool_observation_context,
                "planner_debug": self.planner_debug,
                "closed": closed,
            },
        )

    @staticmethod
    def thread_turn_count(snapshot):
        history = list((snapshot or {}).get("history", []))
        return sum(1 for message in history if isinstance(message, dict) and message.get("role") == "user")

    @staticmethod
    def thread_preview(snapshot):
        history = list((snapshot or {}).get("history", []))
        for message in reversed(history):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "system":
                continue
            content = re.sub(r"\s+", " ", str(message.get("content") or "")).strip()
            if content:
                return content[:72]
        return ""

    # ------------------------------------------------------------------
    # Thread entry normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_thread_entry_fields(
        entry: dict,
        fallback_index: int,
        fallback_timestamp: str,
    ) -> tuple[str, str, str, str]:
        """Normalise the identity fields of a raw thread entry dict.

        Returns ``(thread_id, created_at, updated_at, title)``.
        Extracted from ``normalize_chat_thread_entry`` to reduce cyclomatic
        complexity of that method.
        """
        e = entry or {}
        thread_id = str(e.get("thread_id") or e.get("id") or "").strip() or uuid.uuid4().hex
        created_at = str(e.get("created_at") or fallback_timestamp).strip() or fallback_timestamp
        updated_at = str(e.get("updated_at") or created_at).strip() or created_at
        title = str(e.get("title") or "").strip() or f"Chat {fallback_index}"
        return thread_id, created_at, updated_at, title

    def normalize_chat_thread_entry(self, entry, snapshot=None, fallback_index=1):
        normalized_snapshot = self.normalize_thread_snapshot(snapshot)
        thread_id, created_at, updated_at, title = self._resolve_thread_entry_fields(
            entry or {}, fallback_index, self.thread_timestamp()
        )
        last_message = str(
            (entry or {}).get("last_message") or self.thread_preview(normalized_snapshot),
        )
        turn_count = self._coerce_non_negative_int(
            (entry or {}).get("turn_count") or self.thread_turn_count(normalized_snapshot),
        )
        payload = {
            "thread_id": thread_id,
            "title": title,
            "created_at": created_at,
            "updated_at": updated_at,
            "last_message": last_message,
            "turn_count": turn_count,
            "closed": bool(
                (entry or {}).get("closed") or normalized_snapshot.get("closed"),
            ),
        }
        try:
            validated = ChatThreadState.model_validate(payload)
        except ValidationError:
            validated = ChatThreadState(
                thread_id=thread_id,
                title=f"Chat {fallback_index}",
                created_at=created_at,
                updated_at=updated_at,
                last_message="",
                turn_count=0,
                closed=False,
            )
        return validated.model_dump(mode="python")

    # ------------------------------------------------------------------
    # Snapshot apply / restore
    # ------------------------------------------------------------------

    def apply_thread_snapshot_unlocked(self, snapshot):
        normalized = self.normalize_thread_snapshot(snapshot)
        state = self._container.state
        state.history = [dict(message) for message in normalized["history"]]
        state.session_moods = [str(mood) for mood in normalized["session_moods"]]
        state.session_summary = str(normalized["session_summary"] or "")
        state.session_summary_updated_at = normalized.get("session_summary_updated_at")
        state.session_summary_covered_messages = int(
            normalized.get("session_summary_covered_messages") or 0,
        )
        state.last_relationship_reflection_turn = int(
            normalized.get("last_relationship_reflection_turn") or 0,
        )
        state.pending_daily_checkin_context = bool(
            normalized.get("pending_daily_checkin_context"),
        )
        active_tool = str(
            normalized.get("active_tool_observation_context") or "",
        ).strip()
        state.active_tool_observation_context = active_tool or None
        state.planner_debug = self._normalize_planner_debug_state(
            normalized.get("planner_debug"),
        )
        self.bot._recent_mood_detections.clear()

    # ------------------------------------------------------------------
    # Thread state machine
    # ------------------------------------------------------------------

    def ensure_chat_thread_state(self, preserve_active_runtime=False):
        with self.bot._session_lock:
            state = self._container.state
            seeded_current_snapshot = self.normalize_thread_snapshot(
                {
                    "history": state.history,
                    "session_moods": state.session_moods,
                    "session_summary": state.session_summary,
                    "session_summary_updated_at": state.session_summary_updated_at,
                    "session_summary_covered_messages": state.session_summary_covered_messages,
                    "last_relationship_reflection_turn": state.last_relationship_reflection_turn,
                    "pending_daily_checkin_context": state.pending_daily_checkin_context,
                    "active_tool_observation_context": state.active_tool_observation_context,
                    "planner_debug": state.planner_debug,
                },
            )

            if not state.chat_threads:
                thread_id = state.active_thread_id or uuid.uuid4().hex
                state.thread_snapshots = {thread_id: seeded_current_snapshot}
                state.chat_threads = [
                    self.normalize_chat_thread_entry(
                        {"thread_id": thread_id},
                        seeded_current_snapshot,
                        fallback_index=1,
                    ),
                ]
                state.active_thread_id = thread_id
                self.apply_thread_snapshot_unlocked(seeded_current_snapshot)
                return thread_id

            normalized_snapshots = {}
            normalized_threads = []
            for index, entry in enumerate(state.chat_threads, start=1):
                thread_id = str(entry.get("thread_id") or entry.get("id") or "").strip() or uuid.uuid4().hex
                snapshot = state.thread_snapshots.get(thread_id)
                if (thread_id == state.active_thread_id and preserve_active_runtime) or (
                    snapshot is None and thread_id == state.active_thread_id
                ):
                    snapshot = seeded_current_snapshot
                snapshot = self.normalize_thread_snapshot(snapshot)
                normalized_snapshots[thread_id] = snapshot
                normalized_threads.append(
                    self.normalize_chat_thread_entry(
                        {**dict(entry), "thread_id": thread_id},
                        snapshot,
                        fallback_index=index,
                    ),
                )

            if not normalized_threads:
                thread_id = uuid.uuid4().hex
                normalized_snapshots = {thread_id: seeded_current_snapshot}
                normalized_threads = [
                    self.normalize_chat_thread_entry(
                        {"thread_id": thread_id},
                        seeded_current_snapshot,
                        fallback_index=1,
                    ),
                ]

            state.thread_snapshots = normalized_snapshots
            state.chat_threads = normalized_threads
            active_thread_id = (
                state.active_thread_id
                if state.active_thread_id in normalized_snapshots
                else normalized_threads[0]["thread_id"]
            )
            state.active_thread_id = active_thread_id
            self.apply_thread_snapshot_unlocked(normalized_snapshots[active_thread_id])
            return active_thread_id

    def active_chat_thread(self):
        active_thread_id = self.ensure_chat_thread_state()
        for thread in self.chat_threads:
            if thread.get("thread_id") == active_thread_id:
                return dict(thread)
        return None

    def list_chat_threads(self):
        self.sync_active_thread_snapshot()
        return [dict(thread) for thread in self.chat_threads]

    def sync_active_thread_snapshot(self):
        with self.bot._session_lock:
            active_thread_id = self.ensure_chat_thread_state(
                preserve_active_runtime=True,
            )
            snapshot = self.current_thread_runtime_snapshot()
            self.thread_snapshots[active_thread_id] = snapshot
            updated_threads = []
            now = self.thread_timestamp()
            for index, thread in enumerate(self.chat_threads, start=1):
                if thread.get("thread_id") == active_thread_id:
                    updated_threads.append(
                        self.normalize_chat_thread_entry(
                            {
                                **dict(thread),
                                "updated_at": now,
                                "last_message": self.thread_preview(snapshot),
                                "turn_count": self.thread_turn_count(snapshot),
                                "closed": bool(snapshot.get("closed")),
                            },
                            snapshot,
                            fallback_index=index,
                        ),
                    )
                else:
                    existing_snapshot = self.normalize_thread_snapshot(
                        self.thread_snapshots.get(thread.get("thread_id")),
                    )
                    updated_threads.append(
                        self.normalize_chat_thread_entry(
                            thread,
                            existing_snapshot,
                            fallback_index=index,
                        ),
                    )
            self.chat_threads = updated_threads
            return dict(self.active_chat_thread() or {})

    def create_chat_thread(self, title=""):
        with self.bot._session_lock:
            self.ensure_chat_thread_state(preserve_active_runtime=True)
            self.sync_active_thread_snapshot()
            thread_id = uuid.uuid4().hex
            snapshot = self.initial_thread_snapshot()
            now = self.thread_timestamp()
            self.thread_snapshots[thread_id] = snapshot
            self.chat_threads = [
                self.normalize_chat_thread_entry(
                    thread,
                    self.thread_snapshots.get(thread.get("thread_id")),
                    fallback_index=index,
                )
                for index, thread in enumerate(self.chat_threads, start=1)
            ] + [
                self.normalize_chat_thread_entry(
                    {
                        "thread_id": thread_id,
                        "title": str(title or "").strip(),
                        "created_at": now,
                        "updated_at": now,
                    },
                    snapshot,
                    fallback_index=len(self.chat_threads) + 1,
                ),
            ]
            self.active_thread_id = thread_id
            self.apply_thread_snapshot_unlocked(snapshot)
            return dict(self.active_chat_thread() or {})

    def switch_chat_thread(self, thread_id):
        normalized_thread_id = str(thread_id or "").strip()
        if not normalized_thread_id:
            raise ValueError("thread_id is required")

        with self.bot._session_lock:
            self.ensure_chat_thread_state(preserve_active_runtime=True)
            self.sync_active_thread_snapshot()
            if normalized_thread_id not in self.thread_snapshots:
                raise KeyError(f"Unknown chat thread: {normalized_thread_id}")
            self.active_thread_id = normalized_thread_id
            self.apply_thread_snapshot_unlocked(
                self.thread_snapshots[normalized_thread_id],
            )
            return dict(self.active_chat_thread() or {})

    def mark_chat_thread_closed(self, thread_id=None, closed=True):
        with self.bot._session_lock:
            target_thread_id = str(
                thread_id or self.ensure_chat_thread_state(preserve_active_runtime=True),
            ).strip()
            if target_thread_id not in self.thread_snapshots:
                raise KeyError(f"Unknown chat thread: {target_thread_id}")
            if target_thread_id == self.active_thread_id:
                snapshot = self.current_thread_runtime_snapshot()
            else:
                snapshot = self.normalize_thread_snapshot(
                    self.thread_snapshots[target_thread_id],
                )
            snapshot["closed"] = bool(closed)
            self.thread_snapshots[target_thread_id] = snapshot
            if target_thread_id == self.active_thread_id:
                self.apply_thread_snapshot_unlocked(snapshot)
            self.chat_threads = [
                self.normalize_chat_thread_entry(
                    {
                        **dict(thread),
                        "closed": bool(closed)
                        if thread.get("thread_id") == target_thread_id
                        else bool(thread.get("closed")),
                    },
                    self.thread_snapshots.get(thread.get("thread_id")),
                    fallback_index=index,
                )
                for index, thread in enumerate(self.chat_threads, start=1)
            ]
            return dict(self.active_chat_thread() or {}) if target_thread_id == self.active_thread_id else None
