from __future__ import annotations

import copy
import hashlib
import re
import uuid
from datetime import datetime

from dadbot.models import ChatThreadState, PlannerDebugState, ThreadRuntimeSnapshot
from pydantic import ValidationError


class RuntimeStateManager:
    def __init__(self, bot, container):
        self.bot = bot
        self._container = container
        self._token_count_cache = {}
        self._message_token_cost_cache = {}
        self._prompt_history_cache = {}

    @property
    def container(self):
        return self._container

    @property
    def history(self):
        return self._container.state.history

    @history.setter
    def history(self, value):
        self._container.state.history = [dict(message) for message in (value or []) if isinstance(message, dict)]

    @property
    def session_moods(self):
        return self._container.state.session_moods

    @session_moods.setter
    def session_moods(self, value):
        self._container.state.session_moods = [str(mood) for mood in (value or [])]

    @property
    def session_summary(self):
        return self._container.state.session_summary

    @session_summary.setter
    def session_summary(self, value):
        self._container.state.session_summary = str(value or "")

    @property
    def session_summary_updated_at(self):
        return self._container.state.session_summary_updated_at

    @session_summary_updated_at.setter
    def session_summary_updated_at(self, value):
        self._container.state.session_summary_updated_at = value

    @property
    def session_summary_covered_messages(self):
        return self._container.state.session_summary_covered_messages

    @session_summary_covered_messages.setter
    def session_summary_covered_messages(self, value):
        self._container.state.session_summary_covered_messages = int(value or 0)

    @property
    def last_relationship_reflection_turn(self):
        return self._container.state.last_relationship_reflection_turn

    @last_relationship_reflection_turn.setter
    def last_relationship_reflection_turn(self, value):
        self._container.state.last_relationship_reflection_turn = int(value or 0)

    @property
    def pending_daily_checkin_context(self):
        return self._container.state.pending_daily_checkin_context

    @pending_daily_checkin_context.setter
    def pending_daily_checkin_context(self, value):
        self._container.state.pending_daily_checkin_context = bool(value)

    @property
    def active_tool_observation_context(self):
        return self._container.state.active_tool_observation_context

    @active_tool_observation_context.setter
    def active_tool_observation_context(self, value):
        normalized = str(value or "").strip()
        self._container.state.active_tool_observation_context = normalized or None

    @property
    def planner_debug(self):
        return self._container.state.planner_debug

    @planner_debug.setter
    def planner_debug(self, value):
        self._container.state.planner_debug = self._normalize_planner_debug_state(value)

    @property
    def chat_threads(self):
        return self._container.state.chat_threads

    @chat_threads.setter
    def chat_threads(self, value):
        normalized = []
        snapshots = dict(self._container.state.thread_snapshots or {})
        for index, item in enumerate(value or [], start=1):
            if not isinstance(item, dict):
                continue
            thread_id = str(item.get("thread_id") or item.get("id") or "").strip()
            normalized.append(self.normalize_chat_thread_entry(item, snapshots.get(thread_id), fallback_index=index))
        self._container.state.chat_threads = normalized

    @property
    def active_thread_id(self):
        return self._container.state.active_thread_id

    @active_thread_id.setter
    def active_thread_id(self, value):
        normalized = str(value or "").strip()
        self._container.state.active_thread_id = normalized or None

    @property
    def thread_snapshots(self):
        return self._container.state.thread_snapshots

    @thread_snapshots.setter
    def thread_snapshots(self, value):
        self._container.state.thread_snapshots = {
            str(thread_id): self.normalize_thread_snapshot(snapshot)
            for thread_id, snapshot in (value or {}).items()
            if isinstance(snapshot, dict)
        }

    def _normalize_planner_debug_state(self, value):
        payload = dict(self.bot.default_planner_debug_state())
        if isinstance(value, dict):
            payload.update(value)
        payload["updated_at"] = str(payload.get("updated_at") or "").strip() or None
        payload["user_input"] = str(payload.get("user_input") or "").strip()
        payload["current_mood"] = self.bot.normalize_mood(payload.get("current_mood"))
        for field_name in (
            "planner_status",
            "planner_reason",
            "planner_tool",
            "planner_observation",
            "fallback_status",
            "fallback_reason",
            "fallback_tool",
            "fallback_observation",
            "final_path",
        ):
            payload[field_name] = str(payload.get(field_name) or "").strip()
        payload["planner_parameters"] = dict(payload.get("planner_parameters") or {}) if isinstance(payload.get("planner_parameters"), dict) else {}
        try:
            validated = PlannerDebugState.model_validate(payload)
        except ValidationError:
            validated = PlannerDebugState.model_validate(self.bot.default_planner_debug_state())
        return validated.model_dump(mode="python")

    @staticmethod
    def _coerce_non_negative_int(value):
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    def _normalize_thread_runtime_snapshot(self, snapshot):
        payload = dict(snapshot or {})
        normalized_payload = {
            "history": [dict(message) for message in payload.get("history", []) if isinstance(message, dict)] or self.bot.new_chat_session(),
            "session_moods": [str(mood) for mood in payload.get("session_moods", [])],
            "session_summary": str(payload.get("session_summary") or ""),
            "session_summary_updated_at": str(payload.get("session_summary_updated_at") or "").strip() or None,
            "session_summary_covered_messages": self._coerce_non_negative_int(payload.get("session_summary_covered_messages")),
            "last_relationship_reflection_turn": self._coerce_non_negative_int(payload.get("last_relationship_reflection_turn")),
            "pending_daily_checkin_context": bool(payload.get("pending_daily_checkin_context")),
            "active_tool_observation_context": str(payload.get("active_tool_observation_context") or "").strip() or None,
            "planner_debug": self._normalize_planner_debug_state(payload.get("planner_debug")),
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
                }
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
            }
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
            }
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

    def normalize_chat_thread_entry(self, entry, snapshot=None, fallback_index=1):
        normalized_snapshot = self.normalize_thread_snapshot(snapshot)
        thread_id = str((entry or {}).get("thread_id") or (entry or {}).get("id") or "").strip() or uuid.uuid4().hex
        created_at = str((entry or {}).get("created_at") or self.thread_timestamp()).strip() or self.thread_timestamp()
        updated_at = str((entry or {}).get("updated_at") or created_at).strip() or created_at
        title = str((entry or {}).get("title") or "").strip() or f"Chat {fallback_index}"
        last_message = str((entry or {}).get("last_message") or self.thread_preview(normalized_snapshot))
        turn_count = self._coerce_non_negative_int((entry or {}).get("turn_count") or self.thread_turn_count(normalized_snapshot))
        payload = {
            "thread_id": thread_id,
            "title": title,
            "created_at": created_at,
            "updated_at": updated_at,
            "last_message": last_message,
            "turn_count": turn_count,
            "closed": bool((entry or {}).get("closed") or normalized_snapshot.get("closed")),
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

    def apply_thread_snapshot_unlocked(self, snapshot):
        normalized = self.normalize_thread_snapshot(snapshot)
        state = self._container.state
        state.history = [dict(message) for message in normalized["history"]]
        state.session_moods = [str(mood) for mood in normalized["session_moods"]]
        state.session_summary = str(normalized["session_summary"] or "")
        state.session_summary_updated_at = normalized.get("session_summary_updated_at")
        state.session_summary_covered_messages = int(normalized.get("session_summary_covered_messages") or 0)
        state.last_relationship_reflection_turn = int(normalized.get("last_relationship_reflection_turn") or 0)
        state.pending_daily_checkin_context = bool(normalized.get("pending_daily_checkin_context"))
        active_tool = str(normalized.get("active_tool_observation_context") or "").strip()
        state.active_tool_observation_context = active_tool or None
        state.planner_debug = self._normalize_planner_debug_state(normalized.get("planner_debug"))
        self.bot._recent_mood_detections.clear()

    def estimate_token_count_cached(self, text):
        normalized = str(text or "")
        cache_key = (str(self.bot.ACTIVE_MODEL or "").strip().lower(), normalized)
        cached = self._token_count_cache.get(cache_key)
        if cached is not None:
            return cached

        estimate = self.bot.estimate_token_count(normalized)
        if len(self._token_count_cache) >= 8192:
            self._token_count_cache.pop(next(iter(self._token_count_cache)))
        self._token_count_cache[cache_key] = estimate
        return estimate

    def message_token_cost(self, message):
        role = str((message or {}).get("role", ""))
        content = str((message or {}).get("content", ""))
        cache_key = (str(self.bot.ACTIVE_MODEL or "").strip().lower(), role, content)
        cached = self._message_token_cost_cache.get(cache_key)
        if cached is not None:
            return cached

        total_cost = 4 + self.estimate_token_count_cached(role) + self.estimate_token_count_cached(content)
        if len(self._message_token_cost_cache) >= 4096:
            self._message_token_cost_cache.pop(next(iter(self._message_token_cost_cache)))
        self._message_token_cost_cache[cache_key] = total_cost
        return total_cost

    def prompt_history_cache_key(self, system_prompt, user_input, recent_history):
        signature = hashlib.sha1(
            "\n".join(
                f"{str(message.get('role', ''))}\x1f{str(message.get('content', ''))}"
                for message in recent_history
                if isinstance(message, dict)
            ).encode("utf-8")
        ).hexdigest()
        return (
            str(self.bot.ACTIVE_MODEL or "").strip().lower(),
            self.bot.effective_context_token_budget(),
            int(self.bot.RESERVED_RESPONSE_TOKENS or 0),
            int(self.bot.MAX_HISTORY_MESSAGES_SCAN or 0),
            str(system_prompt or ""),
            str(user_input or ""),
            signature,
        )

    def get_cached_prompt_history(self, cache_key):
        cached = self._prompt_history_cache.get(cache_key)
        if cached is None:
            return None
        return [dict(message) for message in cached]

    def remember_prompt_history(self, cache_key, messages):
        snapshot = [dict(message) for message in messages if isinstance(message, dict)]
        if len(self._prompt_history_cache) >= 512:
            self._prompt_history_cache.pop(next(iter(self._prompt_history_cache)))
        self._prompt_history_cache[cache_key] = snapshot

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
                }
            )

            if not state.chat_threads:
                thread_id = state.active_thread_id or uuid.uuid4().hex
                state.thread_snapshots = {thread_id: seeded_current_snapshot}
                state.chat_threads = [self.normalize_chat_thread_entry({"thread_id": thread_id}, seeded_current_snapshot, fallback_index=1)]
                state.active_thread_id = thread_id
                self.apply_thread_snapshot_unlocked(seeded_current_snapshot)
                return thread_id

            normalized_snapshots = {}
            normalized_threads = []
            for index, entry in enumerate(state.chat_threads, start=1):
                thread_id = str(entry.get("thread_id") or entry.get("id") or "").strip() or uuid.uuid4().hex
                snapshot = state.thread_snapshots.get(thread_id)
                if thread_id == state.active_thread_id and preserve_active_runtime:
                    snapshot = seeded_current_snapshot
                elif snapshot is None and thread_id == state.active_thread_id:
                    snapshot = seeded_current_snapshot
                snapshot = self.normalize_thread_snapshot(snapshot)
                normalized_snapshots[thread_id] = snapshot
                normalized_threads.append(
                    self.normalize_chat_thread_entry({**dict(entry), "thread_id": thread_id}, snapshot, fallback_index=index)
                )

            if not normalized_threads:
                thread_id = uuid.uuid4().hex
                normalized_snapshots = {thread_id: seeded_current_snapshot}
                normalized_threads = [self.normalize_chat_thread_entry({"thread_id": thread_id}, seeded_current_snapshot, fallback_index=1)]

            state.thread_snapshots = normalized_snapshots
            state.chat_threads = normalized_threads
            active_thread_id = state.active_thread_id if state.active_thread_id in normalized_snapshots else normalized_threads[0]["thread_id"]
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
            active_thread_id = self.ensure_chat_thread_state(preserve_active_runtime=True)
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
                        )
                    )
                else:
                    existing_snapshot = self.normalize_thread_snapshot(self.thread_snapshots.get(thread.get("thread_id")))
                    updated_threads.append(self.normalize_chat_thread_entry(thread, existing_snapshot, fallback_index=index))
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
                self.normalize_chat_thread_entry(thread, self.thread_snapshots.get(thread.get("thread_id")), fallback_index=index)
                for index, thread in enumerate(self.chat_threads, start=1)
            ] + [
                self.normalize_chat_thread_entry(
                    {"thread_id": thread_id, "title": str(title or "").strip(), "created_at": now, "updated_at": now},
                    snapshot,
                    fallback_index=len(self.chat_threads) + 1,
                )
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
            self.apply_thread_snapshot_unlocked(self.thread_snapshots[normalized_thread_id])
            return dict(self.active_chat_thread() or {})

    def mark_chat_thread_closed(self, thread_id=None, closed=True):
        with self.bot._session_lock:
            target_thread_id = str(thread_id or self.ensure_chat_thread_state(preserve_active_runtime=True)).strip()
            if target_thread_id not in self.thread_snapshots:
                raise KeyError(f"Unknown chat thread: {target_thread_id}")
            if target_thread_id == self.active_thread_id:
                snapshot = self.current_thread_runtime_snapshot()
            else:
                snapshot = self.normalize_thread_snapshot(self.thread_snapshots[target_thread_id])
            snapshot["closed"] = bool(closed)
            self.thread_snapshots[target_thread_id] = snapshot
            if target_thread_id == self.active_thread_id:
                self.apply_thread_snapshot_unlocked(snapshot)
            self.chat_threads = [
                self.normalize_chat_thread_entry(
                    {**dict(thread), "closed": bool(closed) if thread.get("thread_id") == target_thread_id else bool(thread.get("closed"))},
                    self.thread_snapshots.get(thread.get("thread_id")),
                    fallback_index=index,
                )
                for index, thread in enumerate(self.chat_threads, start=1)
            ]
            return dict(self.active_chat_thread() or {}) if target_thread_id == self.active_thread_id else None

    def reset_session_state(self):
        with self.bot._session_lock:
            self._container.load_snapshot(
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
                    "chat_threads": [],
                    "active_thread_id": None,
                    "thread_snapshots": {},
                }
            )
            self.bot._recent_mood_detections.clear()
            self._prompt_history_cache.clear()
            self.ensure_chat_thread_state()

    def conversation_history(self):
        with self.bot._session_lock:
            history = [dict(message) for message in self.history]
        if history and history[0].get("role") == "system":
            return history[1:]
        return history

    def prompt_history(self):
        history = self.conversation_history()
        if len(history) <= self.bot.RECENT_HISTORY_WINDOW:
            return history
        return history[-self.bot.RECENT_HISTORY_WINDOW :]

    def prompt_history_token_budget(self, system_prompt, user_input):
        return max(
            0,
            self.bot.effective_context_token_budget()
            - self.bot.RESERVED_RESPONSE_TOKENS
            - self.bot.message_token_cost({"role": "system", "content": system_prompt})
            - self.bot.message_token_cost({"role": "user", "content": user_input}),
        )

    def trim_text_to_token_budget(self, text, token_budget):
        normalized = str(text or "")
        if token_budget <= 0:
            return ""
        tokenizer = self.bot.current_tokenizer(model_name=self.bot.ACTIVE_MODEL)
        if tokenizer is not None:
            try:
                encoded = tokenizer.encode(normalized)
                if len(encoded) <= token_budget:
                    return normalized
                trimmed = tokenizer.decode(encoded[:token_budget]).rstrip()
                if trimmed and trimmed != normalized:
                    trimmed = trimmed.rstrip(" ,.;:") + "..."
                return trimmed
            except Exception:
                pass

        estimated_tokens = self.bot.estimate_token_count(normalized)
        if estimated_tokens <= token_budget:
            return normalized

        ratio = token_budget / max(1, estimated_tokens)
        cutoff = max(1, min(len(normalized), int(len(normalized) * ratio)))
        trimmed = normalized[:cutoff].rstrip()
        if trimmed and trimmed != normalized:
            trimmed = trimmed.rstrip(" ,.;:") + "..."
        return trimmed

    def trim_message_to_token_budget(self, message, token_budget):
        role = str(message.get("role", ""))
        overhead = 4 + self.bot.estimate_token_count(role)
        content_budget = max(1, token_budget - overhead)
        trimmed_content = self.bot.trim_text_to_token_budget(message.get("content", ""), content_budget)
        return {
            **message,
            "content": trimmed_content,
        }

    def token_budgeted_prompt_history(self, system_prompt, user_input):
        recent_history = self.conversation_history()[-self.bot.MAX_HISTORY_MESSAGES_SCAN :]
        token_budget = self.prompt_history_token_budget(system_prompt, user_input)
        if token_budget <= 0 or not recent_history:
            return []

        cache_key = self.prompt_history_cache_key(system_prompt, user_input, recent_history)
        cached = self.get_cached_prompt_history(cache_key)
        if cached is not None:
            return cached

        selected_messages = []
        remaining_budget = token_budget
        total_messages = len(recent_history)

        for index, message in enumerate(recent_history):
            remaining_messages = total_messages - index - 1
            reserve_for_rest = remaining_messages * 24
            allowed_budget = max(24, remaining_budget - reserve_for_rest)
            candidate = message
            message_cost = self.bot.message_token_cost(candidate)

            if message_cost > allowed_budget:
                candidate = self.bot.trim_message_to_token_budget(candidate, allowed_budget)
                if not candidate.get("content"):
                    continue
                message_cost = self.bot.message_token_cost(candidate)

            if message_cost > remaining_budget:
                candidate = self.bot.trim_message_to_token_budget(candidate, remaining_budget)
                if not candidate.get("content"):
                    break
                message_cost = self.bot.message_token_cost(candidate)
                if message_cost > remaining_budget:
                    break

            selected_messages.append(candidate)
            remaining_budget -= message_cost
            if remaining_budget <= 0:
                break

        self.remember_prompt_history(cache_key, selected_messages)
        return selected_messages

    def session_turn_count(self):
        return len([message for message in self.conversation_history() if message.get("role") == "user"])

    def snapshot_session_state(self):
        with self.bot._session_lock, self.bot._io_lock:
            self.sync_active_thread_snapshot()
            snapshot = self._container.snapshot()
            history = [dict(message) for message in snapshot.get("history", []) if isinstance(message, dict)]
            if not history:
                snapshot["history"] = self.bot.new_chat_session()
            memory_store_snapshot = copy.deepcopy(self.bot.prepare_memory_store_for_save())
            if isinstance(memory_store_snapshot.get("recent_moods"), list):
                memory_store_snapshot["recent_moods"] = [
                    self.bot.normalize_mood(item.get("mood"))
                    for item in memory_store_snapshot.get("recent_moods", [])
                    if isinstance(item, dict)
                ]
            snapshot["memory_store"] = memory_store_snapshot
            return snapshot

    def load_session_state_snapshot(self, snapshot):
        payload = copy.deepcopy(dict(snapshot or {}))
        normalized_memory_store = None
        memory_store_payload = payload.get("memory_store")
        if isinstance(memory_store_payload, dict):
            normalized_memory_store = self.bot.memory_manager.normalize_memory_store(memory_store_payload)
            payload["memory_store"] = copy.deepcopy(normalized_memory_store)

        with self.bot._session_lock, self.bot._io_lock:
            history = [dict(message) for message in payload.get("history", []) if isinstance(message, dict)]
            if not history:
                payload["history"] = self.bot.new_chat_session()
            if normalized_memory_store is not None:
                self.bot.MEMORY_STORE = normalized_memory_store
            self._container.load_snapshot(payload)
            self.ensure_chat_thread_state()
        return self.snapshot_session_state()

    def set_active_tool_observation(self, observation):
        with self.bot._session_lock:
            self.active_tool_observation_context = str(observation or "").strip() or None

    def begin_planner_debug(self, user_input, current_mood):
        with self.bot._session_lock:
            snapshot = self.bot.default_planner_debug_state()
            snapshot["updated_at"] = datetime.now().isoformat(timespec="seconds")
            snapshot["user_input"] = str(user_input or "").strip()
            snapshot["current_mood"] = self.bot.normalize_mood(current_mood)
            self.planner_debug = snapshot
        return self.planner_debug_snapshot()

    def update_planner_debug(self, **fields):
        with self.bot._session_lock:
            snapshot = self._normalize_planner_debug_state(self.planner_debug)
            if not snapshot.get("updated_at"):
                snapshot["updated_at"] = datetime.now().isoformat(timespec="seconds")
            for key, value in fields.items():
                if key == "planner_parameters":
                    snapshot[key] = dict(value) if isinstance(value, dict) else {}
                elif key == "current_mood":
                    snapshot[key] = self.bot.normalize_mood(value)
                elif key in {
                    "updated_at",
                    "user_input",
                    "planner_status",
                    "planner_reason",
                    "planner_tool",
                    "planner_observation",
                    "fallback_status",
                    "fallback_reason",
                    "fallback_tool",
                    "fallback_observation",
                    "final_path",
                }:
                    snapshot[key] = str(value or "").strip()
                else:
                    snapshot[key] = value
            self.planner_debug = snapshot
        return self.planner_debug_snapshot()

    def planner_debug_snapshot(self):
        with self.bot._session_lock:
            snapshot = self._normalize_planner_debug_state(self.planner_debug)
        snapshot["planner_parameters"] = dict(snapshot.get("planner_parameters") or {})
        return snapshot

    def build_active_tool_observation_context(self):
        context = str(self.active_tool_observation_context or "").strip()
        if not context:
            return None
        return (
            "Fresh tool observation gathered for this reply:\n"
            f"{context}\n"
            "Use it as supporting context, but keep the answer natural and dad-like."
        )
