from __future__ import annotations

import copy
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import Any, Protocol

from .contracts import ChatResponse, DEFAULT_TENANT_ID, EventEnvelope, EventType, normalize_tenant_id

try:
    import redis
except ImportError:
    redis = None

try:
    import psycopg
except ImportError:
    psycopg = None


@dataclass(slots=True)
class SessionRuntimeState:
    history: list[dict[str, Any]] = field(default_factory=list)
    session_moods: list[str] = field(default_factory=list)
    session_summary: str = ""
    session_summary_updated_at: str | None = None
    session_summary_covered_messages: int = 0
    last_relationship_reflection_turn: int = 0
    pending_daily_checkin_context: bool = False
    active_tool_observation_context: str | None = None
    planner_debug: dict[str, Any] = field(default_factory=dict)
    chat_threads: list[dict[str, Any]] = field(default_factory=list)
    active_thread_id: str | None = None
    thread_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    memory_store: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def initial(cls, planner_debug_factory) -> "SessionRuntimeState":
        return cls(planner_debug=dict(planner_debug_factory()))

    @classmethod
    def from_dict(cls, payload: dict[str, Any], planner_debug_factory) -> "SessionRuntimeState":
        state = cls.initial(planner_debug_factory)
        state.history = [dict(message) for message in payload.get("history", []) if isinstance(message, dict)]
        state.session_moods = [str(mood) for mood in payload.get("session_moods", [])]
        state.session_summary = str(payload.get("session_summary") or "")
        state.session_summary_updated_at = payload.get("session_summary_updated_at")
        state.session_summary_covered_messages = int(payload.get("session_summary_covered_messages") or 0)
        state.last_relationship_reflection_turn = int(payload.get("last_relationship_reflection_turn") or 0)
        state.pending_daily_checkin_context = bool(payload.get("pending_daily_checkin_context"))
        active_tool = str(payload.get("active_tool_observation_context") or "").strip()
        state.active_tool_observation_context = active_tool or None
        planner_payload = payload.get("planner_debug")
        if isinstance(planner_payload, dict):
            state.planner_debug = dict(planner_payload)
        threads_payload = payload.get("chat_threads")
        if isinstance(threads_payload, list):
            state.chat_threads = [dict(item) for item in threads_payload if isinstance(item, dict)]
        active_thread_id = str(payload.get("active_thread_id") or "").strip()
        state.active_thread_id = active_thread_id or None
        snapshots_payload = payload.get("thread_snapshots")
        if isinstance(snapshots_payload, dict):
            state.thread_snapshots = {
                str(thread_id): dict(snapshot)
                for thread_id, snapshot in snapshots_payload.items()
                if isinstance(snapshot, dict)
            }
        memory_payload = payload.get("memory_store")
        if isinstance(memory_payload, dict):
            state.memory_store = copy.deepcopy(memory_payload)
        return state

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StateStore(Protocol):
    def save_session_state(self, session_id: str, state: dict[str, Any]) -> None:
        ...

    def load_session_state(self, session_id: str) -> dict[str, Any] | None:
        ...

    def save_task(self, task_id: str, payload: dict[str, Any]) -> None:
        ...

    def load_task(self, task_id: str) -> dict[str, Any] | None:
        ...

    def save_response(self, task_id: str, response: dict[str, Any]) -> None:
        ...

    def load_response(self, task_id: str) -> dict[str, Any] | None:
        ...

    def append_event(self, session_id: str, event: dict[str, Any]) -> None:
        ...

    def list_events(self, session_id: str) -> list[dict[str, Any]]:
        ...


class InMemoryStateStore:
    def __init__(self):
        self._lock = RLock()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, dict[str, Any]] = {}
        self._responses: dict[str, dict[str, Any]] = {}
        self._events: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def save_session_state(self, session_id: str, state: dict[str, Any]) -> None:
        with self._lock:
            self._sessions[session_id] = json.loads(json.dumps(state))

    def load_session_state(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            payload = self._sessions.get(session_id)
            return json.loads(json.dumps(payload)) if payload is not None else None

    def save_task(self, task_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._tasks[task_id] = json.loads(json.dumps(payload))

    def load_task(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            payload = self._tasks.get(task_id)
            return json.loads(json.dumps(payload)) if payload is not None else None

    def save_response(self, task_id: str, response: dict[str, Any]) -> None:
        with self._lock:
            self._responses[task_id] = json.loads(json.dumps(response))

    def load_response(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            payload = self._responses.get(task_id)
            return json.loads(json.dumps(payload)) if payload is not None else None

    def append_event(self, session_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            self._events[session_id].append(json.loads(json.dumps(event)))

    def list_events(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return json.loads(json.dumps(self._events.get(session_id, [])))


class CompositeStateStore:
    def __init__(self, fast_store: StateStore | None = None, durable_store: StateStore | None = None):
        self.fast_store = fast_store
        self.durable_store = durable_store

    def save_session_state(self, session_id: str, state: dict[str, Any]) -> None:
        for store in (self.fast_store, self.durable_store):
            if store is not None:
                store.save_session_state(session_id, state)

    def load_session_state(self, session_id: str) -> dict[str, Any] | None:
        for store in (self.fast_store, self.durable_store):
            if store is None:
                continue
            payload = store.load_session_state(session_id)
            if payload is not None:
                return payload
        return None

    def save_task(self, task_id: str, payload: dict[str, Any]) -> None:
        for store in (self.fast_store, self.durable_store):
            if store is not None:
                store.save_task(task_id, payload)

    def load_task(self, task_id: str) -> dict[str, Any] | None:
        for store in (self.fast_store, self.durable_store):
            if store is None:
                continue
            payload = store.load_task(task_id)
            if payload is not None:
                return payload
        return None

    def save_response(self, task_id: str, response: dict[str, Any]) -> None:
        for store in (self.fast_store, self.durable_store):
            if store is not None:
                store.save_response(task_id, response)

    def load_response(self, task_id: str) -> dict[str, Any] | None:
        for store in (self.fast_store, self.durable_store):
            if store is None:
                continue
            payload = store.load_response(task_id)
            if payload is not None:
                return payload
        return None

    def append_event(self, session_id: str, event: dict[str, Any]) -> None:
        for store in (self.fast_store, self.durable_store):
            if store is not None:
                store.append_event(session_id, event)

    def list_events(self, session_id: str) -> list[dict[str, Any]]:
        if self.fast_store is not None:
            return self.fast_store.list_events(session_id)
        if self.durable_store is not None:
            return self.durable_store.list_events(session_id)
        return []


class NamespacedStateStore:
    def __init__(self, store: StateStore, namespace: str):
        self.store = store
        self.namespace = str(namespace or "").strip()

    def _session_key(self, session_id: str) -> str:
        if not self.namespace:
            return str(session_id or "")
        return f"{self.namespace}:{session_id}"

    def save_session_state(self, session_id: str, state: dict[str, Any]) -> None:
        self.store.save_session_state(self._session_key(session_id), state)

    def load_session_state(self, session_id: str) -> dict[str, Any] | None:
        return self.store.load_session_state(self._session_key(session_id))

    def save_task(self, task_id: str, payload: dict[str, Any]) -> None:
        self.store.save_task(task_id, payload)

    def load_task(self, task_id: str) -> dict[str, Any] | None:
        return self.store.load_task(task_id)

    def save_response(self, task_id: str, response: dict[str, Any]) -> None:
        self.store.save_response(task_id, response)

    def load_response(self, task_id: str) -> dict[str, Any] | None:
        return self.store.load_response(task_id)

    def append_event(self, session_id: str, event: dict[str, Any]) -> None:
        self.store.append_event(self._session_key(session_id), event)

    def list_events(self, session_id: str) -> list[dict[str, Any]]:
        return self.store.list_events(self._session_key(session_id))


class RedisStateStore:
    def __init__(self, redis_url: str, namespace: str = "dadbot"):
        if redis is None:
            raise RuntimeError("redis package is not installed")
        self._client = redis.from_url(redis_url, decode_responses=True)
        self._namespace = namespace

    def _key(self, suffix: str) -> str:
        return f"{self._namespace}:{suffix}"

    def save_session_state(self, session_id: str, state: dict[str, Any]) -> None:
        self._client.set(self._key(f"session:{session_id}"), json.dumps(state))

    def load_session_state(self, session_id: str) -> dict[str, Any] | None:
        payload = self._client.get(self._key(f"session:{session_id}"))
        return json.loads(payload) if payload else None

    def save_task(self, task_id: str, payload: dict[str, Any]) -> None:
        self._client.set(self._key(f"task:{task_id}"), json.dumps(payload))

    def load_task(self, task_id: str) -> dict[str, Any] | None:
        payload = self._client.get(self._key(f"task:{task_id}"))
        return json.loads(payload) if payload else None

    def save_response(self, task_id: str, response: dict[str, Any]) -> None:
        self._client.set(self._key(f"response:{task_id}"), json.dumps(response))

    def load_response(self, task_id: str) -> dict[str, Any] | None:
        payload = self._client.get(self._key(f"response:{task_id}"))
        return json.loads(payload) if payload else None

    def append_event(self, session_id: str, event: dict[str, Any]) -> None:
        self._client.rpush(self._key(f"events:{session_id}"), json.dumps(event))

    def list_events(self, session_id: str) -> list[dict[str, Any]]:
        return [json.loads(item) for item in self._client.lrange(self._key(f"events:{session_id}"), 0, -1)]


class PostgresStateStore:
    def __init__(self, postgres_dsn: str, *, session_table: str, task_table: str, event_table: str):
        if psycopg is None:
            raise RuntimeError("psycopg package is not installed")
        self._dsn = postgres_dsn
        self._session_table = session_table
        self._task_table = task_table
        self._event_table = event_table
        self._ensure_tables()

    def _connect(self):
        return psycopg.connect(self._dsn)

    def _ensure_tables(self) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {self._session_table} (session_id TEXT PRIMARY KEY, payload JSONB NOT NULL)"
            )
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {self._task_table} (task_id TEXT PRIMARY KEY, payload JSONB NOT NULL, response JSONB)"
            )
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {self._event_table} (event_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, payload JSONB NOT NULL)"
            )
            connection.commit()

    def save_session_state(self, session_id: str, state: dict[str, Any]) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {self._session_table} (session_id, payload) VALUES (%s, %s) "
                f"ON CONFLICT (session_id) DO UPDATE SET payload = EXCLUDED.payload",
                (session_id, json.dumps(state)),
            )
            connection.commit()

    def load_session_state(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT payload FROM {self._session_table} WHERE session_id = %s", (session_id,))
            row = cursor.fetchone()
            return row[0] if row else None

    def save_task(self, task_id: str, payload: dict[str, Any]) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {self._task_table} (task_id, payload) VALUES (%s, %s) "
                f"ON CONFLICT (task_id) DO UPDATE SET payload = EXCLUDED.payload",
                (task_id, json.dumps(payload)),
            )
            connection.commit()

    def load_task(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT payload FROM {self._task_table} WHERE task_id = %s", (task_id,))
            row = cursor.fetchone()
            return row[0] if row else None

    def save_response(self, task_id: str, response: dict[str, Any]) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {self._task_table} SET response = %s WHERE task_id = %s",
                (json.dumps(response), task_id),
            )
            connection.commit()

    def load_response(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT response FROM {self._task_table} WHERE task_id = %s", (task_id,))
            row = cursor.fetchone()
            return row[0] if row and row[0] is not None else None

    def append_event(self, session_id: str, event: dict[str, Any]) -> None:
        event_id = str(event.get("event_id") or "")
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {self._event_table} (event_id, session_id, payload) VALUES (%s, %s, %s) "
                f"ON CONFLICT (event_id) DO NOTHING",
                (event_id, session_id, json.dumps(event)),
            )
            connection.commit()

    def list_events(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT payload FROM {self._event_table} WHERE session_id = %s ORDER BY event_id", (session_id,))
            return [row[0] for row in cursor.fetchall()]


class AppStateContainer:
    def __init__(self, session_id: str, planner_debug_factory, *, tenant_id: str = DEFAULT_TENANT_ID, store: StateStore | None = None, event_bus=None):
        self.session_id = session_id
        self.tenant_id = normalize_tenant_id(tenant_id)
        self._planner_debug_factory = planner_debug_factory
        self._lock = RLock()
        self.store = store
        self.event_bus = event_bus
        restored = store.load_session_state(session_id) if store is not None else None
        self._state = SessionRuntimeState.from_dict(restored or {}, planner_debug_factory)

    @property
    def state(self) -> SessionRuntimeState:
        return self._state

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._state.to_dict()

    def load_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._state = SessionRuntimeState.from_dict(payload or {}, self._planner_debug_factory)
            self._persist("state.loaded", {"fields": sorted(self._state.to_dict().keys())})
            return self._state.to_dict()

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self._state = SessionRuntimeState.initial(self._planner_debug_factory)
            self._persist("state.reset", {})
            return self._state.to_dict()

    def update(self, mutator, *, reason: str) -> dict[str, Any]:
        with self._lock:
            mutator(self._state)
            self._persist(reason, {})
            return self._state.to_dict()

    def record_response(self, task_id: str, response: ChatResponse) -> None:
        if self.store is not None:
            self.store.save_response(task_id, response.to_dict())
        self.record_event(
            EventType.RESPONSE_READY,
            {"task_id": task_id, "request_id": response.request_id, "status": response.status},
        )

    def record_event(self, event_type: EventType, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        event = EventEnvelope(
            session_id=self.session_id,
            event_type=event_type,
            tenant_id=self.tenant_id,
            payload=dict(payload or {}),
        )
        self._publish(event)
        return event.to_dict()

    def record_state_event(self, reason: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.record_event(EventType.STATE_UPDATED, {"reason": reason, **dict(payload or {})})

    def _persist(self, reason: str, payload: dict[str, Any]) -> None:
        snapshot = self._state.to_dict()
        if self.store is not None:
            self.store.save_session_state(self.session_id, snapshot)
        self.record_state_event(reason, payload)

    def _publish(self, event: EventEnvelope) -> None:
        if self.store is not None:
            self.store.append_event(self.session_id, event.to_dict())
        if self.event_bus is not None:
            self.event_bus.publish(event)