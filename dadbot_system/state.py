from __future__ import annotations

import copy
import json
import sqlite3
import types
from collections import defaultdict
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

from .contracts import DEFAULT_TENANT_ID, ChatResponse, EventEnvelope, EventType, normalize_tenant_id

try:
    import redis
except ImportError:
    pass

try:
    import psycopg
except ImportError:
    psycopg = types.SimpleNamespace()  # type: ignore[assignment]


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
    def initial(cls, planner_debug_factory) -> SessionRuntimeState:
        return cls(planner_debug=dict(planner_debug_factory()))

    @classmethod
    def from_dict(cls, payload: dict[str, Any], planner_debug_factory) -> SessionRuntimeState:
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
    def save_session_state(self, session_id: str, state: dict[str, Any]) -> None: ...

    def load_session_state(self, session_id: str) -> dict[str, Any] | None: ...

    def save_task(self, task_id: str, payload: dict[str, Any]) -> None: ...

    def save_task_result(self, task_id: str, payload: dict[str, Any], response: dict[str, Any]) -> None: ...

    def load_task(self, task_id: str) -> dict[str, Any] | None: ...

    def list_incomplete_tasks(self, limit: int = 1000) -> list[dict[str, Any]]: ...

    def save_response(self, task_id: str, response: dict[str, Any]) -> None: ...

    def load_response(self, task_id: str) -> dict[str, Any] | None: ...

    def append_event(self, session_id: str, event: dict[str, Any]) -> None: ...

    def list_events(self, session_id: str) -> list[dict[str, Any]]: ...


class SQLiteStateStore:
    def __init__(self, path: str):
        if not path or path == ":memory:":
            raise ValueError("A persistent SQLite state path is required")
        self.path = str(Path(path).resolve())
        self._uri = Path(self.path).as_uri() + "?mode=rw"
        with closing(sqlite3.connect(self.path, timeout=5)) as connection, connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, payload TEXT NOT NULL, response TEXT)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS events (sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id TEXT NOT NULL, payload TEXT NOT NULL)"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS events_session ON events(session_id, sequence)")

    def _connect(self):
        connection = sqlite3.connect(self._uri, timeout=5, uri=True)
        try:
            connection.execute("PRAGMA synchronous=FULL")
        except sqlite3.Error:
            connection.close()
            raise
        return connection

    def save_session_state(self, session_id: str, state: dict[str, Any]) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO sessions VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (session_id, json.dumps(state)),
            )

    def load_session_state(self, session_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT payload FROM sessions WHERE id=?", (session_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def save_task(self, task_id: str, payload: dict[str, Any]) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO tasks (id, payload) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (task_id, json.dumps(payload)),
            )

    def load_task(self, task_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT payload FROM tasks WHERE id=?", (task_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def list_incomplete_tasks(self, limit: int = 1000) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT payload FROM tasks WHERE json_extract(payload, '$.status') IN ('queued', 'running') "
                "ORDER BY json_extract(payload, '$.accepted_at'), id LIMIT ?",
                (max(1, int(limit)),),
            )
            return [json.loads(row[0]) for row in rows]

    def save_task_result(self, task_id: str, payload: dict[str, Any], response: dict[str, Any]) -> None:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "UPDATE tasks SET payload=?, response=? WHERE id=?",
                (json.dumps(payload), json.dumps(response), task_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Cannot persist result for unknown task_id={task_id}")

    def save_response(self, task_id: str, response: dict[str, Any]) -> None:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute("UPDATE tasks SET response=? WHERE id=?", (json.dumps(response), task_id))
            if cursor.rowcount != 1:
                raise RuntimeError(f"Cannot persist response for unknown task_id={task_id}")

    def load_response(self, task_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT response FROM tasks WHERE id=?", (task_id,)).fetchone()
            return json.loads(row[0]) if row and row[0] is not None else None

    def append_event(self, session_id: str, event: dict[str, Any]) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("INSERT INTO events (session_id, payload) VALUES (?, ?)", (session_id, json.dumps(event)))

    def list_events(self, session_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT payload FROM events WHERE session_id=? ORDER BY sequence", (session_id,))
            return [json.loads(row[0]) for row in rows]


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

    def save_task_result(self, task_id: str, payload: dict[str, Any], response: dict[str, Any]) -> None:
        with self._lock:
            self._tasks[task_id] = json.loads(json.dumps(payload))
            self._responses[task_id] = json.loads(json.dumps(response))

    def list_incomplete_tasks(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self._lock:
            tasks = [
                payload
                for payload in self._tasks.values()
                if str(payload.get("status") or "").lower() in {"queued", "running"}
            ]
            return json.loads(json.dumps(tasks[: max(1, int(limit))]))

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

    # Dict-like interface for test compatibility
    def __getitem__(self, key):
        # Try sessions, then tasks, then responses
        if key in self._sessions:
            return self._sessions[key]
        if key in self._tasks:
            return self._tasks[key]
        if key in self._responses:
            return self._responses[key]
        raise KeyError(key)

    def __setitem__(self, key, value):
        # Default to sessions for assignment
        self._sessions[key] = value

    def get(self, key, default=None):
        if key in self._sessions:
            return self._sessions.get(key, default)
        if key in self._tasks:
            return self._tasks.get(key, default)
        if key in self._responses:
            return self._responses.get(key, default)
        return default


class CompositeStateStore:
    def __init__(self, fast_store: StateStore | None = None, durable_store: StateStore | None = None) -> None:
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

    def save_memory_state(self, session_id: str, state: dict[str, Any]) -> None:
        authoritative_store = self.durable_store or self.fast_store
        if authoritative_store is None:
            raise RuntimeError("No authoritative memory state store configured")
        authoritative_store.save_session_state(session_id, state)

    def load_memory_state(self, session_id: str) -> dict[str, Any] | None:
        authoritative_store = self.durable_store or self.fast_store
        if authoritative_store is None:
            raise RuntimeError("No authoritative memory state store configured")
        return authoritative_store.load_session_state(session_id)

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

    def save_task_result(self, task_id: str, payload: dict[str, Any], response: dict[str, Any]) -> None:
        for store in (self.fast_store, self.durable_store):
            if store is not None:
                store.save_task_result(task_id, payload, response)

    def list_incomplete_tasks(self, limit: int = 1000) -> list[dict[str, Any]]:
        authoritative_store = self.durable_store or self.fast_store
        if authoritative_store is None:
            return []
        return authoritative_store.list_incomplete_tasks(limit=limit)

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
    def __init__(self, store: StateStore, namespace: str) -> None:
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

    def save_memory_state(self, session_id: str, state: dict[str, Any]) -> None:
        save = getattr(self.store, "save_memory_state", self.store.save_session_state)
        save(self._session_key(session_id), state)

    def load_memory_state(self, session_id: str) -> dict[str, Any] | None:
        load = getattr(self.store, "load_memory_state", self.store.load_session_state)
        return load(self._session_key(session_id))

    def save_task(self, task_id: str, payload: dict[str, Any]) -> None:
        self.store.save_task(task_id, payload)

    def load_task(self, task_id: str) -> dict[str, Any] | None:
        return self.store.load_task(task_id)

    def save_task_result(self, task_id: str, payload: dict[str, Any], response: dict[str, Any]) -> None:
        self.store.save_task_result(task_id, payload, response)

    def list_incomplete_tasks(self, limit: int = 1000) -> list[dict[str, Any]]:
        return self.store.list_incomplete_tasks(limit=limit)

    def save_response(self, task_id: str, response: dict[str, Any]) -> None:
        self.store.save_response(task_id, response)

    def load_response(self, task_id: str) -> dict[str, Any] | None:
        return self.store.load_response(task_id)

    def append_event(self, session_id: str, event: dict[str, Any]) -> None:
        self.store.append_event(self._session_key(session_id), event)

    def list_events(self, session_id: str) -> list[dict[str, Any]]:
        return self.store.list_events(self._session_key(session_id))


class RedisStateStore:
    def __init__(self, redis_url: str, namespace: str = "dadbot") -> None:
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
        if payload is None:
            return None
        if not isinstance(payload, str):
            raise TypeError(f"Expected str from redis, got {type(payload)}")
        return json.loads(payload)

    def save_task(self, task_id: str, payload: dict[str, Any]) -> None:
        self._client.set(self._key(f"task:{task_id}"), json.dumps(payload))

    def load_task(self, task_id: str) -> dict[str, Any] | None:
        payload = self._client.get(self._key(f"task:{task_id}"))
        if payload is None:
            return None
        if not isinstance(payload, str):
            raise TypeError(f"Expected str from redis, got {type(payload)}")
        return json.loads(payload)

    def save_task_result(self, task_id: str, payload: dict[str, Any], response: dict[str, Any]) -> None:
        with self._client.pipeline(transaction=True) as pipeline:
            pipeline.set(self._key(f"task:{task_id}"), json.dumps(payload))
            pipeline.set(self._key(f"response:{task_id}"), json.dumps(response))
            pipeline.execute()

    def list_incomplete_tasks(self, limit: int = 1000) -> list[dict[str, Any]]:
        incomplete: list[dict[str, Any]] = []
        for key in self._client.scan_iter(match=self._key("task:*")):
            payload = self._client.get(key)
            if payload is None:
                continue
            if not isinstance(payload, str):
                raise TypeError(f"Expected str from redis get, got {type(payload)}")
            task = json.loads(payload)
            if str(task.get("status") or "").lower() in {"queued", "running"}:
                incomplete.append(task)
                if len(incomplete) >= max(1, int(limit)):
                    break
        return incomplete

    def save_response(self, task_id: str, response: dict[str, Any]) -> None:
        self._client.set(self._key(f"response:{task_id}"), json.dumps(response))

    def load_response(self, task_id: str) -> dict[str, Any] | None:
        payload = self._client.get(self._key(f"response:{task_id}"))
        if payload is None:
            return None
        if not isinstance(payload, str):
            raise TypeError(f"Expected str from redis, got {type(payload)}")
        return json.loads(payload)

    def append_event(self, session_id: str, event: dict[str, Any]) -> None:
        self._client.rpush(self._key(f"events:{session_id}"), json.dumps(event))

    def list_events(self, session_id: str) -> list[dict[str, Any]]:
        items = self._client.lrange(self._key(f"events:{session_id}"), 0, -1)
        if not isinstance(items, list):
            raise TypeError(f"Expected list from redis lrange, got {type(items)}")
        result = []
        for item in items:
            if not isinstance(item, str):
                raise TypeError(f"Expected str in redis lrange, got {type(item)}")
            result.append(json.loads(item))
        return result


class PostgresStateStore:
    """Experimental trusted-local persistence; not tenant-isolated for authenticated production.

    See ``docs/decisions/postgres-authenticated-boot.md`` before enabling it
    with authenticated execution.
    """

    def __init__(self, postgres_dsn: str, *, session_table: str, task_table: str, event_table: str) -> None:
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
        from psycopg import sql
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("CREATE TABLE IF NOT EXISTS {} (session_id TEXT PRIMARY KEY, payload JSONB NOT NULL)").format(sql.Identifier(self._session_table))
            )
            cursor.execute(
                sql.SQL("CREATE TABLE IF NOT EXISTS {} (task_id TEXT PRIMARY KEY, payload JSONB NOT NULL, response JSONB)").format(sql.Identifier(self._task_table))
            )
            cursor.execute(
                sql.SQL("CREATE TABLE IF NOT EXISTS {} (event_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, payload JSONB NOT NULL)").format(sql.Identifier(self._event_table))
            )
            connection.commit()

    def save_session_state(self, session_id: str, state: dict[str, Any]) -> None:
        from psycopg import sql
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("INSERT INTO {} (session_id, payload) VALUES (%s, %s) ON CONFLICT (session_id) DO UPDATE SET payload = EXCLUDED.payload").format(sql.Identifier(self._session_table)),
                (session_id, json.dumps(state)),
            )
            connection.commit()

    def load_session_state(self, session_id: str) -> dict[str, Any] | None:
        from psycopg import sql
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("SELECT payload FROM {} WHERE session_id = %s").format(sql.Identifier(self._session_table)),
                (session_id,)
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def save_task(self, task_id: str, payload: dict[str, Any]) -> None:
        from psycopg import sql
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("INSERT INTO {} (task_id, payload) VALUES (%s, %s) ON CONFLICT (task_id) DO UPDATE SET payload = EXCLUDED.payload").format(sql.Identifier(self._task_table)),
                (task_id, json.dumps(payload)),
            )
            connection.commit()

    def load_task(self, task_id: str) -> dict[str, Any] | None:
        from psycopg import sql
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("SELECT payload FROM {} WHERE task_id = %s").format(sql.Identifier(self._task_table)),
                (task_id,)
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def list_incomplete_tasks(self, limit: int = 1000) -> list[dict[str, Any]]:
        from psycopg import sql

        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    "SELECT payload FROM {} "
                    "WHERE payload->>'status' IN ('queued', 'running') "
                    "ORDER BY payload->>'accepted_at' NULLS FIRST LIMIT %s"
                ).format(sql.Identifier(self._task_table)),
                (max(1, int(limit)),),
            )
            return [row[0] for row in cursor.fetchall()]

    def save_task_result(self, task_id: str, payload: dict[str, Any], response: dict[str, Any]) -> None:
        from psycopg import sql

        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("UPDATE {} SET payload = %s, response = %s WHERE task_id = %s").format(
                    sql.Identifier(self._task_table)
                ),
                (json.dumps(payload), json.dumps(response), task_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Cannot persist result for unknown task_id={task_id}")
            connection.commit()

    def save_response(self, task_id: str, response: dict[str, Any]) -> None:
        from psycopg import sql
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("UPDATE {} SET response = %s WHERE task_id = %s").format(sql.Identifier(self._task_table)),
                (json.dumps(response), task_id),
            )
            connection.commit()

    def load_response(self, task_id: str) -> dict[str, Any] | None:
        from psycopg import sql
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("SELECT response FROM {} WHERE task_id = %s").format(sql.Identifier(self._task_table)),
                (task_id,)
            )
            row = cursor.fetchone()
            return row[0] if row and row[0] is not None else None

    def append_event(self, session_id: str, event: dict[str, Any]) -> None:
        from psycopg import sql
        event_id = str(event.get("event_id") or "")
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("INSERT INTO {} (event_id, session_id, payload) VALUES (%s, %s, %s) ON CONFLICT (event_id) DO NOTHING").format(sql.Identifier(self._event_table)),
                (event_id, session_id, json.dumps(event)),
            )
            connection.commit()

    def list_events(self, session_id: str) -> list[dict[str, Any]]:
        from psycopg import sql
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("SELECT payload FROM {} WHERE session_id = %s ORDER BY event_id").format(sql.Identifier(self._event_table)),
                (session_id,)
            )
            return [row[0] for row in cursor.fetchall()]


class AppStateContainer:
    def __init__(
        self,
        session_id: str,
        planner_debug_factory: Any,
        *,
        tenant_id: str = DEFAULT_TENANT_ID,
        store: StateStore | None = None,
        event_bus: Any = None,
    ) -> None:
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

    def update(self, mutator: Any, *, reason: str) -> dict[str, Any]:
        with self._lock:
            mutator(self._state)
            self._persist(reason, {})
            return self._state.to_dict()

    def record_response(self, task_id: str, response: ChatResponse) -> None:
        if self.store is not None:
            # ChatResponseEnvelope is a TypedDict, but store expects dict[str, Any]
            self.store.save_response(task_id, dict(response.to_dict()))
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
        if self.store is not None and not getattr(self.event_bus, "persists_events", False):
            self.store.append_event(self.session_id, event.to_dict())
        if self.event_bus is not None:
            self.event_bus.publish(event)
