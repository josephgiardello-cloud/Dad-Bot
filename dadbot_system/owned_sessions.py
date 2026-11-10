"""Tenant-qualified storage for service session snapshots and events."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from threading import RLock
from typing import Any, Protocol
from urllib.parse import quote

from .execution_identity import require_identifier, require_scoped_tenant_id


class OwnedSessionStore(Protocol):
    def save_session_state(self, tenant_id: str, session_id: str, state: dict[str, Any]) -> None: ...

    def load_session_state(self, tenant_id: str, session_id: str) -> dict[str, Any] | None: ...

    def list_sessions(self, tenant_id: str) -> list[str]: ...

    def delete_session(self, tenant_id: str, session_id: str) -> bool: ...

    def append_event(self, tenant_id: str, session_id: str, event: dict[str, Any]) -> None: ...

    def list_events(self, tenant_id: str, session_id: str) -> list[dict[str, Any]]: ...


def _validate_identity(tenant_id: str, session_id: str) -> None:
    require_scoped_tenant_id(tenant_id)
    require_identifier(session_id, "session")


def _validate_event(tenant_id: str, session_id: str, event: dict[str, Any]) -> None:
    if (
        not isinstance(event, dict)
        or event.get("tenant_id") != tenant_id
        or event.get("session_id") != session_id
    ):
        raise ValueError("Session event ownership identity mismatch")


def _validate_state(state: dict[str, Any]) -> None:
    if not isinstance(state, dict):
        raise TypeError("Session state must be a dictionary")


class SQLiteOwnedSessions:
    def __init__(self, path: str) -> None:
        if not path or path == ":memory:":
            raise ValueError("A persistent SQLite state path is required")
        self.path = str(Path(path).resolve())
        self._uri = Path(self.path).as_uri() + "?mode=rw"
        with closing(sqlite3.connect(self.path, timeout=5)) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS owned_sessions_v1 ("
                "tenant_id TEXT NOT NULL, session_id TEXT NOT NULL, payload TEXT NOT NULL, "
                "PRIMARY KEY (tenant_id, session_id))"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS owned_session_events_v1 ("
                "sequence INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL, "
                "session_id TEXT NOT NULL, payload TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS owned_session_events_v1_owner "
                "ON owned_session_events_v1(tenant_id, session_id, sequence)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._uri, timeout=5, uri=True)
        try:
            connection.execute("PRAGMA synchronous=FULL")
        except sqlite3.Error:
            connection.close()
            raise
        return connection

    @staticmethod
    def _reject_legacy_rows(
        connection: sqlite3.Connection, tenant_id: str, session_id: str
    ) -> None:
        legacy_keys = (
            session_id,
            f"tenant:{tenant_id}:{session_id}",
            f"tenant-doc:{tenant_id}:{session_id}",
        )
        sessions_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions'"
        ).fetchone()
        if sessions_table and connection.execute(
            f"SELECT 1 FROM sessions WHERE id IN ({','.join('?' for _ in legacy_keys)}) LIMIT 1",
            legacy_keys,
        ).fetchone():
            raise RuntimeError("Unowned legacy session state requires explicit migration")
        events_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
        ).fetchone()
        if events_table and connection.execute(
            f"SELECT 1 FROM events WHERE session_id IN ({','.join('?' for _ in legacy_keys)}) LIMIT 1",
            legacy_keys,
        ).fetchone():
            raise RuntimeError("Unowned legacy session events require explicit migration")

    def save_session_state(self, tenant_id: str, session_id: str, state: dict[str, Any]) -> None:
        _validate_identity(tenant_id, session_id)
        with closing(self._connect()) as connection, connection:
            self._reject_legacy_rows(connection, tenant_id, session_id)
            connection.execute(
                "INSERT INTO owned_sessions_v1 (tenant_id, session_id, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(tenant_id, session_id) DO UPDATE SET payload=excluded.payload",
                (tenant_id, session_id, json.dumps(state)),
            )

    def load_session_state(self, tenant_id: str, session_id: str) -> dict[str, Any] | None:
        _validate_identity(tenant_id, session_id)
        with closing(self._connect()) as connection:
            self._reject_legacy_rows(connection, tenant_id, session_id)
            row = connection.execute(
                "SELECT payload FROM owned_sessions_v1 WHERE tenant_id=? AND session_id=?",
                (tenant_id, session_id),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list_sessions(self, tenant_id: str) -> list[str]:
        require_scoped_tenant_id(tenant_id)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT session_id FROM owned_sessions_v1 WHERE tenant_id=? ORDER BY session_id",
                (tenant_id,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def delete_session(self, tenant_id: str, session_id: str) -> bool:
        _validate_identity(tenant_id, session_id)
        with closing(self._connect()) as connection, connection:
            self._reject_legacy_rows(connection, tenant_id, session_id)
            connection.execute(
                "DELETE FROM owned_session_events_v1 WHERE tenant_id=? AND session_id=?",
                (tenant_id, session_id),
            )
            cursor = connection.execute(
                "DELETE FROM owned_sessions_v1 WHERE tenant_id=? AND session_id=?",
                (tenant_id, session_id),
            )
        return cursor.rowcount > 0

    def append_event(self, tenant_id: str, session_id: str, event: dict[str, Any]) -> None:
        _validate_identity(tenant_id, session_id)
        _validate_event(tenant_id, session_id, event)
        with closing(self._connect()) as connection, connection:
            self._reject_legacy_rows(connection, tenant_id, session_id)
            connection.execute(
                "INSERT INTO owned_session_events_v1 (tenant_id, session_id, payload) VALUES (?, ?, ?)",
                (tenant_id, session_id, json.dumps(event)),
            )

    def list_events(self, tenant_id: str, session_id: str) -> list[dict[str, Any]]:
        _validate_identity(tenant_id, session_id)
        with closing(self._connect()) as connection:
            self._reject_legacy_rows(connection, tenant_id, session_id)
            rows = connection.execute(
                "SELECT payload FROM owned_session_events_v1 "
                "WHERE tenant_id=? AND session_id=? ORDER BY sequence",
                (tenant_id, session_id),
            ).fetchall()
        events = [json.loads(row[0]) for row in rows]
        for event in events:
            _validate_event(tenant_id, session_id, event)
        return events


class InMemoryOwnedSessions:
    def __init__(self) -> None:
        self._lock = RLock()
        self._sessions: dict[tuple[str, str], dict[str, Any]] = {}
        self._events: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    def save_session_state(self, tenant_id: str, session_id: str, state: dict[str, Any]) -> None:
        _validate_identity(tenant_id, session_id)
        with self._lock:
            self._sessions[(tenant_id, session_id)] = json.loads(json.dumps(state))

    def load_session_state(self, tenant_id: str, session_id: str) -> dict[str, Any] | None:
        _validate_identity(tenant_id, session_id)
        with self._lock:
            payload = self._sessions.get((tenant_id, session_id))
            return json.loads(json.dumps(payload)) if payload is not None else None

    def list_sessions(self, tenant_id: str) -> list[str]:
        require_scoped_tenant_id(tenant_id)
        with self._lock:
            return sorted(session_id for owner, session_id in self._sessions if owner == tenant_id)

    def delete_session(self, tenant_id: str, session_id: str) -> bool:
        _validate_identity(tenant_id, session_id)
        with self._lock:
            existed = self._sessions.pop((tenant_id, session_id), None) is not None
            self._events.pop((tenant_id, session_id), None)
            return existed

    def append_event(self, tenant_id: str, session_id: str, event: dict[str, Any]) -> None:
        _validate_identity(tenant_id, session_id)
        _validate_event(tenant_id, session_id, event)
        with self._lock:
            self._events[(tenant_id, session_id)].append(json.loads(json.dumps(event)))

    def list_events(self, tenant_id: str, session_id: str) -> list[dict[str, Any]]:
        _validate_identity(tenant_id, session_id)
        with self._lock:
            events = json.loads(json.dumps(self._events.get((tenant_id, session_id), [])))
        for event in events:
            _validate_event(tenant_id, session_id, event)
        return events


class RedisOwnedSessions:
    _GUARDED_SCRIPT = """
for index = 3, #KEYS do
    if redis.call('EXISTS', KEYS[index]) == 1 then
        return -1
    end
end
local state = redis.call('GET', KEYS[1])
if state then
    local ok, record = pcall(cjson.decode, state)
    if not ok or type(record) ~= 'table' or record.tenant_id ~= ARGV[1] or record.session_id ~= ARGV[2] then
        return -2
    end
end
local events = redis.call('LRANGE', KEYS[2], 0, -1)
for _, encoded in ipairs(events) do
    local ok, event = pcall(cjson.decode, encoded)
    if not ok or type(event) ~= 'table' or event.tenant_id ~= ARGV[1] or event.session_id ~= ARGV[2] then
        return -2
    end
end
"""

    def __init__(self, client: Any, namespace: str) -> None:
        self._client = client
        self._namespace = str(namespace or "").strip()
        if not self._namespace:
            raise ValueError("Redis session namespace is required")

    def _keys(self, tenant_id: str, session_id: str) -> tuple[str, ...]:
        tenant = quote(tenant_id, safe="")
        session = quote(session_id, safe="")
        base = f"{self._namespace}:owned-session:{tenant}:{session}"
        return (
            f"{base}:state",
            f"{base}:events",
            f"{self._namespace}:session:{session_id}",
            f"{self._namespace}:events:{session_id}",
            f"{self._namespace}:session:tenant:{tenant_id}:{session_id}",
            f"{self._namespace}:events:tenant:{tenant_id}:{session_id}",
            f"{self._namespace}:session:tenant-doc:{tenant_id}:{session_id}",
            f"{self._namespace}:events:tenant-doc:{tenant_id}:{session_id}",
        )

    def _run_guarded(
        self,
        tenant_id: str,
        session_id: str,
        operation: str,
        *arguments: str,
    ) -> Any:
        keys = self._keys(tenant_id, session_id)
        result = self._client.eval(
            self._GUARDED_SCRIPT + operation,
            len(keys),
            *keys,
            tenant_id,
            session_id,
            *arguments,
        )
        if result == -1:
            raise RuntimeError("Unowned legacy Redis session data requires explicit migration")
        if result == -2:
            raise ValueError("Redis session ownership identity mismatch")
        return result

    @staticmethod
    def _decode(payload: Any) -> dict[str, Any] | None:
        if payload is None:
            return None
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        if not isinstance(payload, str):
            raise TypeError(f"Expected string from Redis, got {type(payload)}")
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise ValueError("Redis session record is not an object")
        return decoded

    def save_session_state(self, tenant_id: str, session_id: str, state: dict[str, Any]) -> None:
        _validate_identity(tenant_id, session_id)
        _validate_state(state)
        record = json.dumps(
            {"tenant_id": tenant_id, "session_id": session_id, "state": state},
            separators=(",", ":"),
        )
        result = self._run_guarded(
            tenant_id,
            session_id,
            "redis.call('SET', KEYS[1], ARGV[3]); return 1",
            record,
        )
        if result != 1:
            raise RuntimeError("Redis session write did not commit")

    def load_session_state(self, tenant_id: str, session_id: str) -> dict[str, Any] | None:
        _validate_identity(tenant_id, session_id)
        raw = self._run_guarded(tenant_id, session_id, "return redis.call('GET', KEYS[1])")
        record = self._decode(raw)
        if record is None:
            return None
        if record.get("tenant_id") != tenant_id or record.get("session_id") != session_id:
            raise ValueError("Redis session ownership identity mismatch")
        state = record.get("state")
        _validate_state(state)
        return state

    def list_sessions(self, tenant_id: str) -> list[str]:
        tenant = require_scoped_tenant_id(tenant_id)
        encoded_tenant = quote(tenant, safe="")
        prefix = f"{self._namespace}:owned-session:{encoded_tenant}:"
        result: set[str] = set()
        for key in self._client.scan_iter(match=f"{prefix}*:state"):
            raw = self._client.get(key)
            record = self._decode(raw)
            if record is None:
                continue
            session_id = record.get("session_id")
            if record.get("tenant_id") != tenant or not isinstance(session_id, str):
                raise ValueError("Redis session listing encountered mismatched ownership metadata")
            _validate_identity(tenant, session_id)
            expected_key = self._keys(tenant, session_id)[0]
            if key != expected_key:
                raise ValueError("Redis session key does not match its ownership metadata")
            result.add(session_id)
        return sorted(result)

    def delete_session(self, tenant_id: str, session_id: str) -> bool:
        _validate_identity(tenant_id, session_id)
        result = self._run_guarded(
            tenant_id,
            session_id,
            "return redis.call('DEL', KEYS[1], KEYS[2])",
        )
        return int(result or 0) > 0

    def append_event(self, tenant_id: str, session_id: str, event: dict[str, Any]) -> None:
        _validate_identity(tenant_id, session_id)
        _validate_event(tenant_id, session_id, event)
        encoded = json.dumps(event, separators=(",", ":"))
        result = self._run_guarded(
            tenant_id,
            session_id,
            "redis.call('RPUSH', KEYS[2], ARGV[3]); return 1",
            encoded,
        )
        if result != 1:
            raise RuntimeError("Redis session event write did not commit")

    def list_events(self, tenant_id: str, session_id: str) -> list[dict[str, Any]]:
        _validate_identity(tenant_id, session_id)
        raw_events = self._run_guarded(
            tenant_id,
            session_id,
            "return redis.call('LRANGE', KEYS[2], 0, -1)",
        )
        if not isinstance(raw_events, list):
            raise TypeError(f"Expected list from Redis, got {type(raw_events)}")
        events = []
        for raw in raw_events:
            event = self._decode(raw)
            if event is None:
                raise ValueError("Redis session event is empty")
            _validate_event(tenant_id, session_id, event)
            events.append(event)
        return events


class PostgresOwnedSessions:
    def __init__(
        self,
        postgres_dsn: str,
        *,
        session_table: str,
        event_table: str,
        legacy_session_table: str,
        legacy_event_table: str,
    ) -> None:
        if not postgres_dsn:
            raise ValueError("PostgreSQL DSN is required")
        self._dsn = postgres_dsn
        digest = hashlib.sha256(f"{session_table}:{event_table}".encode()).hexdigest()[:12]
        self.session_table = f"{session_table[:40]}_owned_{digest}"
        self.event_table = f"{event_table[:40]}_owned_{digest}"
        if self.event_table == self.session_table:
            self.event_table = f"{event_table[:38]}_event_owned_{digest}"
        self._legacy_session_table = legacy_session_table
        self._legacy_event_table = legacy_event_table
        self._policy_prefix = f"owned_sessions_{digest}"
        self._ensure_tables()

    @staticmethod
    def _owned_table(cursor: Any, table_name: str, ddl: Any) -> None:
        from psycopg import sql

        cursor.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = current_schema() AND table_name = %s)",
            (table_name,),
        )
        if cursor.fetchone()[0]:
            cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = %s AND column_name = 'tenant_id')",
                (table_name,),
            )
            if not cursor.fetchone()[0]:
                quarantine = f"{table_name[:40]}_legacy_{table_name[-8:]}"
                cursor.execute(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = current_schema() AND table_name = %s)",
                    (quarantine,),
                )
                if cursor.fetchone()[0]:
                    raise RuntimeError("Legacy PostgreSQL session table quarantine already exists")
                cursor.execute(
                    sql.SQL("ALTER TABLE {} RENAME TO {}").format(
                        sql.Identifier(table_name), sql.Identifier(quarantine)
                    )
                )
        cursor.execute(ddl)

    def _ensure_tables(self) -> None:
        import psycopg
        from psycopg import sql

        with psycopg.connect(self._dsn) as connection:
            with connection.cursor() as cursor:
                self._owned_table(
                    cursor,
                    self.session_table,
                    sql.SQL(
                        "CREATE TABLE IF NOT EXISTS {} ("
                        "tenant_id TEXT NOT NULL, session_id TEXT NOT NULL, payload JSONB NOT NULL, "
                        "PRIMARY KEY (tenant_id, session_id))"
                    ).format(sql.Identifier(self.session_table)),
                )
                self._owned_table(
                    cursor,
                    self.event_table,
                    sql.SQL(
                        "CREATE TABLE IF NOT EXISTS {} ("
                        "sequence BIGSERIAL PRIMARY KEY, tenant_id TEXT NOT NULL, "
                        "session_id TEXT NOT NULL, payload JSONB NOT NULL)"
                    ).format(sql.Identifier(self.event_table)),
                )
                cursor.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (tenant_id, session_id, sequence)").format(
                        sql.Identifier(f"{self.event_table[:52]}_idx"),
                        sql.Identifier(self.event_table),
                    )
                )
                for table_name, suffix in (
                    (self.session_table, "session"),
                    (self.event_table, "event"),
                ):
                    policy = f"{self._policy_prefix}_{suffix}"
                    cursor.execute(
                        sql.SQL("ALTER TABLE {} ENABLE ROW LEVEL SECURITY").format(sql.Identifier(table_name))
                    )
                    cursor.execute(
                        sql.SQL("ALTER TABLE {} FORCE ROW LEVEL SECURITY").format(sql.Identifier(table_name))
                    )
                    cursor.execute(
                        sql.SQL("DROP POLICY IF EXISTS {} ON {}").format(
                            sql.Identifier(policy), sql.Identifier(table_name)
                        )
                    )
                    cursor.execute(
                        sql.SQL(
                            "CREATE POLICY {} ON {} USING ("
                            "tenant_id = current_setting('dadbot.tenant_id', true) "
                            "AND (tenant_id <> 'default' OR "
                            "current_setting('dadbot.allow_legacy_default_tenant', true) = 'true')) "
                            "WITH CHECK (tenant_id = current_setting('dadbot.tenant_id', true) "
                            "AND (tenant_id <> 'default' OR "
                            "current_setting('dadbot.allow_legacy_default_tenant', true) = 'true'))"
                        ).format(sql.Identifier(policy), sql.Identifier(table_name))
                    )
                cursor.execute("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")
                role = cursor.fetchone()
                if role is None or role[0] or role[1]:
                    raise RuntimeError(
                        "PostgreSQL session storage requires a non-superuser NOBYPASSRLS role"
                    )

    def _tenant_connection(self, tenant_id: str):
        from contextlib import contextmanager

        import psycopg

        tenant = require_scoped_tenant_id(tenant_id)

        @contextmanager
        def connection_scope():
            with psycopg.connect(self._dsn) as connection:
                connection.execute("SELECT set_config('dadbot.tenant_id', %s, true)", (tenant,))
                connection.execute(
                    "SELECT set_config('dadbot.allow_legacy_default_tenant', %s, true)",
                    ("true" if tenant == "default" else "false",),
                )
                yield connection

        return connection_scope()

    def _reject_legacy_rows(self, connection: Any, tenant_id: str, session_id: str) -> None:
        from psycopg import sql

        keys = (
            session_id,
            f"tenant:{tenant_id}:{session_id}",
            f"tenant-doc:{tenant_id}:{session_id}",
        )
        for table_name in (self._legacy_session_table, self._legacy_event_table):
            row = connection.execute(
                sql.SQL("SELECT 1 FROM {} WHERE session_id IN (%s, %s, %s) LIMIT 1").format(
                    sql.Identifier(table_name)
                ),
                keys,
            ).fetchone()
            if row:
                raise RuntimeError("Unowned legacy PostgreSQL session data requires explicit migration")

    def _lock_and_check_legacy(self, connection: Any, tenant_id: str, session_id: str) -> None:
        connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"{tenant_id}\x1f{session_id}",),
        )
        self._reject_legacy_rows(connection, tenant_id, session_id)
        from psycopg import sql

        row = connection.execute(
            sql.SQL("SELECT payload FROM {} WHERE tenant_id = %s AND session_id = %s").format(
                sql.Identifier(self.session_table)
            ),
            (tenant_id, session_id),
        ).fetchone()
        if row is not None:
            self._validate_record(row[0], tenant_id, session_id)
        event_rows = connection.execute(
            sql.SQL("SELECT payload FROM {} WHERE tenant_id = %s AND session_id = %s").format(
                sql.Identifier(self.event_table)
            ),
            (tenant_id, session_id),
        ).fetchall()
        for event_row in event_rows:
            _validate_event(tenant_id, session_id, event_row[0])

    @staticmethod
    def _validate_record(record: Any, tenant_id: str, session_id: str) -> None:
        if (
            not isinstance(record, dict)
            or record.get("tenant_id") != tenant_id
            or record.get("session_id") != session_id
        ):
            raise ValueError("PostgreSQL session ownership identity mismatch")

    def save_session_state(self, tenant_id: str, session_id: str, state: dict[str, Any]) -> None:
        _validate_identity(tenant_id, session_id)
        _validate_state(state)
        from psycopg import sql

        record = {"tenant_id": tenant_id, "session_id": session_id, "state": state}
        with self._tenant_connection(tenant_id) as connection:
            self._lock_and_check_legacy(connection, tenant_id, session_id)
            connection.execute(
                sql.SQL(
                    "INSERT INTO {} (tenant_id, session_id, payload) VALUES (%s, %s, %s) "
                    "ON CONFLICT (tenant_id, session_id) DO UPDATE SET payload = EXCLUDED.payload"
                ).format(sql.Identifier(self.session_table)),
                (tenant_id, session_id, json.dumps(record)),
            )

    def load_session_state(self, tenant_id: str, session_id: str) -> dict[str, Any] | None:
        _validate_identity(tenant_id, session_id)
        from psycopg import sql

        with self._tenant_connection(tenant_id) as connection:
            self._lock_and_check_legacy(connection, tenant_id, session_id)
            row = connection.execute(
                sql.SQL("SELECT payload FROM {} WHERE tenant_id = %s AND session_id = %s").format(
                    sql.Identifier(self.session_table)
                ),
                (tenant_id, session_id),
            ).fetchone()
        if row is None:
            return None
        record = row[0]
        self._validate_record(record, tenant_id, session_id)
        state = record.get("state")
        _validate_state(state)
        return state

    def list_sessions(self, tenant_id: str) -> list[str]:
        tenant = require_scoped_tenant_id(tenant_id)
        from psycopg import sql

        with self._tenant_connection(tenant) as connection:
            rows = connection.execute(
                sql.SQL("SELECT session_id, payload FROM {} WHERE tenant_id = %s ORDER BY session_id").format(
                    sql.Identifier(self.session_table)
                ),
                (tenant,),
            ).fetchall()
        sessions = []
        for session_id, record in rows:
            _validate_identity(tenant, session_id)
            self._validate_record(record, tenant, session_id)
            sessions.append(session_id)
        return sessions

    def delete_session(self, tenant_id: str, session_id: str) -> bool:
        _validate_identity(tenant_id, session_id)
        from psycopg import sql

        with self._tenant_connection(tenant_id) as connection:
            self._lock_and_check_legacy(connection, tenant_id, session_id)
            connection.execute(
                sql.SQL("DELETE FROM {} WHERE tenant_id = %s AND session_id = %s").format(
                    sql.Identifier(self.event_table)
                ),
                (tenant_id, session_id),
            )
            cursor = connection.execute(
                sql.SQL("DELETE FROM {} WHERE tenant_id = %s AND session_id = %s").format(
                    sql.Identifier(self.session_table)
                ),
                (tenant_id, session_id),
            )
            return cursor.rowcount > 0

    def append_event(self, tenant_id: str, session_id: str, event: dict[str, Any]) -> None:
        _validate_identity(tenant_id, session_id)
        _validate_event(tenant_id, session_id, event)
        from psycopg import sql

        with self._tenant_connection(tenant_id) as connection:
            self._lock_and_check_legacy(connection, tenant_id, session_id)
            connection.execute(
                sql.SQL("INSERT INTO {} (tenant_id, session_id, payload) VALUES (%s, %s, %s)").format(
                    sql.Identifier(self.event_table)
                ),
                (tenant_id, session_id, json.dumps(event)),
            )

    def list_events(self, tenant_id: str, session_id: str) -> list[dict[str, Any]]:
        _validate_identity(tenant_id, session_id)
        from psycopg import sql

        with self._tenant_connection(tenant_id) as connection:
            self._lock_and_check_legacy(connection, tenant_id, session_id)
            rows = connection.execute(
                sql.SQL(
                    "SELECT payload FROM {} WHERE tenant_id = %s AND session_id = %s ORDER BY sequence"
                ).format(sql.Identifier(self.event_table)),
                (tenant_id, session_id),
            ).fetchall()
        events = [row[0] for row in rows]
        for event in events:
            _validate_event(tenant_id, session_id, event)
        return events


class CompositeOwnedSessions:
    def __init__(self, stores: list[OwnedSessionStore]) -> None:
        if not stores:
            raise ValueError("At least one tenant-owned session store is required")
        self._stores = stores

    def save_session_state(self, tenant_id: str, session_id: str, state: dict[str, Any]) -> None:
        for store in self._stores:
            store.save_session_state(tenant_id, session_id, state)

    def load_session_state(self, tenant_id: str, session_id: str) -> dict[str, Any] | None:
        for store in self._stores:
            payload = store.load_session_state(tenant_id, session_id)
            if payload is not None:
                return payload
        return None

    def list_sessions(self, tenant_id: str) -> list[str]:
        sessions = {
            session_id
            for store in self._stores
            for session_id in store.list_sessions(tenant_id)
        }
        return sorted(sessions)

    def delete_session(self, tenant_id: str, session_id: str) -> bool:
        deleted = False
        for store in self._stores:
            deleted = store.delete_session(tenant_id, session_id) or deleted
        return deleted

    def append_event(self, tenant_id: str, session_id: str, event: dict[str, Any]) -> None:
        for store in self._stores:
            store.append_event(tenant_id, session_id, event)

    def list_events(self, tenant_id: str, session_id: str) -> list[dict[str, Any]]:
        for store in self._stores:
            events = store.list_events(tenant_id, session_id)
            if events:
                return events
        return []
