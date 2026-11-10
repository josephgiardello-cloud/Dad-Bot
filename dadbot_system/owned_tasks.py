"""Task-only storage contracts and local in-memory implementation."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock
from typing import Any, Protocol

from .execution_identity import require_identifier, require_scoped_tenant_id
from .owned_sqlite_tasks import SQLiteOwnedTasks


class OwnedTaskStore(Protocol):
    def save_task(self, tenant_id: str, task_id: str, payload: dict[str, Any]) -> None: ...

    def load_task(self, tenant_id: str, task_id: str) -> dict[str, Any] | None: ...

    def list_task_tenants(self) -> list[str]: ...

    def list_incomplete_tasks(self, tenant_id: str, *, limit: int = 1000) -> list[dict[str, Any]]: ...

    def save_task_result(
        self, tenant_id: str, task_id: str, payload: dict[str, Any], response: dict[str, Any]
    ) -> None: ...

    def load_response(self, tenant_id: str, task_id: str) -> dict[str, Any] | None: ...

    def save_response(self, tenant_id: str, task_id: str, response: dict[str, Any]) -> None: ...

    def reject_legacy_recovery(self) -> None: ...


def build_owned_task_store(
    *,
    sqlite_path: str | None = None,
    in_memory_store: OwnedTaskStore | None = None,
) -> OwnedTaskStore:
    if sqlite_path is not None and in_memory_store is not None:
        raise ValueError("Choose exactly one owned-task store backend")
    if sqlite_path is not None:
        return SQLiteOwnedTasks(sqlite_path)
    if in_memory_store is not None:
        return in_memory_store
    raise ValueError("An owned-task store backend is required")


class RedisOwnedTasks:
    def __init__(self, client, namespace: str) -> None:
        self._client = client
        self._prefix = f"{namespace}:owned-tasks"
        self._tenant_catalog_key = f"{self._prefix}:tenants"

    def _tenant_prefix(self, tenant_id: str) -> str:
        return f"{self._prefix}:{require_scoped_tenant_id(tenant_id)}:"

    def _key(self, tenant_id: str, task_id: str) -> str:
        require_identifier(task_id, "task")
        return f"{self._tenant_prefix(tenant_id)}{task_id}"

    @staticmethod
    def _payload(tenant_id: str, task_id: str, payload: dict[str, Any]) -> None:
        require_scoped_tenant_id(tenant_id)
        require_identifier(task_id, "task")
        if payload.get("tenant_id") != tenant_id or payload.get("task_id") != task_id:
            raise ValueError("Task ownership identity mismatch")

    def save_task(self, tenant_id: str, task_id: str, payload: dict[str, Any]) -> None:
        self._payload(tenant_id, task_id, payload)
        with self._client.pipeline(transaction=True) as pipeline:
            pipeline.hset(self._key(tenant_id, task_id), "payload", json.dumps(payload))
            pipeline.sadd(self._tenant_catalog_key, tenant_id)
            pipeline.execute()

    def load_task(self, tenant_id: str, task_id: str) -> dict[str, Any] | None:
        raw = self._client.hget(self._key(tenant_id, task_id), "payload")
        if raw is None:
            return None
        payload = json.loads(raw)
        self._payload(tenant_id, task_id, payload)
        return payload

    def list_task_tenants(self) -> list[str]:
        return sorted(require_scoped_tenant_id(value) for value in self._client.smembers(self._tenant_catalog_key))

    def list_incomplete_tasks(self, tenant_id: str, *, limit: int = 1000) -> list[dict[str, Any]]:
        tenant_prefix = self._tenant_prefix(tenant_id)
        tasks = []
        for key in self._client.scan_iter(match=f"{tenant_prefix}*"):
            raw = self._client.hget(key, "payload")
            if raw is None:
                continue
            payload = json.loads(raw)
            self._payload(tenant_id, str(payload.get("task_id") or ""), payload)
            if str(payload.get("status") or "").lower() in {"queued", "running"}:
                tasks.append(payload)
        tasks.sort(key=lambda payload: (payload.get("accepted_at") or "", payload["task_id"]))
        return tasks[: max(1, int(limit))]

    def save_task_result(
        self, tenant_id: str, task_id: str, payload: dict[str, Any], response: dict[str, Any]
    ) -> None:
        self._payload(tenant_id, task_id, payload)
        if response.get("tenant_id") != tenant_id:
            raise ValueError("Result ownership identity mismatch")
        key = self._key(tenant_id, task_id)
        if self._client.hget(key, "payload") is None:
            raise RuntimeError("Cannot publish a result for an unknown owned task")
        with self._client.pipeline(transaction=True) as pipeline:
            pipeline.hset(key, mapping={"payload": json.dumps(payload), "response": json.dumps(response)})
            pipeline.execute()

    def load_response(self, tenant_id: str, task_id: str) -> dict[str, Any] | None:
        raw = self._client.hget(self._key(tenant_id, task_id), "response")
        if raw is None:
            return None
        response = json.loads(raw)
        if response.get("tenant_id") != tenant_id:
            raise ValueError("Result ownership identity mismatch")
        return response

    def save_response(self, tenant_id: str, task_id: str, response: dict[str, Any]) -> None:
        if response.get("tenant_id") != tenant_id:
            raise ValueError("Result ownership identity mismatch")
        key = self._key(tenant_id, task_id)
        if self._client.hget(key, "payload") is None:
            raise RuntimeError("Cannot persist a result for an unknown owned task")
        self._client.hset(key, "response", json.dumps(response))

    def reject_legacy_recovery(self) -> None:
        return None


class CompositeOwnedTasks:
    def __init__(self, stores: list[OwnedTaskStore]) -> None:
        if not stores:
            raise ValueError("At least one tenant-qualified task store is required")
        self._stores = stores

    def save_task(self, tenant_id: str, task_id: str, payload: dict[str, Any]) -> None:
        for store in self._stores:
            store.save_task(tenant_id, task_id, payload)

    def load_task(self, tenant_id: str, task_id: str) -> dict[str, Any] | None:
        for store in self._stores:
            payload = store.load_task(tenant_id, task_id)
            if payload is not None:
                return payload
        return None

    def list_task_tenants(self) -> list[str]:
        return sorted({tenant for store in self._stores for tenant in store.list_task_tenants()})

    def list_incomplete_tasks(self, tenant_id: str, *, limit: int = 1000) -> list[dict[str, Any]]:
        tasks: dict[str, dict[str, Any]] = {}
        for store_index, store in enumerate(self._stores):
            for payload in store.list_incomplete_tasks(tenant_id, limit=limit):
                task_id = str(payload.get("task_id") or "")
                authoritative = payload
                for earlier_store in self._stores[:store_index]:
                    persisted = earlier_store.load_task(tenant_id, task_id)
                    if persisted is not None:
                        authoritative = persisted
                        break
                if str(authoritative.get("status") or "").lower() not in {"queued", "running"}:
                    tasks.pop(task_id, None)
                    continue
                if task_id not in tasks or store_index == 0:
                    tasks[task_id] = authoritative
        return sorted(
            tasks.values(),
            key=lambda payload: (payload.get("accepted_at") or "", payload["task_id"]),
        )[: max(1, int(limit))]

    def save_task_result(
        self, tenant_id: str, task_id: str, payload: dict[str, Any], response: dict[str, Any]
    ) -> None:
        for store in self._stores:
            store.save_task_result(tenant_id, task_id, payload, response)

    def load_response(self, tenant_id: str, task_id: str) -> dict[str, Any] | None:
        for store in self._stores:
            response = store.load_response(tenant_id, task_id)
            if response is not None:
                return response
        return None

    def save_response(self, tenant_id: str, task_id: str, response: dict[str, Any]) -> None:
        for store in self._stores:
            store.save_response(tenant_id, task_id, response)

    def reject_legacy_recovery(self) -> None:
        for store in self._stores:
            store.reject_legacy_recovery()


class PostgresOwnedTasks:
    def __init__(self, postgres_dsn: str, table_name: str) -> None:
        if not postgres_dsn or not table_name:
            raise ValueError("PostgreSQL DSN and task table name are required")
        self._dsn = postgres_dsn
        self._table_name = table_name
        self._policy_name = f"owned_tasks_{hashlib.sha256(table_name.encode()).hexdigest()[:12]}_tenant"
        self._tenant_catalog_name = (
            f"{table_name[:40]}_tenants_{hashlib.sha256(table_name.encode()).hexdigest()[:12]}"
        )
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        import psycopg
        from psycopg import sql

        with psycopg.connect(self._dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = current_schema() AND table_name = %s)",
                (self._table_name,),
            )
            table_exists = bool(cursor.fetchone()[0])
            if table_exists:
                cursor.execute(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
                    "WHERE table_schema = current_schema() AND table_name = %s "
                    "AND column_name = 'tenant_id')",
                    (self._table_name,),
                )
                if not cursor.fetchone()[0]:
                    quarantine_name = (
                        f"{self._table_name[:40]}_legacy_"
                        f"{hashlib.sha256(self._table_name.encode()).hexdigest()[:12]}"
                    )
                    cursor.execute(
                        "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                        "WHERE table_schema = current_schema() AND table_name = %s)",
                        (quarantine_name,),
                    )
                    if cursor.fetchone()[0]:
                        raise RuntimeError("Legacy PostgreSQL task table quarantine already exists")
                    cursor.execute(
                        sql.SQL("ALTER TABLE {} RENAME TO {}").format(
                            sql.Identifier(self._table_name),
                            sql.Identifier(quarantine_name),
                        )
                    )
            cursor.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {} ("
                    "tenant_id TEXT NOT NULL, task_id TEXT NOT NULL, "
                    "payload JSONB NOT NULL, response JSONB, "
                    "PRIMARY KEY (tenant_id, task_id))"
                ).format(sql.Identifier(self._table_name))
            )
            cursor.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {} (tenant_id TEXT PRIMARY KEY)"
                ).format(sql.Identifier(self._tenant_catalog_name))
            )
            cursor.execute(
                sql.SQL("ALTER TABLE {} ENABLE ROW LEVEL SECURITY").format(
                    sql.Identifier(self._table_name)
                )
            )
            cursor.execute(
                sql.SQL("ALTER TABLE {} FORCE ROW LEVEL SECURITY").format(
                    sql.Identifier(self._table_name)
                )
            )
            cursor.execute(
                sql.SQL("DROP POLICY IF EXISTS {} ON {}").format(
                    sql.Identifier(self._policy_name),
                    sql.Identifier(self._table_name),
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
                ).format(sql.Identifier(self._policy_name), sql.Identifier(self._table_name))
            )
            cursor.execute(
                "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
            )
            role = cursor.fetchone()
            if role is None or role[0] or role[1]:
                raise RuntimeError("PostgreSQL task storage requires a non-superuser NOBYPASSRLS role")

    def _tenant_id(self, tenant_id: str) -> str:
        return require_scoped_tenant_id(tenant_id)

    @contextmanager
    def _tenant_connection(self, tenant_id: str) -> Iterator[Any]:
        import psycopg

        tenant_id = self._tenant_id(tenant_id)
        with psycopg.connect(self._dsn) as connection:
            connection.execute(
                "SELECT set_config('dadbot.tenant_id', %s, true)",
                (tenant_id,),
            )
            connection.execute(
                "SELECT set_config('dadbot.allow_legacy_default_tenant', %s, true)",
                ("true" if tenant_id == "default" else "false",),
            )
            yield connection

    @staticmethod
    def _payload(tenant_id: str, task_id: str, payload: dict[str, Any]) -> None:
        require_identifier(task_id, "task")
        if payload.get("tenant_id") != tenant_id or payload.get("task_id") != task_id:
            raise ValueError("Task ownership identity mismatch")

    def save_task(self, tenant_id: str, task_id: str, payload: dict[str, Any]) -> None:
        tenant_id = self._tenant_id(tenant_id)
        self._payload(tenant_id, task_id, payload)
        from psycopg import sql

        with self._tenant_connection(tenant_id) as connection:
            connection.execute(
                sql.SQL(
                    "INSERT INTO {} (tenant_id, task_id, payload) VALUES (%s, %s, %s) "
                    "ON CONFLICT (tenant_id, task_id) DO UPDATE SET payload = EXCLUDED.payload"
                ).format(sql.Identifier(self._table_name)),
                (tenant_id, task_id, json.dumps(payload)),
            )
            connection.execute(
                sql.SQL("INSERT INTO {} (tenant_id) VALUES (%s) ON CONFLICT DO NOTHING").format(
                    sql.Identifier(self._tenant_catalog_name)
                ),
                (tenant_id,),
            )

    def load_task(self, tenant_id: str, task_id: str) -> dict[str, Any] | None:
        tenant_id = self._tenant_id(tenant_id)
        require_identifier(task_id, "task")
        from psycopg import sql

        with self._tenant_connection(tenant_id) as connection:
            row = connection.execute(
                sql.SQL("SELECT payload FROM {} WHERE tenant_id = %s AND task_id = %s").format(
                    sql.Identifier(self._table_name)
                ),
                (tenant_id, task_id),
            ).fetchone()
        return row[0] if row else None

    def list_task_tenants(self) -> list[str]:
        import psycopg
        from psycopg import sql

        with psycopg.connect(self._dsn) as connection:
            rows = connection.execute(
                sql.SQL("SELECT tenant_id FROM {} ORDER BY tenant_id").format(
                    sql.Identifier(self._tenant_catalog_name)
                )
            ).fetchall()
        return [require_scoped_tenant_id(row[0]) for row in rows]

    def list_incomplete_tasks(self, tenant_id: str, *, limit: int = 1000) -> list[dict[str, Any]]:
        tenant_id = self._tenant_id(tenant_id)
        from psycopg import sql

        with self._tenant_connection(tenant_id) as connection:
            rows = connection.execute(
                sql.SQL(
                    "SELECT payload FROM {} WHERE tenant_id = %s "
                    "AND payload->>'status' IN ('queued', 'running') "
                    "ORDER BY payload->>'accepted_at' NULLS FIRST, task_id LIMIT %s"
                ).format(sql.Identifier(self._table_name)),
                (tenant_id, max(1, int(limit))),
            ).fetchall()
        return [row[0] for row in rows]

    def save_task_result(
        self, tenant_id: str, task_id: str, payload: dict[str, Any], response: dict[str, Any]
    ) -> None:
        tenant_id = self._tenant_id(tenant_id)
        self._payload(tenant_id, task_id, payload)
        if response.get("tenant_id") != tenant_id:
            raise ValueError("Result ownership identity mismatch")
        from psycopg import sql

        with self._tenant_connection(tenant_id) as connection:
            cursor = connection.execute(
                sql.SQL(
                    "UPDATE {} SET payload = %s, response = %s "
                    "WHERE tenant_id = %s AND task_id = %s"
                ).format(sql.Identifier(self._table_name)),
                (json.dumps(payload), json.dumps(response), tenant_id, task_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Cannot publish a result for an unknown owned task")

    def load_response(self, tenant_id: str, task_id: str) -> dict[str, Any] | None:
        tenant_id = self._tenant_id(tenant_id)
        require_identifier(task_id, "task")
        from psycopg import sql

        with self._tenant_connection(tenant_id) as connection:
            row = connection.execute(
                sql.SQL(
                    "SELECT response FROM {} WHERE tenant_id = %s AND task_id = %s"
                ).format(sql.Identifier(self._table_name)),
                (tenant_id, task_id),
            ).fetchone()
        return row[0] if row and row[0] is not None else None

    def save_response(self, tenant_id: str, task_id: str, response: dict[str, Any]) -> None:
        tenant_id = self._tenant_id(tenant_id)
        require_identifier(task_id, "task")
        if response.get("tenant_id") != tenant_id:
            raise ValueError("Result ownership identity mismatch")
        from psycopg import sql

        with self._tenant_connection(tenant_id) as connection:
            cursor = connection.execute(
                sql.SQL(
                    "UPDATE {} SET response = %s WHERE tenant_id = %s AND task_id = %s"
                ).format(sql.Identifier(self._table_name)),
                (json.dumps(response), tenant_id, task_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Cannot persist a result for an unknown owned task")

    def reject_legacy_recovery(self) -> None:
        return None


class InMemoryOwnedTasks:
    def __init__(self, legacy_tasks: dict[str, dict[str, Any]]) -> None:
        self._legacy_tasks = legacy_tasks
        self._lock = RLock()
        self._tasks: dict[tuple[str, str], dict[str, Any]] = {}
        self._responses: dict[tuple[str, str], dict[str, Any]] = {}

    def save_task(self, tenant_id: str, task_id: str, payload: dict[str, Any]) -> None:
        SQLiteOwnedTasks._payload(tenant_id, task_id, payload)
        with self._lock:
            self._tasks[tenant_id, task_id] = copy.deepcopy(payload)

    def load_task(self, tenant_id: str, task_id: str) -> dict[str, Any] | None:
        SQLiteOwnedTasks._identity(tenant_id, task_id)
        with self._lock:
            return copy.deepcopy(self._tasks.get((tenant_id, task_id)))

    def list_task_tenants(self) -> list[str]:
        with self._lock:
            return sorted({tenant_id for tenant_id, _ in self._tasks})

    def list_incomplete_tasks(self, tenant_id: str, *, limit: int = 1000) -> list[dict[str, Any]]:
        require_scoped_tenant_id(tenant_id)
        with self._lock:
            tasks = [
                payload
                for (owner, _), payload in self._tasks.items()
                if owner == tenant_id and payload.get("status") in {"queued", "running"}
            ]
            tasks.sort(key=lambda payload: (payload.get("accepted_at") or "", payload["task_id"]))
            return copy.deepcopy(tasks[: max(1, int(limit))])

    def save_task_result(
        self, tenant_id: str, task_id: str, payload: dict[str, Any], response: dict[str, Any]
    ) -> None:
        SQLiteOwnedTasks._payload(tenant_id, task_id, payload)
        if response.get("tenant_id") != tenant_id:
            raise ValueError("Result ownership identity mismatch")
        with self._lock:
            if (tenant_id, task_id) not in self._tasks:
                raise RuntimeError("Cannot publish a result for an unknown owned task")
            self._tasks[tenant_id, task_id] = copy.deepcopy(payload)
            self._responses[tenant_id, task_id] = copy.deepcopy(response)

    def load_response(self, tenant_id: str, task_id: str) -> dict[str, Any] | None:
        SQLiteOwnedTasks._identity(tenant_id, task_id)
        with self._lock:
            return copy.deepcopy(self._responses.get((tenant_id, task_id)))

    def reject_legacy_recovery(self) -> None:
        if self._legacy_tasks:
            raise RuntimeError("Unqualified legacy tasks are quarantined; recovery requires migration")
