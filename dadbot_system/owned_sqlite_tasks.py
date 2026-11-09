"""Tenant-qualified SQLite task storage.

This repository does not authenticate callers; its tenant argument must come
from an independently verified execution boundary.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from .execution_identity import require_identifier, require_tenant_id


class SQLiteOwnedTasks:
    def __init__(self, path: str) -> None:
        if not path or path == ":memory:":
            raise ValueError("A persistent SQLite state path is required")
        self.path = str(Path(path).resolve())
        self._uri = Path(self.path).as_uri() + "?mode=rw"
        with closing(sqlite3.connect(self.path, timeout=5)) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS owned_tasks_v1 ("
                "tenant_id TEXT NOT NULL, task_id TEXT NOT NULL, payload TEXT NOT NULL, response TEXT, "
                "PRIMARY KEY (tenant_id, task_id))"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS owned_tasks_v1_pending ON owned_tasks_v1 "
                "(tenant_id, json_extract(payload, '$.status'), "
                "json_extract(payload, '$.accepted_at'), task_id)"
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
    def _identity(tenant_id: str, task_id: str) -> None:
        require_tenant_id(tenant_id)
        require_identifier(task_id, "task")

    @classmethod
    def _payload(cls, tenant_id: str, task_id: str, payload: dict[str, Any]) -> None:
        cls._identity(tenant_id, task_id)
        if payload.get("tenant_id") != tenant_id or payload.get("task_id") != task_id:
            raise ValueError("Task ownership identity mismatch")

    def save_task(self, tenant_id: str, task_id: str, payload: dict[str, Any]) -> None:
        self._payload(tenant_id, task_id, payload)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO owned_tasks_v1 (tenant_id, task_id, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(tenant_id, task_id) DO UPDATE SET payload=excluded.payload",
                (tenant_id, task_id, json.dumps(payload)),
            )

    def load_task(self, tenant_id: str, task_id: str) -> dict[str, Any] | None:
        self._identity(tenant_id, task_id)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload FROM owned_tasks_v1 WHERE tenant_id=? AND task_id=?",
                (tenant_id, task_id),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        self._payload(tenant_id, task_id, payload)
        return payload

    def list_incomplete_tasks(self, tenant_id: str, *, limit: int = 1000) -> list[dict[str, Any]]:
        require_tenant_id(tenant_id)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT task_id, payload FROM owned_tasks_v1 WHERE tenant_id=? "
                "AND json_extract(payload, '$.status') IN ('queued', 'running') "
                "ORDER BY json_extract(payload, '$.accepted_at'), task_id LIMIT ?",
                (tenant_id, max(1, int(limit))),
            ).fetchall()
        tasks = []
        for task_id, raw in rows:
            payload = json.loads(raw)
            self._payload(tenant_id, task_id, payload)
            tasks.append(payload)
        return tasks

    def save_task_result(
        self, tenant_id: str, task_id: str, payload: dict[str, Any], response: dict[str, Any]
    ) -> None:
        self._payload(tenant_id, task_id, payload)
        if response.get("tenant_id") != tenant_id:
            raise ValueError("Result ownership identity mismatch")
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "UPDATE owned_tasks_v1 SET payload=?, response=? WHERE tenant_id=? AND task_id=?",
                (json.dumps(payload), json.dumps(response), tenant_id, task_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Cannot publish a result for an unknown owned task")

    def load_response(self, tenant_id: str, task_id: str) -> dict[str, Any] | None:
        self._identity(tenant_id, task_id)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT response FROM owned_tasks_v1 WHERE tenant_id=? AND task_id=?",
                (tenant_id, task_id),
            ).fetchone()
        if row is None or row[0] is None:
            return None
        response = json.loads(row[0])
        if response.get("tenant_id") != tenant_id:
            raise ValueError("Result ownership identity mismatch")
        return response

    def reject_legacy_recovery(self) -> None:
        with closing(self._connect()) as connection:
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tasks'"
            ).fetchone()
            if table and connection.execute("SELECT 1 FROM tasks LIMIT 1").fetchone():
                raise RuntimeError("Unqualified legacy tasks are quarantined; recovery requires migration")
