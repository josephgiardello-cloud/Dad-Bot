from __future__ import annotations

import json
import logging
import os
import secrets
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from dadbot.pii_scrubber import scrub_text

logger = logging.getLogger(__name__)

DEFAULT_WORKING_MEMORY_TTL_SECONDS = 60 * 60
MAX_WORKING_MEMORY_TTL_SECONDS = 60 * 60
MAX_WORKING_MEMORY_PAYLOAD_BYTES = 32 * 1024


class ToolStepRecord(Protocol):
    idempotency_key: str
    status: str
    result: Any


def new_memory_fact_reference() -> str:
    return secrets.token_hex(16)


def _scope_for(runtime: Any, turn_context: Any) -> tuple[str, str, str] | None:
    config = getattr(runtime, "config", None)
    tenant_id = str(getattr(config, "tenant_id", "") or "").strip()
    metadata = getattr(turn_context, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    control_plane = metadata.get("control_plane")
    control_plane = control_plane if isinstance(control_plane, dict) else {}
    session_id = str(control_plane.get("session_id") or metadata.get("session_id") or "").strip()
    trace_id = str(getattr(turn_context, "trace_id", "") or "").strip()
    if not tenant_id or not session_id or not trace_id:
        return None
    return tenant_id, session_id, trace_id


def working_memory_path(runtime: Any) -> Path:
    configured = str(os.environ.get("DADBOT_WORKING_MEMORY_DB_PATH") or "").strip()
    if configured:
        return Path(configured)
    config = getattr(runtime, "config", None)
    session_log_dir = getattr(config, "session_log_dir", None)
    if session_log_dir is None:
        raise ValueError("Working memory requires config.session_log_dir")
    return Path(session_log_dir) / "working_memory.sqlite3"


def _scrub_value(value: Any) -> Any:
    if isinstance(value, str):
        return scrub_text(value)[0]
    if isinstance(value, dict):
        return {
            scrub_text(str(key))[0]: _scrub_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_scrub_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return scrub_text(str(value))[0]


def _result_status(result: Any) -> str:
    status = result.get("status") if isinstance(result, dict) else getattr(result, "status", "")
    return str(getattr(status, "value", status) or "").strip().lower()


class WorkingMemoryStore:
    """Short-lived, tenant-scoped tool-step records in a dedicated SQLite file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS working_memory_steps (
                    tenant_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    step_key TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    promoted_at REAL,
                    promoted_fact_refs_json TEXT NOT NULL DEFAULT '[]',
                    PRIMARY KEY (tenant_id, session_id, trace_id, step_key)
                )
                """,
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_working_memory_steps_expiry
                ON working_memory_steps (expires_at)
                """,
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        except sqlite3.Error:
            connection.rollback()
            raise
        finally:
            connection.close()

    def write(  # noqa: PLR0913
        self,
        *,
        tenant_id: str,
        session_id: str,
        trace_id: str,
        key: str,
        value: dict[str, Any],
        ttl_seconds: float = DEFAULT_WORKING_MEMORY_TTL_SECONDS,
    ) -> None:
        scope = (tenant_id.strip(), session_id.strip(), trace_id.strip(), key.strip())
        if not all(scope):
            raise ValueError("Working-memory writes require tenant, session, trace, and step keys")
        ttl = float(ttl_seconds)
        if not 0 < ttl <= MAX_WORKING_MEMORY_TTL_SECONDS:
            raise ValueError("Working-memory TTL must be greater than zero and at most one hour")
        if not isinstance(value, dict):
            raise TypeError("Working-memory step values must be dictionaries")

        payload_json = json.dumps(
            _scrub_value(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        if len(payload_json.encode("utf-8")) > MAX_WORKING_MEMORY_PAYLOAD_BYTES:
            raise ValueError("Working-memory step exceeds the 32 KiB payload limit")

        now = time.time()
        with self._connect() as connection:
            connection.execute("DELETE FROM working_memory_steps WHERE expires_at <= ?", (now,))
            connection.execute(
                """
                INSERT INTO working_memory_steps (
                    tenant_id, session_id, trace_id, step_key, payload_json,
                    created_at, expires_at, promoted_at, promoted_fact_refs_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, '[]')
                ON CONFLICT (tenant_id, session_id, trace_id, step_key) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at,
                    promoted_at = NULL,
                    promoted_fact_refs_json = '[]'
                """,
                (*scope, payload_json, now, now + ttl),
            )

    def list_steps(
        self,
        *,
        tenant_id: str,
        session_id: str,
        trace_id: str,
    ) -> list[dict[str, Any]]:
        now = time.time()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT step_key, payload_json, created_at, expires_at,
                       promoted_at, promoted_fact_refs_json
                FROM working_memory_steps
                WHERE tenant_id = ? AND session_id = ? AND trace_id = ? AND expires_at > ?
                ORDER BY created_at, step_key
                """,
                (tenant_id, session_id, trace_id, now),
            ).fetchall()
        return [
            {
                "step_key": row[0],
                "payload": json.loads(row[1]),
                "created_at": row[2],
                "expires_at": row[3],
                "promoted_at": row[4],
                "promoted_fact_refs": json.loads(row[5]),
            }
            for row in rows
        ]

    def promote_for_committed_facts(
        self,
        *,
        tenant_id: str,
        session_id: str,
        trace_id: str,
        fact_refs: list[str],
    ) -> int:
        normalized_refs = sorted({str(item).strip() for item in fact_refs if str(item).strip()})
        if not normalized_refs:
            return 0
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE working_memory_steps
                SET promoted_at = ?, promoted_fact_refs_json = ?
                WHERE tenant_id = ? AND session_id = ? AND trace_id = ?
                  AND expires_at > ?
                """,
                (
                    now,
                    json.dumps(normalized_refs, separators=(",", ":")),
                    tenant_id,
                    session_id,
                    trace_id,
                    now,
                ),
            )
            return int(cursor.rowcount)


def record_successful_tool_step(
    runtime: Any,
    turn_context: Any,
    *,
    tool_name: str,
    parameters: dict[str, Any],
    record: ToolStepRecord,
) -> bool:
    if record.status != "succeeded":
        return False
    if _result_status(record.result) in {
        "blocked",
        "contract_violation",
        "denied",
        "error",
        "fatal",
        "failed",
        "forbidden",
        "not_authorized",
        "policy_denied",
        "rejected",
        "retry",
        "unauthenticated",
        "unauthorized",
    }:
        return False
    if any(
        (
            record.result.get(field) is False
            if isinstance(record.result, dict)
            else getattr(record.result, field, None) is False
        )
        for field in ("authorized", "allowed", "success")
    ):
        return False

    scope = _scope_for(runtime, turn_context)
    if scope is None:
        logger.warning(
            "Working-memory step skipped because tenant/session/trace scope is unavailable",
        )
        return False
    state = getattr(turn_context, "state", None)
    io_ledger = getattr(turn_context, "_tool_io_ledger", None)
    sequence = len(getattr(io_ledger, "records", []) or [])
    step_key = f"{sequence}:{record.idempotency_key}"
    value = {
        "tool_name": str(tool_name),
        "parameters": parameters,
        "result": record.result,
        "status": record.status,
        "idempotency_key": record.idempotency_key,
    }
    try:
        WorkingMemoryStore(working_memory_path(runtime)).write(
            tenant_id=scope[0],
            session_id=scope[1],
            trace_id=scope[2],
            key=step_key,
            value=value,
        )
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        logger.error(
            "Working-memory step write failed (error_type=%s)",
            type(exc).__name__,
            exc_info=True,
        )
        if isinstance(state, dict):
            errors = list(state.get("working_memory_errors") or [])
            errors.append({"operation": "write", "error_type": type(exc).__name__})
            state["working_memory_errors"] = errors
        return False
    return True


def promote_committed_working_memory(runtime: Any, turn_context: Any) -> int:
    state = getattr(turn_context, "state", None)
    if not isinstance(state, dict):
        return 0
    transaction = dict(state.get("_save_transaction") or {})
    if transaction.get("status") != "committed":
        return 0
    fact_refs = list(state.get("_validated_memory_fact_refs") or [])
    scope = _scope_for(runtime, turn_context)
    if scope is None or not fact_refs:
        return 0
    promoted = WorkingMemoryStore(working_memory_path(runtime)).promote_for_committed_facts(
        tenant_id=scope[0],
        session_id=scope[1],
        trace_id=scope[2],
        fact_refs=fact_refs,
    )
    state["working_memory_promoted_steps"] = promoted
    return promoted


__all__ = [
    "DEFAULT_WORKING_MEMORY_TTL_SECONDS",
    "WorkingMemoryStore",
    "new_memory_fact_reference",
    "promote_committed_working_memory",
    "record_successful_tool_step",
    "working_memory_path",
]
