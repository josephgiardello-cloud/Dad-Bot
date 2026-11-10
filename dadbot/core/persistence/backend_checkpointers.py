"""Trusted-local Redis and PostgreSQL checkpoint backends."""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from collections.abc import Mapping
from typing import Any, Literal

from dadbot.core.persistence.base import (
    AbstractCheckpointer,
    CheckpointError,
    CheckpointIntegrityError,
    CheckpointNotFoundError,
)
from dadbot.core.persistence.checkpointer import (
    checkpoint_result,
    decode_loaded_checkpoint,
    prepare_checkpoint_payload,
    resolve_checkpoint_tenant,
    verify_checkpoint_row,
    verify_manifest,
)

logger = logging.getLogger(__name__)


class _TrustedLocalCheckpointer(AbstractCheckpointer):
    def __init__(
        self,
        *,
        default_tenant_id: str,
        trusted_local: Literal[True],
    ) -> None:
        if trusted_local is not True:
            raise CheckpointError("Remote checkpointers require explicit trusted-local opt-in")
        self.default_tenant_id = str(default_tenant_id or "").strip()
        if not self.default_tenant_id:
            raise CheckpointError("default_tenant_id is required")

    def _resolve_tenant(self, tenant_id: str | None) -> str:
        return resolve_checkpoint_tenant(self.default_tenant_id, tenant_id)

    @staticmethod
    def _validate_ids(session_id: str, trace_id: str | None = None) -> tuple[str, str | None]:
        normalized_session = str(session_id or "").strip()
        normalized_trace = str(trace_id or "").strip() if trace_id is not None else None
        if not normalized_session or (trace_id is not None and not normalized_trace):
            raise CheckpointError("session_id and supplied trace_id must be non-empty")
        return normalized_session, normalized_trace

    @staticmethod
    def _prepare(
        tenant_id: str,
        session_id: str,
        trace_id: str,
        checkpoint: dict[str, Any],
        manifest: dict[str, Any],
    ) -> tuple[str, str, str, str]:
        checkpoint_hash, previous_hash, manifest_hash, payload_json, _ = prepare_checkpoint_payload(
            tenant_id,
            session_id,
            trace_id,
            checkpoint,
            manifest,
        )
        return checkpoint_hash, previous_hash, manifest_hash, payload_json

    @staticmethod
    def _loaded_result(
        row: Mapping[str, Any],
        *,
        current_manifest: dict[str, Any] | None,
        strict: bool,
    ) -> dict[str, Any]:
        loaded = decode_loaded_checkpoint(row)
        verify_checkpoint_row(row, loaded.checkpoint)
        verify_manifest(loaded.manifest, current_manifest, strict=strict)
        return checkpoint_result(row, loaded)


class PostgresCheckpointer(_TrustedLocalCheckpointer):
    """Transactional PostgreSQL checkpointer for explicitly trusted-local use."""

    SCHEMA_VERSION = "1"

    def __init__(
        self,
        postgres_dsn: str,
        *,
        default_tenant_id: str = "default",
        table_prefix: str = "dadbot_checkpoint",
        trusted_local: Literal[True],
    ) -> None:
        super().__init__(default_tenant_id=default_tenant_id, trusted_local=trusted_local)
        prefix = str(table_prefix or "").strip()
        if re.fullmatch(r"[a-z][a-z0-9_]{0,39}", prefix) is None:
            raise CheckpointError("table_prefix must be a lowercase SQL identifier prefix")
        if not str(postgres_dsn or "").strip():
            raise CheckpointError("postgres_dsn is required")
        try:
            import psycopg
            from psycopg import sql
            from psycopg.rows import dict_row
            from psycopg.types.json import Jsonb
        except ImportError as exc:
            raise CheckpointError("psycopg is required for PostgresCheckpointer") from exc
        self._psycopg = psycopg
        self._sql = sql
        self._dict_row = dict_row
        self._jsonb = Jsonb
        self._dsn = str(postgres_dsn).strip()
        self._prefix = prefix
        self._checkpoint_table = f"{prefix}_records"
        self._write_table = f"{prefix}_writes"
        self._meta_table = f"{prefix}_meta"
        self.migrate()

    def _connect(self):
        return self._psycopg.connect(self._dsn, row_factory=self._dict_row)

    def _identifier(self, table_name: str):
        return self._sql.Identifier(table_name)

    def migrate(self) -> None:
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    self._sql.SQL("CREATE TABLE IF NOT EXISTS {} (key TEXT PRIMARY KEY, value TEXT NOT NULL)").format(
                        self._identifier(self._meta_table),
                    ),
                )
                cursor.execute(
                    self._sql.SQL("SELECT value FROM {} WHERE key = 'schema_version' FOR UPDATE").format(
                        self._identifier(self._meta_table),
                    ),
                )
                version_row = cursor.fetchone()
                if version_row is not None and str(version_row["value"]) != self.SCHEMA_VERSION:
                    raise CheckpointError(
                        f"Unsupported PostgreSQL checkpoint schema version: {version_row['value']}",
                    )
                if version_row is None:
                    cursor.execute(
                        """
                        SELECT to_regclass(%s) IS NOT NULL AS records_exists,
                               to_regclass(%s) IS NOT NULL AS writes_exists
                        """,
                        (self._checkpoint_table, self._write_table),
                    )
                    existing = cursor.fetchone()
                    if existing and (existing["records_exists"] or existing["writes_exists"]):
                        raise CheckpointError(
                            "Unversioned PostgreSQL checkpoint tables exist; refusing to adopt them",
                        )
                    cursor.execute(
                        self._sql.SQL(
                            "CREATE TABLE {} ("
                            "id BIGSERIAL PRIMARY KEY, tenant_id TEXT NOT NULL, "
                            "session_id TEXT NOT NULL, trace_id TEXT NOT NULL, "
                            "checkpoint_hash TEXT NOT NULL, prev_checkpoint_hash TEXT NOT NULL, "
                            "manifest_hash TEXT NOT NULL, payload JSONB NOT NULL, "
                            "created_at DOUBLE PRECISION NOT NULL, "
                            "UNIQUE (tenant_id, session_id, trace_id))"
                        ).format(self._identifier(self._checkpoint_table)),
                    )
                    cursor.execute(
                        self._sql.SQL(
                            "CREATE TABLE {} ("
                            "id BIGSERIAL PRIMARY KEY, tenant_id TEXT NOT NULL, "
                            "session_id TEXT NOT NULL, trace_id TEXT NOT NULL, "
                            "checkpoint_hash TEXT NOT NULL, status TEXT NOT NULL, "
                            "error TEXT NOT NULL, created_at DOUBLE PRECISION NOT NULL)"
                        ).format(self._identifier(self._write_table)),
                    )
                    cursor.execute(
                        self._sql.SQL(
                            "CREATE INDEX {} ON {} (tenant_id, session_id, created_at DESC, id DESC)"
                        ).format(
                            self._identifier(f"{self._prefix}_records_owner_idx"),
                            self._identifier(self._checkpoint_table),
                        ),
                    )
                    cursor.execute(
                        self._sql.SQL(
                            "CREATE INDEX {} ON {} (tenant_id, session_id, created_at DESC, id DESC)"
                        ).format(
                            self._identifier(f"{self._prefix}_writes_owner_idx"),
                            self._identifier(self._write_table),
                        ),
                    )
                    cursor.execute(
                        self._sql.SQL("INSERT INTO {} (key, value) VALUES ('schema_version', %s)").format(
                            self._identifier(self._meta_table),
                        ),
                        (self.SCHEMA_VERSION,),
                    )
                else:
                    self._validate_postgres_schema(cursor)
        except CheckpointError:
            raise
        except self._psycopg.Error as exc:
            raise CheckpointError(f"PostgreSQL checkpoint migration failed: {exc}") from exc

    def _validate_postgres_schema(self, cursor: Any) -> None:
        cursor.execute(
            """
            SELECT table_name, column_name FROM information_schema.columns
            WHERE table_schema = current_schema() AND table_name = ANY(%s)
            """,
            ([self._checkpoint_table, self._write_table],),
        )
        found: dict[str, set[str]] = {}
        for row in cursor.fetchall():
            found.setdefault(str(row["table_name"]), set()).add(str(row["column_name"]))
        required = {
            self._checkpoint_table: {
                "id", "tenant_id", "session_id", "trace_id", "checkpoint_hash",
                "prev_checkpoint_hash", "manifest_hash", "payload", "created_at",
            },
            self._write_table: {
                "id", "tenant_id", "session_id", "trace_id", "checkpoint_hash",
                "status", "error", "created_at",
            },
        }
        for table_name, columns in required.items():
            if not columns.issubset(found.get(table_name, set())):
                raise CheckpointError(f"PostgreSQL checkpoint schema is incomplete: {table_name}")

    def save_checkpoint(
        self,
        session_id: str,
        trace_id: str,
        checkpoint: dict[str, Any],
        manifest: dict[str, Any],
        *,
        tenant_id: str | None = None,
    ) -> bool:
        resolved_tenant = self._resolve_tenant(tenant_id)
        session_id, normalized_trace = self._validate_ids(session_id, trace_id)
        assert normalized_trace is not None
        checkpoint_hash, previous_hash, manifest_hash, payload_json = self._prepare(
            resolved_tenant,
            session_id,
            normalized_trace,
            checkpoint,
            manifest,
        )
        now = time.time()
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute("SET LOCAL synchronous_commit = on")
                cursor.execute(
                    self._sql.SQL(
                        "INSERT INTO {} "
                        "(tenant_id, session_id, trace_id, checkpoint_hash, prev_checkpoint_hash, "
                        "manifest_hash, payload, created_at) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                        "ON CONFLICT (tenant_id, session_id, trace_id) DO UPDATE SET "
                        "checkpoint_hash = EXCLUDED.checkpoint_hash, "
                        "prev_checkpoint_hash = EXCLUDED.prev_checkpoint_hash, "
                        "manifest_hash = EXCLUDED.manifest_hash, payload = EXCLUDED.payload, "
                        "created_at = EXCLUDED.created_at RETURNING checkpoint_hash"
                    ).format(self._identifier(self._checkpoint_table)),
                    (
                        resolved_tenant, session_id, normalized_trace, checkpoint_hash,
                        previous_hash, manifest_hash, self._jsonb(json.loads(payload_json)), now,
                    ),
                )
                written = cursor.fetchone()
                if written is None or str(written["checkpoint_hash"] or "").strip() != checkpoint_hash:
                    raise CheckpointIntegrityError("PostgreSQL checkpoint write returned an unexpected hash")
                cursor.execute(
                    self._sql.SQL(
                        "INSERT INTO {} "
                        "(tenant_id, session_id, trace_id, checkpoint_hash, status, error, created_at) "
                        "VALUES (%s, %s, %s, %s, 'ok', '', %s)"
                    ).format(self._identifier(self._write_table)),
                    (resolved_tenant, session_id, normalized_trace, checkpoint_hash, now),
                )
        except self._psycopg.Error as exc:
            self._log_write_failure(resolved_tenant, session_id, normalized_trace, str(exc))
            raise CheckpointError(f"Failed to save PostgreSQL checkpoint: {exc}") from exc
        except CheckpointIntegrityError:
            self._log_write_failure(
                resolved_tenant,
                session_id,
                normalized_trace,
                "post-write consistency check failed",
            )
            raise
        return True

    def _log_write_failure(self, tenant_id: str, session_id: str, trace_id: str, error: str) -> None:
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    self._sql.SQL(
                        "INSERT INTO {} "
                        "(tenant_id, session_id, trace_id, checkpoint_hash, status, error, created_at) "
                        "VALUES (%s, %s, %s, '', 'error', %s, %s)"
                    ).format(self._identifier(self._write_table)),
                    (tenant_id, session_id, trace_id, str(error), time.time()),
                )
        except self._psycopg.Error:
            logger.exception("Could not record PostgreSQL checkpoint write failure")

    def _fetch_row(
        self,
        cursor: Any,
        tenant_id: str,
        session_id: str,
        trace_id: str | None,
    ) -> Mapping[str, Any] | None:
        if trace_id:
            cursor.execute(
                self._sql.SQL(
                    "SELECT * FROM {} WHERE tenant_id = %s AND session_id = %s AND trace_id = %s"
                ).format(self._identifier(self._checkpoint_table)),
                (tenant_id, session_id, trace_id),
            )
        else:
            cursor.execute(
                self._sql.SQL(
                    "SELECT * FROM {} WHERE tenant_id = %s AND session_id = %s "
                    "ORDER BY created_at DESC, id DESC LIMIT 1"
                ).format(self._identifier(self._checkpoint_table)),
                (tenant_id, session_id),
            )
        return cursor.fetchone()

    def _verify_prev_link(self, cursor: Any, row: Mapping[str, Any]) -> None:
        previous_hash = str(row["prev_checkpoint_hash"] or "").strip()
        if not previous_hash:
            return
        cursor.execute(
            self._sql.SQL(
                "SELECT checkpoint_hash FROM {} WHERE tenant_id = %s AND session_id = %s "
                "AND (created_at, id) < (%s, %s) "
                "ORDER BY created_at DESC, id DESC LIMIT 1"
            ).format(self._identifier(self._checkpoint_table)),
            (row["tenant_id"], row["session_id"], row["created_at"], row["id"]),
        )
        prior = cursor.fetchone()
        if prior is not None:
            actual_hash = str(prior["checkpoint_hash"] or "").strip()
            if actual_hash != previous_hash:
                raise CheckpointIntegrityError(
                    f"Checkpoint hash-chain broken: expected prev={actual_hash!r}, got {previous_hash!r}",
                )
            return
        cursor.execute(
            self._sql.SQL(
                "SELECT 1 FROM {} WHERE tenant_id = %s AND session_id = %s "
                "AND checkpoint_hash = %s AND status = 'ok' LIMIT 1"
            ).format(self._identifier(self._write_table)),
            (row["tenant_id"], row["session_id"], previous_hash),
        )
        if cursor.fetchone() is None:
            raise CheckpointIntegrityError(
                "Checkpoint hash-chain broken: prev_checkpoint_hash has no prior checkpoint or write evidence",
            )

    def load_checkpoint(
        self,
        session_id: str,
        trace_id: str | None = None,
        *,
        current_manifest: dict[str, Any] | None = None,
        strict: bool = False,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        resolved_tenant = self._resolve_tenant(tenant_id)
        session_id, normalized_trace = self._validate_ids(session_id, trace_id)
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                row = self._fetch_row(cursor, resolved_tenant, session_id, normalized_trace)
                if row is None:
                    raise CheckpointNotFoundError(
                        f"No checkpoint found for session_id={session_id!r}, trace_id={normalized_trace!r}",
                    )
                result = self._loaded_result(
                    row,
                    current_manifest=current_manifest,
                    strict=strict,
                )
                self._verify_prev_link(cursor, row)
                return result
        except (CheckpointError, self._psycopg.Error):
            raise
        except (ValueError, TypeError, KeyError) as exc:
            raise CheckpointError(f"PostgreSQL checkpoint load failed: {exc}") from exc

    def prune_old_checkpoints(
        self,
        session_id: str,
        keep_count: int = 10,
        older_than_days: int | None = None,
        *,
        tenant_id: str | None = None,
    ) -> int:
        resolved_tenant = self._resolve_tenant(tenant_id)
        session_id, _ = self._validate_ids(session_id)
        keep_count = max(int(keep_count or 0), 0)
        cutoff = (
            time.time() - int(older_than_days) * 86400
            if older_than_days is not None and int(older_than_days) >= 0
            else None
        )
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    self._sql.SQL(
                        "SELECT id, created_at FROM {} WHERE tenant_id = %s AND session_id = %s "
                        "ORDER BY created_at DESC, id DESC FOR UPDATE"
                    ).format(self._identifier(self._checkpoint_table)),
                    (resolved_tenant, session_id),
                )
                rows = cursor.fetchall()
                remove_ids = [
                    int(row["id"])
                    for index, row in enumerate(rows)
                    if (keep_count > 0 and index >= keep_count)
                    or (cutoff is not None and float(row["created_at"]) < cutoff)
                ]
                if remove_ids:
                    cursor.execute(
                        self._sql.SQL("DELETE FROM {} WHERE id = ANY(%s)").format(
                            self._identifier(self._checkpoint_table),
                        ),
                        (remove_ids,),
                    )
                return len(remove_ids)
        except self._psycopg.Error as exc:
            raise CheckpointError(f"PostgreSQL checkpoint prune failed: {exc}") from exc

    def prune_checkpoint_write_logs(
        self,
        session_id: str,
        keep_count: int = 200,
        older_than_days: int | None = None,
        *,
        tenant_id: str | None = None,
    ) -> int:
        resolved_tenant = self._resolve_tenant(tenant_id)
        session_id, _ = self._validate_ids(session_id)
        keep_count = max(int(keep_count or 0), 0)
        cutoff = (
            time.time() - int(older_than_days) * 86400
            if older_than_days is not None and int(older_than_days) >= 0
            else None
        )
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    self._sql.SQL(
                        "SELECT id, created_at FROM {} WHERE tenant_id = %s AND session_id = %s "
                        "ORDER BY created_at DESC, id DESC FOR UPDATE"
                    ).format(self._identifier(self._write_table)),
                    (resolved_tenant, session_id),
                )
                rows = cursor.fetchall()
                remove_ids = [
                    int(row["id"])
                    for index, row in enumerate(rows)
                    if (keep_count > 0 and index >= keep_count)
                    or (cutoff is not None and float(row["created_at"]) < cutoff)
                ]
                if remove_ids:
                    cursor.execute(
                        self._sql.SQL("DELETE FROM {} WHERE id = ANY(%s)").format(
                            self._identifier(self._write_table),
                        ),
                        (remove_ids,),
                    )
                return len(remove_ids)
        except self._psycopg.Error as exc:
            raise CheckpointError(f"PostgreSQL checkpoint write-log prune failed: {exc}") from exc

    def delete_session(self, session_id: str, *, tenant_id: str | None = None) -> int:
        resolved_tenant = self._resolve_tenant(tenant_id)
        session_id, _ = self._validate_ids(session_id)
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    self._sql.SQL("DELETE FROM {} WHERE tenant_id = %s AND session_id = %s").format(
                        self._identifier(self._checkpoint_table),
                    ),
                    (resolved_tenant, session_id),
                )
                deleted = int(cursor.rowcount or 0)
                cursor.execute(
                    self._sql.SQL("DELETE FROM {} WHERE tenant_id = %s AND session_id = %s").format(
                        self._identifier(self._write_table),
                    ),
                    (resolved_tenant, session_id),
                )
                return deleted
        except self._psycopg.Error as exc:
            raise CheckpointError(f"PostgreSQL checkpoint deletion failed: {exc}") from exc

    def checkpoint_count(self, session_id: str, *, tenant_id: str | None = None) -> int:
        resolved_tenant = self._resolve_tenant(tenant_id)
        session_id, _ = self._validate_ids(session_id)
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    self._sql.SQL(
                        "SELECT COUNT(*) AS count FROM {} WHERE tenant_id = %s AND session_id = %s"
                    ).format(self._identifier(self._checkpoint_table)),
                    (resolved_tenant, session_id),
                )
                row = cursor.fetchone()
                return int(row["count"] if row is not None else 0)
        except self._psycopg.Error as exc:
            raise CheckpointError(f"PostgreSQL checkpoint count failed: {exc}") from exc

    def validate_consistency(
        self,
        session_id: str,
        *,
        tenant_id: str | None = None,
        strict: bool = False,
    ) -> dict[str, Any]:
        resolved_tenant = self._resolve_tenant(tenant_id)
        session_id, _ = self._validate_ids(session_id)
        issues: list[str] = []
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    self._sql.SQL(
                        "SELECT * FROM {} WHERE tenant_id = %s AND session_id = %s "
                        "ORDER BY created_at ASC, id ASC"
                    ).format(self._identifier(self._checkpoint_table)),
                    (resolved_tenant, session_id),
                )
                rows = cursor.fetchall()
                for row in rows:
                    try:
                        loaded = decode_loaded_checkpoint(row)
                        verify_checkpoint_row(row, loaded.checkpoint)
                        self._verify_prev_link(cursor, row)
                    except CheckpointError as exc:
                        issues.append(str(exc))
                cursor.execute(
                    self._sql.SQL(
                        "SELECT COUNT(*) AS count FROM {} w LEFT JOIN {} c "
                        "ON w.tenant_id = c.tenant_id AND w.session_id = c.session_id "
                        "AND w.trace_id = c.trace_id WHERE w.tenant_id = %s AND w.session_id = %s "
                        "AND w.status = 'ok' AND c.id IS NULL"
                    ).format(
                        self._identifier(self._write_table),
                        self._identifier(self._checkpoint_table),
                    ),
                    (resolved_tenant, session_id),
                )
                orphan_count = int(cursor.fetchone()["count"])
                if orphan_count:
                    issues.append(f"orphan_ok_write_rows:{orphan_count}")
        except self._psycopg.Error as exc:
            issues.append(f"postgres_error:{exc}")
        result = {"ok": not issues, "issues": issues}
        if strict and issues:
            raise CheckpointIntegrityError(f"PostgreSQL checkpoint consistency check failed: {issues}")
        return result


class RedisCheckpointer(_TrustedLocalCheckpointer):
    """Redis checkpointer using atomic Lua writes and tenant/session key slots."""

    SCHEMA_VERSION = "1"

    _SAVE_SCRIPT = """
local cp_type = redis.call('TYPE', KEYS[1]).ok
local index_type = redis.call('TYPE', KEYS[2]).ok
local registry_type = redis.call('TYPE', KEYS[3]).ok
local writes_type = redis.call('TYPE', KEYS[4]).ok
if cp_type ~= 'none' and cp_type ~= 'string' then return redis.error_reply('checkpoint key has wrong type') end
if index_type ~= 'none' and index_type ~= 'zset' then return redis.error_reply('checkpoint index has wrong type') end
if registry_type ~= 'none' and registry_type ~= 'set' then
    return redis.error_reply('checkpoint registry has wrong type')
end
if writes_type ~= 'none' and writes_type ~= 'list' then
    return redis.error_reply('checkpoint write log has wrong type')
end
local old = redis.call('GET', KEYS[1])
if old then
    local ok, old_record = pcall(cjson.decode, old)
    if not ok or type(old_record.sort_key) ~= 'string' then
        return redis.error_reply('existing checkpoint record is malformed')
    end
    redis.call('ZREM', KEYS[2], old_record.sort_key)
end
redis.call('SET', KEYS[1], ARGV[1])
redis.call('ZADD', KEYS[2], 0, ARGV[2])
redis.call('SADD', KEYS[3], ARGV[3])
redis.call('RPUSH', KEYS[4], ARGV[4])
return ARGV[5]
"""

    _PRUNE_SCRIPT = """
local index_type = redis.call('TYPE', KEYS[1]).ok
local registry_type = redis.call('TYPE', KEYS[2]).ok
if index_type ~= 'none' and index_type ~= 'zset' then return redis.error_reply('checkpoint index has wrong type') end
if registry_type ~= 'none' and registry_type ~= 'set' then
    return redis.error_reply('checkpoint registry has wrong type')
end
local removed = 0
for i = 2, #ARGV, 2 do
    local encoded_trace = ARGV[i]
    local expected_sort_key = ARGV[i + 1]
    local checkpoint_key = ARGV[1] .. encoded_trace
    local raw = redis.call('GET', checkpoint_key)
    if raw then
        local ok, record = pcall(cjson.decode, raw)
        if ok and record.sort_key == expected_sort_key then
            redis.call('DEL', checkpoint_key)
            redis.call('ZREM', KEYS[1], expected_sort_key)
            redis.call('SREM', KEYS[2], encoded_trace)
            removed = removed + 1
        end
    end
end
return removed
"""

    _DELETE_SCRIPT = """
local registry_type = redis.call('TYPE', KEYS[2]).ok
if registry_type ~= 'none' and registry_type ~= 'set' then
    return redis.error_reply('checkpoint registry has wrong type')
end
local traces = redis.call('SMEMBERS', KEYS[2])
for _, encoded_trace in ipairs(traces) do
    redis.call('DEL', ARGV[1] .. encoded_trace)
end
local removed = #traces
redis.call('DEL', KEYS[1], KEYS[2], KEYS[3])
return removed
"""

    _PRUNE_LOG_SCRIPT = """
local log_type = redis.call('TYPE', KEYS[1]).ok
if log_type ~= 'none' and log_type ~= 'list' then return redis.error_reply('checkpoint write log has wrong type') end
local values = redis.call('LRANGE', KEYS[1], 0, -1)
local first_kept = {}
local keep_count = tonumber(ARGV[1])
local cutoff = tonumber(ARGV[2])
local total = #values
for i, raw in ipairs(values) do
    local ok, record = pcall(cjson.decode, raw)
    if not ok then return redis.error_reply('checkpoint write log is malformed') end
    local within_count = keep_count <= 0 or i > total - keep_count
    local recent_enough = cutoff <= 0 or tonumber(record.created_at) >= cutoff
    if within_count and recent_enough then table.insert(first_kept, raw) end
end
redis.call('DEL', KEYS[1])
for _, raw in ipairs(first_kept) do redis.call('RPUSH', KEYS[1], raw) end
return total - #first_kept
"""

    def __init__(
        self,
        redis_url: str,
        *,
        namespace: str = "dadbot",
        default_tenant_id: str = "default",
        trusted_local: Literal[True],
    ) -> None:
        super().__init__(default_tenant_id=default_tenant_id, trusted_local=trusted_local)
        if not str(redis_url or "").strip():
            raise CheckpointError("redis_url is required")
        try:
            import redis
        except ImportError as exc:
            raise CheckpointError("redis package is required for RedisCheckpointer") from exc
        self._redis_errors = redis.exceptions.RedisError
        self._client = redis.Redis.from_url(str(redis_url).strip(), decode_responses=True)
        self._namespace = str(namespace or "").strip()
        if not self._namespace:
            raise CheckpointError("namespace is required")
        self._save_script = self._client.register_script(self._SAVE_SCRIPT)
        self._prune_script = self._client.register_script(self._PRUNE_SCRIPT)
        self._delete_script = self._client.register_script(self._DELETE_SCRIPT)
        self._prune_log_script = self._client.register_script(self._PRUNE_LOG_SCRIPT)
        self.migrate()

    @staticmethod
    def _encode(value: str) -> str:
        return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")

    @staticmethod
    def _decode(value: str) -> str:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode("utf-8")

    def _root(self, tenant_id: str, session_id: str) -> str:
        return f"{self._namespace}:checkpointer:{{{self._encode(tenant_id)}:{self._encode(session_id)}}}"

    def _session_keys(self, tenant_id: str, session_id: str) -> tuple[str, str, str]:
        root = self._root(tenant_id, session_id)
        return f"{root}:index", f"{root}:registry", f"{root}:writes"

    def _checkpoint_key(self, tenant_id: str, session_id: str, encoded_trace: str) -> str:
        return f"{self._root(tenant_id, session_id)}:checkpoint:{encoded_trace}"

    @staticmethod
    def _sort_key(created_ns: int, encoded_trace: str) -> str:
        return f"{created_ns:020d}:{encoded_trace}"

    def migrate(self) -> None:
        try:
            self._client.ping()
            schema_key = f"{self._namespace}:checkpointer:schema_version"
            self._client.set(schema_key, self.SCHEMA_VERSION, nx=True)
            version = self._client.get(schema_key)
            if str(version) != self.SCHEMA_VERSION:
                raise CheckpointError(f"Unsupported Redis checkpoint schema version: {version}")
        except CheckpointError:
            raise
        except self._redis_errors as exc:
            raise CheckpointError(f"Redis checkpoint initialization failed: {exc}") from exc

    def _record_to_row(self, raw: str | None) -> Mapping[str, Any] | None:
        if raw is None:
            return None
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CheckpointIntegrityError("Stored Redis checkpoint record is malformed") from exc
        required = {
            "tenant_id", "session_id", "trace_id", "checkpoint_hash",
            "prev_checkpoint_hash", "manifest_hash", "payload", "created_at", "sort_key",
        }
        if not isinstance(record, dict) or not required.issubset(record):
            raise CheckpointIntegrityError("Stored Redis checkpoint record is incomplete")
        return record

    def _get_record(
        self,
        tenant_id: str,
        session_id: str,
        trace_id: str,
    ) -> Mapping[str, Any] | None:
        encoded_trace = self._encode(trace_id)
        raw = self._client.get(self._checkpoint_key(tenant_id, session_id, encoded_trace))
        return self._record_to_row(raw)

    def _iter_rows(self, tenant_id: str, session_id: str) -> list[Mapping[str, Any]]:
        _, registry_key, _ = self._session_keys(tenant_id, session_id)
        rows: list[Mapping[str, Any]] = []
        for encoded_trace in self._client.smembers(registry_key):
            try:
                trace_id = self._decode(str(encoded_trace))
            except (ValueError, UnicodeDecodeError) as exc:
                raise CheckpointIntegrityError("Redis checkpoint registry contains an invalid trace ID") from exc
            row = self._get_record(tenant_id, session_id, trace_id)
            if row is None:
                raise CheckpointIntegrityError("Redis checkpoint registry references a missing record")
            rows.append(row)
        return rows

    def _latest_row(self, tenant_id: str, session_id: str) -> Mapping[str, Any] | None:
        index_key, _, _ = self._session_keys(tenant_id, session_id)
        for sort_key in self._client.zrevrange(index_key, 0, -1):
            try:
                encoded_trace = str(sort_key).rsplit(":", 1)[1]
                trace_id = self._decode(encoded_trace)
            except (IndexError, ValueError, UnicodeDecodeError) as exc:
                raise CheckpointIntegrityError("Redis checkpoint index contains an invalid member") from exc
            row = self._get_record(tenant_id, session_id, trace_id)
            if row is not None and str(row["sort_key"]) == str(sort_key):
                return row
        rows = self._iter_rows(tenant_id, session_id)
        if not rows:
            return None
        return max(rows, key=lambda row: str(row["sort_key"]))

    def _verify_prev_link(self, row: Mapping[str, Any]) -> None:
        previous_hash = str(row["prev_checkpoint_hash"] or "").strip()
        if not previous_hash:
            return
        rows = self._iter_rows(str(row["tenant_id"]), str(row["session_id"]))
        prior_rows = [
            candidate
            for candidate in rows
            if str(candidate["sort_key"]) < str(row["sort_key"])
        ]
        prior = max(prior_rows, key=lambda candidate: str(candidate["sort_key"])) if prior_rows else None
        if prior is not None:
            actual_hash = str(prior["checkpoint_hash"] or "").strip()
            if actual_hash != previous_hash:
                raise CheckpointIntegrityError(
                    f"Checkpoint hash-chain broken: expected prev={actual_hash!r}, got {previous_hash!r}",
                )
            return
        _, _, writes_key = self._session_keys(str(row["tenant_id"]), str(row["session_id"]))
        for raw in self._client.lrange(writes_key, 0, -1):
            try:
                write = json.loads(str(raw))
            except json.JSONDecodeError as exc:
                raise CheckpointIntegrityError("Redis checkpoint write log is malformed") from exc
            if write.get("status") == "ok" and str(write.get("checkpoint_hash") or "") == previous_hash:
                return
        raise CheckpointIntegrityError(
            "Checkpoint hash-chain broken: prev_checkpoint_hash has no prior checkpoint or write evidence",
        )

    def save_checkpoint(
        self,
        session_id: str,
        trace_id: str,
        checkpoint: dict[str, Any],
        manifest: dict[str, Any],
        *,
        tenant_id: str | None = None,
    ) -> bool:
        resolved_tenant = self._resolve_tenant(tenant_id)
        session_id, normalized_trace = self._validate_ids(session_id, trace_id)
        assert normalized_trace is not None
        checkpoint_hash, previous_hash, manifest_hash, payload_json = self._prepare(
            resolved_tenant,
            session_id,
            normalized_trace,
            checkpoint,
            manifest,
        )
        now = time.time()
        encoded_trace = self._encode(normalized_trace)
        sort_key = self._sort_key(time.time_ns(), encoded_trace)
        record = {
            "tenant_id": resolved_tenant,
            "session_id": session_id,
            "trace_id": normalized_trace,
            "checkpoint_hash": checkpoint_hash,
            "prev_checkpoint_hash": previous_hash,
            "manifest_hash": manifest_hash,
            "payload": json.loads(payload_json),
            "created_at": now,
            "sort_key": sort_key,
        }
        index_key, registry_key, writes_key = self._session_keys(resolved_tenant, session_id)
        write_log = json.dumps(
            {
                "trace_id": normalized_trace,
                "checkpoint_hash": checkpoint_hash,
                "status": "ok",
                "error": "",
                "created_at": now,
            },
            sort_keys=True,
        )
        try:
            stored_hash = self._save_script(
                keys=[
                    self._checkpoint_key(resolved_tenant, session_id, encoded_trace),
                    index_key,
                    registry_key,
                    writes_key,
                ],
                args=[
                    json.dumps(record, sort_keys=True),
                    sort_key,
                    encoded_trace,
                    write_log,
                    checkpoint_hash,
                ],
            )
        except self._redis_errors as exc:
            self._log_write_failure(resolved_tenant, session_id, normalized_trace, str(exc))
            raise CheckpointError(f"Failed to save Redis checkpoint: {exc}") from exc
        if str(stored_hash or "").strip() != checkpoint_hash:
            self._log_write_failure(
                resolved_tenant,
                session_id,
                normalized_trace,
                "post-write consistency check failed",
            )
            raise CheckpointIntegrityError("Redis checkpoint post-write verification failed")
        return True

    def _log_write_failure(
        self,
        tenant_id: str,
        session_id: str,
        trace_id: str,
        error: str,
    ) -> None:
        _, _, writes_key = self._session_keys(tenant_id, session_id)
        try:
            self._client.rpush(
                writes_key,
                json.dumps(
                    {
                        "trace_id": trace_id,
                        "checkpoint_hash": "",
                        "status": "error",
                        "error": str(error),
                        "created_at": time.time(),
                    },
                    sort_keys=True,
                ),
            )
        except self._redis_errors:
            logger.exception("Could not record Redis checkpoint write failure")

    def load_checkpoint(
        self,
        session_id: str,
        trace_id: str | None = None,
        *,
        current_manifest: dict[str, Any] | None = None,
        strict: bool = False,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        resolved_tenant = self._resolve_tenant(tenant_id)
        session_id, normalized_trace = self._validate_ids(session_id, trace_id)
        try:
            row = (
                self._get_record(resolved_tenant, session_id, normalized_trace)
                if normalized_trace
                else self._latest_row(resolved_tenant, session_id)
            )
            if row is None:
                raise CheckpointNotFoundError(
                    f"No checkpoint found for session_id={session_id!r}, trace_id={normalized_trace!r}",
                )
            result = self._loaded_result(
                row,
                current_manifest=current_manifest,
                strict=strict,
            )
            self._verify_prev_link(row)
            return result
        except (CheckpointError, self._redis_errors):
            raise
        except (ValueError, TypeError, KeyError) as exc:
            raise CheckpointError(f"Redis checkpoint load failed: {exc}") from exc

    def prune_old_checkpoints(
        self,
        session_id: str,
        keep_count: int = 10,
        older_than_days: int | None = None,
        *,
        tenant_id: str | None = None,
    ) -> int:
        resolved_tenant = self._resolve_tenant(tenant_id)
        session_id, _ = self._validate_ids(session_id)
        keep_count = max(int(keep_count or 0), 0)
        cutoff = (
            time.time() - int(older_than_days) * 86400
            if older_than_days is not None and int(older_than_days) >= 0
            else None
        )
        rows = sorted(
            self._iter_rows(resolved_tenant, session_id),
            key=lambda row: str(row["sort_key"]),
            reverse=True,
        )
        removal = [
            row
            for index, row in enumerate(rows)
            if (keep_count > 0 and index >= keep_count)
            or (cutoff is not None and float(row["created_at"]) < cutoff)
        ]
        if not removal:
            return 0
        index_key, registry_key, _ = self._session_keys(resolved_tenant, session_id)
        prefix = f"{self._root(resolved_tenant, session_id)}:checkpoint:"
        args: list[str] = [prefix]
        for row in removal:
            args.extend(
                [
                    self._encode(str(row["trace_id"])),
                    str(row["sort_key"]),
                ],
            )
        try:
            return int(self._prune_script(keys=[index_key, registry_key], args=args))
        except self._redis_errors as exc:
            raise CheckpointError(f"Redis checkpoint prune failed: {exc}") from exc

    def prune_checkpoint_write_logs(
        self,
        session_id: str,
        keep_count: int = 200,
        older_than_days: int | None = None,
        *,
        tenant_id: str | None = None,
    ) -> int:
        resolved_tenant = self._resolve_tenant(tenant_id)
        session_id, _ = self._validate_ids(session_id)
        _, _, writes_key = self._session_keys(resolved_tenant, session_id)
        cutoff = (
            time.time() - int(older_than_days) * 86400
            if older_than_days is not None and int(older_than_days) >= 0
            else 0
        )
        try:
            return int(
                self._prune_log_script(
                    keys=[writes_key],
                    args=[max(int(keep_count or 0), 0), cutoff],
                ),
            )
        except self._redis_errors as exc:
            raise CheckpointError(f"Redis checkpoint write-log prune failed: {exc}") from exc

    def delete_session(self, session_id: str, *, tenant_id: str | None = None) -> int:
        resolved_tenant = self._resolve_tenant(tenant_id)
        session_id, _ = self._validate_ids(session_id)
        index_key, registry_key, writes_key = self._session_keys(resolved_tenant, session_id)
        prefix = f"{self._root(resolved_tenant, session_id)}:checkpoint:"
        try:
            return int(
                self._delete_script(
                    keys=[index_key, registry_key, writes_key],
                    args=[prefix],
                ),
            )
        except self._redis_errors as exc:
            raise CheckpointError(f"Redis checkpoint deletion failed: {exc}") from exc

    def checkpoint_count(self, session_id: str, *, tenant_id: str | None = None) -> int:
        resolved_tenant = self._resolve_tenant(tenant_id)
        session_id, _ = self._validate_ids(session_id)
        _, registry_key, _ = self._session_keys(resolved_tenant, session_id)
        try:
            return int(self._client.scard(registry_key))
        except self._redis_errors as exc:
            raise CheckpointError(f"Redis checkpoint count failed: {exc}") from exc

    def validate_consistency(
        self,
        session_id: str,
        *,
        tenant_id: str | None = None,
        strict: bool = False,
    ) -> dict[str, Any]:
        resolved_tenant = self._resolve_tenant(tenant_id)
        session_id, _ = self._validate_ids(session_id)
        issues: list[str] = []
        try:
            rows = self._iter_rows(resolved_tenant, session_id)
            for row in rows:
                try:
                    loaded = decode_loaded_checkpoint(row)
                    verify_checkpoint_row(row, loaded.checkpoint)
                    self._verify_prev_link(row)
                except CheckpointError as exc:
                    issues.append(str(exc))
            indexed = {
                str(member)
                for member in self._client.zrange(self._session_keys(resolved_tenant, session_id)[0], 0, -1)
            }
            registered = {str(row["sort_key"]) for row in rows}
            if indexed != registered:
                issues.append("checkpoint_index_mismatch")
        except self._redis_errors as exc:
            issues.append(f"redis_error:{exc}")
        result = {"ok": not issues, "issues": issues}
        if strict and issues:
            raise CheckpointIntegrityError(f"Redis checkpoint consistency check failed: {issues}")
        return result


class RedisPostgresCheckpointer(AbstractCheckpointer):
    """PostgreSQL-authoritative checkpointer with Redis used only as a cache."""

    def __init__(
        self,
        durable: PostgresCheckpointer,
        cache: RedisCheckpointer,
    ) -> None:
        self.durable = durable
        self.cache = cache
        self.last_cache_error: str | None = None

    def migrate(self) -> None:
        self.durable.migrate()
        self.cache.migrate()

    def _refresh_cache(
        self,
        session_id: str,
        trace_id: str,
        checkpoint: dict[str, Any],
        manifest: dict[str, Any],
        tenant_id: str | None,
    ) -> None:
        try:
            self.cache.save_checkpoint(
                session_id,
                trace_id,
                checkpoint,
                manifest,
                tenant_id=tenant_id,
            )
            self.last_cache_error = None
        except CheckpointError as exc:
            self.last_cache_error = str(exc)
            logger.error("Redis checkpoint cache update failed after durable PostgreSQL write: %s", exc)

    def save_checkpoint(
        self,
        session_id: str,
        trace_id: str,
        checkpoint: dict[str, Any],
        manifest: dict[str, Any],
        *,
        tenant_id: str | None = None,
    ) -> bool:
        saved = self.durable.save_checkpoint(
            session_id,
            trace_id,
            checkpoint,
            manifest,
            tenant_id=tenant_id,
        )
        if saved:
            self._refresh_cache(session_id, trace_id, checkpoint, manifest, tenant_id)
        return saved

    def load_checkpoint(
        self,
        session_id: str,
        trace_id: str | None = None,
        *,
        current_manifest: dict[str, Any] | None = None,
        strict: bool = False,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        loaded = self.durable.load_checkpoint(
            session_id,
            trace_id,
            current_manifest=current_manifest,
            strict=strict,
            tenant_id=tenant_id,
        )
        checkpoint_trace_id = str(dict(loaded.get("metadata") or {}).get("trace_id") or "")
        try:
            cached = self.cache.load_checkpoint(
                session_id,
                checkpoint_trace_id,
                tenant_id=tenant_id,
            )
            if str(cached.get("checkpoint_hash") or "") != str(loaded.get("checkpoint_hash") or ""):
                self._refresh_cache(
                    session_id,
                    checkpoint_trace_id,
                    loaded,
                    dict(loaded.get("manifest") or {}),
                    tenant_id,
                )
        except CheckpointNotFoundError:
            self._refresh_cache(
                session_id,
                checkpoint_trace_id,
                loaded,
                dict(loaded.get("manifest") or {}),
                tenant_id,
            )
        except CheckpointError as exc:
            self.last_cache_error = str(exc)
            logger.error("Redis checkpoint cache read failed; using durable PostgreSQL record: %s", exc)
            self._refresh_cache(
                session_id,
                checkpoint_trace_id,
                loaded,
                dict(loaded.get("manifest") or {}),
                tenant_id,
            )
        return loaded

    def prune_old_checkpoints(
        self,
        session_id: str,
        keep_count: int = 10,
        older_than_days: int | None = None,
        *,
        tenant_id: str | None = None,
    ) -> int:
        deleted = self.durable.prune_old_checkpoints(
            session_id,
            keep_count,
            older_than_days,
            tenant_id=tenant_id,
        )
        try:
            self.cache.prune_old_checkpoints(
                session_id,
                keep_count,
                older_than_days,
                tenant_id=tenant_id,
            )
            self.last_cache_error = None
        except CheckpointError as exc:
            self.last_cache_error = str(exc)
            logger.error("Redis checkpoint cache prune failed after durable PostgreSQL prune: %s", exc)
        return deleted

    def delete_session(self, session_id: str, *, tenant_id: str | None = None) -> int:
        deleted = self.durable.delete_session(session_id, tenant_id=tenant_id)
        try:
            self.cache.delete_session(session_id, tenant_id=tenant_id)
            self.last_cache_error = None
        except CheckpointError as exc:
            self.last_cache_error = str(exc)
            logger.error("Redis checkpoint cache delete failed after durable PostgreSQL delete: %s", exc)
        return deleted

    def checkpoint_count(self, session_id: str, *, tenant_id: str | None = None) -> int:
        return self.durable.checkpoint_count(session_id, tenant_id=tenant_id)

    def prune_checkpoint_write_logs(
        self,
        session_id: str,
        keep_count: int = 200,
        older_than_days: int | None = None,
        *,
        tenant_id: str | None = None,
    ) -> int:
        deleted = self.durable.prune_checkpoint_write_logs(
            session_id,
            keep_count,
            older_than_days,
            tenant_id=tenant_id,
        )
        try:
            self.cache.prune_checkpoint_write_logs(
                session_id,
                keep_count,
                older_than_days,
                tenant_id=tenant_id,
            )
            self.last_cache_error = None
        except CheckpointError as exc:
            self.last_cache_error = str(exc)
            logger.error("Redis checkpoint cache log prune failed after durable PostgreSQL prune: %s", exc)
        return deleted

    def validate_consistency(
        self,
        session_id: str,
        *,
        tenant_id: str | None = None,
        strict: bool = False,
    ) -> dict[str, Any]:
        return self.durable.validate_consistency(
            session_id,
            tenant_id=tenant_id,
            strict=strict,
        )
