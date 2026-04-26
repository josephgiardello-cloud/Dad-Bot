"""Dedicated checkpoint saver backends for deterministic resume semantics.

This module provides:
- SQLiteCheckpointer: sync saver with schema migration, hash-chain checks,
  manifest drift checks, and pruning.
- AsyncSQLiteCheckpointer: async facade (thread offload by default, optional
  aiosqlite path when installed).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from dadbot.core.persistence.base import (
    AbstractAsyncCheckpointer,
    AbstractCheckpointer,
    CheckpointError,
    CheckpointIntegrityError,
    CheckpointNotFoundError,
)

logger = logging.getLogger(__name__)


def _stable_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class LoadedCheckpoint:
    checkpoint: dict[str, Any]
    manifest: dict[str, Any]


class SQLiteCheckpointer(AbstractCheckpointer):
    """SQLite-backed durable checkpointer.

    Composite boundary key: (session_id, trace_id)
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        db_path: str,
        *,
        auto_migrate: bool = True,
        prune_every: int = 10,
        default_keep_count: int = 10,
    ):
        self.db_path = Path(db_path)
        self.prune_every = int(prune_every or 0)
        self.default_keep_count = int(default_keep_count or 10)
        self._save_counter = 0
        if auto_migrate:
            self.migrate()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def migrate(self) -> None:
        try:
            with contextlib.closing(self._connect()) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS checkpoint_meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        trace_id TEXT NOT NULL,
                        checkpoint_hash TEXT NOT NULL,
                        prev_checkpoint_hash TEXT,
                        manifest_hash TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        UNIQUE(session_id, trace_id)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS checkpoint_writes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        trace_id TEXT NOT NULL,
                        checkpoint_hash TEXT NOT NULL,
                        status TEXT NOT NULL,
                        error TEXT,
                        created_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_checkpoints_session_created ON checkpoints(session_id, created_at DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_session_created ON checkpoint_writes(session_id, created_at DESC)"
                )
                conn.execute(
                    "INSERT OR REPLACE INTO checkpoint_meta(key, value) VALUES ('schema_version', ?)",
                    (str(self.SCHEMA_VERSION),),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise CheckpointError(f"Checkpoint schema migration failed: {exc}") from exc

    def save_checkpoint(
        self,
        session_id: str,
        trace_id: str,
        checkpoint: Dict[str, Any],
        manifest: Dict[str, Any],
    ) -> bool:
        session_id = str(session_id or "").strip()
        trace_id = str(trace_id or "").strip()
        if not session_id or not trace_id:
            raise CheckpointError("session_id and trace_id are required")

        checkpoint_hash = str(checkpoint.get("checkpoint_hash") or "").strip()
        prev_checkpoint_hash = str(checkpoint.get("prev_checkpoint_hash") or "").strip()
        if not checkpoint_hash:
            raise CheckpointError("checkpoint.checkpoint_hash is required")

        manifest_payload = dict(manifest or {})
        manifest_hash = str(manifest_payload.get("manifest_hash") or _stable_sha256(manifest_payload))
        payload = {
            "checkpoint": dict(checkpoint or {}),
            "manifest": manifest_payload,
        }
        payload_json = json.dumps(payload, sort_keys=True, default=str)
        now_ts = time.time()

        try:
            with contextlib.closing(self._connect()) as conn:
                conn.execute("BEGIN")
                conn.execute(
                    """
                    INSERT OR REPLACE INTO checkpoints(
                        session_id, trace_id, checkpoint_hash, prev_checkpoint_hash,
                        manifest_hash, payload, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        trace_id,
                        checkpoint_hash,
                        prev_checkpoint_hash,
                        manifest_hash,
                        payload_json,
                        now_ts,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO checkpoint_writes(
                        session_id, trace_id, checkpoint_hash, status, error, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        trace_id,
                        checkpoint_hash,
                        "ok",
                        "",
                        now_ts,
                    ),
                )
                conn.commit()
        except sqlite3.Error as exc:
            self._log_write_failure(session_id=session_id, trace_id=trace_id, error=str(exc))
            raise CheckpointError(f"Failed to save checkpoint: {exc}") from exc

        self._save_counter += 1
        if self.prune_every > 0 and self._save_counter % self.prune_every == 0:
            try:
                self.prune_old_checkpoints(session_id, keep_count=self.default_keep_count)
            except Exception as exc:
                logger.warning("Checkpoint prune failed (non-fatal): %s", exc)
        return True

    def _log_write_failure(self, *, session_id: str, trace_id: str, error: str) -> None:
        try:
            with contextlib.closing(self._connect()) as conn:
                conn.execute(
                    """
                    INSERT INTO checkpoint_writes(session_id, trace_id, checkpoint_hash, status, error, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, trace_id, "", "error", str(error or ""), time.time()),
                )
                conn.commit()
        except Exception:
            logger.debug("Checkpoint write failure logging skipped", exc_info=True)

    def load_checkpoint(
        self,
        session_id: str,
        trace_id: Optional[str] = None,
        *,
        current_manifest: Optional[Dict[str, Any]] = None,
        strict: bool = False,
    ) -> Dict[str, Any]:
        session_id = str(session_id or "").strip()
        if not session_id:
            raise CheckpointNotFoundError("session_id is required")

        try:
            with contextlib.closing(self._connect()) as conn:
                if trace_id:
                    cursor = conn.execute(
                        """
                        SELECT * FROM checkpoints
                        WHERE session_id = ? AND trace_id = ?
                        LIMIT 1
                        """,
                        (session_id, str(trace_id)),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT * FROM checkpoints
                        WHERE session_id = ?
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (session_id,),
                    )
                row = cursor.fetchone()
                if row is None:
                    raise CheckpointNotFoundError(
                        f"No checkpoint found for session_id={session_id!r}, trace_id={trace_id!r}"
                    )

                payload = json.loads(str(row["payload"] or "{}"))
                loaded = LoadedCheckpoint(
                    checkpoint=dict(payload.get("checkpoint") or {}),
                    manifest=dict(payload.get("manifest") or {}),
                )

                self._verify_checkpoint_row(row, loaded.checkpoint)
                self._verify_prev_link(conn, row)
                self._verify_manifest(loaded.manifest, current_manifest, strict=strict)

                result = dict(loaded.checkpoint)
                result["manifest"] = dict(loaded.manifest)
                result["manifest_hash"] = str(row["manifest_hash"] or "")
                return result
        except CheckpointNotFoundError:
            raise
        except CheckpointIntegrityError:
            raise
        except sqlite3.Error as exc:
            raise CheckpointError(f"Checkpoint load failed: {exc}") from exc
        except Exception as exc:
            raise CheckpointError(f"Checkpoint load failed: {exc}") from exc

    def _verify_checkpoint_row(self, row: sqlite3.Row, checkpoint: dict[str, Any]) -> None:
        stored_hash = str(row["checkpoint_hash"] or "").strip()
        payload_hash = str(checkpoint.get("checkpoint_hash") or "").strip()
        if stored_hash != payload_hash:
            raise CheckpointIntegrityError(
                f"Checkpoint hash mismatch: stored={stored_hash!r} payload={payload_hash!r}"
            )

        stored_prev = str(row["prev_checkpoint_hash"] or "").strip()
        payload_prev = str(checkpoint.get("prev_checkpoint_hash") or "").strip()
        if stored_prev != payload_prev:
            raise CheckpointIntegrityError(
                f"Prev hash mismatch: stored={stored_prev!r} payload={payload_prev!r}"
            )

    def _verify_prev_link(self, conn: sqlite3.Connection, row: sqlite3.Row) -> None:
        prev_hash = str(row["prev_checkpoint_hash"] or "").strip()
        if not prev_hash:
            return
        prior = conn.execute(
            """
            SELECT checkpoint_hash FROM checkpoints
            WHERE session_id = ? AND created_at < ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (str(row["session_id"]), float(row["created_at"])),
        ).fetchone()
        if prior is None:
            raise CheckpointIntegrityError(
                "Checkpoint hash-chain broken: prev_checkpoint_hash present but no previous checkpoint exists"
            )
        prior_hash = str(prior["checkpoint_hash"] or "").strip()
        if prior_hash != prev_hash:
            raise CheckpointIntegrityError(
                f"Checkpoint hash-chain broken: expected prev={prior_hash!r}, got={prev_hash!r}"
            )

    def _verify_manifest(
        self,
        stored_manifest: dict[str, Any],
        current_manifest: Optional[Dict[str, Any]],
        *,
        strict: bool,
    ) -> None:
        if not current_manifest:
            return
        current_manifest = dict(current_manifest)
        stored_env_hash = str(stored_manifest.get("env_hash") or "")
        current_env_hash = str(current_manifest.get("env_hash") or "")
        stored_python = str(stored_manifest.get("python_version") or "")
        current_python = str(current_manifest.get("python_version") or "")

        env_changed = stored_env_hash and current_env_hash and stored_env_hash != current_env_hash
        py_changed = stored_python and current_python and stored_python != current_python

        if env_changed or py_changed:
            message = (
                "Manifest drift detected on checkpoint load: "
                f"env_hash {stored_env_hash!r}->{current_env_hash!r}, "
                f"python_version {stored_python!r}->{current_python!r}"
            )
            if strict:
                raise CheckpointIntegrityError(message)
            logger.warning(message)

    def prune_old_checkpoints(
        self,
        session_id: str,
        keep_count: int = 10,
        older_than_days: int | None = None,
    ) -> int:
        session_id = str(session_id or "").strip()
        keep_count = max(int(keep_count or 0), 0)
        deleted = 0
        try:
            with contextlib.closing(self._connect()) as conn:
                ids_to_delete: list[int] = []
                if keep_count > 0:
                    rows = conn.execute(
                        """
                        SELECT id FROM checkpoints
                        WHERE session_id = ?
                        ORDER BY created_at DESC
                        LIMIT -1 OFFSET ?
                        """,
                        (session_id, keep_count),
                    ).fetchall()
                    ids_to_delete.extend([int(r["id"]) for r in rows])

                if older_than_days is not None and int(older_than_days) >= 0:
                    cutoff = time.time() - (int(older_than_days) * 86400)
                    rows = conn.execute(
                        """
                        SELECT id FROM checkpoints
                        WHERE session_id = ? AND created_at < ?
                        """,
                        (session_id, float(cutoff)),
                    ).fetchall()
                    ids_to_delete.extend([int(r["id"]) for r in rows])

                ids_to_delete = sorted(set(ids_to_delete))
                if ids_to_delete:
                    placeholders = ",".join("?" for _ in ids_to_delete)
                    conn.execute(
                        f"DELETE FROM checkpoints WHERE id IN ({placeholders})",
                        tuple(ids_to_delete),
                    )
                    conn.commit()
                    deleted = len(ids_to_delete)
            return deleted
        except sqlite3.Error as exc:
            raise CheckpointError(f"Failed to prune checkpoints: {exc}") from exc

    def delete_session(self, session_id: str) -> int:
        session_id = str(session_id or "").strip()
        try:
            with contextlib.closing(self._connect()) as conn:
                cursor = conn.execute("DELETE FROM checkpoints WHERE session_id = ?", (session_id,))
                deleted = int(cursor.rowcount or 0)
                conn.commit()
                return deleted
        except sqlite3.Error as exc:
            raise CheckpointError(f"Failed to delete session checkpoints: {exc}") from exc

    def checkpoint_count(self, session_id: str) -> int:
        session_id = str(session_id or "").strip()
        try:
            with contextlib.closing(self._connect()) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM checkpoints WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                return int((row["n"] if row is not None else 0) or 0)
        except sqlite3.Error:
            return 0


class AsyncSQLiteCheckpointer(AbstractAsyncCheckpointer):
    """Async checkpointer facade.

    Connection pooling stub is intentionally lightweight for Phase 4. It can be
    upgraded to real pooled aiosqlite connections in Phase 5.
    """

    def __init__(self, sync_checkpointer: SQLiteCheckpointer):
        self._sync = sync_checkpointer
        self._pool_enabled = False

    @classmethod
    def from_path(cls, db_path: str, *, auto_migrate: bool = True) -> "AsyncSQLiteCheckpointer":
        return cls(SQLiteCheckpointer(db_path, auto_migrate=auto_migrate))

    async def migrate(self) -> None:
        await asyncio.to_thread(self._sync.migrate)

    async def save_checkpoint(
        self,
        session_id: str,
        trace_id: str,
        checkpoint: Dict[str, Any],
        manifest: Dict[str, Any],
    ) -> bool:
        return await asyncio.to_thread(
            self._sync.save_checkpoint,
            session_id,
            trace_id,
            checkpoint,
            manifest,
        )

    async def load_checkpoint(
        self,
        session_id: str,
        trace_id: Optional[str] = None,
        *,
        current_manifest: Optional[Dict[str, Any]] = None,
        strict: bool = False,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self._sync.load_checkpoint,
            session_id,
            trace_id,
            current_manifest=current_manifest,
            strict=strict,
        )

    async def prune_old_checkpoints(
        self,
        session_id: str,
        keep_count: int = 10,
        older_than_days: int | None = None,
    ) -> int:
        return await asyncio.to_thread(
            self._sync.prune_old_checkpoints,
            session_id,
            keep_count,
            older_than_days,
        )
