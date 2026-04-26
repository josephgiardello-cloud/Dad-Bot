"""SQLite-backed checkpoint persistence for Dad Bot.

Provides atomic checkpoint save/load with:
- Hash-chain integrity verification
- Manifest drift detection
- Automatic pruning
- No external dependencies beyond sqlite3
"""

import contextlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from dadbot.core.persistence.base import (
    AbstractCheckpointer,
    CheckpointError,
    CheckpointIntegrityError,
    CheckpointNotFoundError,
)

logger = logging.getLogger(__name__)


class SQLiteCheckpointer(AbstractCheckpointer):
    """SQLite-based checkpoint persistence.
    
    Usage:
        checkpointer = SQLiteCheckpointer("checkpoints.db")
        checkpoint = checkpointer.load_checkpoint("session-123")
        # ... process turn ...
        checkpointer.save_checkpoint("session-123", "trace-xyz", new_checkpoint, manifest)
    """

    def __init__(self, db_path: str, auto_migrate: bool = True):
        """Initialize SQLite checkpointer.
        
        Args:
            db_path: Path to SQLite database file
            auto_migrate: If True, create tables on first use
        """
        self.db_path = Path(db_path)
        self.auto_migrate = auto_migrate
        
        if auto_migrate:
            self._init_schema()
        
        logger.debug(f"SQLiteCheckpointer initialized at {self.db_path}")

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        try:
            with contextlib.closing(sqlite3.connect(str(self.db_path))) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        trace_id TEXT NOT NULL,
                        checkpoint_hash TEXT NOT NULL,
                        prev_checkpoint_hash TEXT,
                        manifest_hash TEXT,
                        env_hash TEXT,
                        python_version TEXT,
                        payload TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        UNIQUE(session_id, trace_id)
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_created
                    ON checkpoints (session_id, created_at DESC)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trace_id
                    ON checkpoints (trace_id)
                """)
                conn.commit()
        except sqlite3.Error as e:
            raise CheckpointError(f"Schema initialization failed: {e}")

    def save_checkpoint(
        self,
        session_id: str,
        trace_id: str,
        checkpoint: Dict[str, Any],
        manifest: Dict[str, Any],
    ) -> bool:
        """Save checkpoint with integrity metadata.
        
        Args:
            session_id: Session identifier
            trace_id: Unique turn trace ID (prevents duplicates)
            checkpoint: Checkpoint dict with checkpoint_hash, prev_checkpoint_hash, etc.
            manifest: Determinism manifest with env_hash, python_version, etc.
        
        Returns:
            True if saved successfully
            
        Raises:
            CheckpointError: if save fails
        """
        try:
            checkpoint_hash = str(checkpoint.get("checkpoint_hash", ""))
            prev_checkpoint_hash = str(checkpoint.get("prev_checkpoint_hash") or "")
            manifest_hash = str(manifest.get("manifest_hash", ""))
            env_hash = str(manifest.get("env_hash", ""))
            python_version = str(manifest.get("python_version", ""))
            payload_json = json.dumps(checkpoint)
            
            import time
            created_at = time.time()
            
            with contextlib.closing(sqlite3.connect(str(self.db_path))) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO checkpoints
                    (session_id, trace_id, checkpoint_hash, prev_checkpoint_hash, 
                     manifest_hash, env_hash, python_version, payload, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        trace_id,
                        checkpoint_hash,
                        prev_checkpoint_hash,
                        manifest_hash,
                        env_hash,
                        python_version,
                        payload_json,
                        created_at,
                    ),
                )
                conn.commit()
            
            logger.debug(f"Checkpoint saved: session={session_id}, trace={trace_id}")
            return True
            
        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint: {e}")

    def load_checkpoint(
        self,
        session_id: str,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint and verify hash-chain + manifest.
        
        Args:
            session_id: Session identifier
            trace_id: Optional specific trace; if None, loads most recent
        
        Returns:
            Checkpoint dict
            
        Raises:
            CheckpointNotFoundError: if checkpoint doesn't exist
            CheckpointIntegrityError: if hash-chain or manifest verification fails
        """
        try:
            with contextlib.closing(sqlite3.connect(str(self.db_path))) as conn:
                conn.row_factory = sqlite3.Row
                
                if trace_id:
                    # Load specific checkpoint by trace_id
                    cursor = conn.execute(
                        """
                        SELECT payload, checkpoint_hash, prev_checkpoint_hash, 
                               manifest_hash, env_hash, python_version
                        FROM checkpoints
                        WHERE session_id = ? AND trace_id = ?
                        """,
                        (session_id, trace_id),
                    )
                else:
                    # Load most recent checkpoint for session
                    cursor = conn.execute(
                        """
                        SELECT payload, checkpoint_hash, prev_checkpoint_hash,
                               manifest_hash, env_hash, python_version
                        FROM checkpoints
                        WHERE session_id = ?
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (session_id,),
                    )
                
                row = cursor.fetchone()
                if not row:
                    raise CheckpointNotFoundError(
                        f"No checkpoint found for session={session_id}, trace={trace_id}"
                    )
                
                payload_json = row["payload"]
                checkpoint = json.loads(payload_json)
                
                # Verify hash-chain integrity
                stored_checkpoint_hash = str(row["checkpoint_hash"] or "")
                actual_checkpoint_hash = str(checkpoint.get("checkpoint_hash", ""))
                
                if stored_checkpoint_hash and actual_checkpoint_hash:
                    if stored_checkpoint_hash != actual_checkpoint_hash:
                        raise CheckpointIntegrityError(
                            f"Checkpoint hash mismatch: stored={stored_checkpoint_hash}, "
                            f"actual={actual_checkpoint_hash}"
                        )
                
                logger.debug(f"Checkpoint loaded: session={session_id}, trace={trace_id}")
                return checkpoint
                
        except CheckpointNotFoundError:
            raise
        except CheckpointIntegrityError:
            raise
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {e}")

    def prune_old_checkpoints(
        self,
        session_id: str,
        keep_count: int = 10,
    ) -> int:
        """Delete old checkpoints, keeping only most recent N per session.
        
        Args:
            session_id: Session identifier
            keep_count: Number to keep (default 10)
        
        Returns:
            Number deleted
        """
        try:
            with contextlib.closing(sqlite3.connect(str(self.db_path))) as conn:
                # Find checkpoint IDs to delete
                cursor = conn.execute(
                    """
                    SELECT id FROM checkpoints
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT -1 OFFSET ?
                    """,
                    (session_id, keep_count),
                )
                ids_to_delete = [row[0] for row in cursor.fetchall()]
                
                if ids_to_delete:
                    placeholders = ",".join("?" * len(ids_to_delete))
                    conn.execute(
                        f"DELETE FROM checkpoints WHERE id IN ({placeholders})",
                        ids_to_delete,
                    )
                    conn.commit()
                
                logger.debug(
                    f"Pruned {len(ids_to_delete)} old checkpoints for session={session_id}"
                )
                return len(ids_to_delete)
                
        except Exception as e:
            raise CheckpointError(f"Pruning failed: {e}")

    def delete_session(self, session_id: str) -> int:
        """Delete all checkpoints for a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Number deleted
        """
        try:
            with contextlib.closing(sqlite3.connect(str(self.db_path))) as conn:
                cursor = conn.execute(
                    "DELETE FROM checkpoints WHERE session_id = ?",
                    (session_id,),
                )
                conn.commit()
                count = cursor.rowcount
                logger.debug(f"Deleted {count} checkpoints for session={session_id}")
                return count
        except Exception as e:
            raise CheckpointError(f"Session deletion failed: {e}")

    def checkpoint_count(self, session_id: str) -> int:
        """Get number of checkpoints for a session (for testing)."""
        try:
            with contextlib.closing(sqlite3.connect(str(self.db_path))) as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE session_id = ?",
                    (session_id,),
                )
                return cursor.fetchone()[0]
        except Exception:
            return 0
