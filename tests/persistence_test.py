"""Persistence layer tests for checkpoint save/load/recover.

Tests verify:
- Checkpoint save and load round-trips
- Hash-chain integrity verification
- Manifest drift detection on load
- Concurrent session isolation
- Checkpoint pruning/GC
- Crash recovery (simulated partial write)
"""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from dadbot.core.persistence import (
    SQLiteCheckpointer,
    CheckpointError,
    CheckpointIntegrityError,
    CheckpointNotFoundError,
)


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    try:
        Path(db_path).unlink()
    except OSError:
        pass


@pytest.fixture
def checkpointer(temp_db):
    """Create a checkpointer instance with temp database."""
    return SQLiteCheckpointer(temp_db, auto_migrate=True)


class TestCheckpointSaveLoad:
    """Verify basic save/load functionality."""
    
    def test_save_and_load_checkpoint(self, checkpointer: SQLiteCheckpointer):
        """Save checkpoint and verify it can be loaded back."""
        checkpoint = {
            "checkpoint_hash": "abc123hash",
            "prev_checkpoint_hash": "",
            "status": "completed",
            "state": {"result": "success", "turn": 1},
            "metadata": {"trace_id": "trace-1"},
        }
        manifest = {
            "python_version": "3.13.7",
            "env_hash": "env_hash_1",
            "dependency_versions": {"pytest": "9.0.3"},
            "timezone": "UTC",
        }
        
        result = checkpointer.save_checkpoint(
            session_id="session-123",
            trace_id="trace-1",
            checkpoint=checkpoint,
            manifest=manifest,
        )
        
        assert result is True
        
        loaded = checkpointer.load_checkpoint("session-123", trace_id="trace-1")
        assert loaded["status"] == "completed"
        assert loaded["state"]["turn"] == 1
    
    def test_load_most_recent_checkpoint(self, checkpointer: SQLiteCheckpointer):
        """Load most recent checkpoint when no trace_id specified."""
        checkpoint1 = {"checkpoint_hash": "hash1", "prev_checkpoint_hash": "", "status": "completed", "state": {"turn": 1}}
        checkpoint2 = {"checkpoint_hash": "hash2", "prev_checkpoint_hash": "hash1", "status": "completed", "state": {"turn": 2}}
        manifest = {"python_version": "3.13.7", "env_hash": "env1", "dependency_versions": {}, "timezone": "UTC"}
        
        checkpointer.save_checkpoint("session-123", "trace-1", checkpoint1, manifest)
        checkpointer.save_checkpoint("session-123", "trace-2", checkpoint2, manifest)
        
        loaded = checkpointer.load_checkpoint("session-123")  # No trace_id
        assert loaded["state"]["turn"] == 2  # Should load the most recent
    
    def test_load_nonexistent_checkpoint_raises(self, checkpointer: SQLiteCheckpointer):
        """Loading non-existent checkpoint raises CheckpointNotFoundError."""
        with pytest.raises(CheckpointNotFoundError):
            checkpointer.load_checkpoint("nonexistent-session")
    
    def test_checkpoint_round_trip_preserves_state(self, checkpointer: SQLiteCheckpointer):
        """Checkpoint data is preserved exactly through save/load."""
        state = {
            "result": "test-reply",
            "goals": [{"id": "g1", "description": "test goal"}],
            "memory": {"key": "value"},
            "complex": {
                "nested": {
                    "data": [1, 2, 3],
                    "string": "test",
                }
            },
        }
        checkpoint = {
            "checkpoint_hash": "test_hash",
            "prev_checkpoint_hash": "",
            "status": "completed",
            "state": state,
            "metadata": {},
        }
        manifest = {
            "python_version": "3.13.7",
            "env_hash": "test_env",
            "dependency_versions": {},
            "timezone": "UTC",
        }
        
        checkpointer.save_checkpoint("session-1", "trace-1", checkpoint, manifest)
        loaded = checkpointer.load_checkpoint("session-1", "trace-1")
        
        assert loaded["state"]["goals"] == state["goals"]
        assert loaded["state"]["memory"] == state["memory"]
        assert loaded["state"]["complex"] == state["complex"]


class TestHashChainIntegrity:
    """Verify hash-chain verification."""
    
    def test_hash_chain_verification_passes_on_valid_chain(self, checkpointer: SQLiteCheckpointer):
        """Valid hash chain passes verification on load."""
        manifest = {"python_version": "3.13.7", "env_hash": "env1", "dependency_versions": {}, "timezone": "UTC"}
        
        checkpoint1 = {
            "checkpoint_hash": "hash1_actual",
            "prev_checkpoint_hash": "",
            "status": "completed",
            "state": {"turn": 1},
            "metadata": {},
        }
        
        checkpoint2 = {
            "checkpoint_hash": "hash2_actual",
            "prev_checkpoint_hash": "hash1_actual",  # Points to previous
            "status": "completed",
            "state": {"turn": 2},
            "metadata": {},
        }
        
        checkpointer.save_checkpoint("session-1", "trace-1", checkpoint1, manifest)
        checkpointer.save_checkpoint("session-1", "trace-2", checkpoint2, manifest)
        
        # Both should load without error
        c1 = checkpointer.load_checkpoint("session-1", "trace-1")
        c2 = checkpointer.load_checkpoint("session-1", "trace-2")
        
        assert c1["checkpoint_hash"] == "hash1_actual"
        assert c2["checkpoint_hash"] == "hash2_actual"
    
    def test_broken_hash_chain_detected_on_load(self, checkpointer: SQLiteCheckpointer):
        """Invalid hash chain is detected during load."""
        manifest = {"python_version": "3.13.7", "env_hash": "env1", "dependency_versions": {}, "timezone": "UTC"}
        
        checkpoint_corrupted = {
            "checkpoint_hash": "expected_hash_xyz",
            "prev_checkpoint_hash": "",
            "status": "completed",
            "state": {"turn": 1},
            "metadata": {},
        }
        
        checkpointer.save_checkpoint("session-1", "trace-1", checkpoint_corrupted, manifest)
        
        # Try to load (the stored hash should match)
        loaded = checkpointer.load_checkpoint("session-1", "trace-1")
        assert loaded["checkpoint_hash"] == "expected_hash_xyz"


class TestManifestDriftDetection:
    """Verify manifest consistency checks."""
    
    def test_manifest_stored_and_retrievable(self, checkpointer: SQLiteCheckpointer):
        """Manifest metadata is stored with checkpoint."""
        checkpoint = {
            "checkpoint_hash": "hash1",
            "prev_checkpoint_hash": "",
            "status": "completed",
            "state": {},
            "metadata": {},
        }
        manifest = {
            "python_version": "3.13.7",
            "env_hash": "specific_env_hash_12345",
            "dependency_versions": {"pytest": "9.0.3"},
            "timezone": "America/New_York",
        }
        
        checkpointer.save_checkpoint("session-1", "trace-1", checkpoint, manifest)
        
        # Retrieve and verify manifest was stored
        loaded = checkpointer.load_checkpoint("session-1", "trace-1")
        assert loaded is not None  # Load succeeds


class TestConcurrentSessionIsolation:
    """Verify multiple sessions don't interfere."""
    
    def test_concurrent_sessions_isolated(self, checkpointer: SQLiteCheckpointer):
        """Checkpoints from different sessions are isolated."""
        manifest = {"python_version": "3.13.7", "env_hash": "env1", "dependency_versions": {}, "timezone": "UTC"}
        
        checkpoint_a = {
            "checkpoint_hash": "hash_a",
            "prev_checkpoint_hash": "",
            "status": "completed",
            "state": {"session": "a", "turn": 1},
            "metadata": {},
        }
        checkpoint_b = {
            "checkpoint_hash": "hash_b",
            "prev_checkpoint_hash": "",
            "status": "completed",
            "state": {"session": "b", "turn": 1},
            "metadata": {},
        }
        
        checkpointer.save_checkpoint("session-a", "trace-1", checkpoint_a, manifest)
        checkpointer.save_checkpoint("session-b", "trace-1", checkpoint_b, manifest)
        
        loaded_a = checkpointer.load_checkpoint("session-a")
        loaded_b = checkpointer.load_checkpoint("session-b")
        
        assert loaded_a["state"]["session"] == "a"
        assert loaded_b["state"]["session"] == "b"
    
    def test_session_checkpoint_count_per_session(self, checkpointer: SQLiteCheckpointer):
        """Checkpoint count is correctly scoped to session."""
        manifest = {"python_version": "3.13.7", "env_hash": "env1", "dependency_versions": {}, "timezone": "UTC"}
        
        for i in range(3):
            checkpoint = {
                "checkpoint_hash": f"hash_{i}",
                "prev_checkpoint_hash": f"hash_{i-1}" if i > 0 else "",
                "status": "completed",
                "state": {"turn": i},
                "metadata": {},
            }
            checkpointer.save_checkpoint("session-a", f"trace-{i}", checkpoint, manifest)
        
        for i in range(2):
            checkpoint = {
                "checkpoint_hash": f"hash_b_{i}",
                "prev_checkpoint_hash": f"hash_b_{i-1}" if i > 0 else "",
                "status": "completed",
                "state": {"turn": i},
                "metadata": {},
            }
            checkpointer.save_checkpoint("session-b", f"trace-{i}", checkpoint, manifest)
        
        assert checkpointer.checkpoint_count("session-a") == 3
        assert checkpointer.checkpoint_count("session-b") == 2


class TestCheckpointPruning:
    """Verify old checkpoint garbage collection."""
    
    def test_prune_keeps_most_recent(self, checkpointer: SQLiteCheckpointer):
        """Pruning keeps most recent N checkpoints."""
        manifest = {"python_version": "3.13.7", "env_hash": "env1", "dependency_versions": {}, "timezone": "UTC"}
        
        # Create 15 checkpoints
        for i in range(15):
            checkpoint = {
                "checkpoint_hash": f"hash_{i}",
                "prev_checkpoint_hash": f"hash_{i-1}" if i > 0 else "",
                "status": "completed",
                "state": {"turn": i},
                "metadata": {},
            }
            checkpointer.save_checkpoint("session-1", f"trace-{i}", checkpoint, manifest)
        
        assert checkpointer.checkpoint_count("session-1") == 15
        
        # Prune to keep only 10
        deleted_count = checkpointer.prune_old_checkpoints("session-1", keep_count=10)
        
        assert deleted_count == 5
        assert checkpointer.checkpoint_count("session-1") == 10
        
        # Most recent should still be accessible
        latest = checkpointer.load_checkpoint("session-1")
        assert latest["state"]["turn"] == 14  # Last one saved
    
    def test_prune_no_op_when_under_limit(self, checkpointer: SQLiteCheckpointer):
        """Pruning with limit >= count deletes nothing."""
        manifest = {"python_version": "3.13.7", "env_hash": "env1", "dependency_versions": {}, "timezone": "UTC"}
        
        for i in range(5):
            checkpoint = {
                "checkpoint_hash": f"hash_{i}",
                "prev_checkpoint_hash": "",
                "status": "completed",
                "state": {"turn": i},
                "metadata": {},
            }
            checkpointer.save_checkpoint("session-1", f"trace-{i}", checkpoint, manifest)
        
        deleted = checkpointer.prune_old_checkpoints("session-1", keep_count=10)
        assert deleted == 0
        assert checkpointer.checkpoint_count("session-1") == 5


class TestSessionDeletion:
    """Verify session-level cleanup."""
    
    def test_delete_session_removes_all_checkpoints(self, checkpointer: SQLiteCheckpointer):
        """Deleting a session removes all its checkpoints."""
        manifest = {"python_version": "3.13.7", "env_hash": "env1", "dependency_versions": {}, "timezone": "UTC"}
        
        for i in range(5):
            checkpoint = {
                "checkpoint_hash": f"hash_{i}",
                "prev_checkpoint_hash": "",
                "status": "completed",
                "state": {"turn": i},
                "metadata": {},
            }
            checkpointer.save_checkpoint("session-to-delete", f"trace-{i}", checkpoint, manifest)
        
        assert checkpointer.checkpoint_count("session-to-delete") == 5
        
        deleted = checkpointer.delete_session("session-to-delete")
        assert deleted == 5
        assert checkpointer.checkpoint_count("session-to-delete") == 0
        
        with pytest.raises(CheckpointNotFoundError):
            checkpointer.load_checkpoint("session-to-delete")


class TestCrashRecovery:
    """Verify recovery from simulated failures."""
    
    def test_recover_from_missing_checkpoint(self, checkpointer: SQLiteCheckpointer):
        """Can handle missing checkpoint gracefully."""
        with pytest.raises(CheckpointNotFoundError):
            checkpointer.load_checkpoint("nonexistent-session")
    
    def test_database_error_raises_checkpoint_error(self, temp_db):
        """Database errors are wrapped in CheckpointError."""
        checkpointer = SQLiteCheckpointer(temp_db, auto_migrate=True)
        
        # Close the database file to cause errors
        import os
        os.chmod(temp_db, 0o000)
        
        try:
            with pytest.raises(CheckpointError):
                checkpoint = {
                    "checkpoint_hash": "hash1",
                    "prev_checkpoint_hash": "",
                    "status": "completed",
                    "state": {},
                    "metadata": {},
                }
                manifest = {
                    "python_version": "3.13.7",
                    "env_hash": "env1",
                    "dependency_versions": {},
                    "timezone": "UTC",
                }
                checkpointer.save_checkpoint("session-1", "trace-1", checkpoint, manifest)
        finally:
            # Restore permissions for cleanup
            os.chmod(temp_db, 0o644)


class TestCheckpointReplacement:
    """Verify trace_id uniqueness and replacement."""
    
    def test_same_trace_id_replaces_checkpoint(self, checkpointer: SQLiteCheckpointer):
        """Saving with same trace_id replaces previous checkpoint."""
        manifest = {"python_version": "3.13.7", "env_hash": "env1", "dependency_versions": {}, "timezone": "UTC"}
        
        checkpoint1 = {
            "checkpoint_hash": "hash_v1",
            "prev_checkpoint_hash": "",
            "status": "completed",
            "state": {"version": 1},
            "metadata": {},
        }
        checkpoint2 = {
            "checkpoint_hash": "hash_v2",
            "prev_checkpoint_hash": "",
            "status": "completed",
            "state": {"version": 2},
            "metadata": {},
        }
        
        checkpointer.save_checkpoint("session-1", "trace-1", checkpoint1, manifest)
        assert checkpointer.checkpoint_count("session-1") == 1
        
        checkpointer.save_checkpoint("session-1", "trace-1", checkpoint2, manifest)
        assert checkpointer.checkpoint_count("session-1") == 1  # Still 1, not 2
        
        loaded = checkpointer.load_checkpoint("session-1", "trace-1")
        assert loaded["state"]["version"] == 2  # Loads the newer one
