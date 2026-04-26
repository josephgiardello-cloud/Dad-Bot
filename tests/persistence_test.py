"""Persistence layer tests for SQLite checkpointer.

Required coverage:
- TestCheckpointSaveLoad (4)
- TestHashChainIntegrity (2)
- TestManifestDriftDetection (1)
- TestConcurrentSessionIsolation (2)
- TestCheckpointPruning (2)
- TestCrashRecovery (2)
- TestCheckpointReplacement + TestSessionDeletion (2)
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from dadbot.core.persistence import (
    CheckpointError,
    CheckpointIntegrityError,
    CheckpointNotFoundError,
    SQLiteCheckpointer,
)


@pytest.fixture
def temp_db_path() -> str:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        yield path
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


@pytest.fixture
def checkpointer(temp_db_path: str) -> SQLiteCheckpointer:
    return SQLiteCheckpointer(temp_db_path, auto_migrate=True, prune_every=0)


def _manifest(*, env_hash: str = "env-1", python_version: str = "3.13.7") -> dict:
    return {
        "python_version": python_version,
        "env_hash": env_hash,
        "dependency_versions": {"pytest": "9.0.3"},
        "timezone": ["UTC", "UTC"],
    }


def _checkpoint(*, turn: int, checkpoint_hash: str, prev_checkpoint_hash: str = "") -> dict:
    return {
        "checkpoint_hash": checkpoint_hash,
        "prev_checkpoint_hash": prev_checkpoint_hash,
        "status": "completed",
        "state": {"turn": turn, "payload": {"nested": [1, 2, 3]}},
        "metadata": {"kind": "test"},
    }


class TestCheckpointSaveLoad:
    def test_save_and_load_checkpoint(self, checkpointer: SQLiteCheckpointer):
        cp = _checkpoint(turn=1, checkpoint_hash="hash-1")
        assert checkpointer.save_checkpoint("s1", "t1", cp, _manifest()) is True

        loaded = checkpointer.load_checkpoint("s1", "t1")
        assert loaded["checkpoint_hash"] == "hash-1"
        assert loaded["state"]["turn"] == 1

    def test_load_most_recent_checkpoint(self, checkpointer: SQLiteCheckpointer):
        checkpointer.save_checkpoint("s1", "t1", _checkpoint(turn=1, checkpoint_hash="hash-1"), _manifest())
        checkpointer.save_checkpoint(
            "s1",
            "t2",
            _checkpoint(turn=2, checkpoint_hash="hash-2", prev_checkpoint_hash="hash-1"),
            _manifest(),
        )

        loaded = checkpointer.load_checkpoint("s1")
        assert loaded["checkpoint_hash"] == "hash-2"
        assert loaded["state"]["turn"] == 2

    def test_load_nonexistent_checkpoint_raises(self, checkpointer: SQLiteCheckpointer):
        with pytest.raises(CheckpointNotFoundError):
            checkpointer.load_checkpoint("missing")

    def test_checkpoint_round_trip_preserves_state(self, checkpointer: SQLiteCheckpointer):
        original = {
            "checkpoint_hash": "hash-roundtrip",
            "prev_checkpoint_hash": "",
            "status": "completed",
            "state": {
                "turn": 9,
                "goals": [{"id": "g1", "description": "goal"}],
                "memory": {"x": 1, "y": ["a", "b"]},
            },
            "metadata": {"trace": {"id": "abc"}},
        }
        checkpointer.save_checkpoint("s1", "t9", original, _manifest())
        loaded = checkpointer.load_checkpoint("s1", "t9")

        assert loaded["state"] == original["state"]
        assert loaded["metadata"] == original["metadata"]


class TestHashChainIntegrity:
    def test_hash_chain_verification_passes_on_valid_chain(self, checkpointer: SQLiteCheckpointer):
        checkpointer.save_checkpoint("s1", "t1", _checkpoint(turn=1, checkpoint_hash="h1"), _manifest())
        checkpointer.save_checkpoint(
            "s1",
            "t2",
            _checkpoint(turn=2, checkpoint_hash="h2", prev_checkpoint_hash="h1"),
            _manifest(),
        )

        loaded = checkpointer.load_checkpoint("s1", "t2")
        assert loaded["prev_checkpoint_hash"] == "h1"
        assert loaded["checkpoint_hash"] == "h2"

    def test_broken_hash_chain_detected_on_load(self, checkpointer: SQLiteCheckpointer, temp_db_path: str):
        checkpointer.save_checkpoint("s1", "t1", _checkpoint(turn=1, checkpoint_hash="h1"), _manifest())
        checkpointer.save_checkpoint(
            "s1",
            "t2",
            _checkpoint(turn=2, checkpoint_hash="h2", prev_checkpoint_hash="WRONG"),
            _manifest(),
        )

        with pytest.raises(CheckpointIntegrityError, match="hash-chain broken"):
            checkpointer.load_checkpoint("s1", "t2")


class TestManifestDriftDetection:
    def test_manifest_drift_raises_in_strict_mode(self, checkpointer: SQLiteCheckpointer):
        checkpointer.save_checkpoint("s1", "t1", _checkpoint(turn=1, checkpoint_hash="h1"), _manifest(env_hash="env-old"))

        with pytest.raises(CheckpointIntegrityError, match="Manifest drift"):
            checkpointer.load_checkpoint(
                "s1",
                "t1",
                current_manifest=_manifest(env_hash="env-new"),
                strict=True,
            )


class TestConcurrentSessionIsolation:
    def test_concurrent_sessions_isolated(self, checkpointer: SQLiteCheckpointer):
        checkpointer.save_checkpoint("session-a", "t1", _checkpoint(turn=1, checkpoint_hash="ha"), _manifest())
        checkpointer.save_checkpoint("session-b", "t1", _checkpoint(turn=1, checkpoint_hash="hb"), _manifest())

        a = checkpointer.load_checkpoint("session-a")
        b = checkpointer.load_checkpoint("session-b")

        assert a["checkpoint_hash"] == "ha"
        assert b["checkpoint_hash"] == "hb"

    def test_session_checkpoint_count_per_session(self, checkpointer: SQLiteCheckpointer):
        for idx in range(3):
            prev = f"a-{idx - 1}" if idx else ""
            checkpointer.save_checkpoint("a", f"t{idx}", _checkpoint(turn=idx, checkpoint_hash=f"a-{idx}", prev_checkpoint_hash=prev), _manifest())
        for idx in range(2):
            prev = f"b-{idx - 1}" if idx else ""
            checkpointer.save_checkpoint("b", f"t{idx}", _checkpoint(turn=idx, checkpoint_hash=f"b-{idx}", prev_checkpoint_hash=prev), _manifest())

        assert checkpointer.checkpoint_count("a") == 3
        assert checkpointer.checkpoint_count("b") == 2


class TestCheckpointPruning:
    def test_prune_keeps_most_recent(self, checkpointer: SQLiteCheckpointer):
        for idx in range(15):
            prev = f"h-{idx - 1}" if idx else ""
            checkpointer.save_checkpoint("s1", f"t{idx}", _checkpoint(turn=idx, checkpoint_hash=f"h-{idx}", prev_checkpoint_hash=prev), _manifest())

        deleted = checkpointer.prune_old_checkpoints("s1", keep_count=10)
        latest = checkpointer.load_checkpoint("s1")

        assert deleted == 5
        assert checkpointer.checkpoint_count("s1") == 10
        assert latest["state"]["turn"] == 14

    def test_prune_no_op_when_under_limit(self, checkpointer: SQLiteCheckpointer):
        for idx in range(3):
            prev = f"h-{idx - 1}" if idx else ""
            checkpointer.save_checkpoint("s1", f"t{idx}", _checkpoint(turn=idx, checkpoint_hash=f"h-{idx}", prev_checkpoint_hash=prev), _manifest())

        deleted = checkpointer.prune_old_checkpoints("s1", keep_count=10)
        assert deleted == 0
        assert checkpointer.checkpoint_count("s1") == 3


class TestCrashRecovery:
    def test_recover_after_partial_write_payload_corruption(self, checkpointer: SQLiteCheckpointer, temp_db_path: str):
        checkpointer.save_checkpoint("s1", "t1", _checkpoint(turn=1, checkpoint_hash="h1"), _manifest())

        with contextlib.closing(sqlite3.connect(temp_db_path)) as conn:
            conn.execute("UPDATE checkpoints SET payload = ? WHERE session_id = ? AND trace_id = ?", ("{bad-json", "s1", "t1"))
            conn.commit()

        with pytest.raises(CheckpointError):
            checkpointer.load_checkpoint("s1", "t1")

    def test_recover_from_missing_checkpoint(self, checkpointer: SQLiteCheckpointer):
        with pytest.raises(CheckpointNotFoundError):
            checkpointer.load_checkpoint("missing")


class TestCheckpointReplacement:
    def test_same_trace_id_replaces_checkpoint(self, checkpointer: SQLiteCheckpointer):
        checkpointer.save_checkpoint("s1", "t1", _checkpoint(turn=1, checkpoint_hash="h-v1"), _manifest())
        checkpointer.save_checkpoint("s1", "t1", _checkpoint(turn=2, checkpoint_hash="h-v2"), _manifest())

        assert checkpointer.checkpoint_count("s1") == 1
        loaded = checkpointer.load_checkpoint("s1", "t1")
        assert loaded["state"]["turn"] == 2


class TestSessionDeletion:
    def test_delete_session_removes_all_checkpoints(self, checkpointer: SQLiteCheckpointer):
        checkpointer.save_checkpoint("s-del", "t1", _checkpoint(turn=1, checkpoint_hash="h1"), _manifest())
        checkpointer.save_checkpoint("s-del", "t2", _checkpoint(turn=2, checkpoint_hash="h2", prev_checkpoint_hash="h1"), _manifest())

        deleted = checkpointer.delete_session("s-del")
        assert deleted == 2
        assert checkpointer.checkpoint_count("s-del") == 0
        with pytest.raises(CheckpointNotFoundError):
            checkpointer.load_checkpoint("s-del")
