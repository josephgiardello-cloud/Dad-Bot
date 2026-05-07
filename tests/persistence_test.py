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
import sqlite3
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from dadbot.core.graph import TurnContext
from dadbot.core.orchestrator import DadBotOrchestrator
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
        checkpointer.save_checkpoint(
            "s1", "t1", _checkpoint(turn=1, checkpoint_hash="h1"), _manifest(env_hash="env-old")
        )

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
            checkpointer.save_checkpoint(
                "a",
                f"t{idx}",
                _checkpoint(turn=idx, checkpoint_hash=f"a-{idx}", prev_checkpoint_hash=prev),
                _manifest(),
            )
        for idx in range(2):
            prev = f"b-{idx - 1}" if idx else ""
            checkpointer.save_checkpoint(
                "b",
                f"t{idx}",
                _checkpoint(turn=idx, checkpoint_hash=f"b-{idx}", prev_checkpoint_hash=prev),
                _manifest(),
            )

        assert checkpointer.checkpoint_count("a") == 3
        assert checkpointer.checkpoint_count("b") == 2


class TestCheckpointPruning:
    def test_prune_keeps_most_recent(self, checkpointer: SQLiteCheckpointer):
        for idx in range(15):
            prev = f"h-{idx - 1}" if idx else ""
            checkpointer.save_checkpoint(
                "s1",
                f"t{idx}",
                _checkpoint(turn=idx, checkpoint_hash=f"h-{idx}", prev_checkpoint_hash=prev),
                _manifest(),
            )

        deleted = checkpointer.prune_old_checkpoints("s1", keep_count=10)
        latest = checkpointer.load_checkpoint("s1")

        assert deleted == 5
        assert checkpointer.checkpoint_count("s1") == 10
        assert latest["state"]["turn"] == 14

    def test_prune_no_op_when_under_limit(self, checkpointer: SQLiteCheckpointer):
        for idx in range(3):
            prev = f"h-{idx - 1}" if idx else ""
            checkpointer.save_checkpoint(
                "s1",
                f"t{idx}",
                _checkpoint(turn=idx, checkpoint_hash=f"h-{idx}", prev_checkpoint_hash=prev),
                _manifest(),
            )

        deleted = checkpointer.prune_old_checkpoints("s1", keep_count=10)
        assert deleted == 0
        assert checkpointer.checkpoint_count("s1") == 3

    def test_prune_checkpoint_write_logs_keeps_most_recent(self, checkpointer: SQLiteCheckpointer, temp_db_path: str):
        for idx in range(12):
            prev = f"h-{idx - 1}" if idx else ""
            checkpointer.save_checkpoint(
                "s-log",
                f"t{idx}",
                _checkpoint(turn=idx, checkpoint_hash=f"h-{idx}", prev_checkpoint_hash=prev),
                _manifest(),
            )

        deleted = checkpointer.prune_checkpoint_write_logs("s-log", keep_count=5)
        assert deleted == 7

        with contextlib.closing(sqlite3.connect(temp_db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM checkpoint_writes WHERE session_id = ?",
                ("s-log",),
            ).fetchone()
            assert int((row["n"] if row is not None else 0) or 0) == 5

    def test_consistency_check_reports_orphan_ok_write_rows(self, checkpointer: SQLiteCheckpointer, temp_db_path: str):
        with contextlib.closing(sqlite3.connect(temp_db_path)) as conn:
            conn.execute(
                """
                INSERT INTO checkpoint_writes(session_id, trace_id, checkpoint_hash, status, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("orphan-s", "orphan-t", "orphan-h", "ok", "", 0.0),
            )
            conn.commit()

        report = checkpointer.validate_consistency(strict=False)
        assert report["ok"] is False
        assert any("orphan_ok_write_rows" in issue for issue in list(report.get("issues") or []))


class TestCrashRecovery:
    def test_recover_after_partial_write_payload_corruption(self, checkpointer: SQLiteCheckpointer, temp_db_path: str):
        checkpointer.save_checkpoint("s1", "t1", _checkpoint(turn=1, checkpoint_hash="h1"), _manifest())

        with contextlib.closing(sqlite3.connect(temp_db_path)) as conn:
            conn.execute(
                "UPDATE checkpoints SET payload = ? WHERE session_id = ? AND trace_id = ?", ("{bad-json", "s1", "t1")
            )
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
        checkpointer.save_checkpoint(
            "s-del", "t2", _checkpoint(turn=2, checkpoint_hash="h2", prev_checkpoint_hash="h1"), _manifest()
        )

        deleted = checkpointer.delete_session("s-del")
        assert deleted == 2
        assert checkpointer.checkpoint_count("s-del") == 0
        with pytest.raises(CheckpointNotFoundError):
            checkpointer.load_checkpoint("s-del")


class TestPhase4GapClosure:
    def test_process_restart_boundary_replay_preserves_state(self, temp_db_path: str):
        first = SQLiteCheckpointer(temp_db_path, auto_migrate=True, prune_every=0)
        state = {
            "turn": 1,
            "goals": [{"id": "g-1", "description": "persist across restart"}],
            "determinism": {
                "tool_trace_hash": "tool-hash-001",
                "lock_hash_with_tools": "lock-tools-001",
            },
        }
        checkpoint_v1 = {
            "checkpoint_hash": "cp-restart-1",
            "prev_checkpoint_hash": "",
            "status": "completed",
            "state": state,
            "metadata": {"trace_id": "trace-1"},
        }
        first.save_checkpoint("restart-session", "trace-1", checkpoint_v1, _manifest())

        # New checkpointer instance simulates process restart boundary.
        second = SQLiteCheckpointer(temp_db_path, auto_migrate=True, prune_every=0)
        loaded = second.load_checkpoint("restart-session")
        assert loaded["checkpoint_hash"] == "cp-restart-1"
        assert loaded["state"]["goals"][0]["id"] == "g-1"
        assert loaded["state"]["determinism"]["tool_trace_hash"] == "tool-hash-001"

    def test_partial_write_corruption_under_load_is_session_local(self, temp_db_path: str):
        checkpointer = SQLiteCheckpointer(temp_db_path, auto_migrate=True, prune_every=0)
        session_prev: dict[str, str] = {}
        session_turn: dict[str, int] = {}

        for idx in range(60):
            session_id = f"sess-{idx % 3}"
            prev = session_prev.get(session_id, "")
            turn = int(session_turn.get(session_id, 0))
            current_hash = f"{session_id}-h-{turn}"
            checkpointer.save_checkpoint(
                session_id,
                f"trace-{idx}",
                _checkpoint(turn=turn, checkpoint_hash=current_hash, prev_checkpoint_hash=prev),
                _manifest(),
            )
            session_prev[session_id] = current_hash
            session_turn[session_id] = turn + 1

        with contextlib.closing(sqlite3.connect(temp_db_path)) as conn:
            conn.execute(
                "UPDATE checkpoints SET payload = ? WHERE session_id = ?",
                ("{corrupted-json", "sess-1"),
            )
            conn.commit()

        with pytest.raises(CheckpointError):
            checkpointer.load_checkpoint("sess-1")

        # Neighbor sessions remain recoverable despite one corrupted shard.
        assert checkpointer.load_checkpoint("sess-0")["checkpoint_hash"]
        assert checkpointer.load_checkpoint("sess-2")["checkpoint_hash"]

    def test_multi_session_concurrency_counts_are_stable(self, temp_db_path: str):
        checkpointer = SQLiteCheckpointer(temp_db_path, auto_migrate=True, prune_every=0)

        sessions = [f"concurrent-{idx}" for idx in range(6)]
        writes_per_session = 15

        def _write_session(session_id: str) -> None:
            for idx in range(writes_per_session):
                prev = f"{session_id}-h-{idx - 1}" if idx else ""
                checkpointer.save_checkpoint(
                    session_id,
                    f"{session_id}-trace-{idx}",
                    _checkpoint(
                        turn=idx,
                        checkpoint_hash=f"{session_id}-h-{idx}",
                        prev_checkpoint_hash=prev,
                    ),
                    _manifest(),
                )

        with ThreadPoolExecutor(max_workers=len(sessions)) as pool:
            for session_id in sessions:
                pool.submit(_write_session, session_id)

        for session_id in sessions:
            assert checkpointer.checkpoint_count(session_id) == writes_per_session
            latest = checkpointer.load_checkpoint(session_id)
            assert latest["state"]["turn"] == writes_per_session - 1

    def test_tool_determinism_fields_round_trip_on_restore(self, checkpointer: SQLiteCheckpointer):
        deterministic_state = {
            "tool_ir": {
                "execution_plan": [{"tool_name": "web_search", "sequence": 1}],
                "executions": [
                    {
                        "tool_name": "web_search",
                        "sequence": 1,
                        "input_hash": "input-abc",
                        "status": "ok",
                    }
                ],
            },
            "tool_results": [{"tool_name": "web_search", "sequence": 1, "status": "ok"}],
            "determinism": {
                "tool_trace_hash": "stable-tool-trace-hash",
                "lock_hash_with_tools": "stable-lock-hash-with-tools",
            },
        }
        cp = {
            "checkpoint_hash": "tool-cp-1",
            "prev_checkpoint_hash": "",
            "status": "completed",
            "state": deterministic_state,
            "metadata": {},
        }
        checkpointer.save_checkpoint("tool-session", "trace-1", cp, _manifest())

        loaded = checkpointer.load_checkpoint("tool-session", "trace-1")
        det = dict(loaded["state"].get("determinism") or {})
        assert det.get("tool_trace_hash") == "stable-tool-trace-hash"
        assert det.get("lock_hash_with_tools") == "stable-lock-hash-with-tools"
        assert loaded["state"]["tool_ir"]["executions"][0]["input_hash"] == "input-abc"

    def test_manifest_drift_in_lenient_mode_warns_without_aborting(
        self,
        checkpointer: SQLiteCheckpointer,
        caplog: pytest.LogCaptureFixture,
    ):
        checkpointer.save_checkpoint(
            "manifest-session",
            "trace-1",
            _checkpoint(turn=1, checkpoint_hash="manifest-cp-1"),
            _manifest(env_hash="env-old", python_version="3.13.7"),
        )

        with caplog.at_level("WARNING"):
            loaded = checkpointer.load_checkpoint(
                "manifest-session",
                "trace-1",
                current_manifest=_manifest(env_hash="env-new", python_version="3.13.8"),
                strict=False,
            )

        assert loaded["checkpoint_hash"] == "manifest-cp-1"
        assert any("Manifest drift detected on checkpoint load" in msg for msg in caplog.messages)

    def test_checkpoint_write_log_rows_export_as_structured_observability(
        self,
        checkpointer: SQLiteCheckpointer,
        temp_db_path: str,
    ):
        checkpointer.save_checkpoint("obs", "trace-1", _checkpoint(turn=1, checkpoint_hash="obs-h1"), _manifest())
        checkpointer.save_checkpoint(
            "obs",
            "trace-2",
            _checkpoint(turn=2, checkpoint_hash="obs-h2", prev_checkpoint_hash="obs-h1"),
            _manifest(),
        )

        # Ensure explicit failure records are exportable through the same table.
        checkpointer._log_write_failure(session_id="obs", trace_id="trace-3", error="simulated write failure")

        with contextlib.closing(sqlite3.connect(temp_db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT session_id, trace_id, checkpoint_hash, status, error
                FROM checkpoint_writes
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                ("obs",),
            ).fetchall()

        exported = [dict(row) for row in rows]
        assert len(exported) == 3
        assert [row["status"] for row in exported] == ["ok", "ok", "error"]
        assert exported[0]["checkpoint_hash"] == "obs-h1"
        assert exported[1]["checkpoint_hash"] == "obs-h2"
        assert exported[2]["error"] == "simulated write failure"

    @pytest.mark.asyncio
    async def test_orchestrator_restart_boundary_restores_checkpointed_session(
        self,
        bot,
        temp_db_path: str,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Integration proof: session survives orchestrator restart with shared checkpointer DB."""

        async def _stub_agent(context: TurnContext, _rich: dict[str, Any]) -> tuple[str, bool]:
            return (f"restart-proof::{str(context.user_input or '').strip()}", False)

        def _stub_expensive_paths() -> None:
            mc = getattr(bot, "memory_coordinator", None)
            if mc is not None:
                monkeypatch.setattr(mc, "consolidate_memories", lambda **_kw: None)
                monkeypatch.setattr(mc, "apply_controlled_forgetting", lambda **_kw: None)
            rm = getattr(bot, "relationship_manager", None)
            if rm is not None:
                monkeypatch.setattr(rm, "materialize_projection", lambda **_kw: None)
            mm = getattr(bot, "memory_manager", None)
            gm = getattr(mm, "graph_manager", None) if mm is not None else None
            if gm is not None:
                monkeypatch.setattr(gm, "sync_graph_store", lambda **_kw: None)
            if hasattr(bot, "validate_reply"):
                monkeypatch.setattr(bot, "validate_reply", lambda _input, reply: reply)
            if hasattr(bot, "current_runtime_health_snapshot"):
                monkeypatch.setattr(bot, "current_runtime_health_snapshot", lambda **_kw: {})

        session_id = "restart-boundary-int"
        input_text = "Process restart determinism proof"

        checkpointer_a = SQLiteCheckpointer(temp_db_path, auto_migrate=True, prune_every=0)
        orchestrator_a = DadBotOrchestrator(
            bot=bot,
            strict=True,
            checkpointer=checkpointer_a,
        )
        monkeypatch.setattr(orchestrator_a.registry.get("agent_service"), "run_agent", _stub_agent)
        _stub_expensive_paths()

        result_a = await orchestrator_a.handle_turn(input_text, session_id=session_id)
        context_a = getattr(orchestrator_a, "_last_turn_context", None)
        assert isinstance(context_a, TurnContext)
        assert result_a[0] == f"restart-proof::{input_text}"

        first_checkpoint = checkpointer_a.load_checkpoint(session_id)
        first_hash = str(first_checkpoint.get("checkpoint_hash") or "")
        assert first_hash

        # Simulate process kill/restart by replacing orchestrator and checkpointer instances.
        checkpointer_b = SQLiteCheckpointer(temp_db_path, auto_migrate=True, prune_every=0)
        orchestrator_b = DadBotOrchestrator(
            bot=bot,
            strict=True,
            checkpointer=checkpointer_b,
        )
        monkeypatch.setattr(orchestrator_b.registry.get("agent_service"), "run_agent", _stub_agent)

        result_b = await orchestrator_b.handle_turn(input_text, session_id=session_id)
        context_b = getattr(orchestrator_b, "_last_turn_context", None)
        assert isinstance(context_b, TurnContext)
        assert result_b[0] == f"restart-proof::{input_text}"

        rows = []
        with contextlib.closing(sqlite3.connect(temp_db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT trace_id, checkpoint_hash, status
                FROM checkpoint_writes
                WHERE session_id = ? AND status = 'ok'
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

        assert len(rows) >= 2
        assert all(str(r["checkpoint_hash"] or "").strip() for r in rows)

        latest = checkpointer_b.load_checkpoint(session_id)
        assert str(latest.get("checkpoint_hash") or "")
        assert str(latest.get("checkpoint_hash") or "") != first_hash
