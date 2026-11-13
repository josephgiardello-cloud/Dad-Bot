"""Standalone snapshot isolation and diagnostics; these tests do not certify replay."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextvars import copy_context
from dataclasses import asdict, dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dadbot.runtime.agent_driver_loop import AgentDriverLoop, DriverLoopPolicy, DriverLoopResult
from dadbot.runtime.loop_snapshot import LoopSessionSnapshot, LoopSnapshotManager, _snapshot_path
from dadbot_system.security import AuthorizationError

pytestmark = pytest.mark.integration
TENANT = "t1_" + "a" * 26
OTHER_TENANT = "t1_" + "b" * 26
SESSION = "test-resume-session-42"


@pytest.fixture
def canonical_loop_bot(make_test_dadbot, monkeypatch, tmp_path):
    from dadbot.core.execution_receipt import ReceiptSigner

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    monkeypatch.setenv("DADBOT_EXECUTION_LEDGER_BACKEND", "file")
    monkeypatch.setenv("DADBOT_RECEIPT_KEY", os.urandom(32).hex())
    monkeypatch.setattr("dadbot_system.local_mcp_server._project_root", lambda: tmp_path)
    bot = make_test_dadbot(tenant_id=TENANT)
    monkeypatch.setattr(bot.model_port, "generate", lambda *_args, **_kwargs: "verified reply")
    monkeypatch.setattr(bot.model_port, "generate_async", AsyncMock(return_value="verified reply"))
    bot.services.turn_orchestrator.control_plane.graph.configure_receipt_signer(ReceiptSigner())
    try:
        yield bot
    finally:
        bot.shutdown()


def _run_durable(bot, **kwargs):
    from dadbot.assistant_runtime import AssistantRuntime

    return AssistantRuntime(bot).run_agent_loop(
        "bounded goal", session_id=SESSION, policy=DriverLoopPolicy(max_turns=2),
        reflection_hook=lambda _ctx: {"action_input": "perform bounded work"},
        enable_semantic_memory=False, **kwargs,
    )


def test_production_resume_reconciles_committed_dispatch_without_reexecution(canonical_loop_bot, monkeypatch):
    from dadbot.core.session_store import SessionStore

    bot = canonical_loop_bot
    control_plane = bot.services.turn_orchestrator.control_plane
    original = bot.execute_turn
    calls = []

    def crash_after_commit(request, **kwargs):
        calls.append(request.metadata["request_id"])
        original(request, **kwargs)
        raise SystemExit("process lost after canonical completion")

    monkeypatch.setattr(bot, "execute_turn", crash_after_commit)
    with pytest.raises(SystemExit, match="process lost"):
        _run_durable(bot, durable=True)
    state = SessionStore.projected_from_ledger(ledger=control_plane.ledger).get(SESSION, tenant_id=TENANT)
    assert state["agent_loop_cursor"]["pending_request_id"] == calls[0]
    completed = control_plane._ledger_index.completed_metadata(
        session_id=SESSION, tenant_id=TENANT, request_id=calls[0],
    )
    assert completed is not None
    assert completed["metadata"]["execution_receipt_status"] == "verified"
    monkeypatch.setattr(bot, "execute_turn", original)
    resumed = _run_durable(bot, resume=True)
    assert resumed.completed_turns == 2
    assert [record.turn_index for record in resumed.records] == [2]
    assert resumed.stop_reason == "max_turns"
    assert len(control_plane._ledger_index.completed_agent_loop_entries(
        session_id=SESSION, tenant_id=TENANT,
    )) == 2
    again = _run_durable(bot, resume=True)
    assert again.records == []
    assert again.completed_turns == 2
    state = SessionStore.projected_from_ledger(ledger=control_plane.ledger).get(SESSION, tenant_id=TENANT)
    assert state["agent_loop_cursor"]["pending_request_id"] == ""
    assert state["agent_loop_cursor"]["stop_reason"] == "max_turns"


def test_production_resume_in_fresh_process(canonical_loop_bot, monkeypatch):
    bot = canonical_loop_bot
    original = bot.execute_turn

    def crash_after_commit(request, **kwargs):
        original(request, **kwargs)
        raise SystemExit("lost response")

    monkeypatch.setattr(bot, "execute_turn", crash_after_commit)
    with pytest.raises(SystemExit, match="lost response"):
        _run_durable(bot, durable=True)
    bot.shutdown()
    script = """
import json, os, sys
from pathlib import Path
from unittest.mock import AsyncMock
from dadbot.assistant_runtime import AssistantRuntime
from dadbot.config import DadBotConfig
from dadbot.core.dadbot import DadBot
from dadbot.core.execution_receipt import ReceiptSigner
from dadbot.memory.semantic_manager import SemanticIndexManager
from dadbot.runtime.agent_driver_loop import DriverLoopPolicy
import dadbot_system.local_mcp_server as tasks
tasks._project_root = lambda: Path(os.environ["DADBOT_SESSION_LOG_DIR"]).parent
SemanticIndexManager.embed_texts = lambda self, texts, purpose=None: [[1.0] * 12 for _ in texts]
bot = DadBot(config=DadBotConfig(tenant_id=sys.argv[1]))
bot.model_port.generate = lambda *args, **kwargs: "verified reply"
bot.model_port.generate_async = AsyncMock(return_value="verified reply")
control_plane = bot.services.turn_orchestrator.control_plane
control_plane.graph.configure_receipt_signer(ReceiptSigner())
try:
    result = AssistantRuntime(bot).run_agent_loop(
        "bounded goal", session_id=sys.argv[2], policy=DriverLoopPolicy(max_turns=2),
        reflection_hook=lambda ctx: {"action_input": "perform bounded work"},
        enable_semantic_memory=False, resume=True)
    print(json.dumps({
        "completed": result.completed_turns,
        "turns": [record.turn_index for record in result.records],
        "stop": result.stop_reason,
        "commits": len(control_plane._ledger_index.completed_agent_loop_entries(
            session_id=sys.argv[2], tenant_id=sys.argv[1])),
    }))
finally:
    bot.shutdown()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, TENANT, SESSION],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    recovered = json.loads(result.stdout.strip().splitlines()[-1])
    assert recovered == {"completed": 2, "turns": [2], "stop": "max_turns", "commits": 2}


def test_production_resume_rejects_rotated_receipt_key_without_dispatch(canonical_loop_bot, monkeypatch):
    from dadbot.core.execution_receipt import ReceiptSigner

    bot = canonical_loop_bot
    _run_durable(bot, durable=True)
    control_plane = bot.services.turn_orchestrator.control_plane
    control_plane.graph.configure_receipt_signer(ReceiptSigner(os.urandom(32)))

    def forbidden_dispatch(*_args, **_kwargs):
        pytest.fail("Mismatched receipt keys must never authorize dispatch")

    monkeypatch.setattr(bot, "execute_turn", forbidden_dispatch)
    before = control_plane.ledger.event_count()
    with pytest.raises(ValueError, match="receipt envelope"):
        _run_durable(bot, resume=True)
    assert control_plane.ledger.event_count() == before


def test_production_resume_refuses_uncertain_dispatch(canonical_loop_bot, monkeypatch):
    bot = canonical_loop_bot

    def crash_before_completion(_request, **_kwargs):
        raise SystemExit("lost dispatch")

    monkeypatch.setattr(bot, "execute_turn", crash_before_completion)
    with pytest.raises(SystemExit, match="lost dispatch"):
        _run_durable(bot, durable=True)
    with pytest.raises(RuntimeError, match=r"uncertain.*resend"):
        _run_durable(bot, resume=True)
    with pytest.raises(RuntimeError, match="explicitly resumed"):
        _run_durable(bot, durable=True)
    with pytest.raises(RuntimeError, match="explicitly resumed"):
        _run_durable(bot)


@pytest.mark.parametrize("damage", ["owner", "budget"])
def test_production_resume_preserves_corrupt_canonical_cursor(canonical_loop_bot, damage):
    from dadbot.core.session_store import SessionStore

    bot = canonical_loop_bot
    _run_durable(bot, durable=True)
    ledger = bot._execution_ledger
    state = SessionStore.projected_from_ledger(ledger=ledger).get(SESSION, tenant_id=TENANT)
    cursor = state["agent_loop_cursor"]
    cursor.update({"tenant_id": OTHER_TENANT} if damage == "owner" else {"next_turn": -1})
    ledger.write({
        "type": "SESSION_STATE_UPDATED", "tenant_id": TENANT, "session_id": SESSION,
        "trace_id": "corrupt-cursor-test", "kernel_step_id": "test.cursor",
        "timestamp": "2026-11-12", "committed": True,
        "payload": {"tenant_id": TENANT, "state": {"agent_loop_cursor": cursor}},
    })
    before = ledger.event_count()
    with pytest.raises(ValueError, match="cursor"):
        _run_durable(bot, resume=True)
    assert ledger.event_count() == before
    assert SessionStore.projected_from_ledger(ledger=ledger).get(
        SESSION, tenant_id=TENANT,
    )["agent_loop_cursor"] == cursor


def test_production_resume_revalidates_captured_authority(canonical_loop_bot, tmp_path):
    from dadbot.assistant_runtime import AssistantRuntime
    from dadbot_system.delegation import SQLiteDelegationStore, runtime_authority_scope
    from dadbot_system.security import ServiceTokenManager

    bot = canonical_loop_bot
    _run_durable(bot, durable=True)
    path = str(tmp_path / "resume-authority.sqlite")
    SQLiteDelegationStore.initialize(path)
    store = SQLiteDelegationStore(path, "resume-test-authority")
    tokens = ServiceTokenManager("resume-test-authority")
    principal = tokens.verify(tokens.issue(subject="resume-user", tenant_id=TENANT, scopes={"write"}))
    store.grant_membership(issuer="dadbot", subject=principal.subject, tenant_id=TENANT, scopes={"write"})
    with runtime_authority_scope(store, principal, scope="write", tenant_id=TENANT):
        runtime = AssistantRuntime(bot)
        captured = copy_context()
    options = {
        "session_id": SESSION, "policy": DriverLoopPolicy(max_turns=2),
        "enable_semantic_memory": False, "resume": True,
    }
    before = bot._execution_ledger.event_count()
    with pytest.raises(AuthorizationError):
        runtime.run_agent_loop("bounded goal", **options)
    store.revoke_membership(issuer="dadbot", subject=principal.subject, tenant_id=TENANT)
    with pytest.raises(AuthorizationError):
        captured.run(runtime.run_agent_loop, "bounded goal", **options)
    assert bot._execution_ledger.event_count() == before


def test_canonical_loop_claim_excludes_competing_process(canonical_loop_bot):
    from dadbot.core.agent_loop_recovery import agent_loop_recovery_lock

    backend = canonical_loop_bot._execution_ledger._backend
    path = str(backend.path())
    script = """
import sys
from dadbot.core.agent_loop_recovery import agent_loop_recovery_lock
from dadbot.core.ledger_backend import FileWALLedgerBackend
backend = FileWALLedgerBackend(sys.argv[3])
try:
    with agent_loop_recovery_lock(backend, tenant_id=sys.argv[1], session_id=sys.argv[2]):
        raise AssertionError("Competing loop acquired an active claim")
except RuntimeError:
    print("claim rejected")
"""
    with agent_loop_recovery_lock(backend, tenant_id=TENANT, session_id=SESSION):
        result = subprocess.run(
            [sys.executable, "-c", script, TENANT, SESSION, path],
            capture_output=True, text=True, timeout=20, check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "claim rejected"
    with agent_loop_recovery_lock(backend, tenant_id=TENANT, session_id=SESSION):
        pass


def test_production_resume_preserves_noop_budget(canonical_loop_bot):
    from dadbot.assistant_runtime import AssistantRuntime

    bot = canonical_loop_bot
    seen = []

    def interrupted_reflection(ctx):
        seen.append(ctx["turn_index"])
        if ctx["turn_index"] == 2:
            raise SystemExit("reflection interrupted")
        return {"action_input": ""}

    options = {
        "session_id": SESSION, "policy": DriverLoopPolicy(max_turns=4, max_consecutive_noop=2),
        "enable_semantic_memory": False,
    }
    runtime = AssistantRuntime(bot)
    with pytest.raises(SystemExit, match="reflection interrupted"):
        runtime.run_agent_loop("bounded goal", reflection_hook=interrupted_reflection, durable=True, **options)
    resumed = runtime.run_agent_loop(
        "bounded goal", reflection_hook=lambda _ctx: {"action_input": ""}, resume=True, **options,
    )
    assert seen == [1, 2]
    assert [record.turn_index for record in resumed.records] == [2]
    assert resumed.stop_reason == "noop_threshold"
    assert resumed.consecutive_noop == 2
    assert resumed.completed_turns == 0


def test_production_resume_rejects_changed_input_policy_and_foreign_session(canonical_loop_bot):
    from dadbot.assistant_runtime import AssistantRuntime

    runtime = AssistantRuntime(canonical_loop_bot)
    _run_durable(canonical_loop_bot, durable=True)
    for text, session, policy in (
        ("changed goal", SESSION, DriverLoopPolicy(max_turns=2)),
        ("bounded goal", SESSION, DriverLoopPolicy(max_turns=3)),
        ("bounded goal", "foreign-session", DriverLoopPolicy(max_turns=2)),
    ):
        with pytest.raises(ValueError, match="cursor"):
            runtime.run_agent_loop(
                text, session_id=session, policy=policy, resume=True, enable_semantic_memory=False,
            )


@dataclass
class _FakeResponse:
    reply: str
    should_end: bool = False

    def as_result(self) -> tuple[str, bool]:
        return self.reply, self.should_end


class _CountingKernel:
    def __init__(self, *, fail: bool = False) -> None:
        self.call_count = 0
        self.fail = fail

    def execute_turn(self, request: Any, **_: Any) -> _FakeResponse:
        self.call_count += 1
        if self.fail:
            raise RuntimeError("Dispatch failed")
        return _FakeResponse(reply=f"reply_{self.call_count}")


def test_snapshot_lifecycle_and_policy_roundtrip(tmp_path):
    policy = DriverLoopPolicy(max_turns=3, max_failures=1, allowed_tools=["safe_tool"], trigger_type="proactive")
    manager = LoopSnapshotManager(SESSION, "task", policy, tenant_id=TENANT, base_dir=tmp_path)
    snapshot = LoopSessionSnapshot.load(SESSION, tmp_path, tenant_id=TENANT)
    assert snapshot is not None
    assert snapshot.is_incomplete
    assert snapshot.policy == asdict(policy)
    result = AgentDriverLoop(_CountingKernel(), policy=policy).run(
        "task", session_id=SESSION, tenant_id=TENANT, reflection_hook=manager.wrap_reflection(),
    )
    manager.finalize(result)
    snapshot = LoopSessionSnapshot.load(SESSION, tmp_path, tenant_id=TENANT)
    assert snapshot is not None
    assert not snapshot.is_incomplete
    assert snapshot.last_committed_turn == snapshot.completed_turns == 3
    assert snapshot.stop_reason == result.stop_reason
    assert LoopSessionSnapshot.delete(SESSION, tmp_path, tenant_id=TENANT)
    assert not LoopSessionSnapshot.delete(SESSION, tmp_path, tenant_id=TENANT)
    assert LoopSessionSnapshot.load(SESSION, tmp_path, tenant_id=TENANT) is None


def test_reflection_does_not_claim_the_pending_dispatch(tmp_path):
    manager = LoopSnapshotManager(SESSION, "task", tenant_id=TENANT, base_dir=tmp_path)
    hook = manager.wrap_reflection()
    seen = []

    def inspect(ctx):
        output = hook(ctx)
        snapshot = LoopSessionSnapshot.load(SESSION, tmp_path, tenant_id=TENANT)
        assert snapshot is not None
        seen.append((snapshot.last_committed_turn, snapshot.completed_turns))
        return output

    AgentDriverLoop(_CountingKernel(), policy=DriverLoopPolicy(max_turns=3)).run(
        "task", session_id=SESSION, tenant_id=TENANT, reflection_hook=inspect,
    )
    assert seen == [(0, 0), (1, 1), (2, 2)]
    snapshot = LoopSessionSnapshot.load(SESSION, tmp_path, tenant_id=TENANT)
    assert snapshot is not None
    assert snapshot.is_incomplete
    assert snapshot.completed_turns == 2  # Final dispatch is not acknowledged without finalize.


@pytest.mark.parametrize("outcome", ["failed", "skipped", "rejected", "stopped"])
def test_unsuccessful_or_undispatched_actions_do_not_advance_progress(tmp_path, outcome):
    manager = LoopSnapshotManager(SESSION, "task", tenant_id=TENANT, base_dir=tmp_path)
    reflections = {
        "failed": {},
        "skipped": {"action_input": ""},
        "rejected": {"action_input": "task", "goal_updates": {"": "invalid key"}},
        "stopped": {"should_continue": False},
    }
    kernel = _CountingKernel(fail=outcome == "failed")
    result = AgentDriverLoop(kernel, policy=DriverLoopPolicy(max_turns=2)).run(
        "task", session_id=SESSION, tenant_id=TENANT,
        reflection_hook=manager.wrap_reflection(lambda _ctx: reflections[outcome]),
    )
    manager.finalize(result)
    snapshot = LoopSessionSnapshot.load(SESSION, tmp_path, tenant_id=TENANT)
    assert snapshot is not None
    assert snapshot.completed_turns == snapshot.last_committed_turn == 0
    assert snapshot.failures == result.failures
    assert kernel.call_count == (2 if outcome == "failed" else 0)


def test_skipped_turn_indices_are_not_completed_turn_counts(tmp_path):
    manager = LoopSnapshotManager(SESSION, "task", tenant_id=TENANT, base_dir=tmp_path)
    policy = DriverLoopPolicy(max_turns=3)
    result = AgentDriverLoop(_CountingKernel(), policy=policy).run(
        "task", session_id=SESSION, tenant_id=TENANT,
        reflection_hook=manager.wrap_reflection(
            lambda ctx: {"action_input": ""} if ctx["turn_index"] == 2 else {},
        ),
    )
    manager.finalize(result)
    snapshot = LoopSessionSnapshot.load(SESSION, tmp_path, tenant_id=TENANT)
    assert snapshot is not None
    assert snapshot.last_committed_turn == 3
    assert snapshot.completed_turns == 2


def test_snapshot_session_names_and_tenants_do_not_alias(tmp_path):
    paths = []
    for tenant, session, text in (
        (TENANT, "a/b", "first"),
        (TENANT, "a_b", "second"),
        (OTHER_TENANT, "a/b", "foreign"),
        (TENANT, "../" + "task" * 100, "long session"),
        (TENANT, "\u8a18\u61b6", "unicode session"),
    ):
        paths.append(LoopSessionSnapshot(session, text, tenant_id=tenant).save(tmp_path))
        loaded = LoopSessionSnapshot.load(session, tmp_path, tenant_id=tenant)
        assert loaded is not None
        assert loaded.initial_observation == text
    assert len(set(paths)) == len(paths)
    first = LoopSessionSnapshot.load("a/b", tmp_path, tenant_id=TENANT)
    assert first is not None
    assert first.initial_observation == "first"
    assert LoopSessionSnapshot.delete("a/b", tmp_path, tenant_id=OTHER_TENANT)
    assert LoopSessionSnapshot.load("a/b", tmp_path, tenant_id=TENANT) == first


def test_missing_snapshot_reads_require_tenant_and_do_not_create_files(tmp_path):
    directory = tmp_path / "absent"
    with pytest.raises(TypeError):
        LoopSessionSnapshot.load(SESSION, directory)  # pyright: ignore[reportCallIssue]
    with pytest.raises(ValueError):
        LoopSessionSnapshot.load(SESSION, directory, tenant_id="default")
    assert LoopSessionSnapshot.load(SESSION, directory, tenant_id=TENANT) is None
    assert not LoopSessionSnapshot.delete(SESSION, directory, tenant_id=TENANT)
    assert not directory.exists()


@pytest.mark.parametrize(
    "damage",
    ["json", "utf8", "duplicate", "root", "owner", "session", "version", "extra", "negative", "count", "policy"],
)
def test_snapshot_rejects_and_preserves_corruption(tmp_path, damage):
    snapshot = LoopSessionSnapshot(SESSION, "task", tenant_id=TENANT)
    path = snapshot.save(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if damage == "json":
        path.write_text("{", encoding="utf-8")
    elif damage == "utf8":
        path.write_bytes(b"\xff")
    elif damage == "duplicate":
        path.write_text('{"schema_version": 1,' + json.dumps(data)[1:], encoding="utf-8")
    else:
        updates = {
            "owner": {"tenant_id": OTHER_TENANT},
            "session": {"session_id": "another-session"},
            "version": {"schema_version": True},
            "extra": {"unrecognized": 1},
            "negative": {"failures": -1},
            "count": {"completed_turns": 1},
            "policy": {"policy": {**data["policy"], "max_turns": "3"}},
            "root": {},
        }
        data.update(updates[damage])
        path.write_text(json.dumps([] if damage == "root" else data), encoding="utf-8")
    before = path.read_bytes()
    error = AuthorizationError if damage in {"owner", "session"} else ValueError
    with pytest.raises(error):
        LoopSessionSnapshot.load(SESSION, tmp_path, tenant_id=TENANT)
    with pytest.raises(error):
        snapshot.save(tmp_path)
    with pytest.raises(error):
        LoopSessionSnapshot.delete(SESSION, tmp_path, tenant_id=TENANT)
    assert path.read_bytes() == before


def test_legacy_snapshot_is_not_adopted_or_deleted(tmp_path, caplog):
    legacy = tmp_path / f"{SESSION}.loop.json"
    content = '{"session_id": "unowned", "initial_observation": "private"}'
    legacy.write_text(content, encoding="utf-8")
    assert LoopSessionSnapshot.load(SESSION, tmp_path, tenant_id=TENANT) is None
    assert not LoopSessionSnapshot.delete(SESSION, tmp_path, tenant_id=TENANT)
    assert legacy.read_text(encoding="utf-8") == content
    assert "canonical recovery reconciliation" in caplog.text


def test_failed_snapshot_publication_preserves_source_and_cleans_temporary_file(tmp_path, monkeypatch):
    snapshot = LoopSessionSnapshot(SESSION, "original", tenant_id=TENANT)
    path = snapshot.save(tmp_path)
    before = path.read_bytes()
    snapshot.initial_observation = "replacement"

    def reject(_source, _destination):
        raise OSError("Publication failed")

    monkeypatch.setattr("dadbot.runtime.loop_snapshot.os.replace", reject)
    with pytest.raises(OSError, match="Publication failed"):
        snapshot.save(tmp_path)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_snapshot_fresh_process_reload(tmp_path):
    LoopSessionSnapshot(SESSION, "persisted task", tenant_id=TENANT).save(tmp_path)
    script = (
        "import pathlib, sys; from dadbot.runtime.loop_snapshot import LoopSessionSnapshot; "
        "snap = LoopSessionSnapshot.load(sys.argv[1], pathlib.Path(sys.argv[2]), tenant_id=sys.argv[3]); "
        "assert snap is not None; print(snap.initial_observation)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, SESSION, str(tmp_path), TENANT],
        capture_output=True, text=True, check=True, timeout=30,
    )
    assert result.stdout.strip() == "persisted task"


def test_snapshot_revalidates_captured_authority_and_rejects_foreign_access(tmp_path):
    from dadbot_system.delegation import SQLiteDelegationStore, runtime_authority_scope
    from dadbot_system.security import ServiceTokenManager

    authority_path = str(tmp_path / "authority.sqlite")
    SQLiteDelegationStore.initialize(authority_path)
    store = SQLiteDelegationStore(authority_path, "snapshot-test-secret")
    manager = ServiceTokenManager("snapshot-test-secret")
    principal = manager.verify(manager.issue(subject="snapshot-user", tenant_id=TENANT, scopes={"write"}))
    store.grant_membership(issuer="dadbot", subject=principal.subject, tenant_id=TENANT, scopes={"write"})
    with runtime_authority_scope(store, principal, scope="write", tenant_id=TENANT):
        snapshot = LoopSessionSnapshot(SESSION, "task", tenant_id=TENANT)
        path = snapshot.save(tmp_path)
        loaded = LoopSessionSnapshot.load(SESSION, tmp_path, tenant_id=TENANT)
        assert loaded is not None
        for operation in (LoopSessionSnapshot.load, LoopSessionSnapshot.delete):
            with pytest.raises(AuthorizationError):
                operation(SESSION, tmp_path, tenant_id=OTHER_TENANT)
        captured = copy_context()
    before = path.read_bytes()
    for saved in (snapshot, loaded):
        with pytest.raises(AuthorizationError):
            saved.save(tmp_path)
    store.revoke_membership(issuer="dadbot", subject=principal.subject, tenant_id=TENANT)
    for operation in (LoopSessionSnapshot.load, LoopSessionSnapshot.delete):
        with pytest.raises(AuthorizationError):
            captured.run(operation, SESSION, tmp_path, tenant_id=TENANT)
    with pytest.raises(AuthorizationError):
        captured.run(snapshot.save, tmp_path)
    assert path.read_bytes() == before


def test_snapshot_rejects_identity_change_and_inconsistent_final_counters(tmp_path):
    snapshot = LoopSessionSnapshot(SESSION, "task", tenant_id=TENANT)
    path = snapshot.save(tmp_path)
    before = path.read_bytes()
    snapshot.tenant_id = OTHER_TENANT
    with pytest.raises(AuthorizationError, match="identity changed"):
        snapshot.save(tmp_path)
    assert path.read_bytes() == before

    manager = LoopSnapshotManager("another-run", "task", tenant_id=TENANT, base_dir=tmp_path)
    other_path = _snapshot_path("another-run", tmp_path, tenant_id=TENANT)
    before = other_path.read_bytes()
    with pytest.raises(ValueError, match="counters disagree"):
        manager.finalize(DriverLoopResult(tenant_id=TENANT, session_id="another-run", completed_turns=1))
    assert other_path.read_bytes() == before


@pytest.mark.parametrize("mismatch", [{"tenant_id": OTHER_TENANT}, {"session_id": "another-session"}])
def test_snapshot_rejects_another_loops_progress(tmp_path, mismatch):
    manager = LoopSnapshotManager(SESSION, "task", tenant_id=TENANT, base_dir=tmp_path)
    path = _snapshot_path(SESSION, tmp_path, tenant_id=TENANT)
    before = path.read_bytes()
    identity = {"tenant_id": TENANT, "session_id": SESSION, **mismatch}
    with pytest.raises(AuthorizationError, match="dispatch identity mismatch"):
        manager.wrap_reflection()({**identity, "records": []})
    with pytest.raises(AuthorizationError, match="result identity mismatch"):
        manager.finalize(DriverLoopResult(**identity))
    assert path.read_bytes() == before


def test_new_manager_cannot_silently_replace_an_existing_run(tmp_path):
    manager = LoopSnapshotManager(SESSION, "original", tenant_id=TENANT, base_dir=tmp_path)
    path = _snapshot_path(SESSION, tmp_path, tenant_id=TENANT)
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="Reconcile"):
        LoopSnapshotManager(SESSION, "replacement", tenant_id=TENANT, base_dir=tmp_path)
    assert path.read_bytes() == before
    result = AgentDriverLoop(_CountingKernel(), policy=DriverLoopPolicy(max_turns=1)).run(
        "original", session_id=SESSION, tenant_id=TENANT, reflection_hook=manager.wrap_reflection(),
    )
    manager.finalize(result)
    with pytest.raises(FileExistsError):
        LoopSnapshotManager(SESSION, "replacement", tenant_id=TENANT, base_dir=tmp_path)
