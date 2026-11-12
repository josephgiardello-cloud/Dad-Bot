"""Standalone snapshot isolation and diagnostics; these tests do not certify replay."""

from __future__ import annotations

import json
import subprocess
import sys
from contextvars import copy_context
from dataclasses import asdict, dataclass
from typing import Any

import pytest

from dadbot.runtime.agent_driver_loop import AgentDriverLoop, DriverLoopPolicy, DriverLoopResult
from dadbot.runtime.loop_snapshot import LoopSessionSnapshot, LoopSnapshotManager, _snapshot_path
from dadbot_system.security import AuthorizationError

pytestmark = pytest.mark.integration
TENANT = "t1_" + "a" * 26
OTHER_TENANT = "t1_" + "b" * 26
SESSION = "test-resume-session-42"


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
