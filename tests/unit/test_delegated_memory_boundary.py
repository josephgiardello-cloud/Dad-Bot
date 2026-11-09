from dataclasses import replace
import json
import multiprocessing
import os
from threading import Event
from types import SimpleNamespace

import pytest

from dadbot.background import BackgroundTaskManager
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_backend import FileWALLedgerBackend
from dadbot.core.local_tenant_establishment import commit_local_tenant, establish_local_tenant
from dadbot_system.contracts import ChatRequest, WorkerTask
from dadbot_system.delegation import (
    AuthorityUnavailableError,
    SQLiteDelegationStore,
    _execution_authority,
    require_tenant_authority,
)
from dadbot_system.security import AuthorizationError, ServiceTokenManager


SECRET = "delegated-memory-boundary-test-secret"


def _concurrent_revoke(path, delegation, started, attempted, done):
    if not started.wait(15):
        raise RuntimeError("memory write did not begin")
    attempted.set()
    SQLiteDelegationStore(path, SECRET).revoke(delegation)
    done.set()


def _interrupted_commit(path, authority_path, task, started, release):
    class PausingBackend(FileWALLedgerBackend):
        def append(self, event, *, committed=False):
            started.set()
            if not release.wait(20):
                raise RuntimeError("memory write was not released")
            return super().append(event, committed=committed)

    store = SQLiteDelegationStore(authority_path, SECRET)
    token = _execution_authority.set((store, task))
    try:
        ledger = ExecutionLedger(PausingBackend(path))
        ledger.commit_tenant_memory(
            tenant_id=task.tenant_id, expected_revision=0, operation_id="first", state={"memory": "saved"}
        )
    finally:
        _execution_authority.reset(token)


def _exit_after_tenant_append(path, authority_path, task):
    store = SQLiteDelegationStore(authority_path, SECRET)
    token = _execution_authority.set((store, task))
    try:
        commit_local_tenant(
            FileWALLedgerBackend(path), task.tenant_id, 1, "update",
            {"memory": "committed"}, lambda: os._exit(73), lambda _state: None,
        )
    finally:
        _execution_authority.reset(token)


@pytest.fixture
def delegated(tmp_path):
    path = str(tmp_path / "authority.db")
    SQLiteDelegationStore.initialize(path)
    store = SQLiteDelegationStore(path, SECRET)
    tokens = ServiceTokenManager(SECRET)
    tasks = {}
    for tenant in ("tenant-a", "tenant-b"):
        store.grant_membership(issuer="dadbot", subject="alice", tenant_id=tenant, scopes={"write"})
        request = ChatRequest(session_id="session", tenant_id=tenant, user_input="hello")
        task = WorkerTask(session_id=request.session_id, tenant_id=tenant, request=request)
        task.delegation = store.issue(
            tokens.issue(subject="alice", tenant_id=tenant, scopes={"write"}), task
        )
        tasks[tenant] = task
    return store, tasks


def test_background_task_receives_live_delegated_authority(delegated):
    store, tasks = delegated
    manager = BackgroundTaskManager(max_workers=1)
    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        require_tenant_authority("tenant-a")
        future = manager.submit(require_tenant_authority, "tenant-a")
        assert future.result(timeout=10) is None
    finally:
        _execution_authority.reset(token)
        manager.shutdown()


def test_background_task_revalidates_revoked_delegation(delegated):
    store, tasks = delegated
    manager = BackgroundTaskManager(max_workers=1)
    started = Event()
    release = Event()

    def check_authority():
        started.set()
        if not release.wait(10):
            raise TimeoutError("background authority check was not released")
        require_tenant_authority("tenant-a")

    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        future = manager.submit(check_authority)
        assert started.wait(10)
        store.revoke(tasks["tenant-a"].delegation)
        release.set()
        with pytest.raises(AuthorizationError, match="revoked"):
            future.result(timeout=10)
    finally:
        release.set()
        _execution_authority.reset(token)
        manager.shutdown()


def test_background_task_revalidates_expired_delegation(delegated, monkeypatch):
    from dadbot_system import delegation

    store, tasks = delegated
    manager = BackgroundTaskManager(max_workers=1)
    started = Event()
    release = Event()

    def check_authority():
        started.set()
        if not release.wait(10):
            raise TimeoutError("background authority check was not released")
        require_tenant_authority("tenant-a")

    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        future = manager.submit(check_authority)
        assert started.wait(10)
        monkeypatch.setattr(delegation, "time", SimpleNamespace(time=lambda: 10**12))
        release.set()
        with pytest.raises(AuthorizationError, match="expired"):
            future.result(timeout=10)
    finally:
        release.set()
        _execution_authority.reset(token)
        manager.shutdown()


def test_authenticated_ledger_requires_current_tenant_authority(tmp_path, delegated, monkeypatch):
    store, tasks = delegated
    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        ledger = ExecutionLedger(FileWALLedgerBackend(tmp_path / "ledger"))
        first = ledger.commit_tenant_memory(
            tenant_id="tenant-a", expected_revision=0, operation_id="first", state={"memory": "private"}
        )
        assert ledger.replay_tenant_memory("tenant-a").state == {"memory": "private"}
    finally:
        _execution_authority.reset(token)

    with pytest.raises(AuthorizationError, match="missing"):
        ledger.replay_tenant_memory("tenant-a")
    token = _execution_authority.set((store, tasks["tenant-b"]))
    try:
        with pytest.raises(AuthorizationError, match="tenant mismatch"):
            ledger.replay_tenant_memory("tenant-a")
        with pytest.raises(AuthorizationError, match="tenant mismatch"):
            ledger.commit_tenant_memory(
                tenant_id="tenant-a", expected_revision=1, operation_id="delete", state={}
            )
    finally:
        _execution_authority.reset(token)

    forged = replace(tasks["tenant-a"], delegation=tasks["tenant-a"].delegation + "bad")
    token = _execution_authority.set((store, forged))
    try:
        with pytest.raises(AuthorizationError):
            ledger.replay_tenant_memory("tenant-a")
    finally:
        _execution_authority.reset(token)
    with monkeypatch.context() as expired:
        from dadbot_system import delegation

        expired.setattr(delegation.time, "time", lambda: 10**12)
        token = _execution_authority.set((store, tasks["tenant-a"]))
        try:
            with pytest.raises(AuthorizationError, match="expired"):
                ledger.replay_tenant_memory("tenant-a")
        finally:
            _execution_authority.reset(token)
    store.revoke(tasks["tenant-a"].delegation)
    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        with pytest.raises(AuthorizationError, match="revoked"):
            ledger.commit_tenant_memory(
                tenant_id="tenant-a", expected_revision=1, operation_id="overwrite", state={"memory": "changed"}
            )
    finally:
        _execution_authority.reset(token)

    local_ledger = ExecutionLedger(FileWALLedgerBackend(tmp_path / "ledger"))
    local_ledger.load_from_backend()
    assert local_ledger.replay_tenant_memory("tenant-a").revision == 1
    assert local_ledger.replay_tenant_memory("tenant-a").state == {"memory": "private"}
    assert first["payload"]["revision"] == 1


def test_authenticated_memory_storage_denies_missing_and_wrong_tenant_context(make_test_dadbot, delegated):
    store, tasks = delegated
    bot = make_test_dadbot(tenant_id="tenant-a")
    storage = bot.memory_manager._storage
    storage.require_authenticated_authority()
    with pytest.raises(AuthorizationError, match="missing"):
        storage.load_memory_store()
    with pytest.raises(AuthorizationError, match="missing"):
        bot.memory_manager.memory_projection()
    token = _execution_authority.set((store, tasks["tenant-b"]))
    try:
        with pytest.raises(AuthorizationError, match="tenant mismatch"):
            storage.load_memory_store()
        with pytest.raises(AuthorizationError, match="tenant mismatch"):
            storage.clear_memory_projection()
        with pytest.raises(AuthorizationError, match="tenant mismatch"):
            bot.memory_manager.memory_projection()
    finally:
        _execution_authority.reset(token)
    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        assert isinstance(storage.load_memory_store(), dict)
        store.revoke(tasks["tenant-a"].delegation)
        with pytest.raises(AuthorizationError, match="revoked"):
            storage.clear_memory_projection()
    finally:
        _execution_authority.reset(token)
    task = tasks["tenant-a"]
    task.delegation = store.issue(
        ServiceTokenManager(SECRET).issue(subject="alice", tenant_id="tenant-a", scopes={"write"}), task
    )
    token = _execution_authority.set((store, task))
    try:
        bot.shutdown()
    finally:
        _execution_authority.reset(token)


def test_memory_commit_holds_authority_fence_against_cross_process_revocation(tmp_path, delegated):
    store, tasks = delegated
    context = multiprocessing.get_context("spawn")
    started, attempted, done = context.Event(), context.Event(), context.Event()

    class PausingBackend(FileWALLedgerBackend):
        def append(self, event, *, committed=False):
            started.set()
            if not attempted.wait(15):
                raise RuntimeError("revocation did not start")
            assert not done.wait(0.5), "revocation committed before the protected write"
            super().append(event, committed=committed)

    token = _execution_authority.set((store, tasks["tenant-a"]))
    process = context.Process(
        target=_concurrent_revoke,
        args=(store.path, tasks["tenant-a"].delegation, started, attempted, done),
    )
    try:
        ledger = ExecutionLedger(PausingBackend(tmp_path / "ledger"))
        process.start()
        ledger.commit_tenant_memory(
            tenant_id="tenant-a", expected_revision=0, operation_id="first", state={"memory": "saved"}
        )
        process.join(15)
        assert process.exitcode == 0
        assert done.is_set()
        with pytest.raises(AuthorizationError, match="revoked"):
            ledger.commit_tenant_memory(
                tenant_id="tenant-a", expected_revision=1, operation_id="second", state={"memory": "changed"}
            )
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        _execution_authority.reset(token)


def test_memory_commit_rejects_authority_store_outage_before_writing(tmp_path, delegated):
    store, tasks = delegated
    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        ledger = ExecutionLedger(FileWALLedgerBackend(tmp_path / "ledger"))
        authority_path = tmp_path / "authority.db"
        offline = tmp_path / "authority.offline"
        authority_path.rename(offline)
        try:
            with pytest.raises(AuthorityUnavailableError):
                ledger.commit_tenant_memory(
                    tenant_id="tenant-a", expected_revision=0, operation_id="first", state={"memory": "saved"}
                )
        finally:
            offline.rename(authority_path)
        assert ledger.replay_tenant_memory("tenant-a").revision == 0
    finally:
        _execution_authority.reset(token)


def test_terminated_memory_commit_releases_fence_and_can_retry(tmp_path, delegated):
    store, tasks = delegated
    context = multiprocessing.get_context("spawn")
    started, release = context.Event(), context.Event()
    ledger_path = str(tmp_path / "ledger")
    process = context.Process(
        target=_interrupted_commit, args=(ledger_path, store.path, tasks["tenant-a"], started, release)
    )
    process.start()
    try:
        assert started.wait(15), "memory write did not reach append"
        process.terminate()
        process.join(15)
        assert not process.is_alive()
        token = _execution_authority.set((store, tasks["tenant-a"]))
        try:
            ledger = ExecutionLedger(FileWALLedgerBackend(ledger_path))
            ledger.load_from_backend()
            assert ledger.replay_tenant_memory("tenant-a").revision == 0
            ledger.commit_tenant_memory(
                tenant_id="tenant-a", expected_revision=0, operation_id="first", state={"memory": "saved"}
            )
            assert ledger.replay_tenant_memory("tenant-a").state == {"memory": "saved"}
        finally:
            _execution_authority.reset(token)
    finally:
        if process.is_alive():
            process.terminate()
            process.join()


def test_authorized_restart_repairs_snapshot_after_committed_append(tmp_path, delegated):
    store, tasks = delegated
    ledger_path = str(tmp_path / "ledger")
    mirror = tmp_path / "memory.json"

    def repair(state):
        mirror.write_text(json.dumps(state), encoding="utf-8")

    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        assert establish_local_tenant(
            FileWALLedgerBackend(ledger_path), "tenant-a", lambda: {"memory": "initial"}, repair
        ) == {"memory": "initial"}
    finally:
        _execution_authority.reset(token)

    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_exit_after_tenant_append, args=(ledger_path, store.path, tasks["tenant-a"]))
    process.start()
    try:
        process.join(20)
        assert process.exitcode == 73
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
    assert json.loads(mirror.read_text(encoding="utf-8")) == {"memory": "initial"}
    store.revoke(tasks["tenant-a"].delegation)
    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        with pytest.raises(AuthorizationError, match="revoked"):
            establish_local_tenant(
                FileWALLedgerBackend(ledger_path), "tenant-a", lambda: {}, repair
            )
    finally:
        _execution_authority.reset(token)
    original_mirror = mirror.read_text(encoding="utf-8")
    assert json.loads(original_mirror) == {"memory": "initial"}

    token = _execution_authority.set((store, tasks["tenant-b"]))
    try:
        with pytest.raises(AuthorizationError, match="tenant mismatch"):
            establish_local_tenant(FileWALLedgerBackend(ledger_path), "tenant-a", lambda: {}, repair)
    finally:
        _execution_authority.reset(token)
    assert mirror.read_text(encoding="utf-8") == original_mirror

    task = tasks["tenant-a"]
    task.delegation = store.issue(
        ServiceTokenManager(SECRET).issue(subject="alice", tenant_id="tenant-a", scopes={"write"}), task
    )
    token = _execution_authority.set((store, task))
    try:
        backend = FileWALLedgerBackend(ledger_path)
        assert establish_local_tenant(backend, "tenant-a", lambda: {}, repair) == {"memory": "committed"}
        ledger = ExecutionLedger(backend)
        ledger.load_from_backend()
        assert ledger.replay_tenant_memory("tenant-a").revision == 2
        assert len([e for e in ledger.read() if e["type"] == "TENANT_MEMORY_COMMITTED"]) == 2
        assert json.loads(mirror.read_text(encoding="utf-8")) == {"memory": "committed"}
    finally:
        _execution_authority.reset(token)


def test_authenticated_shutdown_requires_authority_and_remains_retryable(make_test_dadbot, monkeypatch, caplog, delegated):
    bot = make_test_dadbot(tenant_id="tenant-a")
    bot.memory_manager._storage.require_authenticated_authority()
    calls = []
    monkeypatch.setattr(bot, "save_memory_store", lambda: calls.append("saved"))

    with pytest.raises(AuthorizationError, match="Final memory flush"):
        bot.shutdown()

    assert calls == []
    assert bot._shutdown_persistence_state == "authority_required"
    assert not getattr(bot, "_shutdown_completed", False)
    assert "Final memory flush blocked" in caplog.text
    store, tasks = delegated
    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        bot.shutdown()
    finally:
        _execution_authority.reset(token)
    assert calls == ["saved"]
    assert bot._shutdown_persistence_state == "flushed"
