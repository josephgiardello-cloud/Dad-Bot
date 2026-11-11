import multiprocessing

import pytest

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_backend import FileWALLedgerBackend, WALCorruptionError
from dadbot_system.contracts import ChatRequest, WorkerTask
from dadbot_system.delegation import AuthorityUnavailableError, SQLiteDelegationStore, _execution_authority
from dadbot_system.security import AuthorizationError, ServiceTokenManager

SECRET = "authenticated-wal-boundary-test-secret"


@pytest.fixture
def authority(tmp_path):
    path = str(tmp_path / "authority.db")
    SQLiteDelegationStore.initialize(path)
    store = SQLiteDelegationStore(path, SECRET)
    tasks = {}
    for tenant in ("tenant-a", "tenant-b"):
        store.grant_membership(issuer="dadbot", subject="user", tenant_id=tenant, scopes={"write"})
        request = ChatRequest(session_id="same-session", tenant_id=tenant, user_input="hello")
        task = WorkerTask(session_id=request.session_id, tenant_id=tenant, request=request)
        task.delegation = store.issue(
            ServiceTokenManager(SECRET).issue(subject="user", tenant_id=tenant, scopes={"write"}), task
        )
        tasks[tenant] = task
    return store, tasks


def _read_in_fresh_process(path, authority_path, task, output):
    store = SQLiteDelegationStore(authority_path, SECRET)
    token = _execution_authority.set((store, task))
    try:
        try:
            output.put(("ok", FileWALLedgerBackend(path).load()))
        except AuthorizationError as exc:
            output.put(("denied", str(exc)))
    finally:
        _execution_authority.reset(token)


def _open_without_authority(path, output):
    try:
        FileWALLedgerBackend(path).load()
    except (AuthorizationError, WALCorruptionError) as exc:
        output.put(("denied", str(exc)))
    else:
        output.put(("allowed", ""))


def _initialize_in_fresh_process(path, authority_path, task, start, output):
    store = SQLiteDelegationStore(authority_path, SECRET)
    token = _execution_authority.set((store, task))
    try:
        if not start.wait(15):
            raise RuntimeError("initialization was not released")
        try:
            FileWALLedgerBackend(path, security_mode="AUTHENTICATED", tenant_id=task.tenant_id)
        except AuthorizationError:
            output.put("denied")
        else:
            output.put("ok")
    finally:
        _execution_authority.reset(token)


def _run_child(target, *args):
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(target=target, args=(*args, output))
    process.start()
    try:
        process.join(20)
        assert process.exitcode == 0
        return output.get(timeout=5)
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        output.close()
        output.join_thread()


def test_raw_backend_enforces_current_delegation_when_context_exists(tmp_path, authority):
    store, tasks = authority
    path = tmp_path / "wal"
    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        backend = FileWALLedgerBackend(path)
        ledger = ExecutionLedger(backend)
        event = ledger.commit_tenant_memory(
            tenant_id="tenant-a", expected_revision=0, operation_id="first", state={"memory": "private"}
        )
        assert backend.load() == [event]
        digest, snapshot = backend.load_if_changed(None)
        assert snapshot == [event]
        assert backend.load_if_changed(digest) == (digest, [event])
    finally:
        _execution_authority.reset(token)
    token = _execution_authority.set((store, tasks["tenant-b"]))
    try:
        with pytest.raises(AuthorizationError, match="tenant mismatch"):
            backend.load()
        with pytest.raises(AuthorizationError, match="tenant mismatch"):
            backend.load_if_changed(digest)
        with pytest.raises(AuthorizationError, match="tenant mismatch"):
            backend.append(event, committed=True)
    finally:
        _execution_authority.reset(token)
    store.revoke(tasks["tenant-a"].delegation)
    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        with pytest.raises(AuthorizationError, match="revoked"):
            backend.load()
        with pytest.raises(AuthorizationError, match="revoked"):
            backend.load_if_changed(digest)
        with pytest.raises(AuthorizationError, match="revoked"):
            backend.append(event, committed=True)
    finally:
        _execution_authority.reset(token)


def test_trusted_local_append_prefix_never_bypasses_delegated_event_checks(tmp_path, authority):
    store, tasks = authority
    path = tmp_path / "wal"
    backend = FileWALLedgerBackend(path)
    ledger = ExecutionLedger(backend)
    ledger.commit_tenant_memory(
        tenant_id="tenant-a", expected_revision=0, operation_id="first", state={"memory": "private"}
    )
    assert backend._append_verifier._snapshot is not None
    event = {
        "type": "TEST", "tenant_id": "tenant-b", "session_id": "same-session", "trace_id": "test",
        "kernel_step_id": "test", "timestamp": "2026-01-01", "payload": {},
    }
    ledger._seal_payload_hash_chain(event, ledger.get_next_sequence())
    before = path.read_bytes()
    token = _execution_authority.set((store, tasks["tenant-b"]))
    try:
        with pytest.raises(AuthorizationError, match="tenant mismatch"):
            backend.append(event, committed=True)
    finally:
        _execution_authority.reset(token)
    assert path.read_bytes() == before


def test_fresh_process_with_different_tenant_cannot_read_raw_backend(tmp_path, authority):
    store, tasks = authority
    path = tmp_path / "wal"
    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        ExecutionLedger(FileWALLedgerBackend(path)).commit_tenant_memory(
            tenant_id="tenant-a", expected_revision=0, operation_id="first", state={"memory": "private"}
        )
    finally:
        _execution_authority.reset(token)
    context = multiprocessing.get_context("spawn")
    for tenant, expected in (("tenant-a", "ok"), ("tenant-b", "denied")):
        output = context.Queue()
        process = context.Process(target=_read_in_fresh_process, args=(str(path), store.path, tasks[tenant], output))
        process.start()
        try:
            process.join(20)
            assert process.exitcode == 0
            assert output.get(timeout=5)[0] == expected
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
            output.close()
            output.join_thread()


def test_raw_backend_rejects_expired_and_unavailable_authority(tmp_path, authority, monkeypatch):
    store, tasks = authority
    backend = FileWALLedgerBackend(tmp_path / "wal")
    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        with monkeypatch.context() as expired:
            from dadbot_system import delegation

            expired.setattr(delegation.time, "time", lambda: 10**12)
            with pytest.raises(AuthorizationError, match="expired"):
                backend.load()
        authority_path = tmp_path / "authority.db"
        offline = tmp_path / "authority.offline"
        authority_path.rename(offline)
        try:
            with pytest.raises(AuthorityUnavailableError):
                backend.load()
            with pytest.raises(AuthorityUnavailableError):
                backend.append({"type": "TEST"}, committed=True)
        finally:
            offline.rename(authority_path)
        assert backend.load() == []
    finally:
        _execution_authority.reset(token)


def test_authenticated_binding_survives_reopen_and_rejects_missing_context(tmp_path, authority):
    store, tasks = authority
    path = tmp_path / "bound.wal"
    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        backend = FileWALLedgerBackend(path, security_mode="AUTHENTICATED", tenant_id="tenant-a")
        ledger = ExecutionLedger(backend)
        event = ledger.commit_tenant_memory(
            tenant_id="tenant-a", expected_revision=0, operation_id="first", state={"memory": "private"}
        )
        assert FileWALLedgerBackend(path).load() == [event]
        with pytest.raises(AuthorizationError, match="trusted-local"):
            FileWALLedgerBackend(path, security_mode="TRUSTED_LOCAL")
    finally:
        _execution_authority.reset(token)
    with pytest.raises(AuthorizationError, match="missing"):
        FileWALLedgerBackend(path)
    with pytest.raises(AuthorizationError, match="missing"):
        backend.load()
    with pytest.raises(AuthorizationError, match="missing"):
        backend.append(event, committed=True)
    assert _run_child(_open_without_authority, str(path))[0] == "denied"
    token = _execution_authority.set((store, tasks["tenant-b"]))
    try:
        with pytest.raises(AuthorizationError, match="tenant mismatch"):
            FileWALLedgerBackend(path)
    finally:
        _execution_authority.reset(token)
    assert _run_child(_read_in_fresh_process, str(path), store.path, tasks["tenant-a"]) == ("ok", [event])


def test_authenticated_constructor_rejects_missing_tenant_or_authority_before_touching_storage(tmp_path):
    path = tmp_path / "new-directory" / "bound.wal"
    with pytest.raises(ValueError):
        FileWALLedgerBackend(path, security_mode="AUTHENTICATED")
    assert not path.parent.exists()
    with pytest.raises(AuthorizationError, match="missing"):
        FileWALLedgerBackend(path, security_mode="AUTHENTICATED", tenant_id="tenant-a")
    assert not path.parent.exists()


def test_authenticated_creation_does_not_claim_legacy_wal(tmp_path, authority):
    store, tasks = authority
    path = tmp_path / "legacy.wal"
    path.write_text('{"type":"legacy"}\n', encoding="utf-8")
    before = path.read_bytes()
    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        with pytest.raises(WALCorruptionError, match="unbound"):
            FileWALLedgerBackend(path, security_mode="AUTHENTICATED", tenant_id="tenant-a")
    finally:
        _execution_authority.reset(token)
    assert path.read_bytes() == before


def test_authenticated_binding_signature_corruption_fails_closed(tmp_path, authority):
    import json

    store, tasks = authority
    path = tmp_path / "bound.wal"
    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        FileWALLedgerBackend(path, security_mode="AUTHENTICATED", tenant_id="tenant-a")
        record = json.loads(path.read_text(encoding="utf-8"))
        record["wal_binding"]["storage_id"] = "0" * 32
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        path.with_name(path.name + ".binding").write_text(json.dumps(record) + "\n", encoding="utf-8")
        before = path.read_bytes()
        with pytest.raises(WALCorruptionError, match="signature"):
            FileWALLedgerBackend(path)
        assert path.read_bytes() == before
    finally:
        _execution_authority.reset(token)


@pytest.mark.parametrize("damage", ["delete_header", "replace_header", "delete_anchor"])
def test_bound_wal_rejects_missing_or_replaced_binding(tmp_path, authority, damage):
    store, tasks = authority
    path = tmp_path / "bound.wal"
    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        FileWALLedgerBackend(path, security_mode="AUTHENTICATED", tenant_id="tenant-a")
        if damage == "delete_header":
            path.write_text('{"type":"legacy"}\n', encoding="utf-8")
        elif damage == "replace_header":
            path.write_text('{"wal_binding":{},"signature":"bad"}\n', encoding="utf-8")
        else:
            path.with_name(path.name + ".binding").unlink()
        before = path.read_bytes()
        with pytest.raises(WALCorruptionError):
            FileWALLedgerBackend(path)
        assert _run_child(_open_without_authority, str(path))[0] == "denied"
        assert path.read_bytes() == before
    finally:
        _execution_authority.reset(token)


def test_fresh_process_rejects_authenticated_wal_when_both_bindings_are_removed(tmp_path, authority):
    store, tasks = authority
    path = tmp_path / "bound.wal"
    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        ledger = ExecutionLedger(
            FileWALLedgerBackend(path, security_mode="AUTHENTICATED", tenant_id="tenant-a")
        )
        ledger.commit_tenant_memory(
            tenant_id="tenant-a",
            expected_revision=0,
            operation_id="first",
            state={"memory": "synthetic-private"},
        )
    finally:
        _execution_authority.reset(token)

    lines = path.read_bytes().splitlines(keepends=True)
    assert len(lines) > 1
    path.write_bytes(b"".join(lines[1:]))
    path.with_name(path.name + ".binding").unlink()
    before = path.read_bytes()

    result = _run_child(_open_without_authority, str(path))
    assert path.read_bytes() == before
    assert result[0] == "denied"


def test_authority_store_identity_rejects_replacement_at_same_path(tmp_path, authority):
    store, tasks = authority
    path = tmp_path / "bound.wal"
    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        FileWALLedgerBackend(path, security_mode="AUTHENTICATED", tenant_id="tenant-a")
    finally:
        _execution_authority.reset(token)
    original_identity = store.storage_identity()
    authority_path = tmp_path / "authority.db"
    authority_path.rename(tmp_path / "old-authority.db")
    SQLiteDelegationStore.initialize(str(authority_path))
    replacement = SQLiteDelegationStore(str(authority_path), SECRET)
    assert replacement.storage_identity() != original_identity
    replacement.grant_membership(issuer="dadbot", subject="user", tenant_id="tenant-a", scopes={"write"})
    task = tasks["tenant-a"]
    task.delegation = replacement.issue(
        ServiceTokenManager(SECRET).issue(subject="user", tenant_id="tenant-a", scopes={"write"}), task
    )
    token = _execution_authority.set((replacement, task))
    try:
        with pytest.raises(AuthorizationError, match="authority store mismatch"):
            FileWALLedgerBackend(path)
    finally:
        _execution_authority.reset(token)


def test_authority_database_copy_preserves_binding_identity(tmp_path, authority):
    import sqlite3

    store, tasks = authority
    path = tmp_path / "bound.wal"
    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        FileWALLedgerBackend(path, security_mode="AUTHENTICATED", tenant_id="tenant-a")
    finally:
        _execution_authority.reset(token)
    restored_path = tmp_path / "restored.db"
    with sqlite3.connect(store.path) as source, sqlite3.connect(restored_path) as destination:
        source.backup(destination)
    restored = SQLiteDelegationStore(str(restored_path), SECRET)
    assert restored.storage_identity() == store.storage_identity()
    token = _execution_authority.set((restored, tasks["tenant-a"]))
    try:
        assert FileWALLedgerBackend(path).load() == []
    finally:
        _execution_authority.reset(token)


def test_real_dadbot_boot_uses_explicit_authenticated_wal(make_test_dadbot, tmp_path, monkeypatch, authority):
    store, tasks = authority
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    sessions = tmp_path / "boot-sessions"
    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        bot = make_test_dadbot(
            session_log_dir=sessions,
            tenant_id="tenant-a",
            dependency_registry={"storage_security_mode": "AUTHENTICATED"},
        )
        path = sessions / "execution_ledger.jsonl"
        assert path.with_name(path.name + ".binding").exists()
        assert bot._execution_ledger._backend._security_mode == "AUTHENTICATED"
        assert _run_child(_read_in_fresh_process, str(path), store.path, tasks["tenant-a"])[0] == "ok"
        bot.shutdown()
    finally:
        _execution_authority.reset(token)


def test_authenticated_dadbot_boot_requires_authority_before_wal_creation(make_test_dadbot, tmp_path, monkeypatch):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    sessions = tmp_path / "boot-sessions"
    with pytest.raises(AuthorizationError, match="missing"):
        make_test_dadbot(
            session_log_dir=sessions,
            tenant_id="tenant-a",
            dependency_registry={"storage_security_mode": "AUTHENTICATED"},
        )
    assert not (sessions / "execution_ledger.jsonl").exists()


def test_real_worker_constructs_authenticated_dadbot(make_test_dadbot, tmp_path, monkeypatch, authority):
    import Dad
    from dadbot_system.contracts import ServiceConfig
    from dadbot_system.worker import DadBotTaskProcessor

    store, tasks = authority
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    sessions = tmp_path / "worker-sessions"
    monkeypatch.setattr(Dad, "DadBot", lambda **kwargs: make_test_dadbot(session_log_dir=sessions, **kwargs))
    config = ServiceConfig()
    config.security.auth_required = True
    config.security.authority_sqlite_path = store.path
    config.security.token_secret = SECRET
    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        bot = DadBotTaskProcessor(config)._get_bot("same-session", "", tenant_id="tenant-a")
        path = sessions / "execution_ledger.jsonl"
        assert bot._execution_ledger._backend._security_mode == "AUTHENTICATED"
        assert _run_child(_read_in_fresh_process, str(path), store.path, tasks["tenant-a"])[0] == "ok"
        bot.shutdown()
    finally:
        _execution_authority.reset(token)


def test_bound_wal_reopen_rejects_expired_revoked_and_unavailable_authority(tmp_path, authority, monkeypatch):
    from dadbot_system import delegation

    store, tasks = authority
    path = tmp_path / "bound.wal"
    token = _execution_authority.set((store, tasks["tenant-a"]))
    try:
        FileWALLedgerBackend(path, security_mode="AUTHENTICATED", tenant_id="tenant-a")
        before = path.read_bytes()
        with monkeypatch.context() as expired:
            expired.setattr(delegation.time, "time", lambda: 10**12)
            with pytest.raises(AuthorizationError, match="expired"):
                FileWALLedgerBackend(path)
        authority_path = tmp_path / "authority.db"
        offline = tmp_path / "authority.offline"
        authority_path.rename(offline)
        try:
            with pytest.raises(AuthorityUnavailableError):
                FileWALLedgerBackend(path)
        finally:
            offline.rename(authority_path)
        store.revoke(tasks["tenant-a"].delegation)
        with pytest.raises(AuthorizationError, match="revoked"):
            FileWALLedgerBackend(path)
        assert path.read_bytes() == before
    finally:
        _execution_authority.reset(token)


def test_concurrent_authenticated_initializers_share_one_binding(tmp_path, authority):
    import json

    store, tasks = authority
    path = tmp_path / "concurrent.wal"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    output = context.Queue()
    processes = [
        context.Process(
            target=_initialize_in_fresh_process, args=(str(path), store.path, tasks["tenant-a"], start, output)
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    try:
        for process in processes:
            process.join(20)
            assert process.exitcode == 0
        assert [output.get(timeout=5), output.get(timeout=5)] == ["ok", "ok"]
        binding = json.loads(path.read_text(encoding="utf-8"))["wal_binding"]
        assert binding["tenant_id"] == "tenant-a"
        assert len(binding["storage_id"]) == 32
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        output.close()
        output.join_thread()


def test_competing_tenants_cannot_both_initialize_wal(tmp_path, authority):
    import json

    store, tasks = authority
    path = tmp_path / "competing.wal"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    output = context.Queue()
    processes = [
        context.Process(target=_initialize_in_fresh_process, args=(str(path), store.path, tasks[tenant], start, output))
        for tenant in ("tenant-a", "tenant-b")
    ]
    for process in processes:
        process.start()
    start.set()
    try:
        for process in processes:
            process.join(20)
            assert process.exitcode == 0
        assert sorted([output.get(timeout=5), output.get(timeout=5)]) == ["denied", "ok"]
        binding = json.loads(path.read_text(encoding="utf-8"))["wal_binding"]
        assert binding["tenant_id"] in {"tenant-a", "tenant-b"}
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        output.close()
        output.join_thread()
