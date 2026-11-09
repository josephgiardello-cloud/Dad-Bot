from __future__ import annotations

import os
import subprocess
import sys
import threading
import uuid

import pytest

from dadbot.core.execution_ledger import ExecutionLedger, _chain_hash, _event_sha256
from dadbot.core.ledger_backend import LedgerHeadConflict, WALCorruptionError
from dadbot.core.postgres_ledger_backend import PostgresLedgerBackend
from dadbot.core.tenant_memory_ledger import TenantMemoryLedgerCorruption, TenantMemoryRevisionConflict
from dadbot_system.state import PostgresStateStore


@pytest.fixture
def postgres_store(postgres_test_dsn):
    from psycopg import sql
    import psycopg

    prefix = f"dadbot_ledger_test_{uuid.uuid4().hex}"
    names = [f"{prefix}_session", f"{prefix}_task", f"{prefix}_event"]
    store = PostgresStateStore(
        postgres_test_dsn, session_table=names[0], task_table=names[1], event_table=names[2]
    )
    try:
        yield store
    finally:
        with psycopg.connect(postgres_test_dsn, autocommit=True) as connection:
            for name in names:
                connection.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(name)))


def event(value):
    return {
        "type": "SESSION_STATE_UPDATED",
        "session_id": "test",
        "trace_id": "trace",
        "kernel_step_id": "step",
        "timestamp": "2026-01-01",
        "payload": {"state": {"value": value}},
        "committed": True,
    }


def test_postgres_ledger_replay_and_isolation(postgres_store):
    first = ExecutionLedger(PostgresLedgerBackend(postgres_store, stream="first"))
    other = ExecutionLedger(PostgresLedgerBackend(postgres_store, stream="other"))
    first.write(event(1))
    other.write(event(2))
    first.write(event(3))
    recovered = ExecutionLedger(PostgresLedgerBackend(postgres_store, stream="first"))
    assert recovered.load_from_backend() == 2
    assert [item["payload"]["state"]["value"] for item in recovered.read()] == [1, 3]
    assert other.read()[0]["payload"]["state"]["value"] == 2
    assert recovered.verify_replay(mode="chain")["ok"]


def test_postgres_tenant_memory_revision_and_retry(postgres_store):
    a = ExecutionLedger(PostgresLedgerBackend(postgres_store, stream="tenant"))
    b = ExecutionLedger(PostgresLedgerBackend(postgres_store, stream="tenant"))
    kwargs = {"tenant_id": "tenant-alpha", "expected_revision": 0, "operation_id": "one", "state": {"value": 1}}
    committed = a.commit_tenant_memory(**kwargs)
    with pytest.raises(LedgerHeadConflict):
        b.commit_tenant_memory(tenant_id="tenant-alpha", expected_revision=0, operation_id="two", state={"value": 2})
    b.load_from_backend()
    assert b.commit_tenant_memory(**kwargs) == committed
    with pytest.raises(TenantMemoryRevisionConflict):
        b.commit_tenant_memory(tenant_id="tenant-alpha", expected_revision=0, operation_id="two", state={"value": 2})
    b.commit_tenant_memory(tenant_id="tenant-beta", expected_revision=0, operation_id="one", state={"value": 3})
    reloaded = ExecutionLedger(PostgresLedgerBackend(postgres_store, stream="tenant"))
    assert reloaded.load_from_backend() == 2
    assert reloaded.replay_tenant_memory("tenant-alpha").state == {"value": 1}
    assert reloaded.replay_tenant_memory("tenant-beta").state == {"value": 3}


@pytest.mark.parametrize("corruption", ["revision", "operation"])
def test_postgres_transaction_rejects_invalid_tenant_event(postgres_store, corruption):
    backend = PostgresLedgerBackend(postgres_store, stream="tenant-invalid")
    ledger = ExecutionLedger(backend)
    first = ledger.commit_tenant_memory(
        tenant_id="tenant-a", expected_revision=0, operation_id="one", state={"value": 1}
    )
    forged = dict(first)
    forged["payload"] = {
        **first["payload"],
        "operation_id": "one" if corruption == "operation" else "two",
        "expected_revision": 1 if corruption == "operation" else 0,
        "revision": 2,
        "state": {"value": 2},
    }
    forged["sequence"] = 2
    forged["_seq"] = 1
    forged["event_id"] = "forged-event"
    forged["prev_chain_hash"] = first["chain_hash"]
    forged["event_sha256"] = _event_sha256(forged)
    forged["chain_hash"] = _chain_hash(forged["prev_chain_hash"], forged["event_sha256"])

    with pytest.raises(TenantMemoryLedgerCorruption, match="revision or operation"):
        backend.append(forged, committed=True)

    recovered = ExecutionLedger(backend)
    assert recovered.load_from_backend() == 1
    assert recovered.replay_tenant_memory("tenant-a").state == {"value": 1}


@pytest.mark.parametrize("crash_after_append", [False, True])
def test_postgres_tenant_event_survives_process_exit(postgres_store, crash_after_append):
    script = """
import os
import sys
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.postgres_ledger_backend import PostgresLedgerBackend
from dadbot_system.state import PostgresStateStore
store = PostgresStateStore(os.environ["DADBOT_TEST_POSTGRES_DSN"],
                           session_table=sys.argv[1], task_table=sys.argv[2], event_table=sys.argv[3])
ledger = ExecutionLedger(PostgresLedgerBackend(store, stream="tenant-crash"))
ledger.load_from_backend()
if sys.argv[4] == "before":
    os._exit(73)
ledger.commit_tenant_memory(tenant_id="tenant-a", expected_revision=1, operation_id="two", state={"value": 2})
os._exit(73)
"""
    backend = PostgresLedgerBackend(postgres_store, stream="tenant-crash")
    ExecutionLedger(backend).commit_tenant_memory(
        tenant_id="tenant-a", expected_revision=0, operation_id="one", state={"value": 1}
    )
    result = subprocess.run(
        [
            sys.executable, "-c", script,
            postgres_store._session_table, postgres_store._task_table, postgres_store._event_table,
            "after" if crash_after_append else "before",
        ],
        env=os.environ.copy(), timeout=30, check=False,
    )
    assert result.returncode == 73
    recovered = ExecutionLedger(backend)
    recovered.load_from_backend()
    assert recovered.replay_tenant_memory("tenant-a").state == {"value": 2 if crash_after_append else 1}


def test_bot_boot_shares_selected_postgres_ledger(postgres_store, make_test_dadbot, monkeypatch):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    monkeypatch.setenv("DADBOT_EXECUTION_LEDGER_BACKEND", "postgres")
    bot = make_test_dadbot(document_store=postgres_store)
    ledger = bot._execution_ledger
    assert ledger is bot.services.turn_orchestrator.control_plane.ledger
    ledger.write(event(1))
    reloaded = ExecutionLedger(PostgresLedgerBackend(postgres_store, stream="default"))
    assert reloaded.load_from_backend() == 1
    assert reloaded.read()[0]["payload"]["state"]["value"] == 1


def test_postgres_committed_tenant_memory_survives_session_restore(
    postgres_store, make_test_dadbot, monkeypatch,
):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    monkeypatch.setenv("DADBOT_EXECUTION_LEDGER_BACKEND", "postgres")
    bot = make_test_dadbot(document_store=postgres_store)
    session = bot.snapshot_session_state()
    state = bot.memory_manager.normalize_memory_store({**bot.MEMORY_STORE, "last_mood": "positive"})
    other = ExecutionLedger(PostgresLedgerBackend(postgres_store, stream="default"))
    other.load_from_backend()
    other.commit_tenant_memory(
        tenant_id=bot.config.tenant_id,
        expected_revision=0,
        operation_id="committed-before-restore",
        state=state,
    )
    bot.load_session_state_snapshot(session)

    assert bot.MEMORY_STORE == state
    recovered = ExecutionLedger(PostgresLedgerBackend(postgres_store, stream="default"))
    recovered.load_from_backend()
    assert recovered.replay_tenant_memory(bot.config.tenant_id).state == state


@pytest.mark.parametrize("durable", ["0", "1"])
def test_switching_from_postgres_ledger_does_not_bypass_tenant_authority(
    postgres_store, make_test_dadbot, monkeypatch, durable,
):
    ledger = ExecutionLedger(PostgresLedgerBackend(postgres_store, stream="default"))
    ledger.commit_tenant_memory(
        tenant_id="test-tenant",
        expected_revision=0,
        operation_id="committed",
        state={"last_mood": "positive"},
    )
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", durable)
    monkeypatch.setenv("DADBOT_EXECUTION_LEDGER_BACKEND", "file")

    with pytest.raises(RuntimeError, match="PostgreSQL.*established tenant memory"):
        make_test_dadbot(document_store=postgres_store, tenant_id="test-tenant")


def test_bot_refuses_to_abandon_existing_file_ledger(postgres_store, make_test_dadbot, monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    monkeypatch.setenv("DADBOT_EXECUTION_LEDGER_BACKEND", "postgres")
    sessions = tmp_path / "existing-sessions"
    sessions.mkdir()
    (sessions / "execution_ledger.jsonl").write_text("existing data\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="migration is not implemented"):
        make_test_dadbot(document_store=postgres_store, session_log_dir=sessions)


def test_postgres_ledger_rejects_simultaneous_stale_writer(postgres_store):
    ledgers = [
        ExecutionLedger(PostgresLedgerBackend(postgres_store, stream="shared")),
        ExecutionLedger(PostgresLedgerBackend(postgres_store, stream="shared")),
    ]
    barrier = threading.Barrier(2)
    outcomes = []

    def write(ledger, value):
        barrier.wait(timeout=10)
        try:
            ledger.write(event(value))
            outcomes.append("committed")
        except LedgerHeadConflict:
            outcomes.append("conflict")

    threads = [threading.Thread(target=write, args=(ledger, i)) for i, ledger in enumerate(ledgers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
        assert not thread.is_alive()
    assert sorted(outcomes) == ["committed", "conflict"]
    recovered = ExecutionLedger(PostgresLedgerBackend(postgres_store, stream="shared"))
    assert recovered.load_from_backend() == 1
    assert recovered.verify_replay(mode="chain")["ok"]


@pytest.mark.parametrize("remove_head", [False, True])
def test_postgres_ledger_detects_missing_record_or_head(postgres_store, remove_head):
    from psycopg import sql

    backend = PostgresLedgerBackend(postgres_store, stream="shared")
    ExecutionLedger(backend).write(event(1))
    with postgres_store._connect() as connection:
        connection.execute(
            sql.SQL("DELETE FROM {} WHERE session_id = %s").format(
                sql.Identifier(postgres_store._session_table if remove_head else postgres_store._event_table)
            ),
            ("execution-ledger:shared:head" if remove_head else "execution-ledger:shared",),
        )
    with pytest.raises(WALCorruptionError, match="head"):
        backend.load()


def test_postgres_ledger_rolls_back_event_when_head_update_fails(postgres_store):
    from psycopg import sql
    import psycopg

    backend = PostgresLedgerBackend(postgres_store, stream="rollback")
    ledger = ExecutionLedger(backend)
    ledger.write(event(1))
    function_name = f"{postgres_store._session_table}_fail_update"
    with postgres_store._connect() as connection:
        connection.execute(
            sql.SQL(
                "CREATE FUNCTION {}() RETURNS trigger LANGUAGE plpgsql "
                "AS 'BEGIN RAISE EXCEPTION ''injected head update failure''; END'"
            ).format(sql.Identifier(function_name))
        )
        connection.execute(
            sql.SQL("CREATE TRIGGER fail_head BEFORE UPDATE ON {} FOR EACH ROW EXECUTE FUNCTION {}()").format(
                sql.Identifier(postgres_store._session_table), sql.Identifier(function_name)
            )
        )
    try:
        with pytest.raises(psycopg.Error, match="injected head update failure"):
            ledger.write(event(2))
        with pytest.raises(RuntimeError, match="reload"):
            ledger.read()
        assert ledger.load_from_backend() == 1
        assert ledger.read()[0]["payload"]["state"]["value"] == 1
    finally:
        with postgres_store._connect() as connection:
            connection.execute(
                sql.SQL("DROP TRIGGER fail_head ON {}").format(sql.Identifier(postgres_store._session_table))
            )
            connection.execute(sql.SQL("DROP FUNCTION {}()").format(sql.Identifier(function_name)))


def test_postgres_ledger_reconciles_uncertain_committed_append(postgres_store):
    class UncertainBackend(PostgresLedgerBackend):
        def append(self, value, *, committed=False):
            super().append(value, committed=committed)
            raise OSError("injected post-commit failure")

    ledger = ExecutionLedger(UncertainBackend(postgres_store, stream="uncertain"))
    with pytest.raises(OSError, match="post-commit"):
        ledger.write(event(1))
    with pytest.raises(RuntimeError, match="reload"):
        ledger.read()
    assert ledger.load_from_backend() == 1
    assert ledger.read()[0]["payload"]["state"]["value"] == 1


@pytest.mark.parametrize("persist", [False, True])
def test_postgres_tenant_operation_reconciliation(postgres_store, persist):
    class UncertainBackend(PostgresLedgerBackend):
        def append(self, value, *, committed=False):
            if persist:
                super().append(value, committed=committed)
            raise OSError("injected uncertain append")

    ledger = ExecutionLedger(UncertainBackend(postgres_store, stream="tenant-reconcile"))
    operation = {"tenant_id": "alpha", "expected_revision": 0, "operation_id": "one", "state": {"value": 1}}
    with pytest.raises(OSError, match="uncertain append"):
        ledger.commit_tenant_memory(**operation)
    with pytest.raises(RuntimeError, match="reload"):
        ledger.read()
    result = ledger.reconcile_tenant_memory_operation(**operation)
    assert (result is not None) is persist
    assert ledger.replay_tenant_memory("alpha").revision == int(persist)


@pytest.mark.parametrize("crash_point", ["before", "before_commit", "after_commit"])
def test_postgres_ledger_subprocess_crash_around_append(postgres_store, crash_point):
    script = """
import os
import sys
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.postgres_ledger_backend import PostgresLedgerBackend
from dadbot_system.state import PostgresStateStore

store = PostgresStateStore(os.environ["DADBOT_TEST_POSTGRES_DSN"],
                           session_table=sys.argv[1], task_table=sys.argv[2], event_table=sys.argv[3])
ledger = ExecutionLedger(PostgresLedgerBackend(store, stream="crash"))
ledger.load_from_backend()
if sys.argv[4] == "before":
    os._exit(73)
if sys.argv[4] == "before_commit":
    connect = store._connect
    class CrashConnection:
        def __init__(self, inner):
            self.inner = inner
        def __enter__(self):
            self.inner.__enter__()
            return self
        def __exit__(self, *args):
            return self.inner.__exit__(*args)
        def cursor(self):
            return self.inner.cursor()
        def commit(self):
            os._exit(73)
    store._connect = lambda: CrashConnection(connect())
ledger.write({"type": "SESSION_STATE_UPDATED", "session_id": "test",
              "trace_id": "trace", "kernel_step_id": "step", "timestamp": "2026-01-01",
              "payload": {"state": {"value": 2}}, "committed": True})
os._exit(73)
"""
    backend = PostgresLedgerBackend(postgres_store, stream="crash")
    ExecutionLedger(backend).write(event(1))
    result = subprocess.run(
        [
            sys.executable, "-c", script,
            postgres_store._session_table, postgres_store._task_table, postgres_store._event_table,
            crash_point,
        ],
        env=os.environ.copy(),
        timeout=30,
        check=False,
    )
    assert result.returncode == 73
    recovered = ExecutionLedger(backend)
    assert recovered.load_from_backend() == 1 + int(crash_point == "after_commit")
    assert recovered.read()[-1]["payload"]["state"]["value"] == (2 if crash_point == "after_commit" else 1)
