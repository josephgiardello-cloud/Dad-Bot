from __future__ import annotations

import os
import subprocess
import sys
import threading
import uuid
import hashlib
import json

import pytest

from dadbot.core.execution_ledger import ExecutionLedger, _chain_hash, _event_sha256
from dadbot.core.ledger_backend import LedgerHeadConflict, WALCorruptionError
from dadbot.core.postgres_ledger_backend import PostgresLedgerBackend
from dadbot.core.tenant_memory_ledger import TenantMemoryLedgerCorruption, TenantMemoryRevisionConflict
from dadbot_system.owned_tasks import PostgresOwnedTasks
from dadbot_system.state import PostgresStateStore


@pytest.fixture
def postgres_store(postgres_test_dsn):
    import psycopg
    from psycopg import sql

    prefix = f"dadbot_ledger_test_{uuid.uuid4().hex}"
    names = [f"{prefix}_session", f"{prefix}_task", f"{prefix}_event"]
    store = PostgresStateStore(
        postgres_test_dsn, session_table=names[0], task_table=names[1], event_table=names[2]
    )
    try:
        yield store
    finally:
        with psycopg.connect(postgres_test_dsn, autocommit=True) as connection:
            for name in (
                store.owned_sessions.session_table,
                store.owned_sessions.event_table,
                store.task_store._tenant_catalog_name,
                *names,
            ):
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


def _recover_api_tasks_in_new_process(
    *,
    postgres_dsn: str,
    table_names: tuple[str, str, str],
    authority_path: str,
    token_secret: str,
    redis_url: str = "",
    redis_namespace: str = "",
) -> list[dict]:
    script = r"""
import json
import os
from fastapi.testclient import TestClient
from dadbot_system.api import create_api_app
from dadbot_system.contracts import ServiceConfig
from dadbot_system.orchestration import DadBotOrchestrator
from dadbot_system.state import CompositeStateStore, PostgresStateStore, RedisStateStore

class Broker:
    def __init__(self):
        self.tasks = []
    def enqueue(self, task):
        self.tasks.append(task)
    def get_result_nowait(self):
        return None

names = json.loads(os.environ["DADBOT_IT_TABLES"])
postgres_store = PostgresStateStore(
    os.environ["DADBOT_TEST_POSTGRES_DSN"],
    session_table=names[0], task_table=names[1], event_table=names[2],
)
state_store = postgres_store
if os.environ.get("DADBOT_IT_REDIS_URL"):
    redis_store = RedisStateStore(
        os.environ["DADBOT_IT_REDIS_URL"],
        namespace=os.environ["DADBOT_IT_REDIS_NAMESPACE"],
    )
    state_store = CompositeStateStore(fast_store=redis_store, durable_store=postgres_store)
broker = Broker()
orchestrator = DadBotOrchestrator(broker, state_store=state_store)
config = ServiceConfig.from_dict({
    "security": {
        "auth_required": True,
        "token_secret": os.environ["DADBOT_IT_TOKEN_SECRET"],
        "token_issuer": "dadbot",
        "authority_sqlite_path": os.environ["DADBOT_IT_AUTHORITY_PATH"],
    }
})
app = create_api_app(orchestrator, config=config)
with TestClient(app):
    pass
print(json.dumps([
    {
        "task_id": task.task_id,
        "tenant_id": task.tenant_id,
        "session_id": task.session_id,
        "attempt": task.attempt,
        "gateway": task.request.metadata.get("gateway"),
    }
    for task in broker.tasks
]))
"""
    env = os.environ.copy()
    env.update(
        {
            "DADBOT_TEST_POSTGRES_DSN": postgres_dsn,
            "DADBOT_IT_TABLES": json.dumps(table_names),
            "DADBOT_IT_AUTHORITY_PATH": authority_path,
            "DADBOT_IT_TOKEN_SECRET": token_secret,
            "DADBOT_IT_REDIS_URL": redis_url,
            "DADBOT_IT_REDIS_NAMESPACE": redis_namespace,
        }
    )
    process = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        env=env,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    output_lines = [line for line in process.stdout.splitlines() if line.strip()]
    assert output_lines, process.stderr
    return json.loads(output_lines[-1])


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


def test_postgres_owned_tasks_enforce_forced_rls_with_nobypass_role(postgres_test_dsn):
    import psycopg
    from psycopg import sql

    table_name = f"dadbot_owned_tasks_{uuid.uuid4().hex}"
    store = PostgresOwnedTasks(postgres_test_dsn, table_name)
    tenant_a = "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"
    tenant_b = "t1_bbbbbbbbbbbbbbbbbbbbbbbbbb"
    tenant_other = "t1_cccccccccccccccccccccccccc"
    task_id = "shared-task"

    def payload(tenant_id: str):
        return {
            "tenant_id": tenant_id,
            "task_id": task_id,
            "status": "queued",
            "accepted_at": "2026-01-01T00:00:00Z",
        }

    try:
        with psycopg.connect(postgres_test_dsn) as connection:
            role = connection.execute(
                "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
            ).fetchone()
            assert role == (False, False)

        store.save_task(tenant_a, task_id, payload(tenant_a))
        store.save_task(tenant_b, task_id, payload(tenant_b))
        store.save_task_result(
            tenant_a,
            task_id,
            payload(tenant_a),
            {"tenant_id": tenant_a, "reply": "tenant A"},
        )
        store.save_task_result(
            tenant_b,
            task_id,
            payload(tenant_b),
            {"tenant_id": tenant_b, "reply": "tenant B"},
        )

        assert store.load_response(tenant_a, task_id) == {
            "tenant_id": tenant_a,
            "reply": "tenant A",
        }
        assert store.load_response(tenant_b, task_id) == {
            "tenant_id": tenant_b,
            "reply": "tenant B",
        }
        assert store.list_task_tenants() == [tenant_a, tenant_b]
        assert store.load_task(tenant_other, task_id) is None
        assert store.load_response(tenant_other, task_id) is None
        assert [item["tenant_id"] for item in store.list_incomplete_tasks(tenant_a)] == [tenant_a]

        with psycopg.connect(postgres_test_dsn) as connection:
            visible_rows = connection.execute(
                sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table_name))
            ).fetchone()[0]
            assert visible_rows == 0
    finally:
        with psycopg.connect(postgres_test_dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(store._tenant_catalog_name))
            )
            connection.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name)))


@pytest.mark.parametrize("persisted_task", [False, True])
def test_postgres_api_lifespan_recovers_empty_and_persisted_tasks(
    postgres_test_dsn, tmp_path, persisted_task
):
    from dadbot_system.contracts import ChatRequest, ChatResponse, ServiceConfig, WorkerResult
    from dadbot_system.delegation import SQLiteDelegationStore
    from dadbot_system.orchestration import DadBotOrchestrator
    from dadbot_system.security import ServiceTokenManager
    from dadbot_system.state import PostgresStateStore
    from dadbot_system.api import create_api_app

    class Broker:
        def __init__(self):
            self.tasks = []

        def enqueue(self, task):
            self.tasks.append(task)

        def get_result_nowait(self):
            return None

    prefix = f"dadbot_api_recovery_{uuid.uuid4().hex}"
    names = (f"{prefix}_session", f"{prefix}_task", f"{prefix}_event")
    authority_path = str(tmp_path / f"{prefix}_authority.sqlite")
    task_catalog = f"{names[1][:40]}_tenants_{hashlib.sha256(names[1].encode()).hexdigest()[:12]}"
    session_digest = hashlib.sha256(f"{names[0]}:{names[2]}".encode()).hexdigest()[:12]
    owned_session_table = f"{names[0][:40]}_owned_{session_digest}"
    owned_event_table = f"{names[2][:40]}_owned_{session_digest}"
    secret = "postgres-task-recovery-test-secret"
    tenant_a = "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"
    tenant_b = "t1_bbbbbbbbbbbbbbbbbbbbbbbbbb"
    tenant_revoked = "t1_cccccccccccccccccccccccccc"
    subject = "postgres-recovery-test"
    SQLiteDelegationStore.initialize(authority_path)
    authority = SQLiteDelegationStore(authority_path, secret)
    tokens = {}
    for tenant_id in (tenant_a, tenant_b, tenant_revoked):
        authority.grant_membership(
            issuer="dadbot",
            subject=subject,
            tenant_id=tenant_id,
            scopes={"read", "write"},
        )
        tokens[tenant_id] = ServiceTokenManager(secret, issuer="dadbot").issue(
            subject=subject,
            tenant_id=tenant_id,
            scopes=("read", "write"),
            allowed_tools=(),
            ttl_seconds=3600,
        )
    config = ServiceConfig.from_dict(
        {
            "security": {
                "auth_required": True,
                "token_secret": secret,
                "token_issuer": "dadbot",
                "authority_sqlite_path": authority_path,
            }
        }
    )

    try:
        seed_store = PostgresStateStore(
            postgres_test_dsn,
            session_table=names[0],
            task_table=names[1],
            event_table=names[2],
        )
        if persisted_task:
            seed_orchestrator = DadBotOrchestrator(Broker(), state_store=seed_store)
            create_api_app(seed_orchestrator, config=config)
            completed_task = seed_orchestrator.submit_chat(
                ChatRequest(
                    session_id="persisted-completed",
                    user_input="complete before restart",
                    tenant_id=tenant_a,
                ),
                bearer_token=tokens[tenant_a],
            )
            recoverable_task = seed_orchestrator.submit_chat(
                ChatRequest(
                    session_id="persisted-recoverable",
                    user_input="recover after restart",
                    tenant_id=tenant_b,
                    metadata={"gateway": {"channel": "teams", "message_id": "persisted-message"}},
                ),
                bearer_token=tokens[tenant_b],
            )
            revoked_task = seed_orchestrator.submit_chat(
                ChatRequest(
                    session_id="persisted-revoked",
                    user_input="must not recover",
                    tenant_id=tenant_revoked,
                ),
                bearer_token=tokens[tenant_revoked],
            )
            assert seed_store.task_store.list_task_tenants() == [
                tenant_a,
                tenant_b,
                tenant_revoked,
            ]
            assert seed_orchestrator.task_status(
                completed_task.task_id, tenant_id=tenant_a
            )["status"] == "queued"

            completed_response = ChatResponse(
                session_id=completed_task.session_id,
                request_id=completed_task.request.request_id,
                tenant_id=tenant_a,
                reply="persisted before process restart",
            )
            completed_result = WorkerResult(
                task_id=completed_task.task_id,
                session_id=completed_task.session_id,
                request_id=completed_task.request.request_id,
                tenant_id=tenant_a,
                status="completed",
                session_state=completed_task.session_state,
                response=completed_response,
            )
            completed_result.authority_proof = authority.sign_result(completed_task, completed_result)
            assert seed_orchestrator.apply_worker_result(completed_result) == completed_response
            assert seed_orchestrator.task_status(
                completed_task.task_id, tenant_id=tenant_a
            )["status"] == "completed"
            authority.revoke(revoked_task.delegation)

            recovered = _recover_api_tasks_in_new_process(
                postgres_dsn=postgres_test_dsn,
                table_names=names,
                authority_path=authority_path,
                token_secret=secret,
            )
            assert [item["task_id"] for item in recovered] == [recoverable_task.task_id]
            assert recovered[0]["tenant_id"] == tenant_b
            assert recovered[0]["session_id"] == "persisted-recoverable"
            assert recovered[0]["attempt"] == 2
            assert recovered[0]["gateway"] == {
                "channel": "teams",
                "delivery_mode": "sync",
                "message_id": "persisted-message",
            }
            assert completed_task.task_id not in {item["task_id"] for item in recovered}
            assert revoked_task.task_id not in {item["task_id"] for item in recovered}
        else:
            recovered = _recover_api_tasks_in_new_process(
                postgres_dsn=postgres_test_dsn,
                table_names=names,
                authority_path=authority_path,
                token_secret=secret,
            )
            assert recovered == []
    finally:
        import psycopg
        from psycopg import sql

        with psycopg.connect(postgres_test_dsn, autocommit=True) as connection:
            for name in (
                owned_session_table,
                owned_event_table,
                task_catalog,
                *names,
            ):
                connection.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(name)))


def test_composite_redis_postgres_task_store_recovers_and_publishes(postgres_store, tmp_path):
    import os

    from dadbot_system.contracts import ChatRequest, ChatResponse, ServiceConfig, WorkerResult
    from dadbot_system.api import create_api_app
    from dadbot_system.delegation import SQLiteDelegationStore
    from dadbot_system.orchestration import DadBotOrchestrator
    from dadbot_system.security import ServiceTokenManager
    from dadbot_system.state import CompositeStateStore, RedisStateStore

    redis_url = str(os.environ.get("DADBOT_TEST_REDIS_URL") or "").strip()
    if not redis_url:
        pytest.skip("Set DADBOT_TEST_REDIS_URL to run Redis + Postgres task integration tests.")

    namespace = f"dadbot-composite-task-test-{uuid.uuid4().hex}"
    fast_store = RedisStateStore(redis_url, namespace=namespace)
    fast_store._client.ping()
    secret = "composite-task-recovery-test-secret"
    authority_path = str(tmp_path / "composite-authority.sqlite")
    tenant_a = "t1_aaaaaaaaaaaaaaaaaaaaaaaaaa"
    tenant_b = "t1_bbbbbbbbbbbbbbbbbbbbbbbbbb"
    tenant_c = "t1_cccccccccccccccccccccccccc"
    subject = "composite-recovery-test"
    SQLiteDelegationStore.initialize(authority_path)
    authority = SQLiteDelegationStore(authority_path, secret)
    tokens = {}
    for tenant_id in (tenant_a, tenant_b, tenant_c):
        authority.grant_membership(
            issuer="dadbot",
            subject=subject,
            tenant_id=tenant_id,
            scopes={"read", "write"},
        )
        tokens[tenant_id] = ServiceTokenManager(secret, issuer="dadbot").issue(
            subject=subject,
            tenant_id=tenant_id,
            scopes=("read", "write"),
            allowed_tools=(),
            ttl_seconds=3600,
        )

    class Broker:
        def __init__(self):
            self.tasks = []

        def enqueue(self, task):
            self.tasks.append(task)

        def get_result_nowait(self):
            return None

    config = ServiceConfig.from_dict(
        {
            "security": {
                "auth_required": True,
                "token_secret": secret,
                "token_issuer": "dadbot",
                "authority_sqlite_path": authority_path,
            }
        }
    )
    restart_names = (
        postgres_store._session_table,
        postgres_store._task_table,
        postgres_store._event_table,
    )
    try:
        composite_store = CompositeStateStore(fast_store=fast_store, durable_store=postgres_store)
        assert composite_store.task_store is not None
        first_broker = Broker()
        first_orchestrator = DadBotOrchestrator(first_broker, state_store=composite_store)
        create_api_app(first_orchestrator, config=config)
        task_a = first_orchestrator.submit_chat(
            ChatRequest(
                session_id="composite-published",
                user_input="publish",
                tenant_id=tenant_a,
            ),
            bearer_token=tokens[tenant_a],
        )
        task_b = first_orchestrator.submit_chat(
            ChatRequest(
                session_id="composite-recoverable",
                user_input="recover after restart",
                tenant_id=tenant_b,
                metadata={"gateway": {"channel": "teams", "message_id": "composite-message"}},
            ),
            bearer_token=tokens[tenant_b],
        )
        task_revoked = first_orchestrator.submit_chat(
            ChatRequest(
                session_id="composite-revoked",
                user_input="do not recover",
                tenant_id=tenant_c,
            ),
            bearer_token=tokens[tenant_c],
        )
        task_durable_completed = first_orchestrator.submit_chat(
            ChatRequest(
                session_id="composite-stale-fast",
                user_input="durable completion wins",
                tenant_id=tenant_b,
            ),
            bearer_token=tokens[tenant_b],
        )
        assert composite_store.task_store.list_task_tenants() == [tenant_a, tenant_b, tenant_c]
        assert len(first_broker.tasks) == 4
        assert first_orchestrator.task_status(task_b.task_id, tenant_id=tenant_b)["status"] == "queued"

        duplicate_task_id = "same-task-id-across-tenants"
        duplicate_payloads = {}
        for tenant_id in (tenant_a, tenant_b):
            duplicate_payloads[tenant_id] = {
                "tenant_id": tenant_id,
                "task_id": duplicate_task_id,
                "status": "completed",
                "owner": tenant_id,
            }
            composite_store.task_store.save_task(tenant_id, duplicate_task_id, duplicate_payloads[tenant_id])
        assert composite_store.task_store.load_task(tenant_a, duplicate_task_id) == duplicate_payloads[tenant_a]
        assert composite_store.task_store.load_task(tenant_b, duplicate_task_id) == duplicate_payloads[tenant_b]

        response = ChatResponse(
            session_id=task_a.session_id,
            request_id=task_a.request.request_id,
            tenant_id=task_a.tenant_id,
            reply="published",
        )
        result = WorkerResult(
            task_id=task_a.task_id,
            session_id=task_a.session_id,
            request_id=task_a.request.request_id,
            tenant_id=task_a.tenant_id,
            status="completed",
            session_state=task_a.session_state,
            response=response,
        )
        result.authority_proof = authority.sign_result(task_a, result)
        assert first_orchestrator.apply_worker_result(result) == response
        assert first_orchestrator.task_status(task_a.task_id, tenant_id=tenant_a)["status"] == "completed"
        assert postgres_store.task_store.load_task(tenant_a, task_a.task_id)["status"] == "completed"
        assert fast_store.task_store.load_task(tenant_a, task_a.task_id)["status"] == "completed"
        assert postgres_store.task_store.load_response(tenant_a, task_a.task_id) == response.to_dict()
        assert fast_store.task_store.load_response(tenant_a, task_a.task_id) == response.to_dict()

        durable_queued = postgres_store.task_store.load_task(tenant_b, task_b.task_id)
        assert durable_queued["status"] == "queued"
        fast_completed = dict(fast_store.task_store.load_task(tenant_b, task_b.task_id))
        fast_completed["status"] = "completed"
        fast_store.task_store.save_task(tenant_b, task_b.task_id, fast_completed)

        durable_completed = dict(
            postgres_store.task_store.load_task(tenant_b, task_durable_completed.task_id)
        )
        durable_completed["status"] = "completed"
        postgres_store.task_store.save_task(
            tenant_b, task_durable_completed.task_id, durable_completed
        )
        assert fast_store.task_store.load_task(
            tenant_b, task_durable_completed.task_id
        )["status"] == "queued"

        authority.revoke(task_revoked.delegation)
        incomplete_for_b = composite_store.task_store.list_incomplete_tasks(tenant_b)
        incomplete_ids_for_b = {item["task_id"] for item in incomplete_for_b}
        assert task_b.task_id in incomplete_ids_for_b
        assert task_durable_completed.task_id not in incomplete_ids_for_b

        recovered = _recover_api_tasks_in_new_process(
            postgres_dsn=os.environ["DADBOT_TEST_POSTGRES_DSN"],
            table_names=restart_names,
            authority_path=authority_path,
            token_secret=secret,
            redis_url=redis_url,
            redis_namespace=namespace,
        )
        assert [item["task_id"] for item in recovered] == [task_b.task_id]
        assert recovered[0]["tenant_id"] == tenant_b
        assert recovered[0]["attempt"] == 2
        assert recovered[0]["gateway"] == {
            "channel": "teams",
            "delivery_mode": "sync",
            "message_id": "composite-message",
        }
    finally:
        keys = list(fast_store._client.scan_iter(match=f"{namespace}:*"))
        if keys:
            fast_store._client.delete(*keys)
        fast_store._client.close()


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


def test_postgres_primitive_certification_does_not_enable_loop_resume(
    postgres_store, make_test_dadbot, monkeypatch,
):
    from dadbot.assistant_runtime import AssistantRuntime
    from dadbot.core.execution_receipt import ReceiptSigner

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    monkeypatch.setenv("DADBOT_EXECUTION_LEDGER_BACKEND", "postgres")
    bot = make_test_dadbot(document_store=postgres_store)
    bot.services.turn_orchestrator.control_plane.graph.configure_receipt_signer(
        ReceiptSigner(b"test-only-loop-receipt-key-000001"),
    )
    before = bot._execution_ledger.event_count()
    for options in ({"durable": True}, {"resume": True}):
        with pytest.raises(RuntimeError, match="non-file recovery is not certified"):
            AssistantRuntime(bot).run_agent_loop("goal", **options)
    assert bot._execution_ledger.event_count() == before


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
    import psycopg
    from psycopg import sql

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
