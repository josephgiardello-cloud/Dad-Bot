import multiprocessing
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from dadbot.core.execution_context import ExecutionTraceRecorder, bind_execution_trace
from dadbot.core.execution_contract import TurnResponse
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.action_mixin import DadBotActionMixin
from dadbot.core.ledger_backend import FileWALLedgerBackend
from dadbot.core.orchestrator import DadBotOrchestrator as CoreDadBotOrchestrator
from dadbot.core.persistence.checkpointer import SQLiteCheckpointer
from dadbot.core.tenant_session import tenant_session_scope
from dadbot.managers.conversation_persistence import ConversationPersistenceManager
from dadbot_system.api import create_api_app
from dadbot_system.contracts import ServiceConfig
from dadbot_system.delegation import (
    AuthorityUnavailableError,
    SQLiteDelegationStore,
    _execution_authority,
)
from dadbot_system.orchestration import DadBotOrchestrator
from dadbot_system.security import AuthorizationError, ServiceTokenManager
from dadbot_system.state import InMemoryStateStore
from dadbot_system.worker import AUTHORITY_UNAVAILABLE, DadBotTaskProcessor, worker_process_main

SECRET = "test-durable-authority-secret"
TENANT_A = "t1_" + "a" * 26
TENANT_B = "t1_" + "b" * 26


class WorkerBot:
    def __init__(self, tenant_id):
        self.config = type("Config", (), {"tenant_id": tenant_id})()

    def load_session_state_snapshot(self, snapshot):
        pass

    async def execute_turn(self, request):
        return TurnResponse(reply="authorized")

    def snapshot_session_state(self):
        return {}


class ProcessWorkerBotFactory:
    def __init__(self, created_queue):
        self.created_queue = created_queue

    def __call__(self, *, model_name, tenant_id):
        self.created_queue.put(tenant_id)
        return WorkerBot(tenant_id)


def _submit_worker_chat(client, broker, tenant_id, bearer_token):
    response = client.post(
        "/v1/sessions/shared/chat",
        headers={"Authorization": f"Bearer {bearer_token}"},
        json={"tenant_id": tenant_id, "user_input": "hello"},
    )
    assert response.status_code == 200
    return broker.tasks[-1]


def _execute_worker_task(task, request_queue, result_queue):
    request_queue.put(task)
    result = result_queue.get(timeout=10)
    assert result.task_id == task.task_id
    assert result.status == "completed"


class Broker:
    def __init__(self):
        self.tasks = []

    def enqueue(self, task):
        self.tasks.append(task)

    def get_result_nowait(self):
        return None


def _verify_in_child(path, task, output):
    store = SQLiteDelegationStore(path, SECRET)
    try:
        store.verify(task.delegation, task, operation="execute")
        config = ServiceConfig()
        config.security.token_secret = SECRET
        config.security.authority_sqlite_path = path
        result = DadBotTaskProcessor(
            config, bot_factory=lambda *, model_name, tenant_id: WorkerBot(tenant_id)
        ).process(task)
        output.put(("verified", result.status, result.error))
    except Exception as exc:
        output.put(type(exc).__name__)


@pytest.fixture
def configured(tmp_path):
    path = str(tmp_path / "authority.sqlite")
    SQLiteDelegationStore.initialize(path)
    store = SQLiteDelegationStore(path, SECRET)
    store.grant_membership(issuer="dadbot", subject="alice", tenant_id=TENANT_A, scopes={"write"})
    config = ServiceConfig()
    config.security.token_secret = SECRET
    config.security.authority_sqlite_path = path
    broker = Broker()
    state = InMemoryStateStore()
    app = create_api_app(DadBotOrchestrator(broker, state_store=state), config=config)
    token = ServiceTokenManager(SECRET).issue(subject="alice", tenant_id=TENANT_A, scopes={"write"})
    return store, app, broker, state, token


def test_authenticated_submission_issues_durable_delegation_verified_in_fresh_process(configured):
    store, app, broker, state, token = configured
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": TENANT_A, "user_input": "hello"},
        )
    assert response.status_code == 200
    task = broker.tasks[0]
    saved = state.owned_tasks.load_task(task.tenant_id, task.task_id)
    assert saved is not None
    assert saved["delegation"] == task.delegation
    store.verify(task.delegation, task, operation="execute")
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(target=_verify_in_child, args=(store.path, task, output))
    process.start()
    process.join(20)
    try:
        assert process.exitcode == 0
        assert output.get(timeout=5) == ("verified", "completed", "")
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        output.close()
        output.join_thread()


def test_revocation_invalidates_durable_submission_authority(configured):
    store, app, broker, _state, token = configured
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": TENANT_A, "user_input": "hello"},
        )
    assert response.status_code == 200
    task = broker.tasks[0]
    store.revoke(task.delegation)
    with pytest.raises(AuthorizationError, match="revoked"):
        store.verify(task.delegation, task, operation="execute")


def test_authority_forgery_identity_mismatch_and_outage_fail_closed(configured):
    store, app, broker, _state, token = configured
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": TENANT_A, "user_input": "hello"},
        )
    assert response.status_code == 200
    task = broker.tasks[0]
    with pytest.raises(AuthorizationError, match="signature"):
        store.verify(task.delegation + "f", task, operation="execute")
    for changed in (
        replace(task, tenant_id="tenant-b"),
        replace(task, task_id="other"),
        replace(task, session_id="other"),
        replace(task, request=replace(task.request, request_id="other")),
    ):
        with pytest.raises(AuthorizationError, match="mismatch"):
            store.verify(task.delegation, changed, operation="execute")
    store.revoke_membership(issuer="dadbot", subject="alice", tenant_id=TENANT_A)
    with pytest.raises(AuthorizationError, match="membership"):
        store.verify(task.delegation, task, operation="execute")
    unavailable = SQLiteDelegationStore(store.path + ".missing", SECRET)
    with pytest.raises(AuthorityUnavailableError):
        unavailable.verify(task.delegation, task, operation="execute")


def test_nonmember_cannot_submit_authenticated_task(configured):
    _store, app, broker, _state, _token = configured
    token = ServiceTokenManager(SECRET).issue(subject="mallory", tenant_id=TENANT_A, scopes={"write"})
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": TENANT_A, "user_input": "hello"},
        )
    assert response.status_code == 403
    assert broker.tasks == []


@pytest.mark.parametrize(
    "path",
    ["/v1/sessions/shared/stream", "/v1/channels/chat/sessions/shared/ingest"],
)
def test_other_async_ingress_routes_issue_delegation(configured, path):
    store, app, broker, _state, token = configured
    with TestClient(app) as client:
        response = client.post(
            path,
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": TENANT_A, "user_input": "hello"},
        )
    assert response.status_code == 200
    store.verify(broker.tasks[0].delegation, broker.tasks[0], operation="execute")


@pytest.mark.parametrize("revoke_before_publication", [False, True])
def test_synchronous_turn_executes_and_publishes_with_verified_authority(configured, revoke_before_publication):
    store, _app, _broker, state, token = configured
    config = ServiceConfig()
    config.security.token_secret = SECRET
    config.security.authority_sqlite_path = store.path
    config.workers.max_retries = 0

    class ExecutingBroker(Broker):
        def __init__(self):
            super().__init__()
            self.results = []

        def enqueue(self, task):
            super().enqueue(task)
            processor = DadBotTaskProcessor(
                config, bot_factory=lambda *, model_name, tenant_id: WorkerBot(tenant_id)
            )
            with ThreadPoolExecutor(max_workers=1) as executor:
                self.results.append(executor.submit(processor.process, task).result())
            if revoke_before_publication:
                store.revoke(task.delegation)

        def get_result_nowait(self):
            return self.results.pop(0) if self.results else None

    broker = ExecutingBroker()
    app = create_api_app(DadBotOrchestrator(broker, state_store=state), config=config)
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/turn",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": TENANT_A, "user_input": "hello"},
        )
    saved = state.owned_tasks.load_response(broker.tasks[0].tenant_id, broker.tasks[0].task_id)
    if revoke_before_publication:
        assert response.status_code == 503
        assert saved is None
    else:
        assert response.status_code == 200
        assert response.json()["response"]["reply"] == "authorized"
        assert saved is not None
        assert saved["reply"] == "authorized"


def test_policy_change_and_expiry_invalidate_authority(configured, monkeypatch):
    store, app, broker, _state, token = configured
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": TENANT_A, "user_input": "hello"},
        )
    assert response.status_code == 200
    task = broker.tasks[0]
    store.grant_membership(issuer="dadbot", subject="alice", tenant_id=TENANT_A, scopes={"write"})
    with pytest.raises(AuthorizationError, match="policy changed"):
        store.verify(task.delegation, task, operation="execute")
    monkeypatch.setattr("dadbot_system.delegation.time.time", lambda: 10**12)
    with pytest.raises(AuthorizationError, match="expired"):
        store.verify(task.delegation, task, operation="execute")


def test_submission_authority_outage_returns_503(configured):
    store, app, broker, _state, token = configured
    from pathlib import Path

    Path(store.path).rename(store.path + ".offline")
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": TENANT_A, "user_input": "hello"},
        )
    assert response.status_code == 503
    assert broker.tasks == []


def test_real_worker_executes_with_verified_delegation(configured):
    store, app, broker, _state, token = configured
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": TENANT_A, "user_input": "hello"},
        )
    assert response.status_code == 200
    config = ServiceConfig()
    config.security.token_secret = SECRET
    config.security.authority_sqlite_path = store.path
    processor = DadBotTaskProcessor(
        config, bot_factory=lambda *, model_name, tenant_id: WorkerBot(tenant_id)
    )
    result = processor.process(broker.tasks[0])
    assert result.status == "completed"
    assert result.response is not None
    assert result.response.reply == "authorized"
    assert result.authority_proof


def test_session_termination_revokes_inflight_worker_authority_and_evicts_cached_bot(configured):
    store, app, broker, state, token = configured
    store.grant_membership(issuer="dadbot", subject="alice", tenant_id=TENANT_B, scopes={"write"})
    other_tenant_token = ServiceTokenManager(SECRET).issue(
        subject="alice",
        tenant_id=TENANT_B,
        scopes={"write"},
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": TENANT_A, "user_input": "hello"},
        )
        other_response = client.post(
            "/v1/sessions/other/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": TENANT_A, "user_input": "hello"},
        )
        other_tenant_response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {other_tenant_token}"},
            json={"tenant_id": TENANT_B, "user_input": "hello"},
        )
    assert response.status_code == other_response.status_code == other_tenant_response.status_code == 200
    task, other_task, other_tenant_task = broker.tasks
    entered = threading.Event()
    release = threading.Event()

    class BlockingWorkerBot(WorkerBot):
        async def execute_turn(self, request):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test worker was not released")
            from dadbot_system.delegation import require_live_authority

            require_live_authority()
            return TurnResponse(reply="authorized")

    config = ServiceConfig()
    config.security.token_secret = SECRET
    config.security.authority_sqlite_path = store.path
    processor = DadBotTaskProcessor(
        config,
        bot_factory=lambda *, model_name, tenant_id: BlockingWorkerBot(tenant_id),
    )
    orchestrator = DadBotOrchestrator(broker, state_store=state)
    orchestrator.require_authenticated_execution()
    orchestrator.configure_delegation_store(store)

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(processor.process, task)
        assert entered.wait(5)
        orchestrator.terminate_session("shared", TENANT_A)
        release.set()
        result = pending.result(timeout=10)

    assert result.status == "failed"
    assert result.error == AUTHORITY_UNAVAILABLE
    assert processor._bots == {}
    with pytest.raises(AuthorizationError, match="revoked"):
        store.verify(task.delegation, task, operation="execute")
    store.verify(other_task.delegation, other_task, operation="execute")
    store.verify(other_tenant_task.delegation, other_tenant_task, operation="execute")


def test_idle_worker_process_invalidates_only_revoked_tenant_session_cache(configured):
    store, _app, broker, state, token = configured
    store.grant_membership(issuer="dadbot", subject="alice", tenant_id=TENANT_B, scopes={"write"})
    config = ServiceConfig()
    config.security.auth_required = True
    config.security.token_secret = SECRET
    config.security.authority_sqlite_path = store.path
    config.workers.max_retries = 0
    orchestrator = DadBotOrchestrator(broker, state_store=state)
    app = create_api_app(orchestrator, config=config)
    token_b = ServiceTokenManager(SECRET).issue(subject="alice", tenant_id=TENANT_B, scopes={"write"})

    context = multiprocessing.get_context("spawn")
    request_queue = context.Queue()
    result_queue = context.Queue()
    event_queue = context.Queue()
    created_queue = context.Queue()
    process = context.Process(
        target=worker_process_main,
        args=(
            config.to_dict(),
            request_queue,
            result_queue,
            event_queue,
            None,
        ),
        kwargs={"bot_factory": ProcessWorkerBotFactory(created_queue)},
    )
    process.start()
    try:
        event_queue.get(timeout=10)
        with TestClient(app) as client:
            task_a = _submit_worker_chat(client, broker, TENANT_A, token)
            _execute_worker_task(task_a, request_queue, result_queue)
            assert created_queue.get(timeout=5) == TENANT_A

            task_b = _submit_worker_chat(client, broker, TENANT_B, token_b)
            _execute_worker_task(task_b, request_queue, result_queue)
            assert created_queue.get(timeout=5) == TENANT_B

            orchestrator.terminate_session("shared", TENANT_A)
            time.sleep(0.25)

            next_a = _submit_worker_chat(client, broker, TENANT_A, token)
            _execute_worker_task(next_a, request_queue, result_queue)
            assert created_queue.get(timeout=5) == TENANT_A

            next_b = _submit_worker_chat(client, broker, TENANT_B, token_b)
            _execute_worker_task(next_b, request_queue, result_queue)
            with pytest.raises(queue.Empty):
                created_queue.get(timeout=0.1)
    finally:
        request_queue.put(None)
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        for channel in (request_queue, result_queue, event_queue, created_queue):
            channel.close()
            channel.join_thread()
    assert process.exitcode == 0


def test_worker_rejects_tasks_when_session_revocation_schema_is_unavailable(configured):
    import sqlite3
    from contextlib import closing

    store, app, broker, _state, token = configured
    with TestClient(app) as client:
        for _ in range(2):
            response = client.post(
                "/v1/sessions/shared/chat",
                headers={"Authorization": f"Bearer {token}"},
                json={"tenant_id": TENANT_A, "user_input": "hello"},
            )
            assert response.status_code == 200
    first_task, second_task = broker.tasks
    created = []
    config = ServiceConfig()
    config.security.token_secret = SECRET
    config.security.authority_sqlite_path = store.path
    processor = DadBotTaskProcessor(
        config,
        bot_factory=lambda *, model_name, tenant_id: created.append(tenant_id) or WorkerBot(tenant_id),
    )
    assert processor.process(first_task).status == "completed"
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute("DROP TABLE session_revocations")

    result = processor.process(second_task)
    assert result.status == "failed"
    assert result.error == AUTHORITY_UNAVAILABLE
    assert created == [TENANT_A]
    assert processor._bots == {}


def test_authority_initializer_upgrades_partial_schema_without_resetting_state(configured):
    import sqlite3
    from contextlib import closing

    store, app, broker, _state, token = configured
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": TENANT_A, "user_input": "hello"},
        )
    assert response.status_code == 200
    task = broker.tasks[0]
    original_identity = store.storage_identity()
    original_revision = store.session_revocation_cursor()

    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute("DROP TABLE session_revocations")
    with pytest.raises(AuthorityUnavailableError):
        store.check_available()

    SQLiteDelegationStore.initialize(store.path)
    store.check_available()
    assert store.storage_identity() == original_identity
    assert store.session_revocation_cursor() == original_revision
    store.verify(task.delegation, task, operation="execute")

    SQLiteDelegationStore.initialize(store.path)
    assert store.storage_identity() == original_identity
    store.check_available()
    assert store.revoke_session(tenant_id=TENANT_A, session_id=task.session_id) == 1
    assert store.session_revocations_after(original_revision) == [
        (original_revision + 1, TENANT_A, task.session_id)
    ]
    with pytest.raises(AuthorizationError, match="revoked"):
        store.verify(task.delegation, task, operation="execute")


def test_interrupted_session_termination_rolls_back_revocation_and_keeps_session(configured):
    import sqlite3
    from contextlib import closing

    store, app, broker, state, token = configured
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": TENANT_A, "user_input": "hello"},
        )
    assert response.status_code == 200
    task = broker.tasks[0]
    orchestrator = DadBotOrchestrator(broker, state_store=state)
    orchestrator.require_authenticated_execution()
    orchestrator.configure_delegation_store(store)
    session = orchestrator.session("shared", TENANT_A)
    revision = store.session_revocation_cursor()
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute(
            """
            CREATE TRIGGER reject_session_revocation
            BEFORE INSERT ON session_revocations
            BEGIN
                SELECT RAISE(ABORT, 'simulated interrupted termination');
            END
            """
        )

    with pytest.raises(AuthorityUnavailableError, match="authority store unavailable"):
        orchestrator.terminate_session("shared", TENANT_A)

    assert orchestrator.session("shared", TENANT_A) is session
    assert store.session_revocation_cursor() == revision
    assert store.session_revocations_after(revision) == []
    store.verify(task.delegation, task, operation="execute")


def test_revoked_delegation_cannot_restore_tenant_session_checkpoint(configured, tmp_path):
    store, app, broker, _state, token = configured
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": TENANT_A, "user_input": "hello"},
        )
    assert response.status_code == 200
    task = broker.tasks[0]
    trace_id = task.request.request_id
    authority_token = _execution_authority.set((store, task))
    try:
        ledger = ExecutionLedger(
            backend=FileWALLedgerBackend(
                Path(tmp_path) / "execution-ledger.jsonl",
                security_mode="AUTHENTICATED",
                tenant_id=TENANT_A,
            ),
        )
        ledger.load_from_backend()
        checkpointer = SQLiteCheckpointer(
            str(Path(tmp_path) / "checkpoints.sqlite"),
            default_tenant_id=TENANT_A,
            prune_every=0,
        )
        checkpointer.save_checkpoint(
            task.session_id,
            trace_id,
            {
                "checkpoint_hash": "checkpoint-hash",
                "prev_checkpoint_hash": "",
                "status": "ok",
                "state": {},
                "metadata": {},
            },
            {},
            tenant_id=TENANT_A,
        )
        assert checkpointer.load_checkpoint(
            task.session_id,
            trace_id,
            tenant_id=TENANT_A,
        )["metadata"]["tenant_id"] == TENANT_A

        class PersistenceBot:
            turn_orchestrator = SimpleNamespace(
                control_plane=SimpleNamespace(ledger=ledger),
            )
            config = SimpleNamespace(tenant_id=TENANT_A)
            _dependency_overrides = {"storage_security_mode": "AUTHENTICATED"}
            _tenant_document_store = None
            _current_turn_time_base = None
            SESSION_LOG_DIR = Path(tmp_path)

            @staticmethod
            def snapshot_session_state():
                return {}

        manager = ConversationPersistenceManager(PersistenceBot())
        recorder = ExecutionTraceRecorder(trace_id=trace_id, prompt="checkpoint recovery")
        with tenant_session_scope(TENANT_A, task.session_id), bind_execution_trace(recorder, required=True):
            manager.persist_graph_checkpoint(
                {
                    "trace_id": trace_id,
                    "session_id": task.session_id,
                    "stage": "save",
                    "status": "after",
                },
            )
            checkpoint = manager.load_latest_graph_checkpoint(
                trace_id,
                session_id=task.session_id,
                tenant_id=TENANT_A,
            )
        assert checkpoint is not None

        store.revoke_session(tenant_id=TENANT_A, session_id=task.session_id)
        recorder = ExecutionTraceRecorder(trace_id=trace_id, prompt="checkpoint recovery")
        with tenant_session_scope(TENANT_A, task.session_id), bind_execution_trace(recorder, required=True):
            with pytest.raises(AuthorizationError, match="revoked"):
                checkpointer.load_checkpoint(
                    task.session_id,
                    trace_id,
                    tenant_id=TENANT_A,
                )
            with pytest.raises(AuthorizationError, match="revoked"):
                manager.resume_graph_checkpoint(
                    trace_id,
                    session_id=task.session_id,
                    tenant_id=TENANT_A,
                )
    finally:
        _execution_authority.reset(authority_token)

    recorder = ExecutionTraceRecorder(trace_id=trace_id, prompt="checkpoint recovery")
    with tenant_session_scope(TENANT_A, task.session_id), bind_execution_trace(recorder, required=True):
        with pytest.raises(AuthorizationError, match="authority is missing"):
            manager.load_latest_graph_checkpoint(
                trace_id,
                session_id=task.session_id,
                tenant_id=TENANT_A,
            )
    orchestrator = object.__new__(CoreDadBotOrchestrator)
    orchestrator.bot = SimpleNamespace(
        _dependency_overrides={"storage_security_mode": "AUTHENTICATED"},
    )
    with pytest.raises(AuthorizationError, match="authority is missing"):
        orchestrator._require_checkpoint_authority(TENANT_A)

@pytest.mark.parametrize("failure", ["missing", "forged", "revoked", "expired", "outage", "modified"])
def test_real_worker_rejects_invalid_or_unavailable_authority(configured, failure, monkeypatch):
    store, app, broker, _state, token = configured
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": TENANT_A, "user_input": "hello"},
        )
    assert response.status_code == 200
    task = broker.tasks[0]
    config = ServiceConfig()
    config.security.token_secret = SECRET
    config.security.authority_sqlite_path = store.path
    if failure == "missing":
        task.delegation = ""
    elif failure == "forged":
        task.delegation += "x"
    elif failure == "revoked":
        store.revoke(task.delegation)
    elif failure == "expired":
        monkeypatch.setattr("dadbot_system.delegation.time.time", lambda: 10**12)
    elif failure == "outage":
        config.security.authority_sqlite_path += ".missing"
    else:
        task.request.user_input = "modified"
    constructed = []
    processor = DadBotTaskProcessor(config, bot_factory=lambda **kwargs: constructed.append(kwargs))
    result = processor.process(task)
    assert result.status == "failed"
    assert result.error == AUTHORITY_UNAVAILABLE
    assert result.retryable is (failure == "outage")
    assert constructed == []


def test_revocation_before_live_tool_prevents_effect(configured):
    from dadbot.core.tool_executor import execute_tool

    store, app, broker, _state, token = configured
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": TENANT_A, "user_input": "hello"},
        )
    assert response.status_code == 200
    task = broker.tasks[0]
    effects = []

    class RevokingBot(WorkerBot):
        async def execute_turn(self, request):
            store.revoke(task.delegation)
            execute_tool(tool_name="effect", executor=lambda: effects.append("executed"))
            return TurnResponse(reply="unexpected")

    config = ServiceConfig()
    config.security.token_secret = SECRET
    config.security.authority_sqlite_path = store.path
    processor = DadBotTaskProcessor(
        config, bot_factory=lambda *, model_name, tenant_id: RevokingBot(tenant_id)
    )
    result = processor.process(task)
    assert result.status == "failed"
    assert result.error == AUTHORITY_UNAVAILABLE
    assert effects == []


def test_authenticated_runtime_tool_requires_authority_but_trusted_local_remains_supported():
    effects = []
    authenticated_runtime = DadBotActionMixin()
    authenticated_runtime._dependency_overrides = {"storage_security_mode": "AUTHENTICATED"}

    with pytest.raises(AuthorizationError, match="authority is required"):
        authenticated_runtime.execute_tool(
            tool_name="set_reminder",
            executor=lambda: effects.append("authenticated"),
        )

    trusted_local_runtime = DadBotActionMixin()
    trusted_local_runtime._dependency_overrides = {"storage_security_mode": "TRUSTED_LOCAL"}
    trusted_local_runtime.execute_tool(
        tool_name="set_reminder",
        executor=lambda: effects.append("trusted-local"),
    )

    assert effects == ["trusted-local"]


def test_task_authority_digest_survives_request_serialization_round_trip():
    from dadbot_system.contracts import ChatRequest, WorkerTask
    from dadbot_system.delegation import SQLiteDelegationStore

    request = ChatRequest(session_id="round-trip", user_input="hello", tenant_id=TENANT_A)
    task = WorkerTask(session_id=request.session_id, request=request, tenant_id=TENANT_A)
    restored_request = ChatRequest.from_dict(request.to_dict())
    restored_task = WorkerTask(
        session_id=task.session_id,
        request=restored_request,
        tenant_id=task.tenant_id,
        session_state=task.session_state,
        task_id=task.task_id,
    )

    assert SQLiteDelegationStore._task_digest(task) == SQLiteDelegationStore._task_digest(restored_task)


def test_retry_rechecks_authority_before_executing_again(configured):
    store, app, broker, _state, token = configured
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": TENANT_A, "user_input": "hello"},
        )
    assert response.status_code == 200
    task = broker.tasks[0]
    calls = []

    class RetryBot(WorkerBot):
        async def execute_turn(self, request):
            calls.append("attempt")
            store.revoke(task.delegation)
            raise RuntimeError("retryable failure")

    config = ServiceConfig()
    config.security.token_secret = SECRET
    config.security.authority_sqlite_path = store.path
    config.workers.max_retries = 2
    config.workers.retry_backoff_seconds = 0
    processor = DadBotTaskProcessor(
        config, bot_factory=lambda *, model_name, tenant_id: RetryBot(tenant_id)
    )
    result = processor.process(task)
    assert result.status == "failed"
    assert result.error == AUTHORITY_UNAVAILABLE
    assert calls == ["attempt"]


def test_authorized_worker_failure_is_published_as_failure(configured):
    store, app, broker, state, token = configured
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": TENANT_A, "user_input": "hello"},
        )
    assert response.status_code == 200
    task = broker.tasks[0]

    class FailingBot(WorkerBot):
        async def execute_turn(self, request):
            raise RuntimeError("model unavailable")

    config = ServiceConfig()
    config.security.token_secret = SECRET
    config.security.authority_sqlite_path = store.path
    config.workers.max_retries = 0
    processor = DadBotTaskProcessor(
        config, bot_factory=lambda *, model_name, tenant_id: FailingBot(tenant_id)
    )
    result = processor.process(task)
    assert result.error == "model unavailable"
    orchestrator = DadBotOrchestrator(Broker(), state_store=state)
    orchestrator.require_authenticated_execution()
    orchestrator.configure_delegation_store(store)
    assert orchestrator.apply_worker_result(result) is None
    persisted = state.owned_tasks.load_task(task.tenant_id, task.task_id)
    assert persisted is not None
    assert persisted["status"] == "failed"
    assert persisted["error"] == "model unavailable"


def test_local_unauthenticated_worker_execution_remains_available(monkeypatch):
    from dadbot_system.contracts import ChatRequest, WorkerTask

    monkeypatch.setenv("DADBOT_ENVIRONMENT", "test")
    monkeypatch.setenv("DADBOT_ALLOW_LEGACY_DEFAULT_TENANT", "true")
    config = ServiceConfig()
    config.security.auth_required = False
    config.workers.max_retries = 0
    request = ChatRequest(session_id="local", user_input="hello")
    task = WorkerTask(session_id=request.session_id, request=request)
    processor = DadBotTaskProcessor(
        config, bot_factory=lambda *, model_name, tenant_id: WorkerBot(tenant_id)
    )
    result = processor.process(task)
    assert result.status == "completed"
    assert result.response is not None
    assert result.response.reply == "authorized"
