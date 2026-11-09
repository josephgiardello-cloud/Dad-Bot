import multiprocessing

import pytest

from dadbot.core.execution_contract import TurnResponse
from dadbot_system.api import create_api_app
from dadbot_system.contracts import ChatRequest, ChatResponse, ServiceConfig, WorkerResult, WorkerTask
from dadbot_system.delegation import AuthorityUnavailableError, SQLiteDelegationStore
from dadbot_system.orchestration import DadBotOrchestrator
from dadbot_system.security import ServiceTokenManager
from dadbot_system.state import InMemoryStateStore
from dadbot_system.worker import AUTHORITY_UNAVAILABLE, DadBotTaskProcessor


class _TenantBot:
    def __init__(self, tenant_id):
        self.config = type("Config", (), {"tenant_id": tenant_id})()

    async def execute_turn(self, request):
        return TurnResponse(reply="executed without verified authority")

    def snapshot_session_state(self):
        return {}


def _configured_authority(tmp_path):
    config = ServiceConfig()
    config.security.token_secret = "delegation-acceptance-test-secret"
    config.security.authority_sqlite_path = str(tmp_path / "authority.sqlite")
    SQLiteDelegationStore.initialize(config.security.authority_sqlite_path)
    authority = SQLiteDelegationStore(config.security.authority_sqlite_path, config.security.token_secret)
    authority.grant_membership(issuer="dadbot", subject="user", tenant_id="tenant-a", scopes={"write"})
    token = ServiceTokenManager(config.security.token_secret).issue(
        subject="user", tenant_id="tenant-a", scopes={"write"},
    )
    return config, authority, token


def _process_uncredentialed_task(result_queue):
    config = ServiceConfig()
    config.security.auth_required = True
    config.security.token_secret = "delegation-acceptance-test-secret"
    config.workers.max_retries = 0
    processor = DadBotTaskProcessor(
        config,
        bot_factory=lambda *, model_name, tenant_id: _TenantBot(tenant_id),
    )
    request = ChatRequest(session_id="same-session", tenant_id="tenant-a", user_input="hello")
    task = WorkerTask(session_id=request.session_id, tenant_id=request.tenant_id, request=request)
    result = processor.process(task)
    result_queue.put((result.status, result.error))


def test_authenticated_worker_rejects_task_without_delegation_in_fresh_process():
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_process_uncredentialed_task, args=(result_queue,))
    process.start()
    process.join(timeout=30)
    try:
        assert not process.is_alive(), "worker process did not finish"
        assert process.exitcode == 0
        status, error = result_queue.get(timeout=5)
        assert status == "failed", f"worker accepted an uncredentialed task: {status}"
        assert "author" in error.lower() or "delegation" in error.lower()
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        result_queue.close()
        result_queue.join_thread()


def test_authority_outage_preserves_queued_task_for_recovery():
    class Broker:
        def enqueue(self, task):
            self.task = task

    broker = Broker()
    store = InMemoryStateStore()
    orchestrator = DadBotOrchestrator(broker, state_store=store)
    request = ChatRequest(session_id="same-session", tenant_id="tenant-a", user_input="hello")
    task = orchestrator.submit_chat(request)
    config = ServiceConfig()
    constructed = []
    processor = DadBotTaskProcessor(config, bot_factory=lambda **kwargs: constructed.append(kwargs))

    result = processor.process(task)
    orchestrator.apply_worker_result(result)

    assert not constructed
    persisted = store.load_task(task.task_id)
    assert persisted is not None
    assert persisted["status"] == "queued"
    assert persisted["error"] == AUTHORITY_UNAVAILABLE
    assert store.load_response(task.task_id) is None
    restarted_broker = Broker()
    assert DadBotOrchestrator(restarted_broker, state_store=store).recover_pending_tasks() == 1
    assert restarted_broker.task.task_id == task.task_id


@pytest.mark.asyncio
async def test_direct_worker_execution_is_denied_without_authority():
    config = ServiceConfig()
    processor = DadBotTaskProcessor(config)
    request = ChatRequest(session_id="same-session", tenant_id="tenant-a", user_input="hello")

    with pytest.raises(RuntimeError, match="authority"):
        await processor._execute_request_async(_TenantBot("tenant-a"), request)


def test_authenticated_api_cannot_publish_identity_consistent_unverified_result(tmp_path):
    class Broker:
        def enqueue(self, task):
            self.task = task

    config, _authority, token = _configured_authority(tmp_path)
    store = InMemoryStateStore()
    broker = Broker()
    orchestrator = DadBotOrchestrator(broker, state_store=store)
    create_api_app(orchestrator, config=config)
    request = ChatRequest(session_id="same-session", tenant_id="tenant-a", user_input="hello")
    task = orchestrator.submit_chat(request, bearer_token=token)
    result = WorkerResult(
        task_id=task.task_id,
        session_id=task.session_id,
        request_id=request.request_id,
        tenant_id=task.tenant_id,
        status="completed",
        response=ChatResponse(
            session_id=task.session_id,
            request_id=request.request_id,
            tenant_id=task.tenant_id,
            reply="unverified",
        ),
    )

    assert orchestrator.apply_worker_result(result) is None
    persisted = store.load_task(task.task_id)
    assert persisted is not None
    assert persisted["status"] == "queued"
    assert store.load_response(task.task_id) is None


def test_authenticated_api_rejects_orchestrator_without_authority_enforcement():
    config = ServiceConfig()
    config.security.token_secret = "delegation-acceptance-test-secret"

    with pytest.raises(RuntimeError, match="authority-enforcing orchestrator"):
        create_api_app(object(), config=config)


@pytest.mark.parametrize("environment", ["production", "development", "test"])
def test_authenticated_api_rejects_missing_authority_at_startup(monkeypatch, environment):
    monkeypatch.setenv("DADBOT_ENVIRONMENT", environment)
    config = ServiceConfig()
    config.security.token_secret = "delegation-acceptance-test-secret"
    orchestrator = DadBotOrchestrator(object(), state_store=InMemoryStateStore())
    with pytest.raises(RuntimeError, match="authority"):
        create_api_app(orchestrator, config=config)


def test_authenticated_api_rejects_unavailable_configured_authority_at_startup(tmp_path):
    config = ServiceConfig()
    config.security.token_secret = "delegation-acceptance-test-secret"
    config.security.authority_sqlite_path = str(tmp_path / "missing.db")
    orchestrator = DadBotOrchestrator(object(), state_store=InMemoryStateStore())
    with pytest.raises(RuntimeError, match="authority"):
        create_api_app(orchestrator, config=config)
    assert not (tmp_path / "missing.db").exists()


def test_authenticated_api_does_not_requeue_tasks_without_authority_on_restart(tmp_path):
    class Broker:
        def __init__(self):
            self.tasks = []

        def enqueue(self, task):
            self.tasks.append(task)

    config, _authority, _token = _configured_authority(tmp_path)
    store = InMemoryStateStore()
    initial = DadBotOrchestrator(Broker(), state_store=store)
    task = initial.submit_chat(ChatRequest(session_id="same-session", tenant_id="tenant-a", user_input="hello"))
    broker = Broker()
    restarted = DadBotOrchestrator(broker, state_store=store)
    create_api_app(restarted, config=config)

    assert restarted.recover_pending_tasks() == 0
    assert broker.tasks == []
    persisted = store.load_task(task.task_id)
    assert persisted is not None
    assert persisted["status"] == "queued"


def test_synchronous_api_reports_authority_outage_without_losing_task(tmp_path):
    from fastapi.testclient import TestClient

    config, _authority, token = _configured_authority(tmp_path)
    worker_config = ServiceConfig()
    worker_config.security.token_secret = config.security.token_secret
    worker_config.security.authority_sqlite_path = str(tmp_path / "unavailable.sqlite")
    processor = DadBotTaskProcessor(worker_config)

    class Broker:
        def __init__(self):
            self.results = []

        def enqueue(self, task):
            self.results.append(processor.process(task))

        def get_result_nowait(self):
            return self.results.pop(0) if self.results else None

    store = InMemoryStateStore()
    app = create_api_app(DadBotOrchestrator(Broker(), state_store=store), config=config)
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/same-session/turn",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": "tenant-a", "user_input": "hello", "timeout_seconds": 2},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == AUTHORITY_UNAVAILABLE
    assert len(store.list_incomplete_tasks()) == 1


def test_direct_authenticated_orchestrator_rejects_submission_without_authority():
    store = InMemoryStateStore()
    orchestrator = DadBotOrchestrator(object(), state_store=store)
    orchestrator.require_authenticated_execution()
    request = ChatRequest(session_id="same-session", tenant_id="tenant-a", user_input="hello")

    with pytest.raises(AuthorityUnavailableError, match="authority"):
        orchestrator.submit_chat(request)
    assert store.list_incomplete_tasks() == []
