import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from dadbot.core.execution_contract import TurnResponse
from dadbot_system.api import create_api_app
from dadbot_system.contracts import ServiceConfig
from dadbot_system.delegation import AuthorityUnavailableError, SQLiteDelegationStore
from dadbot_system.orchestration import DadBotOrchestrator
from dadbot_system.security import AuthorizationError, ServiceTokenManager
from dadbot_system.state import InMemoryStateStore
from dadbot_system.worker import AUTHORITY_UNAVAILABLE, DadBotTaskProcessor

SECRET = "test-durable-authority-secret"


class WorkerBot:
    def __init__(self, tenant_id):
        self.config = type("Config", (), {"tenant_id": tenant_id})()

    def load_session_state_snapshot(self, snapshot):
        pass

    async def execute_turn(self, request):
        return TurnResponse(reply="authorized")

    def snapshot_session_state(self):
        return {}


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
    store.grant_membership(issuer="dadbot", subject="alice", tenant_id="tenant-a", scopes={"write"})
    config = ServiceConfig()
    config.security.token_secret = SECRET
    config.security.authority_sqlite_path = path
    broker = Broker()
    state = InMemoryStateStore()
    app = create_api_app(DadBotOrchestrator(broker, state_store=state), config=config)
    token = ServiceTokenManager(SECRET).issue(subject="alice", tenant_id="tenant-a", scopes={"write"})
    return store, app, broker, state, token


def test_authenticated_submission_issues_durable_delegation_verified_in_fresh_process(configured):
    store, app, broker, state, token = configured
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": "tenant-a", "user_input": "hello"},
        )
    assert response.status_code == 200
    task = broker.tasks[0]
    saved = state.load_task(task.task_id)
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
            json={"tenant_id": "tenant-a", "user_input": "hello"},
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
            json={"tenant_id": "tenant-a", "user_input": "hello"},
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
    store.revoke_membership(issuer="dadbot", subject="alice", tenant_id="tenant-a")
    with pytest.raises(AuthorizationError, match="membership"):
        store.verify(task.delegation, task, operation="execute")
    unavailable = SQLiteDelegationStore(store.path + ".missing", SECRET)
    with pytest.raises(AuthorityUnavailableError):
        unavailable.verify(task.delegation, task, operation="execute")


def test_nonmember_cannot_submit_authenticated_task(configured):
    _store, app, broker, _state, _token = configured
    token = ServiceTokenManager(SECRET).issue(subject="mallory", tenant_id="tenant-a", scopes={"write"})
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": "tenant-a", "user_input": "hello"},
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
            json={"tenant_id": "tenant-a", "user_input": "hello"},
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
            json={"tenant_id": "tenant-a", "user_input": "hello"},
        )
    saved = state.load_response(broker.tasks[0].task_id)
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
            json={"tenant_id": "tenant-a", "user_input": "hello"},
        )
    assert response.status_code == 200
    task = broker.tasks[0]
    store.grant_membership(issuer="dadbot", subject="alice", tenant_id="tenant-a", scopes={"write"})
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
            json={"tenant_id": "tenant-a", "user_input": "hello"},
        )
    assert response.status_code == 503
    assert broker.tasks == []


def test_real_worker_executes_with_verified_delegation(configured):
    store, app, broker, _state, token = configured
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": "tenant-a", "user_input": "hello"},
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


@pytest.mark.parametrize("failure", ["missing", "forged", "revoked", "expired", "outage", "modified"])
def test_real_worker_rejects_invalid_or_unavailable_authority(configured, failure, monkeypatch):
    store, app, broker, _state, token = configured
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": "tenant-a", "user_input": "hello"},
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
    assert constructed == []


def test_revocation_before_live_tool_prevents_effect(configured):
    from dadbot.core.tool_executor import execute_tool

    store, app, broker, _state, token = configured
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": "tenant-a", "user_input": "hello"},
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


def test_retry_rechecks_authority_before_executing_again(configured):
    store, app, broker, _state, token = configured
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": "tenant-a", "user_input": "hello"},
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
            json={"tenant_id": "tenant-a", "user_input": "hello"},
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
    persisted = state.load_task(task.task_id)
    assert persisted is not None
    assert persisted["status"] == "failed"
    assert persisted["error"] == "model unavailable"


def test_local_unauthenticated_worker_execution_remains_available():
    from dadbot_system.contracts import ChatRequest, WorkerTask

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
