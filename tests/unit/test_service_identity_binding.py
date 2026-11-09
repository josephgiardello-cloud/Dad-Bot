import pytest

from dadbot.core.execution_contract import TurnResponse
from dadbot_system.contracts import ChatRequest, ChatResponse, ServiceConfig, WorkerResult, WorkerTask
from dadbot_system.delegation import SQLiteDelegationStore, _execution_authority
from dadbot_system.orchestration import DadBotOrchestrator
from dadbot_system.security import ServiceTokenManager
from dadbot_system.state import InMemoryStateStore
from dadbot_system.worker import DadBotTaskProcessor


class Broker:
    def __init__(self):
        self.tasks = []

    def enqueue(self, task):
        self.tasks.append(task)

    def get_result_nowait(self):
        return None


class Bot:
    def __init__(self, tenant_id):
        self.config = type("Config", (), {"tenant_id": tenant_id})()
        self.requests = []
        self.snapshots = []

    def load_session_state_snapshot(self, snapshot):
        self.snapshots.append(snapshot)

    async def execute_turn(self, request):
        self.requests.append(request)
        return TurnResponse(reply="ok", should_end=False)

    def snapshot_session_state(self):
        return {}


def task_for(tenant="tenant-a", session="same-session"):
    request = ChatRequest(session_id=session, tenant_id=tenant, user_input="hello")
    return WorkerTask(session_id=session, tenant_id=tenant, request=request, session_state={"history": []})


@pytest.fixture
def worker_authority(tmp_path):
    config = ServiceConfig()
    config.security.token_secret = "worker-identity-test-secret"
    config.security.authority_sqlite_path = str(tmp_path / "authority.db")
    SQLiteDelegationStore.initialize(config.security.authority_sqlite_path)
    store = SQLiteDelegationStore(config.security.authority_sqlite_path, config.security.token_secret)
    tokens = ServiceTokenManager(config.security.token_secret)
    for tenant in ("tenant-a", "tenant-b"):
        store.grant_membership(issuer="dadbot", subject="worker-test", tenant_id=tenant, scopes={"write"})

    def authorize(task):
        token = tokens.issue(subject="worker-test", tenant_id=task.tenant_id, scopes={"write"})
        task.delegation = store.issue(token, task)
        return task

    return config, authorize


@pytest.mark.parametrize("field", ["tenant", "session"])
def test_worker_rejects_mismatched_task_before_constructing_bot(field, worker_authority):
    config, authorize = worker_authority
    task = authorize(task_for())
    if field == "tenant":
        task.request.tenant_id = "tenant-b"
    else:
        task.request.session_id = "another-session"
    created = []
    processor = DadBotTaskProcessor(config, bot_factory=lambda **kw: created.append(kw))

    result = processor.process(task)

    assert result.status == "failed"
    assert "authority" in result.error.lower()
    assert not created


def test_worker_preserves_identity_for_two_tenants_with_same_session(worker_authority):
    config, authorize = worker_authority
    bots = {}

    def factory(*, model_name, tenant_id):
        bots[tenant_id] = Bot(tenant_id)
        return bots[tenant_id]

    processor = DadBotTaskProcessor(config, bot_factory=factory)
    for tenant in ("tenant-a", "tenant-b"):
        assert processor.process(authorize(task_for(tenant))).status == "completed"
        assert bots[tenant].requests[0].context.tenant_id == tenant
        assert bots[tenant].requests[0].context.session_id == "same-session"
    assert bots["tenant-a"] is not bots["tenant-b"]


def test_worker_binds_real_dadbot_instances_for_two_tenants(make_test_dadbot, monkeypatch, worker_authority):
    config, authorize = worker_authority
    bots = {tenant: make_test_dadbot(tenant_id=tenant) for tenant in ("tenant-a", "tenant-b")}
    calls = []

    async def execute(request):
        calls.append((request.context.tenant_id, request.context.session_id))
        return TurnResponse(reply="ok", should_end=False)

    for bot in bots.values():
        monkeypatch.setattr(bot, "execute_turn", execute)
    processor = DadBotTaskProcessor(
        config,
        bot_factory=lambda *, model_name, tenant_id: bots[tenant_id],
    )

    for tenant in bots:
        task = task_for(tenant)
        task.session_state = {}
        assert processor.process(authorize(task)).status == "completed"
        token = _execution_authority.set(
            (SQLiteDelegationStore(config.security.authority_sqlite_path, config.security.token_secret), task)
        )
        try:
            bots[tenant].shutdown()
        finally:
            _execution_authority.reset(token)
    assert calls == [("tenant-a", "same-session"), ("tenant-b", "same-session")]


def test_worker_rejects_factory_returning_bot_for_another_tenant(worker_authority):
    config, authorize = worker_authority
    bot = Bot("tenant-b")
    processor = DadBotTaskProcessor(config, bot_factory=lambda **kw: bot)

    result = processor.process(authorize(task_for()))

    assert result.status == "failed"
    assert "tenant" in result.error.lower()
    assert not bot.snapshots
    assert not bot.requests


def test_worker_does_not_retry_factory_without_tenant_on_type_error(worker_authority):
    config, authorize = worker_authority
    calls = []

    def factory(*, model_name, **kwargs):
        calls.append(kwargs)
        if "tenant_id" in kwargs:
            raise TypeError("factory failed internally")
        return Bot("default")

    config.workers.max_retries = 0
    result = DadBotTaskProcessor(config, bot_factory=factory).process(authorize(task_for()))

    assert result.status == "failed"
    assert calls == [{"tenant_id": "tenant-a"}]


@pytest.mark.parametrize("field", ["tenant_id", "session_id"])
def test_recovery_rejects_conflicting_persisted_request_identity(field):
    store = InMemoryStateStore()
    broker = Broker()
    task = DadBotOrchestrator(broker, state_store=store).submit_chat(
        ChatRequest(session_id="same-session", tenant_id="tenant-a", user_input="hello")
    )
    saved = store.load_task(task.task_id)
    saved["request"][field] = "tenant-b" if field == "tenant_id" else "other-session"
    store.save_task(task.task_id, saved)
    recovered = Broker()

    with pytest.raises(ValueError, match="mismatch"):
        DadBotOrchestrator(recovered, state_store=store).recover_pending_tasks()
    assert not recovered.tasks


@pytest.mark.parametrize("field", ["tenant_id", "session_id", "request_id", "response_tenant"])
def test_orchestrator_rejects_mismatched_worker_result_without_state_change(field):
    store = InMemoryStateStore()
    orchestrator = DadBotOrchestrator(Broker(), state_store=store)
    task = orchestrator.submit_chat(ChatRequest(session_id="same-session", tenant_id="tenant-a", user_input="hello"))
    before = store.load_task(task.task_id)
    response = ChatResponse(
        session_id=task.session_id,
        request_id=task.request.request_id,
        tenant_id=task.tenant_id,
        reply="ok",
    )
    result = WorkerResult(
        task_id=task.task_id,
        session_id=task.session_id,
        request_id=task.request.request_id,
        tenant_id=task.tenant_id,
        status="completed",
        response=response,
    )
    if field == "response_tenant":
        result.response.tenant_id = "tenant-b"
    else:
        setattr(result, field, "different")

    with pytest.raises(ValueError, match="mismatch"):
        orchestrator.apply_worker_result(result)
    assert store.load_task(task.task_id) == before
    assert store.load_response(task.task_id) is None


def test_orchestrator_rejects_lossy_tenant_alias_before_state_or_enqueue():
    broker = Broker()
    orchestrator = DadBotOrchestrator(broker)
    with pytest.raises(ValueError, match="tenant"):
        orchestrator.submit_chat(ChatRequest(session_id="same-session", tenant_id="tenant/a", user_input="hello"))
    assert not broker.tasks


def test_orchestrator_rejects_alias_for_existing_tenant_session():
    orchestrator = DadBotOrchestrator(Broker())
    orchestrator.submit_chat(ChatRequest(session_id="same-session", tenant_id="tenant-a", user_input="hello"))

    with pytest.raises(ValueError, match="tenant"):
        orchestrator.session_events("same-session", tenant_id="tenant/a")


@pytest.mark.parametrize("tenant", ["tenant/a", "Tenant-A", " tenant-a", ""])
def test_request_deserialization_does_not_normalize_tenant_aliases(tenant):
    with pytest.raises(ValueError, match="tenant"):
        ChatRequest.from_dict({"session_id": "same-session", "tenant_id": tenant, "user_input": "hello"})


def test_worker_rechecks_cached_bot_tenant_before_next_turn(worker_authority):
    config, authorize = worker_authority
    bot = Bot("tenant-a")
    processor = DadBotTaskProcessor(config, bot_factory=lambda **kw: bot)
    assert processor.process(authorize(task_for())).status == "completed"
    bot.config.tenant_id = "tenant-b"

    result = processor.process(authorize(task_for()))

    assert result.status == "failed"
    assert len(bot.requests) == 1


def test_result_rejects_tampered_persisted_request_before_publication():
    store = InMemoryStateStore()
    orchestrator = DadBotOrchestrator(Broker(), state_store=store)
    task = orchestrator.submit_chat(ChatRequest(session_id="same-session", tenant_id="tenant-a", user_input="hello"))
    saved = store.load_task(task.task_id)
    saved["request"]["tenant_id"] = "tenant-b"
    store.save_task(task.task_id, saved)
    result = WorkerResult(
        task_id=task.task_id,
        session_id=task.session_id,
        request_id=task.request.request_id,
        tenant_id=task.tenant_id,
        status="completed",
        response=ChatResponse(
            session_id=task.session_id,
            request_id=task.request.request_id,
            tenant_id=task.tenant_id,
            reply="ok",
        ),
    )

    with pytest.raises(ValueError, match="mismatch"):
        orchestrator.apply_worker_result(result)
    assert store.load_task(task.task_id) == saved
    assert store.load_response(task.task_id) is None
