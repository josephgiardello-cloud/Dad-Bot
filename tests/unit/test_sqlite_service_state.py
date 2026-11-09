import json
import multiprocessing
import sqlite3
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from dadbot.app_runtime import build_service_state_store
from dadbot.core.execution_contract import TurnResponse
from dadbot_system.api import create_api_app
from dadbot_system.contracts import ChatRequest, ServiceConfig
from dadbot_system.delegation import AuthorityUnavailableError, SQLiteDelegationStore
from dadbot_system.orchestration import DadBotOrchestrator
from dadbot_system.security import AuthorizationError, ServiceTokenManager
from dadbot_system.state import NamespacedStateStore, SQLiteStateStore
from dadbot_system.worker import DadBotTaskProcessor

SECRET = "sqlite-state-integration-secret"


class Broker:
    def __init__(self):
        self.tasks = []

    def enqueue(self, task):
        self.tasks.append(task)

    def get_result_nowait(self):
        return None


def _read_persisted_task(path, task_id, output):
    task = SQLiteStateStore(path).load_task(task_id)
    output.put(task)


def _read_namespaced_task(path, namespace, task_id, output):
    task = NamespacedStateStore(SQLiteStateStore(path), namespace).load_task(task_id)
    output.put(task)


def _recover_namespaced_tasks(path, output):
    broker = Broker()
    try:
        count = DadBotOrchestrator(broker, state_store=SQLiteStateStore(path)).recover_pending_tasks()
        output.put(("recovered", count, [(task.tenant_id, task.task_id) for task in broker.tasks]))
    except Exception as exc:
        output.put(("error", type(exc).__name__, str(exc)))


def test_task_namespace_isolated_across_processes(tmp_path):
    path = str(tmp_path / "state.db")
    store = NamespacedStateStore(SQLiteStateStore(path), "tenant:tenant-a")
    store.save_task("same-task", {"tenant_id": "tenant-a", "marker": "synthetic-private"})
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(
        target=_read_namespaced_task,
        args=(path, "tenant:tenant-b", "same-task", output),
    )
    process.start()
    try:
        process.join(20)
        assert process.exitcode == 0
        assert output.get(timeout=5) is None
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        output.close()
        output.join_thread()


def test_identical_task_ids_do_not_overwrite_another_namespace(tmp_path):
    state = SQLiteStateStore(str(tmp_path / "state.db"))
    first = NamespacedStateStore(state, "tenant:tenant-a")
    second = NamespacedStateStore(state, "tenant:tenant-b")
    first.save_task("same-task", {"tenant_id": "tenant-a", "marker": "first"})
    second.save_task("same-task", {"tenant_id": "tenant-b", "marker": "second"})

    assert first.load_task("same-task") == {"tenant_id": "tenant-a", "marker": "first"}
    assert second.load_task("same-task") == {"tenant_id": "tenant-b", "marker": "second"}


def test_task_result_and_response_are_namespace_owned(tmp_path):
    state = SQLiteStateStore(str(tmp_path / "state.db"))
    first = NamespacedStateStore(state, "tenant:tenant-a")
    second = NamespacedStateStore(state, "tenant:tenant-b")
    first.save_task("same-task", {"tenant_id": "tenant-a", "status": "queued"})
    second.save_task("same-task", {"tenant_id": "tenant-b", "status": "queued"})

    first.save_task_result(
        "same-task", {"tenant_id": "tenant-a", "status": "completed"}, {"reply": "first"}
    )
    second.save_task_result(
        "same-task", {"tenant_id": "tenant-b", "status": "completed"}, {"reply": "second"}
    )

    assert first.load_response("same-task") == {"reply": "first"}
    assert second.load_response("same-task") == {"reply": "second"}
    assert first.load_task("same-task")["tenant_id"] == "tenant-a"
    assert second.load_task("same-task")["tenant_id"] == "tenant-b"


def test_incomplete_task_listing_filters_before_limit(tmp_path):
    state = SQLiteStateStore(str(tmp_path / "state.db"))
    first = NamespacedStateStore(state, "tenant:tenant-a")
    second = NamespacedStateStore(state, "tenant:tenant-b")
    first.save_task(
        "a", {"task_id": "a", "tenant_id": "tenant-a", "status": "queued", "accepted_at": "2020"}
    )
    second.save_task(
        "b", {"task_id": "b", "tenant_id": "tenant-b", "status": "queued", "accepted_at": "2021"}
    )

    assert second.list_incomplete_tasks(limit=1) == [
        {"task_id": "b", "tenant_id": "tenant-b", "status": "queued", "accepted_at": "2021"}
    ]


def test_legacy_task_is_quarantined_without_implicit_ownership(tmp_path):
    path = str(tmp_path / "state.db")
    state = SQLiteStateStore(path)
    request = ChatRequest(session_id="same-session", tenant_id="tenant-a", user_input="hello")
    legacy = {
        "task_id": "legacy",
        "tenant_id": "tenant-a",
        "session_id": request.session_id,
        "request_id": request.request_id,
        "request": request.to_dict(),
        "status": "queued",
        "accepted_at": "2020-01-01T00:00:00+00:00",
    }
    state.save_task("legacy", legacy)
    broker = Broker()

    with pytest.raises(RuntimeError, match="(?i)(legacy|unqualified|quarantin)"):
        DadBotOrchestrator(broker, state_store=state).recover_pending_tasks()
    assert not broker.tasks
    with sqlite3.connect(path) as connection:
        row = connection.execute("SELECT payload FROM tasks WHERE id=?", ("legacy",)).fetchone()
    assert row is not None
    assert json.loads(row[0]) == legacy


def test_raw_store_rejects_unqualified_task_lookup(tmp_path):
    state = SQLiteStateStore(str(tmp_path / "state.db"))
    NamespacedStateStore(state, "tenant:tenant-a").save_task(
        "same-task", {"tenant_id": "tenant-a", "marker": "synthetic-private"}
    )

    with pytest.raises((TypeError, ValueError, PermissionError)):
        state.load_task("same-task")


def test_fresh_process_recovers_same_task_id_for_both_owners(tmp_path):
    path = str(tmp_path / "state.db")
    state = SQLiteStateStore(path)
    for tenant in ("tenant-a", "tenant-b"):
        request = ChatRequest(session_id="same-session", tenant_id=tenant, user_input="hello")
        NamespacedStateStore(state, f"tenant:{tenant}").save_task(
            "same-task",
            {
                "task_id": "same-task",
                "tenant_id": tenant,
                "session_id": request.session_id,
                "request_id": request.request_id,
                "request": request.to_dict(),
                "session_state": {},
                "status": "queued",
                "accepted_at": "2020-01-01T00:00:00+00:00",
            },
        )
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(target=_recover_namespaced_tasks, args=(path, output))
    process.start()
    try:
        process.join(20)
        assert process.exitcode == 0
        outcome = output.get(timeout=5)
        assert outcome[0] == "recovered", outcome
        assert outcome[1] == 2
        assert set(outcome[2]) == {("tenant-a", "same-task"), ("tenant-b", "same-task")}
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        output.close()
        output.join_thread()


class _WorkerBot:
    def __init__(self, tenant_id):
        self.config = type("Config", (), {"tenant_id": tenant_id})()

    def load_session_state_snapshot(self, snapshot):
        pass

    async def execute_turn(self, request):
        return TurnResponse(reply="authorized execution")

    def snapshot_session_state(self):
        return {}


def _run_recovered_worker(state_path, authority_path, output):
    config = ServiceConfig()
    config.security.token_secret = SECRET
    config.security.authority_sqlite_path = authority_path
    config.workers.max_retries = 0
    broker = Broker()
    orchestrator = DadBotOrchestrator(broker, state_store=SQLiteStateStore(state_path))
    orchestrator.require_authenticated_execution()
    orchestrator.configure_delegation_store(SQLiteDelegationStore(authority_path, SECRET))
    recovered = orchestrator.recover_pending_tasks()
    if recovered != 1:
        output.put(("recovery-failed", recovered))
        return
    processor = DadBotTaskProcessor(
        config, bot_factory=lambda *, model_name, tenant_id: _WorkerBot(tenant_id)
    )
    output.put(processor.process(broker.tasks[0]))


@pytest.mark.parametrize("publication", ["valid", "revoked", "modified"])
def test_authorized_recovered_worker_executes_and_publishes(tmp_path, publication):
    authority_path = str(tmp_path / "authority.db")
    state_path = str(tmp_path / "service.db")
    SQLiteDelegationStore.initialize(authority_path)
    authority = SQLiteDelegationStore(authority_path, SECRET)
    authority.grant_membership(issuer="dadbot", subject="alice", tenant_id="tenant-a", scopes={"write"})
    config = ServiceConfig()
    config.security.token_secret = SECRET
    config.security.authority_sqlite_path = authority_path
    config.persistence.sqlite_path = state_path
    broker = Broker()
    orchestrator = DadBotOrchestrator(broker, state_store=build_service_state_store(config))
    app = create_api_app(orchestrator, config=config)
    token = ServiceTokenManager(SECRET).issue(subject="alice", tenant_id="tenant-a", scopes={"write"})
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": "tenant-a", "user_input": "hello"},
        )
    assert response.status_code == 200
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(target=_run_recovered_worker, args=(state_path, authority_path, output))
    process.start()
    process.join(20)
    try:
        assert process.exitcode == 0
        result = output.get(timeout=5)
        assert result.status == "completed", result.error
        if publication == "revoked":
            authority.revoke(broker.tasks[0].delegation)
        if publication == "modified":
            result.response = replace(result.response, reply="forged")
        published = orchestrator.apply_worker_result(result)
        persisted = SQLiteStateStore(state_path)
        if publication == "valid":
            assert published is not None
            assert published.reply == "authorized execution"
            saved_response = persisted.load_response(result.task_id)
            assert saved_response is not None
            assert saved_response["reply"] == "authorized execution"
        else:
            assert published is None
            assert persisted.load_response(result.task_id) is None
            saved_task = persisted.load_task(result.task_id)
            assert saved_task is not None
            assert saved_task["status"] == "queued"
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        output.close()
        output.join_thread()


def test_authenticated_api_persists_task_and_delegation_across_processes(tmp_path):
    authority_path = str(tmp_path / "authority.db")
    state_path = str(tmp_path / "service.db")
    SQLiteDelegationStore.initialize(authority_path)
    authority = SQLiteDelegationStore(authority_path, SECRET)
    authority.grant_membership(issuer="dadbot", subject="alice", tenant_id="tenant-a", scopes={"write"})
    config = ServiceConfig()
    config.security.token_secret = SECRET
    config.security.authority_sqlite_path = authority_path
    config.persistence.sqlite_path = state_path
    state = build_service_state_store(config)
    broker = Broker()
    app = create_api_app(DadBotOrchestrator(broker, state_store=state), config=config)
    token = ServiceTokenManager(SECRET).issue(subject="alice", tenant_id="tenant-a", scopes={"write"})
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": "tenant-a", "user_input": "hello"},
        )
    assert response.status_code == 200
    task = broker.tasks[0]
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(target=_read_persisted_task, args=(state_path, task.task_id, output))
    process.start()
    process.join(20)
    try:
        assert process.exitcode == 0
        persisted = output.get(timeout=5)
        assert persisted["delegation"] == task.delegation
        assert persisted["status"] == "queued"
        authority.verify(persisted["delegation"], task, operation="execute")
        authority.revoke(task.delegation)
        with pytest.raises(AuthorizationError, match="revoked"):
            authority.verify(persisted["delegation"], task, operation="execute")
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        output.close()
        output.join_thread()


def test_sqlite_result_update_for_unknown_task_fails_without_inserting(tmp_path):
    store = SQLiteStateStore(str(tmp_path / "state.db"))
    with pytest.raises(RuntimeError, match="unknown task"):
        store.save_task_result("missing", {"status": "completed"}, {"reply": "no"})
    assert store.load_task("missing") is None
    assert store.load_response("missing") is None


def _revoke_during_publication(authority_path, delegation, started, attempted, done):
    if not started.wait(15):
        raise RuntimeError("Publication did not start")
    attempted.set()
    SQLiteDelegationStore(authority_path, SECRET).revoke(delegation)
    done.set()


def _pause_during_publication(authority_path, state_path, result, started, proceed):
    class PausingStateStore(SQLiteStateStore):
        def save_task_result(self, task_id, payload, response):
            started.set()
            if not proceed.wait(20):
                raise RuntimeError("Publication was not released")
            return super().save_task_result(task_id, payload, response)

    orchestrator = DadBotOrchestrator(Broker(), state_store=PausingStateStore(state_path))
    orchestrator.configure_delegation_store(SQLiteDelegationStore(authority_path, SECRET))
    orchestrator.require_authenticated_execution()
    orchestrator.apply_worker_result(result)


def test_process_termination_during_publication_leaves_queued_task_recoverable(tmp_path):
    authority_path = str(tmp_path / "authority.db")
    state_path = str(tmp_path / "state.db")
    SQLiteDelegationStore.initialize(authority_path)
    authority = SQLiteDelegationStore(authority_path, SECRET)
    authority.grant_membership(issuer="dadbot", subject="alice", tenant_id="tenant-a", scopes={"write"})
    config = ServiceConfig()
    config.security.token_secret = SECRET
    config.security.authority_sqlite_path = authority_path
    broker = Broker()
    state = SQLiteStateStore(state_path)
    orchestrator = DadBotOrchestrator(broker, state_store=state)
    app = create_api_app(orchestrator, config=config)
    token = ServiceTokenManager(SECRET).issue(subject="alice", tenant_id="tenant-a", scopes={"write"})
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": "tenant-a", "user_input": "hello"},
        )
    assert response.status_code == 200

    class SnapshotBot(_WorkerBot):
        def snapshot_session_state(self):
            return {"history": [{"role": "assistant", "content": "partially persisted"}]}

    result = DadBotTaskProcessor(
        config, bot_factory=lambda *, model_name, tenant_id: SnapshotBot(tenant_id)
    ).process(broker.tasks[0])
    assert result.status == "completed"
    context = multiprocessing.get_context("spawn")
    started, proceed = context.Event(), context.Event()
    process = context.Process(
        target=_pause_during_publication, args=(authority_path, state_path, result, started, proceed)
    )
    process.start()
    try:
        assert started.wait(15), "publication did not reach task persistence"
        process.terminate()
        process.join(15)
        assert not process.is_alive()
        assert state.load_task(result.task_id)["status"] == "queued"
        assert state.load_response(result.task_id) is None
        assert state.load_session_state("tenant:tenant-a:shared")["history"] == [
            {"role": "assistant", "content": "partially persisted"}
        ]
        assert orchestrator.apply_worker_result(result).reply == "authorized execution"
    finally:
        if process.is_alive():
            process.terminate()
            process.join()


@pytest.mark.parametrize("revoke_first", [False, True])
def test_revocation_is_ordered_after_publication_that_has_authorized(tmp_path, revoke_first):
    authority_path = str(tmp_path / "authority.db")
    state_path = str(tmp_path / "service.db")
    SQLiteDelegationStore.initialize(authority_path)
    authority = SQLiteDelegationStore(authority_path, SECRET)
    authority.grant_membership(issuer="dadbot", subject="alice", tenant_id="tenant-a", scopes={"write"})
    context = multiprocessing.get_context("spawn")
    started, attempted, done = context.Event(), context.Event(), context.Event()

    class ObservedStore(SQLiteStateStore):
        def save_task_result(self, task_id, payload, response):
            assert not revoke_first, "revoked result must not be persisted"
            started.set()
            assert attempted.wait(10)
            assert not done.wait(0.5), "revocation committed before publication"
            super().save_task_result(task_id, payload, response)

    config = ServiceConfig()
    config.security.token_secret = SECRET
    config.security.authority_sqlite_path = authority_path
    config.workers.max_retries = 0
    broker = Broker()
    state = ObservedStore(state_path)
    orchestrator = DadBotOrchestrator(broker, state_store=state)
    orchestrator.require_authenticated_execution()
    orchestrator.configure_delegation_store(authority)
    from dadbot_system.contracts import ChatRequest

    token = ServiceTokenManager(SECRET).issue(subject="alice", tenant_id="tenant-a", scopes={"write"})
    request = ChatRequest.from_dict({"session_id": "shared", "tenant_id": "tenant-a", "user_input": "hi"})
    task = orchestrator.submit_chat(request, bearer_token=token)
    processor = DadBotTaskProcessor(config, bot_factory=lambda *, model_name, tenant_id: _WorkerBot(tenant_id))
    result = processor.process(task)
    assert result.status == "completed"
    process = context.Process(
        target=_revoke_during_publication, args=(authority_path, task.delegation, started, attempted, done)
    )
    process.start()
    try:
        if revoke_first:
            started.set()
            assert done.wait(10)
            assert orchestrator.apply_worker_result(result) is None
        else:
            assert orchestrator.apply_worker_result(result) is not None
        process.join(15)
        assert process.exitcode == 0
        assert done.is_set()
        assert (state.load_response(task.task_id) is None) == revoke_first
    finally:
        if process.is_alive():
            process.terminate()
            process.join()


def test_sqlite_state_outage_does_not_create_empty_replacement(tmp_path):
    path = tmp_path / "state.db"
    store = SQLiteStateStore(str(path))
    path.rename(tmp_path / "state-offline.db")
    with pytest.raises(sqlite3.OperationalError):
        store.load_task("task")
    assert not path.exists()


@pytest.mark.parametrize("offline", ["authority", "state"])
def test_publication_fails_closed_during_database_outage_and_can_be_retried(tmp_path, offline):
    authority_path = tmp_path / "authority.db"
    state_path = tmp_path / "state.db"
    SQLiteDelegationStore.initialize(str(authority_path))
    authority = SQLiteDelegationStore(str(authority_path), SECRET)
    authority.grant_membership(issuer="dadbot", subject="alice", tenant_id="tenant-a", scopes={"write"})
    config = ServiceConfig()
    config.security.token_secret = SECRET
    config.security.authority_sqlite_path = str(authority_path)
    broker = Broker()
    state = SQLiteStateStore(str(state_path))
    orchestrator = DadBotOrchestrator(broker, state_store=state)
    app = create_api_app(orchestrator, config=config)
    token = ServiceTokenManager(SECRET).issue(subject="alice", tenant_id="tenant-a", scopes={"write"})
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": "tenant-a", "user_input": "hello"},
        )
    assert response.status_code == 200
    result = DadBotTaskProcessor(
        config, bot_factory=lambda *, model_name, tenant_id: _WorkerBot(tenant_id)
    ).process(broker.tasks[0])
    assert result.status == "completed"
    path = authority_path if offline == "authority" else state_path
    moved = path.with_suffix(".offline")
    path.rename(moved)
    try:
        error = AuthorityUnavailableError if offline == "authority" else sqlite3.OperationalError
        with pytest.raises(error):
            orchestrator.apply_worker_result(result)
        assert not path.exists()
    finally:
        moved.rename(path)
    assert state.load_task(result.task_id)["status"] == "queued"
    assert state.load_response(result.task_id) is None
    assert orchestrator.apply_worker_result(result).reply == "authorized execution"
    assert state.load_response(result.task_id)["reply"] == "authorized execution"
    events = state.list_events("tenant:tenant-a:shared")
    assert orchestrator.apply_worker_result(result).reply == "authorized execution"
    assert state.list_events("tenant:tenant-a:shared") == events
    conflicting = replace(result, response=replace(result.response, reply="different"), authority_proof="")
    conflicting.authority_proof = authority.sign_result(broker.tasks[0], conflicting)
    with pytest.raises(ValueError, match="already completed"):
        orchestrator.apply_worker_result(conflicting)
    authority.revoke(broker.tasks[0].delegation)
    with pytest.raises(ValueError, match="authority rejection"):
        orchestrator.apply_worker_result(result)
    assert state.load_response(result.task_id)["reply"] == "authorized execution"
    assert state.list_events("tenant:tenant-a:shared") == events


@pytest.mark.parametrize("configured_state", [False, True])
def test_authority_and_service_state_cannot_share_sqlite_file(tmp_path, configured_state):
    path = str(tmp_path / "shared.db")
    SQLiteDelegationStore.initialize(path)
    config = ServiceConfig()
    config.security.token_secret = SECRET
    config.security.authority_sqlite_path = path
    if configured_state:
        config.persistence.sqlite_path = path
    orchestrator = DadBotOrchestrator(Broker(), state_store=SQLiteStateStore(path))
    with pytest.raises(RuntimeError, match="separate"):
        create_api_app(orchestrator, config=config)


@pytest.mark.parametrize("revoke", [False, True])
def test_restart_only_requeues_tasks_with_current_authority(tmp_path, revoke):
    authority_path = str(tmp_path / "authority.db")
    state_path = str(tmp_path / "service.db")
    SQLiteDelegationStore.initialize(authority_path)
    authority = SQLiteDelegationStore(authority_path, SECRET)
    authority.grant_membership(issuer="dadbot", subject="alice", tenant_id="tenant-a", scopes={"write"})
    config = ServiceConfig()
    config.security.token_secret = SECRET
    config.security.authority_sqlite_path = authority_path
    config.persistence.sqlite_path = state_path
    broker = Broker()
    app = create_api_app(DadBotOrchestrator(broker, state_store=build_service_state_store(config)), config=config)
    token = ServiceTokenManager(SECRET).issue(subject="alice", tenant_id="tenant-a", scopes={"write"})
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/shared/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"tenant_id": "tenant-a", "user_input": "hello"},
        )
    assert response.status_code == 200
    task = broker.tasks[0]
    if revoke:
        authority.revoke(task.delegation)
    restarted_broker = Broker()
    restarted = DadBotOrchestrator(restarted_broker, state_store=build_service_state_store(config))
    restarted.require_authenticated_execution()
    restarted.configure_delegation_store(authority)
    assert restarted.recover_pending_tasks() == (0 if revoke else 1)
    assert len(restarted_broker.tasks) == (0 if revoke else 1)
    if not revoke:
        assert restarted_broker.tasks[0].delegation == task.delegation
        assert restarted_broker.tasks[0].task_id == task.task_id
    saved = restarted.state_store.load_task(task.task_id)
    assert saved is not None
    assert saved["status"] == "queued"
