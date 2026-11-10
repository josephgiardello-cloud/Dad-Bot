import multiprocessing
import sqlite3

import pytest

from dadbot_system.contracts import ChatRequest, ChatResponse, WorkerResult
from dadbot_system.orchestration import DadBotOrchestrator
from dadbot_system.owned_sqlite_tasks import SQLiteOwnedTasks
from dadbot_system.state import InMemoryStateStore, SQLiteStateStore


def _write_task(path, tenant, ready, start, output):
    store = SQLiteOwnedTasks(path)
    ready.put(tenant)
    if not start.wait(20):
        raise TimeoutError("Concurrent writers did not start")
    payload = {
        "tenant_id": tenant,
        "task_id": "same",
        "session_id": "shared-session",
        "status": "queued",
        "accepted_at": "2026-11-09T00:00:00+00:00",
    }
    store.save_task(tenant, "same", payload)
    store.save_task_result(tenant, "same", payload, {"tenant_id": tenant, "reply": tenant})
    output.put(tenant)


def test_owned_tasks_and_results_are_independent(tmp_path):
    store = SQLiteOwnedTasks(str(tmp_path / "state.db"))
    a = {"tenant_id": "tenant-a", "task_id": "same", "status": "queued", "accepted_at": "1"}
    b = {"tenant_id": "tenant-b", "task_id": "same", "status": "queued", "accepted_at": "2"}
    store.save_task("tenant-a", "same", a)
    assert store.load_task("tenant-b", "same") is None
    store.save_task("tenant-b", "same", b)
    assert store.load_task("tenant-a", "same") == a
    assert store.list_incomplete_tasks("tenant-b", limit=1) == [b]
    with pytest.raises(ValueError, match="identity"):
        store.save_task("tenant-b", "same", a)
    store.save_task_result("tenant-a", "same", a, {"tenant_id": "tenant-a", "reply": "first"})
    assert store.load_response("tenant-b", "same") is None
    store.save_task_result("tenant-b", "same", b, {"tenant_id": "tenant-b", "reply": "second"})
    assert store.load_response("tenant-a", "same")["reply"] == "first"
    assert store.load_response("tenant-b", "same")["reply"] == "second"
    with pytest.raises(TypeError):
        store.load_task("same")


def test_recovery_tenants_come_only_from_owned_task_rows(tmp_path):
    path = str(tmp_path / "state.db")
    store = SQLiteOwnedTasks(path)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE tenants (tenant_id TEXT)")
        connection.execute("INSERT INTO tenants VALUES ('catalog-only')")
        connection.execute("CREATE TABLE tasks (id TEXT, payload TEXT)")
        connection.execute("INSERT INTO tasks VALUES ('legacy', '{\"tenant_id\":\"legacy-only\"}')")
    assert store.list_task_tenants() == []
    for tenant in ("tenant-b", "tenant-a"):
        store.save_task(tenant, "same", {"tenant_id": tenant, "task_id": "same", "status": "queued"})
    assert store.list_task_tenants() == ["tenant-a", "tenant-b"]
    with pytest.raises(RuntimeError, match="quarantined"):
        store.reject_legacy_recovery()


class _Broker:
    def __init__(self):
        self.tasks = []

    def enqueue(self, task):
        self.tasks.append(task)

    def get_result_nowait(self):
        return None


@pytest.mark.parametrize("backend", ["sqlite", "memory"])
def test_orchestration_task_paths_are_owned_and_keep_state_store_for_sessions(
    tmp_path, monkeypatch, backend
):
    state = (
        SQLiteStateStore(str(tmp_path / "state.db")) if backend == "sqlite" else InMemoryStateStore()
    )
    broker = _Broker()
    orchestrator = DadBotOrchestrator(broker, state_store=state)

    def reject_unqualified_access(*args, **kwargs):
        raise AssertionError("orchestration must not use legacy task storage")

    for method in ("save_task", "load_task", "save_task_result", "load_response", "save_response",
                   "list_incomplete_tasks"):
        monkeypatch.setattr(state, method, reject_unqualified_access)
    tasks = [
        orchestrator.submit_chat(ChatRequest(
            session_id="shared", request_id="same-request", tenant_id=tenant, user_input="hello"
        ))
        for tenant in ("tenant-a", "tenant-b")
    ]
    for task in tasks:
        payload = orchestrator.task_store.load_task(task.tenant_id, task.task_id)
        payload["task_id"] = "shared-task"
        orchestrator.task_store.save_task(task.tenant_id, "shared-task", payload)
        result = WorkerResult(
            task_id="shared-task", session_id=task.session_id, request_id=task.request.request_id,
            tenant_id=task.tenant_id, status="completed",
            session_state={"history": [{"role": "assistant", "content": task.tenant_id}]},
            response=ChatResponse(
                session_id=task.session_id, request_id=task.request.request_id,
                tenant_id=task.tenant_id, reply=task.tenant_id,
            ),
        )
        assert orchestrator.apply_worker_result(result).reply == task.tenant_id

    for tenant in ("tenant-a", "tenant-b"):
        assert orchestrator.task_status("shared-task", tenant_id=tenant)["status"] == "completed"
        assert orchestrator.response_for_task("shared-task", tenant_id=tenant)["reply"] == tenant
        assert state.owned_sessions.load_session_state(tenant, "shared") is not None
        assert state.owned_sessions.list_events(tenant, "shared")
    assert orchestrator.task_status("shared-task", tenant_id="tenant-c") is None
    assert orchestrator.response_for_task("shared-task", tenant_id="tenant-c") is None
    with pytest.raises(TypeError):
        orchestrator.task_status("shared-task")
    with pytest.raises(TypeError):
        orchestrator.response_for_task("shared-task")


@pytest.mark.parametrize("backend", ["sqlite", "memory"])
def test_recovery_deduplication_is_tenant_qualified(tmp_path, backend):
    state = (
        SQLiteStateStore(str(tmp_path / "state.db")) if backend == "sqlite" else InMemoryStateStore()
    )
    orchestrator = DadBotOrchestrator(_Broker(), state_store=state)
    for tenant in ("tenant-a", "tenant-b"):
        request = ChatRequest(session_id="shared", request_id="same-request", tenant_id=tenant, user_input="hello")
        orchestrator.task_store.save_task(tenant, "same-task", {
            "task_id": "same-task", "tenant_id": tenant, "session_id": request.session_id,
            "request_id": request.request_id, "request": request.to_dict(), "session_state": {},
            "status": "queued", "accepted_at": "2020-01-01T00:00:00+00:00",
        })
    restarted_broker = _Broker()
    restarted = DadBotOrchestrator(restarted_broker, state_store=state)
    assert restarted.recover_pending_tasks() == 2
    assert {(task.tenant_id, task.task_id) for task in restarted_broker.tasks} == {
        ("tenant-a", "same-task"), ("tenant-b", "same-task")
    }
    assert restarted.recover_pending_tasks() == 0


def test_worker_result_cannot_address_another_tenants_task(tmp_path):
    orchestrator = DadBotOrchestrator(
        _Broker(), state_store=SQLiteStateStore(str(tmp_path / "state.db"))
    )
    task = orchestrator.submit_chat(ChatRequest(session_id="shared", tenant_id="tenant-a", user_input="hello"))
    before = orchestrator.task_status(task.task_id, tenant_id=task.tenant_id)
    result = WorkerResult(
        task_id=task.task_id, session_id=task.session_id, request_id=task.request.request_id,
        tenant_id="tenant-b", status="failed", error="forged",
    )
    with pytest.raises(ValueError, match="task not found"):
        orchestrator.apply_worker_result(result)
    assert orchestrator.task_status(task.task_id, tenant_id=task.tenant_id) == before
    assert orchestrator.response_for_task(task.task_id, tenant_id=task.tenant_id) is None


def test_legacy_recovery_is_rejected_without_changing_original_row(tmp_path):
    path = str(tmp_path / "state.db")
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, payload TEXT NOT NULL, response TEXT)")
        connection.execute("INSERT INTO tasks VALUES (?, ?, ?)", ("old", '{"status":"queued"}', '{"result":1}'))
    store = SQLiteOwnedTasks(path)
    with pytest.raises(RuntimeError, match="quarantined"):
        store.reject_legacy_recovery()
    assert store.load_task("tenant-a", "old") is None
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT id, payload, response FROM tasks").fetchall() == [
            ("old", '{"status":"queued"}', '{"result":1}')
        ]


def test_concurrent_tenant_writes_survive_fresh_construction(tmp_path):
    path = str(tmp_path / "state.db")
    SQLiteOwnedTasks(path)
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    output = context.Queue()
    start = context.Event()
    processes = [
        context.Process(target=_write_task, args=(path, tenant, ready, start, output))
        for tenant in ("tenant-a", "tenant-b")
    ]
    for process in processes:
        process.start()
    try:
        assert {ready.get(timeout=20), ready.get(timeout=20)} == {"tenant-a", "tenant-b"}
        start.set()
        assert {output.get(timeout=20), output.get(timeout=20)} == {"tenant-a", "tenant-b"}
        for process in processes:
            process.join(20)
            assert process.exitcode == 0
        store = SQLiteOwnedTasks(path)
        for tenant in ("tenant-a", "tenant-b"):
            task = store.load_task(tenant, "same")
            assert task["tenant_id"] == tenant
            assert task["session_id"] == "shared-session"
            assert store.list_incomplete_tasks(tenant) == [task]
            assert store.load_response(tenant, "same") == {"tenant_id": tenant, "reply": tenant}
    finally:
        start.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        ready.close()
        ready.join_thread()
        output.close()
        output.join_thread()
