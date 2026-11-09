import multiprocessing
import sqlite3

import pytest

from dadbot_system.owned_sqlite_tasks import SQLiteOwnedTasks


def _write_task(path, tenant, ready, start, output):
    store = SQLiteOwnedTasks(path)
    ready.put(tenant)
    if not start.wait(20):
        raise TimeoutError("Concurrent writers did not start")
    store.save_task(tenant, "same", {"tenant_id": tenant, "task_id": "same", "status": "queued"})
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
        assert store.load_task("tenant-a", "same")["tenant_id"] == "tenant-a"
        assert store.load_task("tenant-b", "same")["tenant_id"] == "tenant-b"
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
