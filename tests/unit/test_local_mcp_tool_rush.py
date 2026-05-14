from __future__ import annotations

import pathlib

import pytest

from dadbot_system import local_mcp_server as lms

pytestmark = pytest.mark.unit


def test_markdown_task_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.setattr(lms, "_project_root", lambda: tmp_path)

    path = lms._task_file_path()
    tasks = [
        {"id": "tsk1", "title": "Ship release notes", "due": "2026-05-20T10:00:00", "priority": "high", "done": False},
        {"id": "tsk2", "title": "Archive old logs", "due": "", "priority": "low", "done": True},
    ]

    lms._write_markdown_tasks(path, tasks)
    loaded = lms._load_markdown_tasks(path)

    assert len(loaded) == 2
    assert loaded[0]["id"] == "tsk1"
    assert loaded[0]["done"] is False
    assert loaded[1]["id"] == "tsk2"
    assert loaded[1]["done"] is True


def test_python_function_refactor_rename_and_docstring(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.setattr(lms, "_project_root", lambda: tmp_path)

    target = tmp_path / "module.py"
    target.write_text(
        "def sample(value):\n"
        "    return value * 2\n",
        encoding="utf-8",
    )

    result = lms._python_function_refactor(
        target,
        function_name="sample",
        new_name="sample_v2",
        prepend_docstring="Doubles the input.",
    )

    updated = target.read_text(encoding="utf-8")
    assert result["changed"] is True
    assert "def sample_v2(value):" in updated
    assert '"""Doubles the input."""' in updated


def test_python_function_refactor_missing_function(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.setattr(lms, "_project_root", lambda: tmp_path)

    target = tmp_path / "module.py"
    target.write_text("def alpha():\n    return 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not found"):
        lms._python_function_refactor(target, function_name="beta")


def test_workspace_file_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.setattr(lms, "_project_root", lambda: tmp_path)

    (tmp_path / "docs").mkdir()
    file_a = tmp_path / "docs" / "notes.md"
    file_b = tmp_path / "main.py"
    file_a.write_text("alpha beta gamma", encoding="utf-8")
    file_b.write_text("print('hello')", encoding="utf-8")

    files = lms._iter_workspace_files(tmp_path)
    names = sorted(path.name for path in files)

    assert names == ["main.py", "notes.md"]
    assert "alpha beta" in lms._safe_read_text(file_a)


def test_get_pending_executive_tasks_filters_and_limits(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.setattr(lms, "_project_root", lambda: tmp_path)

    path = lms._task_file_path()
    lms._write_markdown_tasks(
        path,
        [
            {"id": "a", "title": "Open 1", "due": "2026-05-20", "priority": "normal", "done": False},
            {"id": "b", "title": "Done", "due": "", "priority": "low", "done": True},
            {"id": "c", "title": "Open 2", "due": "2026-05-19", "priority": "high", "done": False},
        ],
    )

    pending = lms.get_pending_executive_tasks(limit=1)

    assert len(pending) == 1
    assert pending[0]["done"] is False


def test_persist_research_memory_syncs_and_writes_event():
    calls = {"sync": 0, "event": 0}

    class _Writer:
        def write_event(self, *_args, **_kwargs):
            calls["event"] += 1
            return {"event_id": "evt-1"}

    class _Bot:
        ledger_writer = _Writer()

        @staticmethod
        def sync_semantic_memory_index(_rows):
            calls["sync"] += 1

    result = lms._persist_research_memory(_Bot(), url="https://example.com", preview="important finding")

    assert result["memory_indexed"] is True
    assert result["memory_event_written"] is True
    assert calls["sync"] == 1
    assert calls["event"] == 1
