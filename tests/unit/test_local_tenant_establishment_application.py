from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from dadbot.core.ledger_backend import WALCorruptionError
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_backend import FileWALLedgerBackend


_STARTUP = r"""
import os
import sys
from dadbot.core.dadbot import DadBot
import dadbot.core.local_tenant_establishment as establishment
from dadbot.memory.storage import MemoryStorageBackend
point = sys.argv[1]
original = establishment._write_evidence
def evidence(path, payload):
    if point == "after_append" and payload["status"] == "established":
        os._exit(73)
    original(path, payload)
    if point == "after_intent" and payload["status"] == "pending":
        os._exit(73)
    if point == "after_evidence" and payload["status"] == "established":
        os._exit(73)
establishment._write_evidence = evidence
original_repair = MemoryStorageBackend._repair_established_snapshot
def repair(self, state):
    if point == "before_repair":
        os._exit(73)
    return original_repair(self, state)
MemoryStorageBackend._repair_established_snapshot = repair
bot = DadBot()
assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).revision == 1
assert isinstance(bot.MEMORY_STORE, dict)
bot.shutdown()
"""


def _environment(tmp_path):
    root = tmp_path
    env = os.environ.copy()
    env.update({
        "DADBOT_DURABLE_EXECUTION_LEDGER": "1",
        "DADBOT_EXECUTION_LEDGER_BACKEND": "file",
        "DADBOT_MEMORY_PATH": str(root / "memory.json"),
        "DADBOT_PROFILE_PATH": str(root / "profile.json"),
        "DADBOT_SEMANTIC_DB_PATH": str(root / "semantic.sqlite"),
        "DADBOT_GRAPH_DB_PATH": str(root / "graph.sqlite"),
        "DADBOT_SESSION_LOG_DIR": str(root / "sessions"),
        "PYTEST_CURRENT_TEST": "authority-subprocess",
    })
    return env


@pytest.mark.parametrize("point", ["after_intent", "after_append", "after_evidence", "before_repair"])
def test_real_startup_reconciles_process_termination(tmp_path, point):
    env = _environment(tmp_path)
    interrupted = subprocess.run([sys.executable, "-c", _STARTUP, point], env=env, timeout=60, check=False)
    assert interrupted.returncode == 73
    resumed = subprocess.run([sys.executable, "-c", _STARTUP, "resume"], env=env, timeout=60, check=False)
    assert resumed.returncode == 0
    ledger = ExecutionLedger(FileWALLedgerBackend(tmp_path / "sessions" / "execution_ledger.jsonl"))
    ledger.load_from_backend()
    assert sum(e["type"] == "TENANT_MEMORY_COMMITTED" for e in ledger.read()) == 1


@pytest.mark.parametrize("attempt", range(3))
def test_concurrent_first_startup_establishes_one_event(tmp_path, attempt):
    env = _environment(tmp_path)
    processes = [
        subprocess.Popen([sys.executable, "-c", _STARTUP, "resume"], env=env)
        for _ in range(2)
    ]
    for process in processes:
        assert process.wait(timeout=60) == 0
    ledger = ExecutionLedger(FileWALLedgerBackend(tmp_path / "sessions" / "execution_ledger.jsonl"))
    ledger.load_from_backend()
    events = [event for event in ledger.read() if event["type"] == "TENANT_MEMORY_COMMITTED"]
    assert len(events) == 1
    evidence = list((tmp_path / "sessions").glob("execution_ledger.jsonl.tenant-*"))
    assert len(evidence) == 1
    marker = json.loads(evidence[0].read_text(encoding="utf-8"))
    assert marker["status"] == "established"
    assert marker["initial_event_sha256"] == events[0]["event_sha256"]


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing semantics")
def test_windows_evidence_replacement_sharing_violation_preserves_old_evidence(tmp_path):
    import ctypes
    from ctypes import wintypes

    from dadbot.core.local_tenant_establishment import _write_evidence

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    path = tmp_path / "evidence"
    initial = {"status": "pending", "revision": 0}
    _write_evidence(path, initial)
    handle = kernel.CreateFileW(str(path), 0x80000000, 0x1 | 0x2, None, 3, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        with pytest.raises(PermissionError):
            _write_evidence(path, {"status": "established", "revision": 1})
        assert json.loads(path.read_text(encoding="utf-8")) == initial
    finally:
        if not kernel.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())
    for revision in range(1, 21):
        _write_evidence(path, {"status": "established", "revision": revision})
        assert json.loads(path.read_text(encoding="utf-8"))["revision"] == revision


def test_unowned_legacy_snapshot_does_not_establish(tmp_path):
    (tmp_path / "memory.json").write_text("{invalid", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-c", _STARTUP, "resume"], env=_environment(tmp_path), timeout=60, check=False,
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "no tenant ownership evidence" in result.stderr
    ledger = ExecutionLedger(FileWALLedgerBackend(tmp_path / "sessions" / "execution_ledger.jsonl"))
    assert ledger.load_from_backend() == 0


def test_local_startup_refuses_unowned_shared_snapshot(make_test_dadbot, monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    sessions = tmp_path / "sessions"
    memory = tmp_path / "memory.json"
    memory.write_text(json.dumps({"last_mood": "positive"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="no tenant ownership evidence"):
        make_test_dadbot(memory_path=memory, session_log_dir=sessions)


def test_missing_established_local_wal_fails_closed(make_test_dadbot, monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    sessions = tmp_path / "sessions"
    first = make_test_dadbot(session_log_dir=sessions)
    first.shutdown()
    (sessions / "execution_ledger.jsonl").unlink()
    with pytest.raises(WALCorruptionError, match="missing"):
        make_test_dadbot(session_log_dir=sessions)


def test_missing_evidence_is_rebuilt_from_existing_stream(make_test_dadbot, monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    sessions = tmp_path / "sessions"
    memory = tmp_path / "memory.json"
    first = make_test_dadbot(memory_path=memory, session_log_dir=sessions)
    tenant = first.config.tenant_id
    expected_memory = first.MEMORY_STORE
    first.shutdown()
    markers = list(sessions.glob("execution_ledger.jsonl.tenant-*"))
    assert len(markers) == 1
    markers[0].unlink()
    memory.write_text(json.dumps({"last_mood": "stale"}), encoding="utf-8")
    second = make_test_dadbot(memory_path=memory, session_log_dir=sessions)
    assert second._execution_ledger.replay_tenant_memory(tenant).revision == 1
    assert second.MEMORY_STORE == expected_memory
    assert markers[0].exists()


def test_local_evidence_rejects_disabling_ledger(make_test_dadbot, monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    sessions = tmp_path / "sessions"
    first = make_test_dadbot(session_log_dir=sessions)
    first.shutdown()
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "0")
    with pytest.raises(RuntimeError, match="Local tenant authority"):
        make_test_dadbot(session_log_dir=sessions)
