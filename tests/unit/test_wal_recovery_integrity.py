from __future__ import annotations

import json
import subprocess
import sys
import threading

import pytest

from dadbot.core.compaction import CompactionPolicy, EventCompactor
from dadbot.core.durability import CRC32LineCodec, FileLockMutex
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_backend import (
    BatchWriteBackend,
    CRCFileWALLedgerBackend,
    EventualConsistencyBackend,
    FileWALLedgerBackend,
    InMemoryLedgerBackend,
    WALCorruptionError,
)
from dadbot.core.ledger_index import LedgerIndex


def _session_event(value):
    return {
        "type": "SESSION_STATE_UPDATED",
        "session_id": "test",
        "trace_id": "trace",
        "kernel_step_id": "step",
        "timestamp": "2026-01-01",
        "payload": {"state": {"value": value}},
    }


def test_ledger_events_are_isolated_from_caller_mutation():
    ledger = ExecutionLedger()
    event = _session_event(1)
    written = ledger.write(event)
    event["payload"]["state"]["value"] = 2
    written["payload"]["state"]["value"] = 3
    full = ledger.read()
    full[0]["payload"]["state"]["value"] = 4
    full.clear()
    tail = ledger.read(full=False)
    tail[0]["payload"]["state"]["value"] = 5
    assert ledger.read()[0]["payload"]["state"]["value"] == 1
    assert ledger.verify_replay(mode="chain")["ok"]


def test_selective_ledger_reads_filter_before_copy_without_changing_full_reads():
    ledger = ExecutionLedger()
    ledger.write(_session_event(1))
    ledger.write({**_session_event(2), "type": "GRAPH_CHECKPOINT"})
    full = ledger.read()
    selected = ledger.read(event_types=frozenset({"SESSION_STATE_UPDATED"}))
    assert selected == full[:1]
    compact = ledger.read(payload_types=frozenset({"SESSION_STATE_UPDATED"}))
    assert compact[0] == full[0]
    assert compact[1] == {key: value for key, value in full[1].items() if key != "payload"}
    selected[0]["payload"]["state"]["value"] = 9
    compact[0]["payload"]["state"]["value"] = 8
    assert ledger.read() == full
    ledger._backend_uncertain = True
    with pytest.raises(RuntimeError, match="reload"):
        ledger.read(event_types=frozenset())


def test_hot_path_index_does_not_copy_checkpoint_payloads_and_preserves_trace_roots(monkeypatch):
    ledger = ExecutionLedger()
    root = ledger.write({**_session_event(1), "type": "GRAPH_CHECKPOINT"})
    started = ledger.write({**_session_event(2), "type": "JOB_STARTED", "payload": {"request_id": "request"}})
    index = LedgerIndex(ledger)
    from dadbot.core import execution_ledger

    original_copy = execution_ledger.deepcopy
    snapshots = []

    def capture_copy(value):
        snapshots.append(value)
        return original_copy(value)

    monkeypatch.setattr(execution_ledger, "deepcopy", capture_copy)
    assert index.trace_root("trace") == root["event_id"]
    assert index.has_ambiguous_request_inflight(session_id="test", request_id="request")
    assert snapshots[0][1] == started
    assert "payload" not in snapshots[0][0]
    assert len(snapshots) == 1
    monkeypatch.setattr(execution_ledger, "deepcopy", original_copy)
    ledger.write({
        **_session_event(3), "type": "JOB_COMPLETED", "payload": {"request_id": "request", "result": ["ok", False]},
    })
    assert not index.has_ambiguous_request_inflight(session_id="test", request_id="request")
    assert index.completed_result(session_id="test", request_id="request") == ("ok", False)
    assert index.trace_root("trace") == root["event_id"]


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_replay_preserves_next_sequence_for_legacy_gap(tmp_path, backend_type):
    path = tmp_path / "events.wal"
    backend = backend_type(path)
    old_event = _session_event(1)
    old_event["sequence"] = 10
    backend.append(old_event)
    ledger = ExecutionLedger(backend)
    assert ledger.load_from_backend() == 1
    assert ledger.get_next_sequence() == 11
    assert ledger.write(_session_event(2))["sequence"] == 11
    reloaded = ExecutionLedger(backend_type(path))
    assert reloaded.load_from_backend() == 2
    assert reloaded.get_next_sequence() == 12


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_load_waits_for_in_progress_writer(tmp_path, backend_type):
    path = tmp_path / "events.wal"
    backend = backend_type(path)
    record = {"type": "TEST", "value": 1}
    line = CRC32LineCodec.encode(record) if backend_type is CRCFileWALLedgerBackend else json.dumps(record) + "\n"
    started = threading.Event()
    done = threading.Event()
    result = []

    def read():
        started.set()
        try:
            result.append(backend.load())
        except Exception as exc:
            result.append(exc)
        finally:
            done.set()

    with FileLockMutex(path.with_name(path.name + ".lock")).locked():
        with path.open("w", encoding="utf-8") as handle:
            split = len(line) // 2
            handle.write(line[:split])
            handle.flush()
            reader = threading.Thread(target=read)
            reader.start()
            assert started.wait(1)
            assert not done.wait(0.2)
            handle.write(line[split:])
            handle.flush()
    reader.join(timeout=3)
    assert not reader.is_alive()
    assert result == [[record]]


@pytest.mark.parametrize("persist_before_failure", [False, True])
def test_failed_append_requires_replay_before_read_or_write(persist_before_failure):
    class FailingBackend(InMemoryLedgerBackend):
        fail = False

        def append(self, event, *, committed=False):
            if self.fail:
                if persist_before_failure:
                    super().append(event, committed=committed)
                raise OSError("injected append failure")
            return super().append(event, committed=committed)

    backend = FailingBackend()
    ledger = ExecutionLedger(backend)
    ledger.write(_session_event(1))
    backend.fail = True
    with pytest.raises(OSError, match="injected append failure"):
        ledger.write(_session_event(2))
    with pytest.raises(RuntimeError, match="reload"):
        ledger.read()
    with pytest.raises(RuntimeError, match="reload"):
        ledger.write(_session_event(3))
    with pytest.raises(RuntimeError, match="reload"):
        ledger.event_count()
    with pytest.raises(RuntimeError, match="reload"):
        ledger.get_next_sequence()
    backend.fail = False
    assert ledger.load_from_backend() == 1 + int(persist_before_failure)
    ledger.write(_session_event(3))
    assert len(ledger.read()) == 2 + int(persist_before_failure)
    assert ledger.verify_replay(mode="chain")["ok"]


def test_failed_reload_does_not_clear_uncertain_state(tmp_path):
    path = tmp_path / "events.wal"
    ledger = ExecutionLedger(FileWALLedgerBackend(path))
    ledger.write(_session_event(1))
    with path.open("ab") as handle:
        handle.write(b'{"type":')
    with pytest.raises(WALCorruptionError):
        ledger.write(_session_event(2))
    with pytest.raises(WALCorruptionError):
        ledger.load_from_backend()
    with pytest.raises(RuntimeError, match="reload"):
        ledger.read()


def test_failure_after_backend_append_requires_replay(monkeypatch):
    backend = InMemoryLedgerBackend()
    ledger = ExecutionLedger(backend)
    original_update = ledger._update_session_heads

    def fail_update(*args):
        raise RuntimeError("injected post-append failure")

    monkeypatch.setattr(ledger, "_update_session_heads", fail_update)
    with pytest.raises(RuntimeError, match="post-append"):
        ledger.write(_session_event(1))
    with pytest.raises(RuntimeError, match="reload"):
        ledger.read()
    monkeypatch.setattr(ledger, "_update_session_heads", original_update)
    assert ledger.load_from_backend() == 1
    ledger.write(_session_event(2))
    assert ledger.verify_replay(mode="chain")["ok"]


@pytest.mark.parametrize("wrapper_type", [EventualConsistencyBackend, BatchWriteBackend])
@pytest.mark.parametrize("persist_before_failure", [False, True])
def test_buffer_flush_failure_does_not_retry_uncertain_batch(wrapper_type, persist_before_failure):
    class FailingBackend(InMemoryLedgerBackend):
        def append(self, event, *, committed=False):
            if event["value"] == 2:
                if persist_before_failure:
                    super().append(event, committed=committed)
                raise OSError("uncertain batch write")
            return super().append(event, committed=committed)

    inner = FailingBackend()
    wrapper = wrapper_type(inner)
    wrapper.append({"value": 1})
    wrapper.append({"value": 2})
    with pytest.raises(OSError, match="uncertain batch write"):
        wrapper.flush()
    before = inner.load()
    with pytest.raises(RuntimeError, match="uncertain"):
        wrapper.load()
    with pytest.raises(RuntimeError, match="uncertain"):
        wrapper.flush()
    with pytest.raises(RuntimeError, match="uncertain"):
        wrapper.append({"value": 3})
    assert inner.load() == before


@pytest.mark.parametrize("wrapper_type", [EventualConsistencyBackend, BatchWriteBackend])
def test_ledger_reload_cannot_retry_uncertain_buffer(wrapper_type):
    class FailingBackend(InMemoryLedgerBackend):
        def append(self, event, *, committed=False):
            super().append(event, committed=committed)
            raise OSError("write outcome uncertain")

    inner = FailingBackend()
    ledger = ExecutionLedger(wrapper_type(inner))
    with pytest.raises(OSError, match="write outcome uncertain"):
        ledger.write({**_session_event(1), "committed": True})
    before = inner.load()
    with pytest.raises(RuntimeError, match="uncertain"):
        ledger.load_from_backend()
    with pytest.raises(RuntimeError, match="reload"):
        ledger.read()
    assert inner.load() == before


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_compaction_does_not_break_wal_sequence_or_recovery(tmp_path, backend_type):
    path = tmp_path / "events.wal"
    ledger = ExecutionLedger(backend_type(path))
    for value in (1, 2, 3):
        ledger.write({**_session_event(value), "timestamp": 1.0})
    report = EventCompactor(
        policy=CompactionPolicy(max_events=1, min_snapshot_distance=0),
    ).compact(ledger=ledger, snapshot={"head_sequence": 3}, force=True)
    assert not report["compacted"]
    assert report["reason"] == "durable_checkpoint_required"
    ledger.write(_session_event(4))
    restarted = ExecutionLedger(backend_type(path))
    assert restarted.load_from_backend() == 4
    assert restarted.verify_replay(mode="chain")["ok"]


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_torn_tail_survives_restart_without_silent_loss(tmp_path, backend_type):
    path = tmp_path / "events.wal"
    backend = backend_type(path)
    backend.append({"type": "SESSION_STATE_UPDATED", "value": 1}, committed=True)
    crashed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os,sys; f=open(sys.argv[1], 'ab', buffering=0); "
            "f.write(b'{\"type\":'); os.fsync(f.fileno()); os._exit(17)",
            str(path),
        ],
        check=False,
    )
    assert crashed.returncode == 17
    before = path.read_bytes()
    restarted = backend_type(path)
    with pytest.raises(WALCorruptionError, match="Incomplete WAL tail"):
        restarted.load()
    with pytest.raises(WALCorruptionError, match="Incomplete WAL tail"):
        restarted.append({"type": "SESSION_STATE_UPDATED", "value": 2}, committed=True)
    assert path.read_bytes() == before


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
@pytest.mark.parametrize("bad_record", [b"bad\n", b"[]\n", b"\n"])
def test_complete_invalid_record_fails_without_rewriting(tmp_path, backend_type, bad_record):
    path = tmp_path / "events.wal"
    backend = backend_type(path)
    backend.append({"type": "SESSION_STATE_UPDATED"}, committed=True)
    with path.open("ab") as handle:
        handle.write(bad_record)
    before = path.read_bytes()
    with pytest.raises(WALCorruptionError, match="WAL record"):
        backend.load()
    assert path.read_bytes() == before


@pytest.mark.parametrize("field", ["event_sha256", "chain_hash", "prev_chain_hash", "event_id"])
def test_missing_checksum_fields_cannot_bypass_replay_integrity(tmp_path, field):
    path = tmp_path / "events.wal"
    ledger = ExecutionLedger(FileWALLedgerBackend(path))
    for sequence in (1, 2):
        ledger.write({
            "type": "SESSION_STATE_UPDATED", "session_id": "test",
            "trace_id": "trace", "kernel_step_id": "step", "timestamp": "2026-01-01",
            "payload": {"state": {"value": sequence}},
        })
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[1].pop(field)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum chain verification failed"):
        ExecutionLedger(FileWALLedgerBackend(path)).load_from_backend()


def test_legacy_ledger_can_be_reloaded_after_new_chained_event(tmp_path):
    path = tmp_path / "events.wal"
    backend = FileWALLedgerBackend(path)
    backend.append({
        "type": "SESSION_STATE_UPDATED", "session_id": "test", "session_index": 1,
        "trace_id": "trace", "kernel_step_id": "step", "timestamp": "2026-01-01",
        "payload": {"state": {"value": 1}},
    }, committed=True)
    ledger = ExecutionLedger(backend)
    ledger.load_from_backend()
    ledger.write({
        "type": "SESSION_STATE_UPDATED", "session_id": "test",
        "trace_id": "trace", "kernel_step_id": "step", "timestamp": "2026-01-01",
        "payload": {"state": {"value": 2}},
    })
    assert ExecutionLedger(FileWALLedgerBackend(path)).load_from_backend() == 2


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_stale_file_ledger_writer_cannot_append_to_changed_head(tmp_path, backend_type):
    path = tmp_path / "events.wal"
    first = ExecutionLedger(backend_type(path))
    stale = ExecutionLedger(backend_type(path))
    first.write(_session_event(1))
    with pytest.raises(RuntimeError, match="ledger head changed"):
        stale.write(_session_event(2))
    assert stale.load_from_backend() == 1
    stale.write(_session_event(2))
    restarted = ExecutionLedger(backend_type(path))
    assert restarted.load_from_backend() == 2
    assert restarted.verify_replay(mode="chain")["ok"]


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_legacy_backend_append_cannot_break_an_existing_chain(tmp_path, backend_type):
    path = tmp_path / "events.wal"
    ledger = ExecutionLedger(backend_type(path))
    ledger.write(_session_event(1))
    with pytest.raises(RuntimeError, match="Unchained events"):
        backend_type(path).append(_session_event(2))
    restarted = ExecutionLedger(backend_type(path))
    assert restarted.load_from_backend() == 1


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_competing_processes_cannot_commit_same_ledger_head(tmp_path, backend_type):
    path = tmp_path / "events.wal"
    gate = tmp_path / "gate"
    script = (
        "import os,sys,time; "
        "from dadbot.core.execution_ledger import ExecutionLedger; "
        "from dadbot.core import ledger_backend; "
        "ledger=ExecutionLedger(getattr(ledger_backend,sys.argv[1])(sys.argv[2])); "
        "ledger.load_from_backend(); print('ready',flush=True)\n"
        "while not os.path.exists(sys.argv[3]): time.sleep(.005)\n"
        "event={'type':'SESSION_STATE_UPDATED','session_id':'s','trace_id':'t',"
        "'kernel_step_id':'test','timestamp':1,'payload':{'state':{'pid':os.getpid()}}}\n"
        "try:\n ledger.write(event)\n"
        "except ledger_backend.LedgerHeadConflict:\n sys.exit(12)\n"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, backend_type.__name__, str(path), str(gate)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    try:
        for process in processes:
            assert process.stdout is not None
            assert process.stdout.readline().strip() == "ready"
        gate.touch()
        outcomes = [process.communicate(timeout=15) for process in processes]
        assert sorted(process.returncode for process in processes) == [0, 12], outcomes
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    restarted = ExecutionLedger(backend_type(path))
    assert restarted.load_from_backend() == 1
    assert restarted.verify_replay(mode="chain")["ok"]
