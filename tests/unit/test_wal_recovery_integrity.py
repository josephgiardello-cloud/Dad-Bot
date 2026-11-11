from __future__ import annotations

import hashlib
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
    LedgerHeadConflict,
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


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_json_normalized_append_keeps_verified_digest_and_original_runtime_types(tmp_path, backend_type):
    path = tmp_path / "ledger"
    ledger = ExecutionLedger(backend_type(path))
    first = ledger.write(_session_event({"tuple": (1, 2), "bytes": b"value"}))
    assert first["payload"]["state"]["value"] == {"tuple": (1, 2), "bytes": b"value"}
    assert ledger._backend_content_digest == hashlib.sha256(path.read_bytes()).digest()
    second = ledger.write(_session_event(2))
    assert ledger._backend_content_digest == hashlib.sha256(path.read_bytes()).digest()
    assert second["parent_event_id"] == first["event_id"]
    restored = ExecutionLedger(backend_type(path))
    restored.load_from_backend()
    assert restored.read()[0]["payload"]["state"]["value"] == {"tuple": [1, 2], "bytes": "b'value'"}
    assert ledger.replay_hash() == restored.replay_hash()
    assert ledger.read()[0]["payload"]["state"]["value"]["tuple"] == (1, 2)


def test_digest_records_hash_complete_payloads_and_return_only_isolated_scalars(monkeypatch):
    import hashlib

    from dadbot.core import execution_ledger
    from dadbot.core._control_plane_compaction import CompactionMixin

    ledger = ExecutionLedger()
    for value in ({"nested": [1, {"text": "retained"}]}, {"other": (2, 3)}):
        ledger.write(_session_event(value))
    before = ledger.read()
    copied = []
    original_copy = execution_ledger.deepcopy

    def capture_copy(value):
        copied.append(value)
        return original_copy(value)

    with monkeypatch.context() as patch:
        patch.setattr(execution_ledger, "deepcopy", capture_copy)
        records = ledger.read_digest_records()
    assert not copied
    assert records == [
        {
            "sequence": int(event["sequence"]), "type": event["type"],
            "trace_id": event["trace_id"], "session_id": event["session_id"],
            "kernel_step_id": event["kernel_step_id"],
            "payload_hash": hashlib.sha256(
                json.dumps(event["payload"], sort_keys=True, default=str).encode("utf-8"),
            ).hexdigest(),
        }
        for event in before
    ]
    assert CompactionMixin._digest_records(records) == CompactionMixin._event_stream_digest(before)
    assert all(type(value) in (str, int) for record in records for value in record.values())
    before_payload_hash = records[0]["payload_hash"]
    records[0]["payload_hash"] = "changed-copy"
    assert ledger.read() == before
    ledger._events[0]["payload"]["state"]["value"]["nested"].append("changed-payload")
    assert ledger.read_digest_records()[0]["payload_hash"] != before_payload_hash
    assert CompactionMixin._digest_records(ledger.read_digest_records()) != CompactionMixin._event_stream_digest(before)


def test_digest_records_validate_cache_and_reject_uncertain_state():
    ledger = ExecutionLedger()
    assert ledger.read_digest_records() == []
    ledger.write(_session_event(1))
    ledger._cache["event_count"] = 100
    assert len(ledger.read_digest_records()) == 1
    assert ledger.event_count() == 1
    ledger._backend_uncertain = True
    with pytest.raises(RuntimeError, match="reload"):
        ledger.read_digest_records()


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_unchanged_append_verifies_only_new_event_after_verified_prefix(tmp_path, monkeypatch, backend_type):
    from dadbot.core import execution_ledger

    backend = backend_type(tmp_path / "ledger")
    ledger = ExecutionLedger(backend)
    ledger.write(_session_event(0))
    verified_lengths = []
    verify = execution_ledger._verify_persisted_chain
    loads = []
    load = backend._load_unlocked

    def capture_verify(events, **kwargs):
        verified_lengths.append(len(events))
        return verify(events, **kwargs)

    def capture_load():
        loads.append(True)
        return load()

    monkeypatch.setattr(execution_ledger, "_verify_persisted_chain", capture_verify)
    monkeypatch.setattr(backend, "_load_unlocked", capture_load)
    for value in range(1, 12):
        ledger.write(_session_event(value))
    assert verified_lengths == [1] * 11
    assert not loads
    assert ledger.verify_replay(mode="chain")["ok"]
    assert ExecutionLedger(backend_type(tmp_path / "ledger")).load_from_backend() == 12


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_successful_local_append_preserves_byte_verified_refresh(tmp_path, monkeypatch, backend_type):
    path = tmp_path / "ledger"
    backend = backend_type(path)
    ledger = ExecutionLedger(backend)
    ledger.commit_tenant_memory(
        tenant_id="a", expected_revision=0, operation_id="one", state={"items": [1]},
    )
    loads = []
    load = backend._load_unlocked

    def capture_load():
        loads.append(True)
        return load()

    monkeypatch.setattr(backend, "_load_unlocked", capture_load)
    rebuilds = ledger.telemetry_snapshot()["cache_rebuild_count"]
    for value in range(3):
        ledger.write(_session_event(value))
        assert ledger._backend_content_digest == hashlib.sha256(path.read_bytes()).digest()
        assert ledger.refresh_tenant_memory("a").state == {"items": [1]}
        assert ledger.telemetry_snapshot()["cache_rebuild_count"] == rebuilds
    assert not loads
    restored = ExecutionLedger(backend_type(path))
    restored.load_from_backend()
    assert restored.read() == ledger.read()
    assert restored.verify_replay(mode="chain")["ok"]


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_append_refresh_reuses_verified_runtime_prefix_without_historical_comparisons(tmp_path, backend_type):
    class NoHistoricalComparison(dict):
        __hash__ = None

        def __eq__(self, other):
            pytest.fail("Unchanged runtime prefix must not compare historical payloads")

    ledger = ExecutionLedger(backend_type(tmp_path / "ledger"))
    ledger.write(_session_event(1))
    ledger.load_from_backend(only_if_changed=True)
    ledger._events[0] = NoHistoricalComparison(ledger._events[0])
    ledger.write(_session_event(2))
    assert ledger._backend_content_digest is not None


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_append_refresh_keeps_runtime_type_and_fresh_restore_normalizes(tmp_path, backend_type):
    path = tmp_path / "ledger"
    ledger = ExecutionLedger(backend_type(path))
    ledger.write(_session_event(1))
    ledger.load_from_backend(only_if_changed=True)
    ledger.write(_session_event((2, 3)))
    assert ledger._backend_content_digest == hashlib.sha256(path.read_bytes()).digest()
    ledger.load_from_backend(only_if_changed=True)
    assert ledger.read()[-1]["payload"]["state"]["value"] == (2, 3)

    restored = ExecutionLedger(backend_type(path))
    restored.load_from_backend()
    assert restored.read()[-1]["payload"]["state"]["value"] == [2, 3]


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_append_refresh_does_not_bless_a_mismatched_verified_prefix(tmp_path, monkeypatch, backend_type):
    path = tmp_path / "ledger"
    backend = backend_type(path)
    ledger = ExecutionLedger(backend)
    ledger.write(_session_event(1))
    ledger.load_from_backend()
    # Non-replay metadata can differ without breaking the canonical checksum chain.
    ledger._events[0]["timestamp"] = "different-runtime-metadata"
    ledger.write(_session_event(2))
    assert ledger._backend_content_digest is None
    loads = []
    load = backend._load_unlocked

    def capture_load():
        loads.append(True)
        return load()

    monkeypatch.setattr(backend, "_load_unlocked", capture_load)
    ledger.load_from_backend(only_if_changed=True)
    assert loads == [True]
    assert ledger.read()[0]["timestamp"] == "2026-01-01"


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_append_refresh_reloads_resealed_metadata_changes_with_an_unchanged_chain(tmp_path, backend_type):
    path = tmp_path / "ledger"
    ledger = ExecutionLedger(backend_type(path))
    ledger.write(_session_event(1))
    ledger.load_from_backend(only_if_changed=True)
    event = ledger.read()[0]
    event["timestamp"] = "2026-01-02"
    line = (
        json.dumps(event) + "\n"
        if backend_type is FileWALLedgerBackend
        else CRC32LineCodec.encode(event)
    )
    path.write_bytes(line.encode("utf-8"))
    ledger.write(_session_event(2))
    assert ledger._backend_content_digest is None
    assert ledger.load_from_backend(only_if_changed=True) == 2
    assert ledger.read()[0]["timestamp"] == "2026-01-02"
    assert ledger.verify_replay(mode="chain")["ok"]


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_append_refresh_observes_same_backend_concurrent_successor(tmp_path, monkeypatch, backend_type):
    backend = backend_type(tmp_path / "ledger")
    ledger = ExecutionLedger(backend)
    ledger.write(_session_event(1))
    append = backend.append

    def append_with_successor(event, *, committed=False):
        append(event, committed=committed)
        other = ExecutionLedger(backend)
        other.load_from_backend()
        with monkeypatch.context() as patch:
            patch.setattr(backend, "append", append)
            other.write(_session_event(3))

    monkeypatch.setattr(backend, "append", append_with_successor)
    ledger.write(_session_event(2))
    assert ledger._backend_content_digest is None
    assert ledger.load_from_backend(only_if_changed=True) == 3
    assert ledger.read()[-1]["payload"]["state"]["value"] == 3


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
@pytest.mark.parametrize("mode", ["filtered", "ledger_subclass", "backend_subclass", "authenticated"])
def test_append_refresh_retains_full_paths_for_nonlocal_or_custom_views(tmp_path, monkeypatch, backend_type, mode):
    class CustomBackend(backend_type):
        pass

    class CustomLedger(ExecutionLedger):
        pass

    backend = (CustomBackend if mode == "backend_subclass" else backend_type)(tmp_path / "ledger")
    ledger = (CustomLedger if mode == "ledger_subclass" else ExecutionLedger)(backend)
    if mode == "filtered":
        ledger.add_replay_filter(lambda events: events)
    elif mode == "authenticated":
        ledger.require_authenticated_authority()
    ledger.write(_session_event(1))
    assert ledger._backend_content_digest is None
    loads = []
    load = backend._load_unlocked

    def capture_load():
        loads.append(True)
        return load()

    monkeypatch.setattr(backend, "_load_unlocked", capture_load)
    assert ledger.load_from_backend(only_if_changed=True) == 1
    assert loads == [True]


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_append_refresh_rejects_post_fsync_tampering_and_requires_recovery(tmp_path, monkeypatch, backend_type):
    from dadbot.core import ledger_backend

    path = tmp_path / "ledger"
    ledger = ExecutionLedger(backend_type(path))
    ledger.write(_session_event(1))
    original = path.read_bytes()
    fsync = ledger_backend.os.fsync

    def tamper_after_fsync(fd):
        fsync(fd)
        path.write_bytes(path.read_bytes().replace(b'"value": 1', b'"value": 9'))

    monkeypatch.setattr(ledger_backend.os, "fsync", tamper_after_fsync)
    ledger.write(_session_event(2))
    assert ledger._backend_content_digest != hashlib.sha256(path.read_bytes()).digest()
    with pytest.raises(RuntimeError):
        ledger.load_from_backend(only_if_changed=True)
    path.write_bytes(original)
    with pytest.raises(RuntimeError, match="reload"):
        ledger.read()
    with pytest.raises(RuntimeError, match="reload"):
        ledger.load_from_backend(only_if_changed=True)
    assert ledger.load_from_backend() == 1


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_append_prefix_detects_same_size_tampering_and_external_writer(tmp_path, backend_type):
    import os

    path = tmp_path / "ledger"
    ledger = ExecutionLedger(backend_type(path))
    ledger.write(_session_event(1))
    original = path.read_bytes()
    stat = path.stat()
    changed = original.replace(b'"value": 1', b'"value": 9')
    assert changed != original and len(changed) == len(original)
    path.write_bytes(changed)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises((WALCorruptionError, LedgerHeadConflict)):
        ledger.write(_session_event(2))
    assert path.read_bytes() == changed
    path.write_bytes(original)
    ledger.load_from_backend()
    other = ExecutionLedger(backend_type(path))
    other.load_from_backend()
    other.write(_session_event(2))
    with pytest.raises(LedgerHeadConflict):
        ledger.write(_session_event(3))
    ledger.load_from_backend()
    ledger.write(_session_event(3))
    assert ledger.verify_replay(mode="chain")["ok"]
    assert len(ledger.read()) == 3


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_append_prefix_failure_keeps_fsync_ambiguity_and_requires_full_recovery(tmp_path, monkeypatch, backend_type):
    from dadbot.core import ledger_backend

    path = tmp_path / "ledger"
    backend = backend_type(path)
    ledger = ExecutionLedger(backend)
    ledger.write(_session_event(1))
    original_fsync = ledger_backend.os.fsync

    def fail_fsync(*args):
        raise OSError("synthetic durable append failure")

    monkeypatch.setattr(ledger_backend.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="durable append"):
        ledger.write(_session_event(2))
    with pytest.raises(RuntimeError, match="reload"):
        ledger.read()
    monkeypatch.setattr(ledger_backend.os, "fsync", original_fsync)
    assert ledger.load_from_backend() == 2
    ledger.write(_session_event(3))
    assert ExecutionLedger(backend_type(path)).load_from_backend() == 3


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_append_prefix_digest_binds_expected_bytes_not_post_write_tampering(tmp_path, monkeypatch, backend_type):
    from dadbot.core import ledger_backend

    path = tmp_path / "ledger"
    ledger = ExecutionLedger(backend_type(path))
    ledger.write(_session_event(1))
    fsync = ledger_backend.os.fsync

    def tamper_after_fsync(fd):
        fsync(fd)
        raw = path.read_bytes()
        changed = raw.replace(b'"value": 1', b'"value": 9')
        assert changed != raw
        path.write_bytes(changed)

    monkeypatch.setattr(ledger_backend.os, "fsync", tamper_after_fsync)
    ledger.write(_session_event(2))
    monkeypatch.setattr(ledger_backend.os, "fsync", fsync)
    corrupted = path.read_bytes()
    with pytest.raises((WALCorruptionError, LedgerHeadConflict)):
        ledger.write(_session_event(3))
    assert path.read_bytes() == corrupted


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_custom_append_backends_do_not_skip_their_loaders(tmp_path, monkeypatch, backend_type):
    class CustomBackend(backend_type):
        pass

    backend = CustomBackend(tmp_path / "ledger")
    ledger = ExecutionLedger(backend)
    loads = []
    load = backend._load_unlocked

    def capture_load():
        loads.append(True)
        return load()

    monkeypatch.setattr(backend, "_load_unlocked", capture_load)
    ledger.write(_session_event(1))
    ledger.write(_session_event(2))
    assert len(loads) == 2


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_full_reload_verifies_sealed_history_once_and_preserves_isolated_lineage(tmp_path, monkeypatch, backend_type):
    from dadbot.core import execution_ledger

    path = tmp_path / "ledger"
    writer = ExecutionLedger(backend_type(path))
    for value in range(8):
        writer.write(_session_event(value))
    hashes = []
    hash_event = execution_ledger._event_sha256

    def capture_hash(event):
        hashes.append(event["event_id"])
        return hash_event(event)

    monkeypatch.setattr(execution_ledger, "_event_sha256", capture_hash)
    restored = ExecutionLedger(backend_type(path))
    assert restored.load_from_backend() == 8
    assert len(hashes) == 8
    assert restored.read() == writer.read()
    copied = restored.read()
    copied[0]["payload"]["state"]["value"] = 99
    assert restored.read() == writer.read()
    restored.write(_session_event(8))
    assert restored.read()[-1]["parent_event_id"] == writer.read()[-1]["event_id"]
    assert restored.verify_replay(mode="chain")["ok"]


def test_filtered_reload_still_reseals_its_derived_chain(tmp_path):
    path = tmp_path / "ledger"
    writer = ExecutionLedger(FileWALLedgerBackend(path))
    for value in range(4):
        writer.write(_session_event(value))
    restored = ExecutionLedger(FileWALLedgerBackend(path))
    restored.add_replay_filter(lambda events: events[1:])
    assert restored.load_from_backend() == 3
    assert restored.read()[0]["prev_chain_hash"] == ""
    assert restored.verify_replay(mode="chain")["ok"]


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


def test_trace_job_read_selection_preserves_union_order_and_payload_isolation():
    ledger = ExecutionLedger()
    ledger.write({**_session_event(1), "trace_id": " wanted "})
    ledger.write({**_session_event(2), "trace_id": "other", "payload": {"job_id": " job-1 "}})
    ledger.write({**_session_event(3), "trace_id": "other", "payload": {"job_id": "job-2"}})
    full = ledger.read()
    selected = ledger.read(trace_id="wanted", job_id="job-1")
    assert selected == full[:2]
    assert ledger.read(trace_id="wanted") == full[:1]
    assert ledger.read(job_id="job-1") == full[1:2]
    assert ledger.read(full=False, trace_id="wanted", job_id="job-1") == selected
    headers = ledger.read(trace_id="wanted", job_id="job-1", payload_types=frozenset())
    assert headers == [{key: value for key, value in e.items() if key != "payload"} for e in selected]
    selected[0]["payload"]["state"]["value"] = 9
    assert ledger.read() == full
    ledger._backend_uncertain = True
    with pytest.raises(RuntimeError, match="reload"):
        ledger.read(trace_id="wanted", job_id="job-1")


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


@pytest.mark.parametrize("empty,snapshot,reason", [
    (True, None, "empty"),
    (False, None, "no_snapshot"),
    (False, {"head_sequence": 1}, "durable_checkpoint_required"),
])
def test_disabled_compactor_needs_no_payload_copy(monkeypatch, empty, snapshot, reason):
    ledger = ExecutionLedger()
    if not empty:
        ledger.write(_session_event(1))
        ledger._cache["event_count"] = 0

    def forbid_read(*_args, **_kwargs):
        raise AssertionError("Disabled compaction must not copy payloads")

    monkeypatch.setattr(ledger, "read", forbid_read)
    report = EventCompactor().compact(ledger=ledger, snapshot=snapshot, force=True)
    assert report["reason"] == reason
    assert not report["compacted"]
    assert report["events_removed"] == 0
    ledger._backend_uncertain = True
    with pytest.raises(RuntimeError, match="reload"):
        EventCompactor().compact(ledger=ledger, snapshot=snapshot, force=True)


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
