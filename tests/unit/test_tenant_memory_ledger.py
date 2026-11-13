from __future__ import annotations

import json
import subprocess
import sys
import threading
from copy import deepcopy
from types import SimpleNamespace

import pytest

from dadbot.core.execution_ledger import (
    ExecutionLedger,
    WriteBoundaryViolationError,
    _chain_hash,
    _event_sha256,
)
from dadbot.core.ledger_backend import (
    CRCFileWALLedgerBackend,
    FileWALLedgerBackend,
    LedgerHeadConflict,
)
from dadbot.core.tenant_memory_ledger import (
    TenantMemoryLedgerCorruption,
    TenantMemoryOperationConflict,
    TenantMemoryRevisionConflict,
)
from dadbot.memory.storage import MemoryStorageBackend


def _established_local_ledger(tmp_path):
    from dadbot.core.local_tenant_establishment import establish_local_tenant

    backend = FileWALLedgerBackend(tmp_path / "local")
    establish_local_tenant(backend, "a", lambda: {"memories": []}, lambda state: None)
    ledger = ExecutionLedger(backend)
    ledger.load_from_backend()
    ledger.write({
        "type": "TEST", "session_id": "test", "trace_id": "test", "kernel_step_id": "test",
        "timestamp": 1.0, "payload": {},
    })
    return backend, ledger


def test_local_commit_prefix_detects_same_size_tampering_before_publication(tmp_path):
    import os

    from dadbot.core.local_tenant_establishment import commit_local_tenant

    backend, ledger = _established_local_ledger(tmp_path)
    raw = backend._path.read_bytes()
    stat = backend._path.stat()
    changed = raw.replace(b'"memories": []', b'"memories": {}')
    assert changed != raw and len(changed) == len(raw)
    backend._path.write_bytes(changed)
    os.utime(backend._path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    published = []
    with pytest.raises(RuntimeError, match="checksum chain"):
        commit_local_tenant(
            backend, "a", 1, "update", {"memories": []}, lambda: published.append(True),
            lambda state: published.append(True), execution_ledger=ledger,
        )
    assert published == []
    assert backend._path.read_bytes() == changed
    assert backend._append_verifier._snapshot is None


def test_local_commit_with_changed_head_requires_full_validation_and_refresh(tmp_path, monkeypatch):
    from dadbot.core.local_tenant_establishment import commit_local_tenant

    backend, ledger = _established_local_ledger(tmp_path)
    other = ExecutionLedger(FileWALLedgerBackend(backend._path))
    other.load_from_backend()
    other.write({
        "type": "OTHER", "session_id": "other", "trace_id": "other", "kernel_step_id": "other",
        "timestamp": 1.0, "payload": {},
    })
    loads = []
    rehydrate = ExecutionLedger._rehydrate_loaded_events

    def capture(self, *args, **kwargs):
        loads.append(True)
        return rehydrate(self, *args, **kwargs)

    monkeypatch.setattr(ExecutionLedger, "_rehydrate_loaded_events", capture)
    assert commit_local_tenant(
        backend, "a", 1, "update", {"memories": [], "updated": True}, lambda: None,
        lambda state: None, execution_ledger=ledger,
    ) == 2
    assert loads == [True]
    with pytest.raises(RuntimeError, match="uncertain"):
        ledger.read()
    ledger.load_from_backend()
    assert ledger.replay_tenant_memory("a").revision == 2
    assert sum(event["type"] == "OTHER" for event in ledger.read()) == 1
    assert ledger.verify_replay(mode="chain")["ok"]


def test_local_prefix_does_not_reuse_replay_filtered_runtime_head(tmp_path, monkeypatch):
    from dadbot.core.local_tenant_establishment import commit_local_tenant

    backend, ledger = _established_local_ledger(tmp_path)
    ledger.add_replay_filter(lambda events: events)
    loads = []
    rehydrate = ExecutionLedger._rehydrate_loaded_events

    def capture(self, *args, **kwargs):
        loads.append(True)
        return rehydrate(self, *args, **kwargs)

    monkeypatch.setattr(ExecutionLedger, "_rehydrate_loaded_events", capture)
    commit_local_tenant(
        backend, "a", 1, "update", {"memories": []}, lambda: None, lambda state: None, execution_ledger=ledger,
    )
    assert loads == [True]
    ledger.load_from_backend()
    assert ledger.replay_tenant_memory("a").revision == 2


def test_local_prefix_fsync_failure_requires_explicit_recovery_and_idempotent_retry(tmp_path, monkeypatch):
    from dadbot.core import local_tenant_establishment

    backend, ledger = _established_local_ledger(tmp_path)
    fsync = local_tenant_establishment.os.fsync
    published = []

    def fail_fsync(fd):
        raise OSError("synthetic tenant fsync failure")

    monkeypatch.setattr(local_tenant_establishment.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="tenant fsync"):
        local_tenant_establishment.commit_local_tenant(
            backend, "a", 1, "update", {"memories": []}, lambda: published.append(True),
            lambda state: None, execution_ledger=ledger,
        )
    assert not published
    assert backend._append_verifier._snapshot is None
    with pytest.raises(RuntimeError, match="uncertain"):
        ledger.read()
    monkeypatch.setattr(local_tenant_establishment.os, "fsync", fsync)
    ledger.load_from_backend()
    assert local_tenant_establishment.commit_local_tenant(
        backend, "a", 1, "update", {"memories": []}, lambda: published.append(True),
        lambda state: None, execution_ledger=ledger,
    ) == 2
    assert published == [True]
    assert ledger.replay_tenant_memory("a").revision == 2
    assert len([
        event for event in backend.load()
        if event["type"] == "TENANT_MEMORY_COMMITTED" and event["payload"]["operation_id"] == "update"
    ]) == 1


def test_graph_and_semantic_fences_reuse_only_byte_identical_committed_projection(tmp_path, monkeypatch):
    path = tmp_path / "ledger"
    backend = FileWALLedgerBackend(path)
    ledger = ExecutionLedger(backend)
    ledger.commit_tenant_memory(tenant_id="a", expected_revision=0, operation_id="one", state={"memories": []})
    storage = MemoryStorageBackend(
        SimpleNamespace(config=SimpleNamespace(tenant_id="a"), _execution_ledger=ledger, _io_lock=threading.RLock()),
        SimpleNamespace(normalize_memory_store=deepcopy),
    )
    loads = []
    load = backend._load_unlocked

    def capture_load():
        loads.append(True)
        return load()

    monkeypatch.setattr(backend, "_load_unlocked", capture_load)
    with storage.fenced_semantic_index_write([], expected_revision=1) as revision:
        assert revision == 1
    with storage.fenced_graph_index_write(1, {"memories": []}):
        pass
    with storage.fenced_semantic_index_write([], expected_revision=1):
        pass
    assert len(loads) == 0
    other = ExecutionLedger(FileWALLedgerBackend(path))
    other.load_from_backend()
    other.commit_tenant_memory(
        tenant_id="a", expected_revision=1, operation_id="two", state={"memories": [], "changed": True}
    )
    with pytest.raises(TenantMemoryRevisionConflict):
        with storage.fenced_graph_index_write(1, {"memories": []}):
            pytest.fail("A stale graph projection crossed the WAL fence")
    assert len(loads) == 1
    with storage.fenced_graph_index_write(2, {"memories": [], "changed": True}):
        pass
    with pytest.raises(TenantMemoryRevisionConflict):
        with storage.fenced_semantic_index_write([], expected_revision=1):
            pytest.fail("A stale semantic revision crossed the WAL fence")


def test_cached_projection_fence_revalidates_corruption_and_forgets_failed_snapshot(tmp_path):
    path = tmp_path / "ledger"
    backend = FileWALLedgerBackend(path)
    ledger = ExecutionLedger(backend)
    ledger.commit_tenant_memory(tenant_id="a", expected_revision=0, operation_id="one", state={"memories": []})
    storage = MemoryStorageBackend(
        SimpleNamespace(config=SimpleNamespace(tenant_id="a"), _execution_ledger=ledger, _io_lock=threading.RLock()),
        SimpleNamespace(normalize_memory_store=deepcopy),
    )
    with storage.fenced_semantic_index_write([], expected_revision=1):
        pass
    original = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(b'{"partial":')
    with pytest.raises(RuntimeError):
        with storage.fenced_graph_index_write(1, {"memories": []}):
            pytest.fail("Corrupt WAL crossed the projection fence")
    assert storage._fenced_tenant_snapshot is None
    path.write_bytes(original)
    with storage.fenced_graph_index_write(1, {"memories": []}):
        pass


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_unchanged_refresh_does_not_reparse_or_rebuild_verified_ledger(tmp_path, monkeypatch, backend_type):
    backend = backend_type(tmp_path / "ledger")
    ledger = ExecutionLedger(backend)
    ledger.commit_tenant_memory(tenant_id="a", expected_revision=0, operation_id="one", state={"items": [1]})
    first = ledger.refresh_tenant_memory("a")
    rebuilds = ledger.telemetry_snapshot()["cache_rebuild_count"]
    loads = []
    load = backend._load_unlocked

    def capture_load():
        loads.append(True)
        return load()

    monkeypatch.setattr(backend, "_load_unlocked", capture_load)
    second = ledger.refresh_tenant_memory("a")
    assert second.state == first.state
    second.state["items"].append(2)
    assert ledger.refresh_tenant_memory("a").state == {"items": [1]}
    assert not loads
    assert ledger.telemetry_snapshot()["cache_rebuild_count"] == rebuilds


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_refresh_detects_other_writer_and_invalidates_after_local_write(tmp_path, backend_type):
    path = tmp_path / "ledger"
    reader = ExecutionLedger(backend_type(path))
    reader.commit_tenant_memory(tenant_id="a", expected_revision=0, operation_id="one", state={"items": [1]})
    reader.refresh_tenant_memory("a")
    other = ExecutionLedger(backend_type(path))
    other.load_from_backend()
    other.commit_tenant_memory(tenant_id="a", expected_revision=1, operation_id="two", state={"items": [2]})
    assert reader.refresh_tenant_memory("a").state == {"items": [2]}
    reader.commit_tenant_memory(tenant_id="a", expected_revision=2, operation_id="three", state={"items": [3]})
    assert reader.refresh_tenant_memory("a").revision == 3
    assert reader.refresh_tenant_memory("a").state == {"items": [3]}


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_refresh_reverifies_same_size_tampering_even_with_restored_mtime(tmp_path, backend_type):
    import os

    path = tmp_path / "ledger"
    ledger = ExecutionLedger(backend_type(path))
    ledger.commit_tenant_memory(tenant_id="a", expected_revision=0, operation_id="one", state={"value": 1})
    ledger.refresh_tenant_memory("a")
    stat = path.stat()
    raw = path.read_bytes()
    changed = raw.replace(b'"value": 1', b'"value": 9')
    assert changed != raw and len(changed) == len(raw)
    path.write_bytes(changed)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(RuntimeError):
        ledger.refresh_tenant_memory("a")
    with pytest.raises(RuntimeError, match="reload"):
        ledger.read()


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_explicit_reload_never_uses_unchanged_fast_path(tmp_path, monkeypatch, backend_type):
    backend = backend_type(tmp_path / "ledger")
    ledger = ExecutionLedger(backend)
    ledger.commit_tenant_memory(tenant_id="a", expected_revision=0, operation_id="one", state={})
    ledger.refresh_tenant_memory("a")
    loads = []
    load = backend._load_unlocked

    def capture_load():
        loads.append(True)
        return load()

    monkeypatch.setattr(backend, "_load_unlocked", capture_load)
    ledger.load_from_backend()
    assert loads == [True]


def test_new_replay_filter_invalidates_verified_refresh(tmp_path):
    ledger = ExecutionLedger(FileWALLedgerBackend(tmp_path / "ledger"))
    ledger.commit_tenant_memory(tenant_id="a", expected_revision=0, operation_id="one", state={})
    ledger.refresh_tenant_memory("a")
    ledger.add_replay_filter(lambda events: [])
    with pytest.raises(RuntimeError, match="filters cannot modify tenant memory"):
        ledger.refresh_tenant_memory("a")
    with pytest.raises(RuntimeError, match="reload"):
        ledger.read()


def test_stateful_filters_and_custom_backend_loaders_are_not_skipped(tmp_path):
    class CustomBackend(FileWALLedgerBackend):
        loads = 0

        def load(self):
            self.loads += 1
            return super().load()

    backend = CustomBackend(tmp_path / "custom-ledger")
    ledger = ExecutionLedger(backend)
    ledger.commit_tenant_memory(tenant_id="a", expected_revision=0, operation_id="one", state={})
    ledger.refresh_tenant_memory("a")
    ledger.refresh_tenant_memory("a")
    assert backend.loads == 2
    standard = ExecutionLedger(FileWALLedgerBackend(tmp_path / "filtered-ledger"))
    standard.commit_tenant_memory(tenant_id="a", expected_revision=0, operation_id="one", state={})
    calls = []
    drop = []

    def filter_events(events):
        calls.append(True)
        return [] if drop else events

    standard.add_replay_filter(filter_events)
    standard.refresh_tenant_memory("a")
    drop.append(True)
    with pytest.raises(RuntimeError, match="filters cannot modify tenant memory"):
        standard.refresh_tenant_memory("a")
    assert len(calls) == 2


def test_failed_refresh_requires_explicit_recovery_even_if_original_bytes_return(tmp_path):
    path = tmp_path / "ledger"
    ledger = ExecutionLedger(FileWALLedgerBackend(path))
    ledger.commit_tenant_memory(tenant_id="a", expected_revision=0, operation_id="one", state={})
    ledger.refresh_tenant_memory("a")
    raw = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(b'{"partial":')
    with pytest.raises(RuntimeError):
        ledger.refresh_tenant_memory("a")
    path.write_bytes(raw)
    with pytest.raises(RuntimeError, match="reload"):
        ledger.refresh_tenant_memory("a")
    ledger.load_from_backend()
    assert ledger.refresh_tenant_memory("a").revision == 1


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_tenant_memory_replay_isolation_and_idempotency(tmp_path, backend_type):
    path = tmp_path / "ledger"
    ledger = ExecutionLedger(backend_type(path))
    first = ledger.commit_tenant_memory(
        tenant_id="a", expected_revision=0, operation_id="one", state={"items": [1]}
    )
    ledger.commit_tenant_memory(tenant_id="b", expected_revision=0, operation_id="one", state={"items": [2]})
    ledger.commit_tenant_memory(tenant_id="a", expected_revision=1, operation_id="two", state={"items": [3]})
    assert ledger.commit_tenant_memory(
        tenant_id="a", expected_revision=0, operation_id="one", state={"items": [1]}
    ) == first
    with pytest.raises(TenantMemoryOperationConflict):
        ledger.commit_tenant_memory(tenant_id="a", expected_revision=0, operation_id="one", state={"items": [9]})
    with pytest.raises(TenantMemoryRevisionConflict):
        ledger.commit_tenant_memory(tenant_id="a", expected_revision=0, operation_id="three", state={})
    reloaded = ExecutionLedger(backend_type(path))
    assert reloaded.load_from_backend() == 3
    assert reloaded.replay_tenant_memory("a").revision == 2
    assert reloaded.replay_tenant_memory("a").state == {"items": [3]}
    assert reloaded.replay_tenant_memory("b").state == {"items": [2]}
    assert reloaded.commit_tenant_memory(
        tenant_id="a", expected_revision=0, operation_id="one", state={"items": [1]}
    ) == first


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_tenant_memory_replays_exact_committed_source_revision(tmp_path, backend_type):
    ledger = ExecutionLedger(backend_type(tmp_path / "ledger"))
    ledger.commit_tenant_memory(
        tenant_id="a", expected_revision=0, operation_id="one", state={"memories": ["first"]}
    )
    ledger.commit_tenant_memory(
        tenant_id="a", expected_revision=1, operation_id="two", state={"memories": ["second"]}
    )

    source = ledger.replay_tenant_memory_at_revision("a", 1)

    assert source.revision == 1
    assert source.state == {"memories": ["first"]}
    assert list(source.operations) == ["one"]
    with pytest.raises(ValueError, match="not committed"):
        ledger.replay_tenant_memory_at_revision("a", 3)
    with pytest.raises(ValueError, match="nonnegative integer"):
        ledger.replay_tenant_memory_at_revision("a", True)


@pytest.mark.parametrize("state", [{"invalid": float("nan")}, {"bad": (1, 2)}, {1: "not a string"}])
def test_invalid_memory_is_rejected_before_append(tmp_path, state):
    ledger = ExecutionLedger(FileWALLedgerBackend(tmp_path / "ledger"))
    with pytest.raises(ValueError, match="JSON values"):
        ledger.commit_tenant_memory(tenant_id="a", expected_revision=0, operation_id="one", state=state)
    assert ledger.event_count() == 0


def test_unsupported_or_nondurable_backend_is_rejected(tmp_path):
    for ledger in (ExecutionLedger(), ExecutionLedger(FileWALLedgerBackend(tmp_path / "ledger", fsync=False))):
        with pytest.raises(RuntimeError, match="requires"):
            ledger.commit_tenant_memory(tenant_id="a", expected_revision=0, operation_id="one", state={})


@pytest.mark.parametrize("persist", [False, True])
def test_uncertain_append_is_reconciled_by_operation_id(tmp_path, persist):
    class UncertainBackend(FileWALLedgerBackend):
        fail = True

        def append(self, event, *, committed=False):
            if self.fail:
                if persist:
                    super().append(event, committed=committed)
                raise OSError("injected append uncertainty")
            return super().append(event, committed=committed)

    backend = UncertainBackend(tmp_path / "ledger")
    ledger = ExecutionLedger(backend)
    kwargs = {"tenant_id": "a", "expected_revision": 0, "operation_id": "one", "state": {"value": 1}}
    with pytest.raises(OSError, match="uncertainty"):
        ledger.commit_tenant_memory(**kwargs)
    with pytest.raises(RuntimeError, match="reload"):
        ledger.replay_tenant_memory("a")
    backend.fail = False
    ledger.load_from_backend()
    result = ledger.commit_tenant_memory(**kwargs)
    assert result["payload"]["operation_id"] == "one"
    assert ledger.event_count() == 1


@pytest.mark.parametrize("persist", [False, True])
def test_operation_reconciliation_reports_whether_uncertain_append_committed(tmp_path, persist):
    class UncertainBackend(FileWALLedgerBackend):
        fail = True

        def append(self, event, *, committed=False):
            if not self.fail:
                return super().append(event, committed=committed)
            if persist:
                super().append(event, committed=committed)
            raise OSError("uncertain append")

    backend = UncertainBackend(tmp_path / "ledger")
    ledger = ExecutionLedger(backend)
    operation = {"tenant_id": "a", "expected_revision": 0, "operation_id": "one", "state": {"value": 1}}
    with pytest.raises(OSError, match="uncertain append"):
        ledger.commit_tenant_memory(**operation)
    with pytest.raises(RuntimeError, match="reload"):
        ledger.read()

    event = ledger.reconcile_tenant_memory_operation(**operation)
    if persist:
        assert event is not None
        assert event["payload"]["operation_id"] == "one"
        with pytest.raises(TenantMemoryOperationConflict):
            ledger.reconcile_tenant_memory_operation(**{**operation, "state": {"value": 2}})
    else:
        assert event is None
    assert ledger.replay_tenant_memory("a").revision == int(persist)
    backend.fail = False
    result = ledger.commit_tenant_memory(**operation)
    assert result["payload"]["operation_id"] == "one"
    assert ledger.event_count() == 1


def test_failed_operation_reconciliation_keeps_ledger_blocked(tmp_path, monkeypatch):
    backend = FileWALLedgerBackend(tmp_path / "ledger")
    ledger = ExecutionLedger(backend)
    ledger._backend_uncertain = True

    def fail_load():
        raise OSError("cannot read ledger")

    monkeypatch.setattr(backend, "load", fail_load)
    with pytest.raises(OSError, match="cannot read ledger"):
        ledger.reconcile_tenant_memory_operation(
            tenant_id="a", expected_revision=0, operation_id="one", state={}
        )
    with pytest.raises(RuntimeError, match="reload"):
        ledger.read()


def test_stale_writer_reloads_before_retry(tmp_path):
    path = tmp_path / "ledger"
    a = ExecutionLedger(FileWALLedgerBackend(path))
    b = ExecutionLedger(FileWALLedgerBackend(path))
    a.commit_tenant_memory(tenant_id="a", expected_revision=0, operation_id="one", state={"value": 1})
    with pytest.raises(LedgerHeadConflict):
        b.commit_tenant_memory(tenant_id="a", expected_revision=0, operation_id="two", state={"value": 2})
    b.load_from_backend()
    with pytest.raises(TenantMemoryRevisionConflict):
        b.commit_tenant_memory(tenant_id="a", expected_revision=0, operation_id="two", state={"value": 2})
    assert b.replay_tenant_memory("a").state == {"value": 1}


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
@pytest.mark.parametrize("corruption", ["revision", "operation"])
def test_backend_rejects_invalid_sealed_tenant_append_under_writer_lock(tmp_path, backend_type, corruption):
    backend = backend_type(tmp_path / "ledger")
    ledger = ExecutionLedger(backend)
    first = ledger.commit_tenant_memory(
        tenant_id="a", expected_revision=0, operation_id="one", state={"value": 1}
    )
    forged = dict(first)
    forged["payload"] = {
        **first["payload"],
        "operation_id": "one" if corruption == "operation" else "two",
        "expected_revision": 1 if corruption == "operation" else 0,
        "revision": 2,
        "state": {"value": 2},
    }
    forged["sequence"] = 2
    forged["_seq"] = 1
    forged["event_id"] = "forged-event"
    forged["prev_chain_hash"] = first["chain_hash"]
    forged["event_sha256"] = _event_sha256(forged)
    forged["chain_hash"] = _chain_hash(forged["prev_chain_hash"], forged["event_sha256"])

    with pytest.raises(TenantMemoryLedgerCorruption, match="revision or operation"):
        backend.append(forged, committed=True)

    recovered = ExecutionLedger(backend_type(tmp_path / "ledger"))
    assert recovered.load_from_backend() == 1
    assert recovered.replay_tenant_memory("a").state == {"value": 1}


def test_bad_revision_in_event_is_not_replayed(tmp_path):
    backend = FileWALLedgerBackend(tmp_path / "ledger")
    invalid_event = {
        "type": "TENANT_MEMORY_COMMITTED",
        "session_id": "",
        "trace_id": "",
        "kernel_step_id": "tenant-memory",
        "timestamp": "2026-01-01",
        "committed": True,
        "payload": {"tenant_id": "a", "operation_id": "x", "expected_revision": 0, "revision": 2, "state": {}},
    }
    with pytest.raises(WriteBoundaryViolationError):
        ExecutionLedger(backend).write(invalid_event)
    (tmp_path / "ledger").write_text(json.dumps(invalid_event) + "\n", encoding="utf-8")
    ledger = ExecutionLedger(backend)
    with pytest.raises(TenantMemoryLedgerCorruption):
        ledger.load_from_backend()
    with pytest.raises(RuntimeError, match="reload"):
        ledger.read()


def test_load_rejects_malformed_event_for_another_tenant(tmp_path):
    backend = FileWALLedgerBackend(tmp_path / "ledger")
    invalid_event = {
        "type": "TENANT_MEMORY_COMMITTED",
        "session_id": "",
        "trace_id": "",
        "kernel_step_id": "tenant-memory",
        "timestamp": "2026-01-01",
        "committed": True,
        "payload": {"tenant_id": "other", "operation_id": "x", "expected_revision": 0, "revision": 2, "state": {}},
    }
    (tmp_path / "ledger").write_text(json.dumps(invalid_event) + "\n", encoding="utf-8")
    ledger = ExecutionLedger(backend)
    with pytest.raises(TenantMemoryLedgerCorruption):
        ledger.load_from_backend()
    with pytest.raises(RuntimeError, match="reload"):
        ledger.replay_tenant_memory("healthy")


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_backend_rejects_unsealed_tenant_event_before_writing(tmp_path, backend_type):
    backend = backend_type(tmp_path / "ledger")
    event = {
        "type": "TENANT_MEMORY_COMMITTED",
        "committed": True,
        "payload": {
            "tenant_id": "a",
            "operation_id": "one",
            "expected_revision": 0,
            "revision": 1,
            "state": {"value": 1},
        },
    }

    with pytest.raises(TenantMemoryLedgerCorruption, match="sealed"):
        backend.append(event, committed=True)

    assert backend.load() == []


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_load_rejects_unsealed_tenant_event_even_if_payload_is_valid(tmp_path, backend_type):
    backend = backend_type(tmp_path / "ledger")
    event = {
        "type": "TENANT_MEMORY_COMMITTED",
        "session_id": "",
        "trace_id": "",
        "kernel_step_id": "tenant-memory",
        "timestamp": "2026-01-01",
        "committed": True,
        "payload": {
            "tenant_id": "a",
            "operation_id": "one",
            "expected_revision": 0,
            "revision": 1,
            "state": {"value": 1},
        },
    }
    if backend_type is CRCFileWALLedgerBackend:
        from dadbot.core.durability import CRC32LineCodec

        encoded = CRC32LineCodec.encode(event)
    else:
        encoded = json.dumps(event) + "\n"
    (tmp_path / "ledger").write_text(encoded, encoding="utf-8")
    ledger = ExecutionLedger(backend)
    with pytest.raises(TenantMemoryLedgerCorruption, match="sealed"):
        ledger.load_from_backend()
    with pytest.raises(RuntimeError, match="reload"):
        ledger.read()


def test_replay_filter_cannot_discard_tenant_events(tmp_path):
    backend = FileWALLedgerBackend(tmp_path / "ledger")
    ExecutionLedger(backend).commit_tenant_memory(
        tenant_id="a", expected_revision=0, operation_id="one", state={}
    )
    reloaded = ExecutionLedger(backend)
    reloaded.add_replay_filter(lambda events: [])
    with pytest.raises(RuntimeError, match="cannot modify tenant memory"):
        reloaded.load_from_backend()
    with pytest.raises(RuntimeError, match="reload"):
        reloaded.read()


@pytest.mark.parametrize("after_append", [False, True])
def test_subprocess_exit_around_tenant_append(tmp_path, after_append):
    path = tmp_path / "ledger"
    ledger = ExecutionLedger(FileWALLedgerBackend(path))
    ledger.commit_tenant_memory(tenant_id="a", expected_revision=0, operation_id="one", state={"value": 1})
    script = """
import os
import sys
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_backend import FileWALLedgerBackend
ledger = ExecutionLedger(FileWALLedgerBackend(sys.argv[1]))
ledger.load_from_backend()
if sys.argv[2] == "before":
    os._exit(73)
ledger.commit_tenant_memory(tenant_id="a", expected_revision=1, operation_id="two", state={"value": 2})
os._exit(73)
"""
    proc = subprocess.run(
        [sys.executable, "-c", script, str(path), "after" if after_append else "before"],
        timeout=30, check=False,
    )
    assert proc.returncode == 73
    recovered = ExecutionLedger(FileWALLedgerBackend(path))
    recovered.load_from_backend()
    assert recovered.replay_tenant_memory("a").state == {"value": 2 if after_append else 1}
