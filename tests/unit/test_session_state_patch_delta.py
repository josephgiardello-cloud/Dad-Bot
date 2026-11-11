from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from dadbot.core.control_plane import ControlPlaneOptions, ExecutionControlPlane, SessionRegistry
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.kernel_gateway import KernelGateway
from dadbot.core.ledger_backend import CRCFileWALLedgerBackend, FileWALLedgerBackend
from dadbot.core.ledger_writer import LedgerWriter
from dadbot.core.session_store import SessionStore
from dadbot.core.tenant_session import tenant_session_scope


def _plane(ledger: ExecutionLedger) -> ExecutionControlPlane:
    async def executor(_session: dict[str, Any], _job: Any) -> tuple[str, bool]:
        return "ok", False

    return ExecutionControlPlane(
        registry=SessionRegistry(),
        kernel_executor=executor,
        options=ControlPlaneOptions(ledger=ledger),
    )


def _merge(plane: ExecutionControlPlane, patch: dict[str, Any], *, tenant: str = "tenant-a") -> None:
    with KernelGateway.open_scope():
        plane._merge_session_state(
            session_id="shared", tenant_id=tenant,
            state_patch=patch, trace_token="test-trace",
        )


def _state_events(ledger: ExecutionLedger) -> list[dict[str, Any]]:
    return ledger.read(event_types=frozenset({"SESSION_STATE_UPDATED"}))


def _state(plane: ExecutionControlPlane) -> dict[str, Any]:
    session = plane.registry.get("shared", tenant_id="tenant-a")
    assert session is not None
    return session["state"]


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_session_state_writes_omit_unchanged_history_without_losing_commits(
    tmp_path: Path, backend_type: type[FileWALLedgerBackend] | type[CRCFileWALLedgerBackend],
) -> None:
    ledger = ExecutionLedger(backend_type(tmp_path / "state.wal"))
    plane = _plane(ledger)
    history = [{"event_id": str(index), "payload": {"value": "x" * 1000}} for index in range(20)]
    initial = {"core_state": {"events": history}, "counter": 1}
    _merge(plane, initial)
    _merge(plane, {**initial, "counter": 2})
    _merge(plane, {**initial, "counter": 2})

    events = _state_events(ledger)
    assert len(events) == 3
    assert events[0]["payload"]["state"] == initial
    assert events[1]["payload"]["state"] == {"counter": 2}
    assert events[2]["payload"]["state"] == {}
    assert all(event["committed"] for event in events)
    assert _state(plane) == {**initial, "counter": 2}
    restored = ExecutionLedger(backend_type(tmp_path / "state.wal"))
    restored.load_from_backend()
    store = SessionStore(projection_only=True)
    store.rebuild_from_ledger(restored.read())
    assert store.get("shared", tenant_id="tenant-a") == {**initial, "counter": 2}
    assert restored.verify_replay(mode="chain")["ok"] is True
    assert restored.replay_hash() == ledger.replay_hash()


@pytest.mark.parametrize("backend_type", [FileWALLedgerBackend, CRCFileWALLedgerBackend])
def test_empty_state_patch_still_fsyncs_its_committed_event(
    tmp_path: Path, backend_type: type[FileWALLedgerBackend] | type[CRCFileWALLedgerBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dadbot.core import ledger_backend

    ledger = ExecutionLedger(backend_type(tmp_path / "state.wal"))
    plane = _plane(ledger)
    _merge(plane, {"value": "same"})
    calls: list[int] = []
    fsync = ledger_backend.os.fsync

    def record_fsync(fd: int) -> None:
        calls.append(fd)
        fsync(fd)

    monkeypatch.setattr(ledger_backend.os, "fsync", record_fsync)
    _merge(plane, {"value": "same"})
    assert len(calls) == 1
    assert _state_events(ledger)[-1]["payload"]["state"] == {}
    assert _state_events(ledger)[-1]["committed"] is True


def test_state_comparison_and_append_share_the_ledger_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FileWALLedgerBackend(tmp_path / "state.wal")
    ledger = ExecutionLedger(backend)
    plane = _plane(ledger)
    _merge(plane, {"value": "same"})
    original_delta = ExecutionLedger.session_state_delta
    original_append = backend.append
    checked: list[str] = []

    def lock_available_to_another_thread() -> bool:
        acquired = ledger._lock.acquire(blocking=False)
        if acquired:
            ledger._lock.release()
        return acquired

    with ThreadPoolExecutor(max_workers=1) as worker:
        def delta(
            self: ExecutionLedger, *, tenant_id: str, session_id: str, state_patch: dict[str, Any],
        ) -> dict[str, Any]:
            result = original_delta(self, tenant_id=tenant_id, session_id=session_id, state_patch=state_patch)
            assert worker.submit(lock_available_to_another_thread).result(timeout=2) is False
            checked.append("comparison_complete")
            return result

        def append(event: dict[str, Any], *, committed: bool = False) -> None:
            assert worker.submit(lock_available_to_another_thread).result(timeout=2) is False
            checked.append("append")
            original_append(event, committed=committed)

        monkeypatch.setattr(ExecutionLedger, "session_state_delta", delta)
        monkeypatch.setattr(backend, "append", append)
        _merge(plane, {"value": "same"})
    assert checked == ["comparison_complete", "append"]


def test_state_delta_uses_committed_values_not_mutated_registry(tmp_path: Path) -> None:
    ledger = ExecutionLedger(FileWALLedgerBackend(tmp_path / "state.wal"))
    plane = _plane(ledger)
    _merge(plane, {"nested": {"values": ["before"]}})
    state = _state(plane)
    state["nested"]["values"].append("after")
    _merge(plane, state)
    assert _state_events(ledger)[-1]["payload"]["state"] == {"nested": {"values": ["before", "after"]}}


def test_state_delta_preserves_json_types_nulls_and_normalization(tmp_path: Path) -> None:
    ledger = ExecutionLedger(FileWALLedgerBackend(tmp_path / "state.wal"))
    plane = _plane(ledger)
    _merge(plane, {"flag": True, "number": 1, "zero": 0.0, "tuple": (1, 2), "missing": None})
    changed = {"flag": 1, "number": 1.0, "zero": -0.0, "tuple": [1, 2], "missing": None}
    _merge(plane, changed)
    delta = _state_events(ledger)[-1]["payload"]["state"]
    assert set(delta) == {"flag", "number", "zero"}
    assert json.dumps(delta, sort_keys=True) == json.dumps(
        {"flag": 1, "number": 1.0, "zero": -0.0}, sort_keys=True,
    )


def test_state_delta_isolates_tenants_and_respects_partial_prior_patches(tmp_path: Path) -> None:
    ledger = ExecutionLedger(FileWALLedgerBackend(tmp_path / "state.wal"))
    plane = _plane(ledger)
    _merge(plane, {"value": "a", "nested": {"keep": True}})
    _merge(plane, {"value": "b"}, tenant="tenant-b")
    _merge(plane, {"other": "a"})
    _merge(plane, {"value": "a", "nested": {"keep": True}, "other": "a"})
    assert _state_events(ledger)[-1]["payload"]["state"] == {}
    _merge(plane, {"value": "a", "nested": {"keep": True}}, tenant="tenant-b")
    assert _state_events(ledger)[-1]["payload"]["state"] == {"value": "a", "nested": {"keep": True}}


def test_state_delta_respects_session_deletion_and_job_result_projection(tmp_path: Path) -> None:
    ledger = ExecutionLedger(FileWALLedgerBackend(tmp_path / "state.wal"))
    plane = _plane(ledger)
    _merge(plane, {"value": "before", "last_result": ["old", False]})
    writer = LedgerWriter(ledger)
    with KernelGateway.open_scope():
        writer.write_event(
            "JOB_COMPLETED", session_id="shared", tenant_id="tenant-a",
            trace_token="test-trace", step_key="test", payload={"job_id": "job", "result": ["done", False]},
        )
    _merge(plane, {"last_result": ["done", False]})
    assert _state_events(ledger)[-1]["payload"]["state"] == {}
    with KernelGateway.open_scope():
        writer.write_event(
            "SESSION_STATE_DELETED", session_id="shared", tenant_id="tenant-a",
            trace_token="test-trace", step_key="test", payload={},
        )
    _merge(plane, {"value": "before"})
    assert _state_events(ledger)[-1]["payload"]["state"] == {"value": "before"}


def test_state_delta_revalidates_reload_and_rejects_uncertainty(tmp_path: Path) -> None:
    path = tmp_path / "state.wal"
    ledger = ExecutionLedger(FileWALLedgerBackend(path))
    plane = _plane(ledger)
    _merge(plane, {"value": "before"})
    other = ExecutionLedger(FileWALLedgerBackend(path))
    other.load_from_backend()
    _merge(_plane(other), {"value": "outside"})
    ledger.load_from_backend(only_if_changed=True)
    _merge(plane, {"value": "before"})
    assert _state_events(ledger)[-1]["payload"]["state"] == {"value": "before"}
    ledger._backend_uncertain = True
    with pytest.raises(RuntimeError, match="reload and verify"):
        _merge(plane, {"value": "before"})


def test_state_delta_revalidates_same_count_replacement(tmp_path: Path) -> None:
    path = tmp_path / "state.wal"
    ledger = ExecutionLedger(FileWALLedgerBackend(path))
    plane = _plane(ledger)
    _merge(plane, {"value": "before"})
    replacement = tmp_path / "replacement.wal"
    _merge(_plane(ExecutionLedger(FileWALLedgerBackend(replacement))), {"value": "replacement"})
    path.write_bytes(replacement.read_bytes())
    ledger.load_from_backend(only_if_changed=True)
    assert len(_state_events(ledger)) == 1
    _merge(plane, {"value": "before"})
    assert _state_events(ledger)[-1]["payload"]["state"] == {"value": "before"}


def test_state_delta_preserves_snapshot_and_reducer_projection(tmp_path: Path) -> None:
    from dadbot.core.event_reducer import CanonicalEventReducer
    from dadbot.core.snapshot_engine import SnapshotEngine

    ledger = ExecutionLedger(FileWALLedgerBackend(tmp_path / "state.wal"))
    plane = _plane(ledger)
    full_patch = {"nested": {"events": [{"type": "retained"}]}, "counter": 1}
    _merge(plane, full_patch)
    _merge(plane, {**full_patch, "counter": 2})
    expected = {**full_patch, "counter": 2}
    assert CanonicalEventReducer().reduce(ledger.read())["sessions"]["shared"] == expected
    store = SessionStore(projection_only=True)
    store.rebuild_from_ledger(ledger.read())
    engine = SnapshotEngine()
    with KernelGateway.open_scope():
        snapshot = engine.take_snapshot(ledger=ledger, session_store=store)
    restored = SessionStore(projection_only=True)
    engine.restore_from_snapshot(snapshot, session_store=restored)
    assert restored.get("shared", tenant_id="tenant-a") == expected


@pytest.mark.parametrize("mode", ["memory", "custom_ledger", "custom_backend", "filtered", "authenticated"])
def test_custom_or_authority_bound_state_writes_keep_complete_patches(tmp_path: Path, mode: str) -> None:
    class CustomLedger(ExecutionLedger):
        pass

    class CustomBackend(FileWALLedgerBackend):
        pass

    ledger_type = CustomLedger if mode == "custom_ledger" else ExecutionLedger
    backend_type = CustomBackend if mode == "custom_backend" else FileWALLedgerBackend
    ledger = ledger_type() if mode == "memory" else ledger_type(backend_type(tmp_path / "state.wal"))
    plane = _plane(ledger)
    patch = {"nested": {"values": ["retained"]}}
    _merge(plane, patch)
    if mode == "filtered":
        ledger.add_replay_filter(lambda events: events)
    elif mode == "authenticated":
        ledger.require_authenticated_authority()
    _merge(plane, deepcopy(patch))
    assert _state_events(ledger)[-1]["payload"]["state"] == patch


def test_session_state_failure_never_publishes_registry_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = ExecutionLedger(FileWALLedgerBackend(tmp_path / "state.wal"))
    plane = _plane(ledger)
    _merge(plane, {"counter": 1})

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("failed append")

    monkeypatch.setattr(ledger._backend, "append", fail)
    with pytest.raises(OSError, match="failed append"):
        _merge(plane, {"counter": 2})
    assert _state(plane)["counter"] == 1
    assert ledger._backend_uncertain is True


def test_session_state_patch_rejects_cross_tenant_scope(tmp_path: Path) -> None:
    ledger = ExecutionLedger(FileWALLedgerBackend(tmp_path / "state.wal"))
    plane = _plane(ledger)
    _merge(plane, {"value": "a"})
    with tenant_session_scope("tenant-a", "shared"):
        with pytest.raises(ValueError, match="tenant identity"):
            _merge(plane, {"value": "a"}, tenant="tenant-b")


def test_stale_state_delta_append_fails_before_registry_publication(tmp_path: Path) -> None:
    path = tmp_path / "state.wal"
    ledger = ExecutionLedger(FileWALLedgerBackend(path))
    plane = _plane(ledger)
    _merge(plane, {"value": "before"})
    outside = ExecutionLedger(FileWALLedgerBackend(path))
    outside.load_from_backend()
    _merge(_plane(outside), {"value": "outside"})
    with pytest.raises(RuntimeError, match="head changed"):
        _merge(plane, {"value": "before"})
    assert _state(plane) == {"value": "before"}
    assert ledger._backend_uncertain is True
    recovered = ExecutionLedger(FileWALLedgerBackend(path))
    recovered.load_from_backend()
    assert _state_events(recovered)[-1]["payload"]["state"] == {"value": "outside"}


def test_state_delta_results_are_isolated_and_custom_method_retains_full_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = ExecutionLedger(FileWALLedgerBackend(tmp_path / "state.wal"))
    plane = _plane(ledger)
    _merge(plane, {"nested": {"values": ["before"]}})
    source = {"nested": {"values": ["after"]}}
    delta = ledger.session_state_delta(tenant_id="tenant-a", session_id="shared", state_patch=source)
    delta["nested"]["values"].append("changed")
    assert source["nested"]["values"] == ["after"]
    assert _state_events(ledger)[-1]["payload"]["state"]["nested"]["values"] == ["before"]

    def custom(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("Custom delta methods must retain full-write compatibility")

    monkeypatch.setattr(ledger, "session_state_delta", custom)
    _merge(plane, {"nested": {"values": ["before"]}})
    assert _state_events(ledger)[-1]["payload"]["state"] == {"nested": {"values": ["before"]}}


@pytest.mark.parametrize("mode", ["delegated", "authenticated_backend", "no_fsync", "unverified"])
def test_state_delta_direct_authority_and_verification_guards(
    tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dadbot_system import delegation

    backend = FileWALLedgerBackend(tmp_path / "state.wal")
    ledger = ExecutionLedger(backend)
    _merge(_plane(ledger), {"value": "before"})
    if mode == "authenticated_backend":
        backend._security_mode = "AUTHENTICATED"
    elif mode == "no_fsync":
        backend._fsync_enabled = False
    elif mode == "unverified":
        ledger._backend_content_digest = None
    if mode == "delegated":
        authority: ContextVar[bool] = ContextVar("test_delta_authority", default=True)
        monkeypatch.setattr(delegation, "_execution_authority", authority)
    assert ledger.session_state_delta(
        tenant_id="tenant-a", session_id="shared", state_patch={"value": "before"},
    ) == {"value": "before"}
