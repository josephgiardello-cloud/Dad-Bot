from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import dadbot.core.execution_ledger as ledger_module
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.kernel_gateway import KernelGateway
from dadbot.core.ledger_backend import CRCFileWALLedgerBackend, FileWALLedgerBackend, InMemoryLedgerBackend
from dadbot.core.ledger_writer import LedgerWriter
from dadbot.managers.conversation_persistence import ConversationPersistenceManager
from dadbot.services.persistence import PersistenceService, StateDivergenceError

pytestmark = pytest.mark.unit


def _service(ledger: Any, monkeypatch: pytest.MonkeyPatch) -> PersistenceService:
    manager = ConversationPersistenceManager.__new__(ConversationPersistenceManager)
    monkeypatch.setattr(manager, "_execution_ledger", lambda: ledger)
    monkeypatch.setattr(manager, "load_latest_graph_checkpoint", lambda **_kwargs: {"fallback": True})
    return PersistenceService(manager)


def _invoke(service: PersistenceService, method: str, *args: Any, **kwargs: Any) -> Any:
    return getattr(service, method)(*args, **kwargs)


def _append(
    ledger: ExecutionLedger, event_type: str, payload: dict[str, Any], *, tenant_id: str = "tenant-a",
) -> None:
    with KernelGateway.open_scope():
        LedgerWriter(ledger).write_event(
            event_type,
            tenant_id=tenant_id,
            session_id="session-a",
            trace_token="header-trace",
            step_key="test.checkpoint_lookup",
            payload=payload,
        )


def _payload(trace_id: str, value: str) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "checkpoint": {"trace_id": trace_id, "state": {"result": [value]}},
        "session_state": {"history": [value]},
    }


def test_authority_checkpoint_lookup_copies_only_checkpoint_events(monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = ExecutionLedger()
    _append(ledger, "UNRELATED", {"state": {"large": ["history"] * 100}})
    _append(ledger, "GRAPH_CHECKPOINT", _payload("old-trace", "history"))
    payload = _payload("payload-trace", "saved")
    _append(ledger, "GRAPH_CHECKPOINT", payload)
    service = _service(ledger, monkeypatch)
    copied_types: list[list[str]] = []

    def checked_copy(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        types = [event["type"] for event in events]
        copied_types.append(types)
        assert types == ["GRAPH_CHECKPOINT"]
        assert events[0]["payload"]["trace_id"] == "payload-trace"
        return deepcopy(events)

    monkeypatch.setattr(ledger_module, "_copy_ledger_value", checked_copy)
    result = _invoke(service, "_load_event_sourced_checkpoint", " payload-trace ")
    expected = {**payload["checkpoint"], "session_state": payload["session_state"]}
    assert result == expected
    result["state"]["result"].append("changed")
    result["session_state"]["history"].append("changed")
    assert _invoke(service, "_load_event_sourced_checkpoint", "payload-trace") == expected
    assert copied_types == [["GRAPH_CHECKPOINT"], ["GRAPH_CHECKPOINT"]]


@pytest.mark.parametrize("trace_id", ["target", "", "missing"])
def test_authority_checkpoint_lookup_preserves_latest_payload_trace_and_fallback(
    monkeypatch: pytest.MonkeyPatch, trace_id: str,
) -> None:
    ledger = ExecutionLedger()
    _append(ledger, "GRAPH_CHECKPOINT", _payload("target", "old"))
    _append(ledger, "GRAPH_CHECKPOINT", _payload("target", "new"))
    latest = _payload("other", "latest")
    _append(ledger, "GRAPH_CHECKPOINT", latest)
    _append(ledger, "GRAPH_CHECKPOINT", {"trace_id": trace_id, "checkpoint": {}})
    _append(ledger, "UNRELATED", _payload("target", "not-a-checkpoint"))
    service = _service(ledger, monkeypatch)
    expected = {"fallback": True}
    if trace_id == "target":
        payload = _payload("target", "new")
        expected = {**payload["checkpoint"], "session_state": payload["session_state"]}
    elif not trace_id:
        expected = {**latest["checkpoint"], "session_state": latest["session_state"]}
    assert _invoke(service, "_load_event_sourced_checkpoint", trace_id) == expected


def test_authority_checkpoint_lookup_rejects_uncertain_ledger_before_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = ExecutionLedger()
    _append(ledger, "GRAPH_CHECKPOINT", _payload("target", "saved"))
    service = _service(ledger, monkeypatch)
    ledger._backend_uncertain = True
    with pytest.raises(RuntimeError, match="reload and verify"):
        _invoke(service, "_load_event_sourced_checkpoint", "target")


@pytest.mark.parametrize("custom_kind", ["duck", "subclass", "instance"])
def test_authority_checkpoint_lookup_keeps_custom_full_read_contract(
    monkeypatch: pytest.MonkeyPatch, custom_kind: str,
) -> None:
    payload = _payload("target", "custom")
    events = [{"type": "GRAPH_CHECKPOINT", "payload": payload}]
    calls: list[bool] = []

    class CustomLedger(ExecutionLedger):
        pass

    def read() -> list[dict[str, Any]]:
        calls.append(True)
        return deepcopy(events)

    if custom_kind == "instance":
        ledger = ExecutionLedger()
    elif custom_kind == "subclass":
        ledger = CustomLedger()
    else:
        ledger = SimpleNamespace(read=read)
    monkeypatch.setattr(ledger, "read", read)
    assert _invoke(_service(ledger, monkeypatch), "_load_event_sourced_checkpoint", "target") == {
        **payload["checkpoint"], "session_state": payload["session_state"],
    }
    assert calls == [True]


@pytest.mark.parametrize("mutation", ["none", "checkpoint", "session", "missing_session"])
def test_selective_checkpoint_lookup_preserves_strict_full_state_authority(
    monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    ledger = ExecutionLedger()
    payload = _payload("target", "saved")
    stored = deepcopy(payload)
    if mutation == "checkpoint":
        stored["checkpoint"]["state"]["result"] = ["tampered"]
    elif mutation == "session":
        stored["session_state"]["history"] = ["tampered"]
    elif mutation == "missing_session":
        stored.pop("session_state")
    _append(ledger, "GRAPH_CHECKPOINT", stored)
    service = _service(ledger, monkeypatch)
    service.strict_mode = True
    context = SimpleNamespace(trace_id="target", state={})
    runtime = SimpleNamespace(snapshot_session_state=lambda: deepcopy(payload["session_state"]))

    if mutation != "none":
        with pytest.raises(StateDivergenceError, match="Memory authority divergence"):
            _invoke(service, "_enforce_memory_authority", runtime, context, checkpoint=payload["checkpoint"])
        return
    _invoke(service, "_enforce_memory_authority", runtime, context, checkpoint=payload["checkpoint"])
    report = context.state["memory_authority_check"]
    assert report["consistent"] is True
    assert report["projected_hash"] == report["event_sourced_hash"]


@pytest.mark.parametrize("backend_type", [InMemoryLedgerBackend, FileWALLedgerBackend, CRCFileWALLedgerBackend])
@pytest.mark.parametrize("rolling", [False, True])
def test_checkpoint_references_materialize_exact_history_across_restart(
    tmp_path: Path, backend_type: type, rolling: bool,
) -> None:
    backend = backend_type() if backend_type is InMemoryLedgerBackend else backend_type(tmp_path / "checkpoints.wal")
    ledger = ExecutionLedger(backend)
    history = [{"id": index, "text": "x" * 1000} for index in range(32)]
    expected = []
    for turn in range(4):
        checkpoint = {
            "trace_id": f"trace-{turn}", "history": [*(history[turn:] if rolling else history), *range(turn)],
            "nested": {"history": history, **({"removed": True} if turn == 0 else {})},
            "tool_io_ledger": {"entries": [{"request_id": "original", "result": ["retained"]}]},
            **({"obsolete": True} if turn == 0 else {}),
        }
        expected.append(checkpoint)
        _append(ledger, "GRAPH_CHECKPOINT", {
            "trace_id": checkpoint["trace_id"], "checkpoint": checkpoint,
            "session_state": {"history": checkpoint["history"]},
        })
        if turn == 0:
            _append(ledger, "GRAPH_CHECKPOINT", {"checkpoint": {"history": ["other"]}}, tenant_id="tenant-b")
    wire = [event for event in ledger.read() if event["tenant_id"] == "tenant-a"]
    assert all("checkpoint_history_delta" in event["payload"] for event in wire[1:])
    assert sum(len(json.dumps(event["payload"])) for event in wire) < sum(
        len(json.dumps(item)) for item in expected
    ) / 2
    restored = ExecutionLedger(backend)
    restored.load_from_backend()
    assert restored.verify_replay(mode="chain")["ok"]
    assert restored.replay_hash() == ledger.replay_hash()
    views = [event for event in restored.read_checkpoints() if event["tenant_id"] == "tenant-a"]
    assert [event["payload"]["checkpoint"] for event in views] == expected
    assert [event["payload"]["session_state"] for event in views] == [
        {"history": checkpoint["history"]} for checkpoint in expected
    ]
    assert any(
        field["path"][0] == "session_state"
        for event in wire[1:] for field in event["payload"]["session_state_history_delta"]["fields"]
    )
    assert all("chain_hash" not in event and "checkpoint_history_delta" not in event["payload"] for event in views)
    for turn, checkpoint in enumerate(expected):
        view = restored.read_checkpoints(payload_trace_id=f"trace-{turn}")[0]
        assert view["payload"]["checkpoint"] == checkpoint
        view["payload"]["checkpoint"]["history"].clear()
        assert restored.read_checkpoints(payload_trace_id=f"trace-{turn}")[0]["payload"]["checkpoint"] == checkpoint
    assert restored.read() == ledger.read()


def test_checkpoint_references_reject_corrupt_or_absent_bases_before_append_and_on_reload() -> None:
    from dadbot.core.session_state_codec import encode_session_state_update

    ledger = ExecutionLedger()
    before = {"history": ["retained" * 200]}
    _append(ledger, "GRAPH_CHECKPOINT", {"checkpoint": before})
    encoded = encode_session_state_update(
        {"checkpoint": before}, {"checkpoint": {"history": [*before["history"], "next"]}},
    )
    malformed = {
        **encoded["state"], "checkpoint_history_delta": encoded["state_history_delta"],
    }
    malformed["checkpoint_history_delta"]["fields"][0]["base_hash"] = "0" * 64
    for target in (ledger, ExecutionLedger()):
        records = target.read()
        with pytest.raises(RuntimeError, match="history delta base"):
            _append(target, "GRAPH_CHECKPOINT", malformed)
        assert target.read() == records
        assert target._backend.load() == records
    _append(ledger, "GRAPH_CHECKPOINT", {"checkpoint": {"history": [*before["history"], "next"]}})
    records = ledger.read()
    records[-1]["payload"]["checkpoint_history_delta"]["fields"][0]["base_hash"] = "0" * 64
    ledger._backend._events = records
    with pytest.raises(RuntimeError, match="history delta base"):
        ExecutionLedger(ledger._backend).load_from_backend()


def test_checkpoint_session_history_references_remain_checksum_protected() -> None:
    ledger = ExecutionLedger()
    session = {"history": ["retained" * 200]}
    _append(ledger, "GRAPH_CHECKPOINT", {"checkpoint": {"step": 1}, "session_state": session})
    _append(ledger, "GRAPH_CHECKPOINT", {
        "checkpoint": {"step": 2}, "session_state": {"history": [*session["history"], "next"]},
    })
    records = ledger.read()
    records[-1]["payload"]["session_state_history_delta"]["fields"][0]["append"].append("tampered")
    ledger._backend._events = records
    with pytest.raises(RuntimeError, match="checksum chain"):
        ExecutionLedger(ledger._backend).load_from_backend()


def test_materialized_checkpoint_cache_does_not_mask_changed_wire_payload() -> None:
    ledger = ExecutionLedger()
    before = {"trace_id": "target", "session_id": "session-a", "history": ["retained" * 200]}
    _append(ledger, "GRAPH_CHECKPOINT", {"trace_id": "target", "checkpoint": before})
    _append(ledger, "GRAPH_CHECKPOINT", {
        "trace_id": "target", "checkpoint": {**before, "history": [*before["history"], "next"]},
    })
    assert "checkpoint_history_delta" in ledger.read()[-1]["payload"]
    ledger._events[-1]["payload"]["checkpoint"]["session_id"] = "different"
    latest = ledger.read_checkpoints(payload_trace_id="target")[-1]
    assert latest["payload"]["checkpoint"]["session_id"] == "different"


@pytest.mark.parametrize("custom_kind", ["ledger", "backend", "authenticated"])
def test_custom_or_authenticated_checkpoints_keep_full_payloads(custom_kind: str) -> None:
    class CustomLedger(ExecutionLedger):
        pass

    class CustomBackend(InMemoryLedgerBackend):
        pass

    ledger = CustomLedger() if custom_kind == "ledger" else ExecutionLedger(
        CustomBackend() if custom_kind == "backend" else InMemoryLedgerBackend(),
    )
    if custom_kind == "authenticated":
        ledger.require_authenticated_authority()
    before = {"history": ["retained" * 200]}
    after = {"history": [*before["history"], "next"]}
    _append(ledger, "GRAPH_CHECKPOINT", {"checkpoint": before, "session_state": before})
    _append(ledger, "GRAPH_CHECKPOINT", {"checkpoint": after, "session_state": after})
    assert ledger.read()[-1]["payload"] == {"checkpoint": after, "session_state": after}


@pytest.mark.parametrize("custom_kind", ["duck", "instance"])
def test_manager_checkpoint_lookup_keeps_custom_read_contract(
    monkeypatch: pytest.MonkeyPatch, custom_kind: str,
) -> None:
    checkpoint = {"session_id": "session-a", "trace_id": "target", "state": {"result": ["custom"]}}
    events = [{"type": "GRAPH_CHECKPOINT", "session_id": "session-a", "payload": {
        "trace_id": "target", "checkpoint": checkpoint,
    }}]
    ledger = ExecutionLedger() if custom_kind == "instance" else SimpleNamespace()
    monkeypatch.setattr(ledger, "read", lambda: deepcopy(events), raising=False)
    manager = ConversationPersistenceManager.__new__(ConversationPersistenceManager)
    monkeypatch.setattr(manager, "_execution_ledger", lambda: ledger)
    result = manager._latest_checkpoint_for_trace("target", session_id="session-a")
    assert result is not None
    assert result["execution_state"]["state"] == checkpoint["state"]
