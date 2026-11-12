from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

import dadbot.core.execution_ledger as ledger_module
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.kernel_gateway import KernelGateway
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


def _append(ledger: ExecutionLedger, event_type: str, payload: dict[str, Any]) -> None:
    with KernelGateway.open_scope():
        LedgerWriter(ledger).write_event(
            event_type,
            tenant_id="tenant-a",
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

    monkeypatch.setattr(ledger_module, "deepcopy", checked_copy)
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
