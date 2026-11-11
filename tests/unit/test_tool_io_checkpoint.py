from copy import deepcopy

import pytest

from dadbot.core.graph_context import TurnContext
from dadbot.core.persistence_record_schema import normalize_checkpoint_record
from dadbot.core.tool_executor import execute_tool
from dadbot.core.tool_recording import ToolIOLedger
from dadbot.managers.conversation_persistence import ConversationPersistenceManager


def _recorded_context():
    context = TurnContext(
        user_input="Approve this Home Assistant action.",
        trace_id="decision-trace",
        metadata={
            "tenant_id": "tenant-a", "session_id": "thread-a",
            "request_id": "decision-request", "tool_version": "test.v1",
        },
    )
    execute_tool(
        tool_name="checkpoint_test", parameters={"value": "input"},
        executor=lambda: {"value": "output"}, turn_context=context,
    )
    assert context._tool_io_ledger is not None
    return context


def test_checkpoint_normalization_retains_complete_kernel_io():
    context = _recorded_context()
    assert context._tool_io_ledger is not None
    expected = context._tool_io_ledger.to_dict()
    snapshot = context.checkpoint_snapshot(stage="save", status="atomic_commit")
    assert snapshot["tool_io_ledger"] == expected
    normalized = normalize_checkpoint_record(snapshot)
    assert normalized["tool_io_ledger"] == expected
    manager = ConversationPersistenceManager.__new__(ConversationPersistenceManager)
    assert manager._thin_checkpoint_payload(normalized)["tool_io_ledger"] == expected
    assert normalize_checkpoint_record(normalized)["tool_io_ledger"] == expected
    restored = ToolIOLedger.from_checkpoint(normalized["tool_io_ledger"])
    assert restored.to_dict() == expected
    normalized["tool_io_ledger"]["records"][0]["output_payload"]["value"] = "changed"
    assert context._tool_io_ledger.to_dict() == expected


@pytest.mark.parametrize("corruption", ["missing_field", "output", "input", "counts", "shape", "nonfinite"])
def test_checkpoint_tool_io_corruption_fails_closed(corruption):
    context = _recorded_context()
    assert context._tool_io_ledger is not None
    data = deepcopy(context._tool_io_ledger.to_dict())
    if corruption == "missing_field":
        del data["records"][0]["tool_name"]
    elif corruption == "output":
        data["records"][0]["output_payload"]["value"] = "tampered"
    elif corruption == "input":
        data["records"][0]["input_payload"]["value"] = "tampered"
    elif corruption == "counts":
        data["sequence_count"] += 1
    elif corruption == "shape":
        data["records"][0] = "not-a-record"
    else:
        data["records"][0]["latency_ms"] = float("nan")
    with pytest.raises(ValueError, match="tool IO"):
        normalize_checkpoint_record({"tool_io_ledger": data})


def test_legacy_checkpoint_without_tool_io_remains_compatible():
    assert "tool_io_ledger" not in normalize_checkpoint_record({"trace_id": "old-trace"})
