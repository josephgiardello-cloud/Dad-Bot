from __future__ import annotations

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_reader import LedgerReader
from dadbot.core.ledger_writer import LedgerWriter


def test_summarize_policy_events_rolls_up_actions_and_policy() -> None:
    ledger = ExecutionLedger()
    writer = LedgerWriter(ledger)
    reader = LedgerReader(ledger)

    writer.write_event(
        event_type="PolicyTraceEvent",
        session_id="s-1",
        trace_id="t-1",
        kernel_step_id="save_node.policy_trace",
        payload={
            "summary": {
                "policy": "safety",
                "decision_action": "handled",
                "step_name": "validate",
            },
        },
        committed=False,
    )
    writer.write_event(
        event_type="PolicyTraceEvent",
        session_id="s-1",
        trace_id="t-1",
        kernel_step_id="save_node.policy_trace",
        payload={
            "summary": {
                "policy": "safety",
                "decision_action": "passthrough",
                "step_name": "passthrough",
            },
        },
        committed=False,
    )

    summary = reader.summarize_policy_trace_events(session_id="s-1", trace_id="t-1")

    assert summary["event_type"] == "PolicyTraceEvent"
    assert summary["event_count"] == 2
    assert summary["policies"] == ["safety"]
    assert summary["action_counts"] == {"handled": 1, "passthrough": 1}
    assert summary["latest_action"] == "passthrough"
    assert summary["latest_step_name"] == "passthrough"
    assert summary["latest_trace_id"] == "t-1"
