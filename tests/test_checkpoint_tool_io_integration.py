"""Integration test: checkpoint persistence with tool_io_ledger.

Validates that:
1. tool_io_ledger is persisted in checkpoint.to_dict()
2. tool_io_ledger is restored from checkpoint.from_dict()
3. Round-trip maintains all tool IO records
"""

from __future__ import annotations

import pytest

from dadbot.core.execution_checkpoint import ExecutionCheckpoint, NodeExecutionSnapshot, NodeState
from dadbot.core.tool_recording import ToolIOLedger, ToolIORecord

pytestmark = pytest.mark.unit


class TestCheckpointToolIOIntegration:
    """Checkpoint persistence with tool_io_ledger."""

    def test_checkpoint_with_empty_tool_io_ledger(self) -> None:
        checkpoint = ExecutionCheckpoint(
            checkpoint_id="ckpt-1",
            label="test",
            created_at=0.0,
            node_snapshots={},
            prev_checkpoint_hash="",
            tool_io_ledger=ToolIOLedger(),
        )
        
        data = checkpoint.to_dict()
        assert "tool_io_ledger" in data
        assert data["tool_io_ledger"]["sequence_count"] == 0

    def test_checkpoint_with_populated_tool_io_ledger(self) -> None:
        ledger = ToolIOLedger()
        record = ToolIORecord(
            sequence=1,
            tool_name="set_reminder",
            input_hash="h1",
            input_payload={"title": "Test"},
            output_payload={"id": "r1"},
            output_hash="o1",
            status="ok",
            latency_ms=10.0,
        )
        ledger.append(record)
        
        checkpoint = ExecutionCheckpoint(
            checkpoint_id="ckpt-1",
            label="test",
            created_at=0.0,
            node_snapshots={},
            prev_checkpoint_hash="",
            tool_io_ledger=ledger,
        )
        
        data = checkpoint.to_dict()
        assert data["tool_io_ledger"]["sequence_count"] == 1
        assert len(data["tool_io_ledger"]["records"]) == 1
        assert data["tool_io_ledger"]["records"][0]["tool_name"] == "set_reminder"

    def test_checkpoint_round_trip_preserves_tool_io_ledger(self) -> None:
        # Build original checkpoint
        ledger = ToolIOLedger()
        records = [
            ToolIORecord(
                sequence=1,
                tool_name="set_reminder",
                input_hash="h1",
                input_payload={"title": "Reminder 1"},
                output_payload={"id": "r1"},
                output_hash="o1",
                status="ok",
                latency_ms=10.0,
            ),
            ToolIORecord(
                sequence=2,
                tool_name="web_search",
                input_hash="h2",
                input_payload={"query": "coffee"},
                output_payload={"results": []},
                output_hash="o2",
                status="ok",
                latency_ms=50.0,
            ),
        ]
        for rec in records:
            ledger.append(rec)
        
        original_checkpoint = ExecutionCheckpoint(
            checkpoint_id="ckpt-1",
            label="test",
            created_at=123.456,
            node_snapshots={},
            prev_checkpoint_hash="prev",
            checkpoint_hash="hash",
            tool_io_ledger=ledger,
        )
        
        # Serialize
        data = original_checkpoint.to_dict()
        
        # Deserialize
        restored_checkpoint = ExecutionCheckpoint.from_dict(data)
        
        # Verify
        assert restored_checkpoint.checkpoint_id == "ckpt-1"
        assert restored_checkpoint.label == "test"
        assert len(restored_checkpoint.tool_io_ledger.records) == 2
        assert restored_checkpoint.tool_io_ledger.successful_calls() == 2
        
        # Verify ledger contents
        first_rec = restored_checkpoint.tool_io_ledger.records[0]
        assert first_rec.tool_name == "set_reminder"
        assert first_rec.input_payload["title"] == "Reminder 1"
        
        second_rec = restored_checkpoint.tool_io_ledger.records[1]
        assert second_rec.tool_name == "web_search"
        assert second_rec.lookup_key() == "web_search|h2||"

    def test_checkpoint_lookup_after_restore(self) -> None:
        # Build and populate ledger
        ledger = ToolIOLedger()
        record = ToolIORecord(
            sequence=1,
            tool_name="set_reminder",
            input_hash="reminder_hash",
            input_payload={"title": "Test"},
            output_payload={"id": "r1"},
            output_hash="o1",
            status="ok",
            latency_ms=10.0,
        )
        ledger.append(record)
        
        checkpoint = ExecutionCheckpoint(
            checkpoint_id="ckpt-1",
            label="test",
            created_at=0.0,
            node_snapshots={},
            prev_checkpoint_hash="",
            tool_io_ledger=ledger,
        )
        
        # Round trip
        data = checkpoint.to_dict()
        restored = ExecutionCheckpoint.from_dict(data)
        
        # Test fast lookup works after restore
        found = restored.tool_io_ledger.lookup("set_reminder", "reminder_hash")
        assert found is not None
        assert found.output_payload == {"id": "r1"}

    def test_checkpoint_from_dict_with_missing_tool_io_ledger(self) -> None:
        # Simulate old checkpoint without tool_io_ledger field
        data = {
            "checkpoint_id": "ckpt-1",
            "label": "test",
            "created_at": 0.0,
            "nodes": {},
            "prev_checkpoint_hash": "",
            "checkpoint_hash": "hash",
            # No tool_io_ledger field
        }
        
        checkpoint = ExecutionCheckpoint.from_dict(data)
        assert checkpoint.tool_io_ledger.is_empty()
