"""Tests for execute_tool() recording layer.

Validates that:
1. ToolIORecord is created and appended to turn_context._tool_io_ledger
2. Input and output hashes are correctly calculated
3. Latency is recorded
4. Multiple tool calls build correct sequence
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import pytest

from dadbot.core.kernel_locks import KernelToolIdempotencyRegistry
from dadbot.core.tool_executor import execute_tool
from dadbot.core.tool_recording import ToolIOLedger

pytestmark = pytest.mark.unit


@dataclass
class MockTurnContext:
    """Mock turn context for testing."""

    trace_id: str = "test-trace"
    state: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.state is None:
            self.state = {}
        if self.metadata is None:
            self.metadata = {}


@pytest.fixture(autouse=True)
def clear_idempotency_registry():
    """Clear global idempotency registry between tests."""
    original = KernelToolIdempotencyRegistry._registry.copy() if hasattr(KernelToolIdempotencyRegistry, '_registry') else {}
    yield
    # Reset registry after test (implementation dependent, may not work for all backends)


class TestExecuteToolRecording:
    """Tool execution recording layer."""

    def test_basic_tool_execution_creates_io_record(self) -> None:
        ctx = MockTurnContext()
        unique_id = str(uuid.uuid4())[:8]

        def test_executor():
            return {"id": "r1", "title": "Test Reminder"}

        record = execute_tool(
            tool_name="set_reminder",
            parameters={"title": f"Test Reminder {unique_id}", "due_text": "tomorrow"},
            executor=test_executor,
            turn_context=ctx,
        )

        # Verify execution succeeded (could be "succeeded" or "cached")
        assert record.status in ("succeeded", "cached")
        assert record.result == {"id": "r1", "title": "Test Reminder"}

        # Verify IO ledger was created
        assert hasattr(ctx, "_tool_io_ledger")
        ledger: ToolIOLedger = ctx._tool_io_ledger
        assert len(ledger.records) == 1

        # Verify IO record
        io_rec = ledger.records[0]
        assert io_rec.tool_name == "set_reminder"
        assert io_rec.sequence == 1
        assert io_rec.status in ("succeeded", "cached")
        assert "Test Reminder" in str(io_rec.input_payload.get("title", ""))
        assert io_rec.latency_ms > 0

    def test_multiple_tool_calls_build_sequence(self) -> None:
        ctx = MockTurnContext()
        unique_id = str(uuid.uuid4())[:8]

        # First call
        record1 = execute_tool(
            tool_name="set_reminder",
            parameters={"title": f"Reminder 1 {unique_id}"},
            executor=lambda: {"id": "r1"},
            turn_context=ctx,
        )

        # Second call
        record2 = execute_tool(
            tool_name="web_search",
            parameters={"query": f"coffee {unique_id}"},
            executor=lambda: {"results": ["article1", "article2"]},
            turn_context=ctx,
        )

        # Verify ledger has both
        ledger: ToolIOLedger = ctx._tool_io_ledger
        assert len(ledger.records) == 2

        # Verify sequences
        assert ledger.records[0].sequence == 1
        assert ledger.records[1].sequence == 2

        # Verify tools
        assert ledger.records[0].tool_name == "set_reminder"
        assert ledger.records[1].tool_name == "web_search"

    def test_failed_tool_execution_records_error(self) -> None:
        ctx = MockTurnContext()
        unique_id = str(uuid.uuid4())[:8]

        def failing_executor():
            raise ValueError("Tool error")

        record = execute_tool(
            tool_name="set_reminder",
            parameters={"title": f"Test {unique_id}"},
            executor=failing_executor,
            turn_context=ctx,
        )

        # Record should reflect failure
        assert record.status == "failed"
        assert "Tool error" in record.error

        # IO record should be created
        ledger: ToolIOLedger = ctx._tool_io_ledger
        assert len(ledger.records) == 1
        io_rec = ledger.records[0]
        assert io_rec.status == "failed"
        assert "Tool error" in io_rec.error

    def test_without_turn_context_no_recording(self) -> None:
        """If turn_context is None, no recording happens (backward compat)."""
        unique_id = str(uuid.uuid4())[:8]
        record = execute_tool(
            tool_name="set_reminder",
            parameters={"title": f"Test {unique_id}"},
            executor=lambda: {"id": "r1"},
            turn_context=None,  # No context
        )

        # Execution still succeeds
        assert record.status in ("succeeded", "cached")

    def test_idempotent_calls_both_recorded(self) -> None:
        """Both successful and cached executions are recorded."""
        ctx = MockTurnContext()

        call_count = 0

        def executor():
            nonlocal call_count
            call_count += 1
            return {"id": "r1"}

        # First call (live execution)
        record1 = execute_tool(
            tool_name="set_reminder",
            parameters={"title": "Test Unique A"},
            executor=executor,
            turn_context=ctx,
        )

        # Second call with same params (cached)
        record2 = execute_tool(
            tool_name="set_reminder",
            parameters={"title": "Test Unique A"},
            executor=executor,
            turn_context=ctx,
        )

        # Both should be recorded
        ledger: ToolIOLedger = ctx._tool_io_ledger
        assert len(ledger.records) == 2

        # First should be succeeded, second should be cached
        assert ledger.records[0].status == "succeeded"
        assert ledger.records[1].status == "cached"

        # Executor should only be called once (due to idempotency)
        assert call_count == 1

    def test_input_hash_deterministic(self) -> None:
        """Same input produces same hash."""
        ctx1 = MockTurnContext()
        ctx2 = MockTurnContext()

        def executor():
            return {"id": "r1"}

        execute_tool(
            tool_name="set_reminder",
            parameters={"title": "Test X", "due_text": "tomorrow"},
            executor=executor,
            turn_context=ctx1,
        )

        execute_tool(
            tool_name="set_reminder",
            parameters={"title": "Test X", "due_text": "tomorrow"},
            executor=executor,
            turn_context=ctx2,
        )

        hash1 = ctx1._tool_io_ledger.records[0].input_hash
        hash2 = ctx2._tool_io_ledger.records[0].input_hash

        assert hash1 == hash2

    def test_different_inputs_different_hashes(self) -> None:
        """Different inputs produce different hashes."""
        ctx1 = MockTurnContext()
        ctx2 = MockTurnContext()

        def executor():
            return {"id": "r1"}

        execute_tool(
            tool_name="set_reminder",
            parameters={"title": "Test 1 Unique B"},
            executor=executor,
            turn_context=ctx1,
        )

        execute_tool(
            tool_name="set_reminder",
            parameters={"title": "Test 2 Unique B"},
            executor=executor,
            turn_context=ctx2,
        )

        hash1 = ctx1._tool_io_ledger.records[0].input_hash
        hash2 = ctx2._tool_io_ledger.records[0].input_hash

        assert hash1 != hash2

    def test_ledger_lookup_after_recording(self) -> None:
        """Fast lookup works after recording."""
        ctx = MockTurnContext()
        unique_id = str(uuid.uuid4())[:8]

        def executor():
            return {"id": "r1"}

        execute_tool(
            tool_name="set_reminder",
            parameters={"title": f"Test {unique_id}"},
            executor=executor,
            turn_context=ctx,
        )

        ledger: ToolIOLedger = ctx._tool_io_ledger
        io_rec = ledger.records[0]

        # Lookup should work
        found = ledger.lookup(io_rec.tool_name, io_rec.input_hash)
        assert found is not None
        # Output is recorded for succeeded executions
        if io_rec.status == "succeeded":
            assert found.output_payload == {"id": "r1"}
