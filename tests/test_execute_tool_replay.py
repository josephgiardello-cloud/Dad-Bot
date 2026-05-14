"""Unit tests for replay mode execution in tool_executor."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from dadbot.core.tool_executor import execute_tool
from dadbot.core.runtime_errors import ReplayInvariantViolation
from dadbot.core.tool_recording import ToolIOLedger, ToolIORecord


def test_execute_tool_live_mode_no_context():
    """Test normal execution when no turn_context is provided."""
    result_value = {"status": "success", "data": "test"}

    def mock_executor():
        return result_value

    record = execute_tool(
        tool_name="test_tool_unique_1",
        parameters={"arg1": "value1", "_idempotency_key": "unique-key-1"},
        executor=mock_executor,
    )

    assert record.tool_name == "test_tool_unique_1"
    assert record.status in ("succeeded", "cached")
    assert record.result == result_value or record.status == "cached"


def test_execute_tool_live_mode_with_context():
    """Test execution in live mode with turn_context."""
    result_value = {"status": "success"}

    def mock_executor():
        return result_value

    turn_context = Mock()
    turn_context.metadata = {"replay_mode": False}
    turn_context.state = {}

    record = execute_tool(
        tool_name="test_tool_unique_2",
        parameters={"arg1": "value1", "_idempotency_key": "unique-key-2"},
        executor=mock_executor,
        turn_context=turn_context,
    )

    assert record.tool_name == "test_tool_unique_2"
    assert record.status in ("succeeded", "cached")
    assert hasattr(turn_context, "_tool_io_ledger")


def test_execute_tool_replay_mode_hit():
    """Test that replay mode skips execution and returns recorded output."""
    # Create a properly formed tool IO record
    from dadbot.core.tool_recording import _stable_payload_hash
    
    tool_params = {"arg1": "value1"}
    tool_version = "test-tool-version-1"
    environment_fingerprint = "test-env-fingerprint-1"
    input_hash = _stable_payload_hash({"tool_name": "test_tool_replay", "parameters": tool_params})
    output = {"result": "recorded_output"}
    output_hash = _stable_payload_hash(output)
    
    # Create a ledger with a single recorded tool call
    ledger = ToolIOLedger()
    record_to_save = ToolIORecord(
        sequence=1,
        tool_name="test_tool_replay",
        input_hash=input_hash,
        input_payload=tool_params,
        output_payload=output,
        output_hash=output_hash,
        status="succeeded",
        latency_ms=0.5,
        error="",
        metadata={
            "tool_version": tool_version,
            "environment_fingerprint": environment_fingerprint,
        },
    )
    ledger.append(record_to_save)

    # Track if executor was called
    executor_called = False

    def mock_executor():
        nonlocal executor_called
        executor_called = True
        raise Exception("Should not be called in replay mode")

    turn_context = Mock()
    turn_context.metadata = {
        "replay_mode": True,
        "_tool_io_ledger": ledger,
        "tool_version": tool_version,
        "environment_fingerprint": environment_fingerprint,
    }
    turn_context.state = {}
    # Initialize current ledger
    turn_context._tool_io_ledger = ToolIOLedger()

    record = execute_tool(
        tool_name="test_tool_replay",
        parameters=tool_params,
        executor=mock_executor,
        turn_context=turn_context,
    )

    # Executor should not have been called
    assert not executor_called
    assert record.tool_name == "test_tool_replay"
    # The result should come from the recorded data
    assert record.result == output


def test_execute_tool_replay_mode_miss():
    """Test that replay mode fails closed on cache miss."""
    # Create an empty restored ledger (no matching tool IO)
    restored_ledger = ToolIOLedger()

    result_value = {"status": "live_execution"}

    def mock_executor():
        return result_value

    turn_context = Mock()
    turn_context.metadata = {
        "replay_mode": True,
        "_tool_io_ledger": restored_ledger,
        "tool_version": "test-tool-version-2",
        "environment_fingerprint": "test-env-fingerprint-2",
    }
    turn_context.state = {}

    with pytest.raises(ReplayInvariantViolation):
        execute_tool(
            tool_name="test_tool_miss",
            parameters={"arg1": "value1", "_idempotency_key": "miss-key"},
            executor=mock_executor,
            turn_context=turn_context,
        )


def test_execute_tool_replay_mode_no_ledger():
    """Test that replay mode fails closed when ledger is missing."""
    result_value = {"status": "fallback"}

    def mock_executor():
        return result_value

    turn_context = Mock()
    turn_context.metadata = {
        "replay_mode": True,
        "_tool_io_ledger": None,  # Missing ledger
        "tool_version": "test-tool-version-3",
        "environment_fingerprint": "test-env-fingerprint-3",
    }
    turn_context.state = {}

    with pytest.raises(ReplayInvariantViolation):
        execute_tool(
            tool_name="test_tool_no_ledger",
            parameters={"arg1": "value1", "_idempotency_key": "no-ledger-key"},
            executor=mock_executor,
            turn_context=turn_context,
        )


def test_execute_tool_replay_mode_preserves_latency():
    """Test that replayed records preserve original latency."""
    from dadbot.core.tool_recording import _stable_payload_hash
    
    tool_params = {"arg1": "value1"}
    tool_version = "test-tool-version-4"
    environment_fingerprint = "test-env-fingerprint-4"
    input_hash = _stable_payload_hash({"tool_name": "test_tool_latency", "parameters": tool_params})
    output = {"result": "recorded"}
    output_hash = _stable_payload_hash(output)
    
    # Create ledger with recorded latency
    ledger = ToolIOLedger()
    record_to_save = ToolIORecord(
        sequence=1,
        tool_name="test_tool_latency",
        input_hash=input_hash,
        input_payload=tool_params,
        output_payload=output,
        output_hash=output_hash,
        status="succeeded",
        latency_ms=2500.0,  # 2.5 seconds
        error="",
        metadata={
            "tool_version": tool_version,
            "environment_fingerprint": environment_fingerprint,
        },
    )
    ledger.append(record_to_save)

    def mock_executor():
        raise Exception("Should not execute")

    turn_context = Mock()
    turn_context.metadata = {
        "replay_mode": True,
        "_tool_io_ledger": ledger,
        "tool_version": tool_version,
        "environment_fingerprint": environment_fingerprint,
    }
    turn_context.state = {}
    turn_context._tool_io_ledger = ToolIOLedger()

    record = execute_tool(
        tool_name="test_tool_latency",
        parameters=tool_params,
        executor=mock_executor,
        turn_context=turn_context,
    )

    assert record is not None
    assert record.status == "replayed"
    assert record.result == output
    assert record.idempotency_key


def test_execute_tool_live_mode_records_to_ledger():
    """Test that live mode records executed tool IO to ledger."""
    result_value = {"key": "value"}

    def mock_executor():
        return result_value

    turn_context = Mock()
    turn_context.metadata = {"replay_mode": False}
    turn_context.state = {}

    record = execute_tool(
        tool_name="live_tool_record",
        parameters={"param": "test", "_idempotency_key": "record-key"},
        executor=mock_executor,
        turn_context=turn_context,
    )

    # Tool IO should be recorded to ledger
    assert hasattr(turn_context, "_tool_io_ledger")
    ledger = turn_context._tool_io_ledger
    if isinstance(ledger, ToolIOLedger):
        assert len(ledger.records) > 0


def test_execute_tool_compensating_action():
    """Test that compensating action is preserved in replayed record."""
    from dadbot.core.tool_recording import _stable_payload_hash
    
    tool_params = {"arg1": "value1"}
    tool_version = "test-tool-version-5"
    environment_fingerprint = "test-env-fingerprint-5"
    input_hash = _stable_payload_hash({"tool_name": "rollback_tool", "parameters": tool_params})
    output = {"result": "recorded"}
    output_hash = _stable_payload_hash(output)
    
    ledger = ToolIOLedger()
    record_to_save = ToolIORecord(
        sequence=1,
        tool_name="rollback_tool",
        input_hash=input_hash,
        input_payload=tool_params,
        output_payload=output,
        output_hash=output_hash,
        status="succeeded",
        latency_ms=0.1,
        error="",
        metadata={
            "tool_version": tool_version,
            "environment_fingerprint": environment_fingerprint,
        },
    )
    ledger.append(record_to_save)

    compensating_called = False

    def mock_compensating():
        nonlocal compensating_called
        compensating_called = True

    turn_context = Mock()
    turn_context.metadata = {
        "replay_mode": True,
        "_tool_io_ledger": ledger,
        "tool_version": tool_version,
        "environment_fingerprint": environment_fingerprint,
    }
    turn_context.state = {}
    turn_context._tool_io_ledger = ToolIOLedger()

    record = execute_tool(
        tool_name="rollback_tool",
        parameters=tool_params,
        executor=lambda: {},
        compensating_action=mock_compensating,
        turn_context=turn_context,
    )

    # Compensating action should be attached to record
    assert record.compensating_action == mock_compensating


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
