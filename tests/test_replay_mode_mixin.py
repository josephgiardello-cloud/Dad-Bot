"""Unit tests for replay mode detection and checkpoint restoration."""

from unittest.mock import Mock

import pytest

from dadbot.core.replay_mode_mixin import ReplayModeMixin
from dadbot.core.tool_recording import ToolIOLedger


class _ReplayModeTestHelper(ReplayModeMixin):
    """Concrete test subclass of ReplayModeMixin."""

    def __init__(self):
        self.services = None


def test_replay_mode_detect_no_checkpoint():
    """Test that live mode is set when no checkpoint exists."""
    mixin = _ReplayModeTestHelper()
    mixin.services = None

    metadata = mixin._detect_and_prepare_replay_mode(
        session_id="session-123",
        trace_id="trace-456",
    )

    assert metadata["replay_mode"] is False
    assert metadata["checkpoint_available"] is False


def test_replay_mode_detect_with_checkpoint():
    """Test that replay mode is enabled when checkpoint exists."""
    mixin = _ReplayModeTestHelper()

    # Mock the persistence layer
    mock_persistence = Mock()
    mock_persistence.load_latest_graph_checkpoint.return_value = {
        "stage": "complete",
    }

    mock_services = Mock()
    mock_services.get_persistence_service.return_value = mock_persistence
    mixin.services = mock_services

    metadata = mixin._detect_and_prepare_replay_mode(
        session_id="session-123",
        trace_id="trace-456",
    )

    assert metadata["replay_mode"] is True
    assert metadata["checkpoint_available"] is True
    mock_persistence.load_latest_graph_checkpoint.assert_called_once_with(
        trace_token="trace-456",
        session_id="session-123",
    )


def test_replay_mode_restore_tool_io_ledger():
    """Test that tool_io_ledger is restored from checkpoint."""
    mixin = _ReplayModeTestHelper()

    tool_io_dict = ToolIOLedger().to_dict()

    mock_persistence = Mock()
    mock_persistence.load_latest_graph_checkpoint.return_value = {
        "stage": "complete",
        "tool_io_ledger": tool_io_dict,
    }

    mock_services = Mock()
    mock_services.get_persistence_service.return_value = mock_persistence
    mixin.services = mock_services

    metadata = mixin._detect_and_prepare_replay_mode(
        session_id="session-123",
        trace_id="trace-456",
    )

    assert metadata["replay_mode"] is True
    assert "_tool_io_ledger" in metadata
    assert isinstance(metadata["_tool_io_ledger"], ToolIOLedger)
    assert metadata.get("tool_io_ledger_restored") is True


def test_replay_mode_corrupted_tool_io_ledger():
    """Corrupt durable IO must not become an empty replay ledger."""
    mixin = _ReplayModeTestHelper()

    # Invalid tool_io_ledger data
    mock_persistence = Mock()
    mock_persistence.load_latest_graph_checkpoint.return_value = {
        "stage": "complete",
        "tool_io_ledger": {"invalid": "data"},
    }

    mock_services = Mock()
    mock_services.get_persistence_service.return_value = mock_persistence
    mixin.services = mock_services

    with pytest.raises(ValueError, match="tool IO"):
        mixin._detect_and_prepare_replay_mode(session_id="session-123", trace_id="trace-456")


def test_replay_mode_schema_failure_does_not_fall_back_to_live():
    from dadbot.core.persistence_record_schema import PersistenceSchemaError

    mixin = _ReplayModeTestHelper()
    persistence = Mock()
    persistence.load_latest_graph_checkpoint.side_effect = PersistenceSchemaError("Corrupt checkpoint tool IO")
    mixin.services = Mock()
    mixin.services.get_persistence_service.return_value = persistence
    with pytest.raises(PersistenceSchemaError, match="Corrupt checkpoint"):
        mixin._detect_and_prepare_replay_mode(session_id="session-123", trace_id="trace-456")


def test_replay_mode_no_persistence_service():
    """Test live mode when persistence service is unavailable."""
    mixin = _ReplayModeTestHelper()
    mixin.services = None

    metadata = mixin._detect_and_prepare_replay_mode(
        session_id="session-123",
        trace_id="trace-456",
    )

    assert metadata["replay_mode"] is False
    assert metadata["checkpoint_available"] is False


def test_replay_mode_checkpoint_load_error():
    """Test live mode when checkpoint load fails."""
    mixin = _ReplayModeTestHelper()

    mock_persistence = Mock()
    mock_persistence.load_latest_graph_checkpoint.side_effect = RuntimeError(
        "Checkpoint not found"
    )

    mock_services = Mock()
    mock_services.get_persistence_service.return_value = mock_persistence
    mixin.services = mock_services

    metadata = mixin._detect_and_prepare_replay_mode(
        session_id="session-123",
        trace_id="trace-456",
    )

    assert metadata["replay_mode"] is False
    assert metadata["checkpoint_available"] is False


def test_replay_mode_preserves_existing_metadata():
    """Test that existing metadata is preserved when replay mode is initialized."""
    mixin = _ReplayModeTestHelper()
    mixin.services = None

    input_metadata = {
        "user_id": "user-123",
        "session_tags": ["test", "dev"],
    }

    metadata = mixin._detect_and_prepare_replay_mode(
        session_id="session-123",
        trace_id="trace-456",
        metadata=input_metadata.copy(),
    )

    assert metadata["user_id"] == "user-123"
    assert metadata["session_tags"] == ["test", "dev"]
    assert metadata["replay_mode"] is False
    assert metadata["checkpoint_available"] is False


def test_replay_mode_empty_tool_io_ledger():
    """Test that empty tool_io_ledger is handled correctly."""
    mixin = _ReplayModeTestHelper()

    mock_persistence = Mock()
    mock_persistence.load_latest_graph_checkpoint.return_value = {
        "stage": "complete",
        "tool_io_ledger": ToolIOLedger().to_dict(),
    }

    mock_services = Mock()
    mock_services.get_persistence_service.return_value = mock_persistence
    mixin.services = mock_services

    metadata = mixin._detect_and_prepare_replay_mode(
        session_id="session-123",
        trace_id="trace-456",
    )

    assert metadata["replay_mode"] is True
    assert metadata["_tool_io_ledger"].is_empty()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
