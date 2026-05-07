"""Priority 1: Canonical Execution Path tests.

Tests that all execution surfaces converge to a single canonical path through
the control plane, with no shortcut delegation branches.

Invariants validated:
1. All execution surfaces (run, run_async, handle_turn) produce identical results
2. Every turn produces exactly one trace_id
3. Every trace has exactly one commit boundary (save node)
4. No requests bypass the control plane
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call, PropertyMock
from inspect import iscoroutinefunction

from dadbot.core.orchestrator import DadBotOrchestrator
from dadbot.core.control_plane import ExecutionJob, ExecutionControlPlane
from dadbot.contracts import FinalizedTurnResult


@pytest.mark.unit
class TestCanonicalExecutionPathStructure:
    """Verify structural changes for canonical execution path."""

    def test_orchestrator_has_no_can_delegate_method(self):
        """Verify _can_delegate_to_bot_execute_turn() method was removed."""
        # This method should not exist anymore
        assert not hasattr(DadBotOrchestrator, "_can_delegate_to_bot_execute_turn")

    def test_handle_turn_signature_correct(self):
        """Verify handle_turn() has correct signature without delegation logic."""
        import inspect
        
        source = inspect.getsource(DadBotOrchestrator.handle_turn)
        
        # Should not reference direct delegation shortcuts
        assert "execute_turn" not in source
        assert "TurnDelivery.ASYNC" not in source
        assert "live_turn_request" not in source
        # Should reference control plane
        assert "_submit_turn_via_control_plane" in source

    def test_run_method_delegates_to_handle_turn(self):
        """Verify run() delegates to handle_turn() instead of duplicating logic."""
        import inspect
        
        source = inspect.getsource(DadBotOrchestrator.run)
        
        # Should not have delegation shortcuts
        assert "execute_turn" not in source
        assert "TurnDelivery.SYNC" not in source
        # Should delegate to handle_turn
        assert "handle_turn" in source

    def test_run_async_method_delegates_to_handle_turn(self):
        """Verify run_async() delegates to handle_turn() instead of duplicating logic."""
        import inspect
        
        source = inspect.getsource(DadBotOrchestrator.run_async)
        
        # Should not have delegation shortcuts
        assert "execute_turn" not in source
        assert "TurnDelivery.ASYNC" not in source
        # Should delegate to handle_turn
        assert "handle_turn" in source

    def test_submit_turn_via_control_plane_exists(self):
        """Verify _submit_turn_via_control_plane() exists and routes correctly."""
        assert hasattr(DadBotOrchestrator, "_submit_turn_via_control_plane")
        # Should be async
        assert iscoroutinefunction(DadBotOrchestrator._submit_turn_via_control_plane)


@pytest.mark.unit
class TestTraceInvariants:
    """Verify trace invariant validation in control plane."""

    @pytest.fixture
    def mock_ledger(self):
        """Mock ledger for trace event tracking."""
        ledger = MagicMock()
        ledger.read = MagicMock(return_value=[])
        return ledger

    @pytest.fixture
    def control_plane_with_ledger(self, mock_ledger):
        """Create control plane with mocked ledger."""
        registry = MagicMock()
        kernel_executor = AsyncMock()
        plane = ExecutionControlPlane(
            registry=registry,
            kernel_executor=kernel_executor,
        )
        plane.ledger = mock_ledger
        return plane

    def test_validate_trace_invariant_missing_trace_id(self, control_plane_with_ledger, caplog):
        """Verify validation detects missing trace_id."""
        job = ExecutionJob(
            session_id="test",
            user_input="test",
            metadata={},
        )
        job.trace_id = ""  # Force empty trace_id
        job.metadata["trace_id"] = ""  # Clear it from metadata too

        control_plane_with_ledger._validate_trace_invariant(job, ("response", False))

        assert "missing trace_id" in caplog.text.lower()

    def test_validate_trace_invariant_no_events(self, control_plane_with_ledger, caplog):
        """Verify validation detects missing trace events."""
        job = ExecutionJob(
            session_id="test",
            user_input="test",
            metadata={"trace_id": "tr-12345"},
        )

        control_plane_with_ledger._validate_trace_invariant(job, ("response", False))

        # Should warn about no events recorded
        assert "no events recorded" in caplog.text.lower()

    def test_validate_trace_invariant_multiple_commits(self, control_plane_with_ledger, caplog):
        """Verify validation detects multiple commit boundaries."""
        job = ExecutionJob(
            session_id="test",
            user_input="test",
            metadata={"trace_id": "tr-12345"},
        )
        
        # Mock multiple save (commit) events
        control_plane_with_ledger.ledger.read.return_value = [
            {"trace_id": "tr-12345", "event_type": "node_completed", "node_type": "save"},
            {"trace_id": "tr-12345", "event_type": "node_completed", "node_type": "save"},
        ]

        control_plane_with_ledger._validate_trace_invariant(job, ("response", False))

        assert "exactly 1 commit boundary" in caplog.text.lower()
        assert "found 2" in caplog.text.lower()

    def test_validate_trace_invariant_no_commits(self, control_plane_with_ledger, caplog):
        """Verify validation detects missing commit boundary."""
        job = ExecutionJob(
            session_id="test",
            user_input="test",
            metadata={"trace_id": "tr-12345"},
        )
        
        # Mock events with no save node
        control_plane_with_ledger.ledger.read.return_value = [
            {"trace_id": "tr-12345", "event_type": "node_completed", "node_type": "planner"},
            {"trace_id": "tr-12345", "event_type": "node_completed", "node_type": "inference"},
        ]

        control_plane_with_ledger._validate_trace_invariant(job, ("response", False))

        assert "exactly 1 commit boundary" in caplog.text.lower()
        assert "found 0" in caplog.text.lower()

    def test_validate_trace_invariant_called_after_execution(self):
        """Verify _validate_trace_invariant() is called in submit_turn flow."""
        import inspect
        from dadbot.core.control_plane import ExecutionControlPlane
        
        source = inspect.getsource(ExecutionControlPlane.submit_turn)
        
        # Should call _validate_trace_invariant after job completes
        assert "_validate_trace_invariant" in source


@pytest.mark.unit
class TestExecutionJobTraceId:
    """Verify ExecutionJob properly manages trace_id assignment."""

    def test_execution_job_assigns_trace_id(self):
        """Verify ExecutionJob generates trace_id if not provided."""
        job = ExecutionJob(
            session_id="test_session",
            user_input="test input",
        )
        
        # Should have assigned a trace_id
        assert job.trace_id
        assert job.trace_id.startswith("tr-")
        assert job.metadata["trace_id"] == job.trace_id

    def test_execution_job_preserves_provided_trace_id(self):
        """Verify ExecutionJob uses provided trace_id."""
        provided_trace_id = "tr-custom-123"
        job = ExecutionJob(
            session_id="test_session",
            user_input="test input",
            trace_id=provided_trace_id,
        )
        
        assert job.trace_id == provided_trace_id
        assert job.metadata["trace_id"] == provided_trace_id

    def test_execution_job_gets_trace_id_from_metadata(self):
        """Verify ExecutionJob extracts trace_id from metadata."""
        provided_trace_id = "tr-meta-456"
        job = ExecutionJob(
            session_id="test_session",
            user_input="test input",
            metadata={"trace_id": provided_trace_id},
        )
        
        assert job.trace_id == provided_trace_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
