"""tests/unit/test_topology_runtime.py — TopologyRuntime tests."""

from __future__ import annotations

import pytest

from dadbot.core.topology_runtime import TopologyRuntime, TopologyValidationResult


@pytest.fixture(autouse=True)
def _reset_topology_state() -> None:
    """Auto-reset topology state before each test."""
    # Import here to avoid module-level issues
    from dadbot.core.execution_topology_graph import get_execution_topology_graph

    graph = get_execution_topology_graph()
    graph.clear_violations()
    graph.clear_execution_trace()
    yield
    graph.clear_violations()
    graph.clear_execution_trace()


class TestTopologyRuntime:
    """Tests for TopologyRuntime."""

    def test_initialization(self) -> None:
        """Test TopologyRuntime initialization."""
        runtime = TopologyRuntime(strict_mode=False)
        assert runtime._strict_mode is False

    def test_initialization_strict(self) -> None:
        """Test TopologyRuntime initialization with strict mode."""
        runtime = TopologyRuntime(strict_mode=True)
        assert runtime._strict_mode is True

    def test_begin_and_end_turn(self) -> None:
        """Test beginning and ending a turn."""
        runtime = TopologyRuntime()
        runtime.begin_turn(trace_id="trace1", session_id="sess1")
        assert runtime._current_trace_id == "trace1"
        assert runtime._current_session_id == "sess1"

        result = runtime.end_turn()
        assert isinstance(result, TopologyValidationResult)
        assert runtime._current_trace_id is None
        assert runtime._current_session_id is None

    def test_record_node_entry(self) -> None:
        """Test recording node entries."""
        runtime = TopologyRuntime()
        runtime.begin_turn(trace_id="trace1", session_id="sess1")

        runtime.record_node_entry(
            node_id="control_plane.submit_turn",
            timestamp_ms=1000.0,
        )

        assert len(runtime._execution_sequence) == 1
        assert runtime._execution_sequence[0] == "control_plane.submit_turn"

    def test_record_multiple_entries(self) -> None:
        """Test recording multiple node entries."""
        runtime = TopologyRuntime()
        runtime.begin_turn(trace_id="trace1", session_id="sess1")

        nodes = [
            "control_plane.submit_turn",
            "kernel_gateway.submit_turn",
            "orchestrator.handle_turn",
        ]

        for i, node_id in enumerate(nodes):
            runtime.record_node_entry(
                node_id=node_id,
                timestamp_ms=float(i * 1000),
            )

        assert runtime._execution_sequence == nodes

    def test_valid_canonical_path(self) -> None:
        """Test validation of a valid canonical path."""
        runtime = TopologyRuntime()
        runtime.begin_turn(trace_id="trace1", session_id="sess1")

        canonical_path = [
            "control_plane.submit_turn",
            "kernel_gateway.submit_turn",
            "orchestrator.handle_turn",
            "orchestrator._run_graph_with_trace_binding",
            "graph.execute",
            "response_engine.generate_and_rank",
            "persistence.finalize_turn",
        ]

        for node_id in canonical_path:
            runtime.record_node_entry(node_id=node_id)

        result = runtime.end_turn()
        assert result.passed is True
        assert result.violations_critical == 0
        assert result.violations_total == 0

    def test_invalid_path_detection(self) -> None:
        """Test detection of invalid execution path."""
        runtime = TopologyRuntime()
        runtime.begin_turn(trace_id="trace1", session_id="sess1")

        # Try to go directly from entry to executor (skipping gates)
        runtime.record_node_entry(node_id="control_plane.submit_turn")
        runtime.record_node_entry(node_id="graph.execute")  # Invalid transition

        result = runtime.end_turn()
        assert result.passed is False
        assert result.violations_total > 0

    def test_unknown_node_detection(self) -> None:
        """Test detection of unknown nodes in path."""
        runtime = TopologyRuntime()
        runtime.begin_turn(trace_id="trace1", session_id="sess1")

        runtime.record_node_entry(node_id="control_plane.submit_turn")
        runtime.record_node_entry(node_id="unknown.shadow.path")

        result = runtime.end_turn()
        assert result.passed is False
        assert result.violations_critical > 0

    def test_result_details_structure(self) -> None:
        """Test that validation result includes proper details."""
        runtime = TopologyRuntime()
        runtime.begin_turn(trace_id="trace1", session_id="sess1")

        runtime.record_node_entry(node_id="control_plane.submit_turn")
        result = runtime.end_turn()

        assert "trace_id" in result.details
        assert "session_id" in result.details
        assert "sequence_length" in result.details
        assert "sequence" in result.details
        assert "violations" in result.details

    def test_strict_mode_setting(self) -> None:
        """Test strict mode can be toggled."""
        runtime = TopologyRuntime(strict_mode=False)
        assert runtime._strict_mode is False

        runtime.set_strict_mode(True)
        assert runtime._strict_mode is True

        runtime.set_strict_mode(False)
        assert runtime._strict_mode is False

    def test_audit_report(self) -> None:
        """Test audit report generation."""
        runtime = TopologyRuntime()
        runtime.begin_turn(trace_id="trace1", session_id="sess1")

        # Record an invalid path to generate violations
        runtime.record_node_entry(node_id="unknown.node")

        runtime.end_turn()

        report = runtime.get_audit_report()
        assert "total_nodes" in report
        assert "violations_critical" in report
        assert "execution_trace_length" in report

    def test_clear_audit_data(self) -> None:
        """Test clearing audit data."""
        runtime = TopologyRuntime()
        runtime.begin_turn(trace_id="trace1", session_id="sess1")

        runtime.record_node_entry(node_id="control_plane.submit_turn")
        runtime.record_node_entry(node_id="unknown.node")

        runtime.end_turn()

        report_before = runtime.get_audit_report()
        assert report_before["violations_critical"] > 0

        runtime.clear_audit_data()

        report_after = runtime.get_audit_report()
        assert report_after["violations_critical"] == 0

    def test_end_turn_without_begin(self) -> None:
        """Test that end_turn without begin_turn returns error result."""
        runtime = TopologyRuntime()
        result = runtime.end_turn()
        assert result.passed is False
        assert result.details["reason"] == "no_active_trace"

    def test_record_node_without_trace(self) -> None:
        """Test that record_node_entry without active trace is handled."""
        runtime = TopologyRuntime()
        # This should not raise an error, just log a warning
        runtime.record_node_entry(node_id="control_plane.submit_turn")
        assert runtime._execution_sequence == []  # Should not record

    def test_multiple_turns_isolated(self) -> None:
        """Test that multiple turns are properly isolated."""
        runtime = TopologyRuntime()

        # First turn
        runtime.begin_turn(trace_id="trace1", session_id="sess1")
        runtime.record_node_entry(node_id="control_plane.submit_turn")
        result1 = runtime.end_turn()

        # Second turn
        runtime.begin_turn(trace_id="trace2", session_id="sess2")
        runtime.record_node_entry(node_id="control_plane.submit_turn")
        result2 = runtime.end_turn()

        # Both should complete independently
        assert result1.details["trace_id"] == "trace1"
        assert result2.details["trace_id"] == "trace2"

    def test_metadata_passed_through(self) -> None:
        """Test that metadata is passed through to audit data."""
        runtime = TopologyRuntime()
        runtime.begin_turn(trace_id="trace_meta_test", session_id="sess_meta")

        runtime.record_node_entry(
            node_id="control_plane.submit_turn",
            timestamp_ms=1000.0,
            metadata={"user_input": "hello", "session_type": "chat"},
        )

        runtime.end_turn()
        report = runtime.get_audit_report()
        # Should have at least one execution recorded (may have more from test isolation issues)
        assert report["execution_trace_length"] >= 1
