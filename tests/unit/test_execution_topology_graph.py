"""tests/unit/test_execution_topology_graph.py — ExecutionTopologyGraph tests."""

from __future__ import annotations

import pytest

from dadbot.core.execution_topology_graph import (
    ExecutionNode,
    ExecutionTopologyGraph,
    NodeSeverity,
    NodeType,
    TopologyViolation,
    get_execution_topology_graph,
)


class TestExecutionNode:
    """Tests for ExecutionNode dataclass."""

    def test_node_creation(self) -> None:
        """Test basic ExecutionNode creation."""
        node = ExecutionNode(
            node_id="test.node",
            node_type=NodeType.ENTRY,
            module_path="test.module",
            callable_name="test_func",
        )
        assert node.node_id == "test.node"
        assert node.node_type == NodeType.ENTRY
        assert node.module_path == "test.module"
        assert node.callable_name == "test_func"

    def test_node_equality(self) -> None:
        """Test ExecutionNode equality based on node_id."""
        node1 = ExecutionNode(
            node_id="same.id",
            node_type=NodeType.ENTRY,
            module_path="module1",
            callable_name="func1",
        )
        node2 = ExecutionNode(
            node_id="same.id",
            node_type=NodeType.EXECUTOR,
            module_path="module2",
            callable_name="func2",
        )
        assert node1 == node2

    def test_node_hash(self) -> None:
        """Test ExecutionNode hash based on node_id."""
        node = ExecutionNode(
            node_id="hash.test",
            node_type=NodeType.PLANNER,
            module_path="module",
            callable_name="func",
        )
        assert hash(node) == hash("hash.test")

    def test_node_frozen(self) -> None:
        """Test that ExecutionNode is immutable (frozen)."""
        node = ExecutionNode(
            node_id="frozen.test",
            node_type=NodeType.GATE,
            module_path="module",
            callable_name="func",
        )
        with pytest.raises(AttributeError):
            node.node_id = "modified"


class TestTopologyViolation:
    """Tests for TopologyViolation dataclass."""

    def test_violation_creation(self) -> None:
        """Test basic TopologyViolation creation."""
        violation = TopologyViolation(
            violation_id="v1",
            node_id="bad.node",
            violation_type="unknown_node",
            severity=NodeSeverity.CRITICAL,
            detail="Unknown node encountered",
        )
        assert violation.violation_id == "v1"
        assert violation.node_id == "bad.node"
        assert violation.severity == NodeSeverity.CRITICAL

    def test_violation_with_context(self) -> None:
        """Test TopologyViolation with context metadata."""
        violation = TopologyViolation(
            violation_id="v2",
            node_id="bad.node",
            violation_type="unauthorized_transition",
            severity=NodeSeverity.HIGH,
            detail="Invalid transition",
            context={"trace_id": "trace123", "session_id": "sess456"},
        )
        assert violation.context["trace_id"] == "trace123"
        assert violation.context["session_id"] == "sess456"


class TestExecutionTopologyGraph:
    """Tests for ExecutionTopologyGraph."""

    def test_initialization(self) -> None:
        """Test ExecutionTopologyGraph initialization with canonical topology."""
        graph = ExecutionTopologyGraph()
        nodes = graph.get_all_nodes()
        assert len(nodes) > 0
        # Should have at least entry, executor, and commit nodes
        node_ids = {n.node_id for n in nodes}
        assert "control_plane.submit_turn" in node_ids
        assert "graph.execute" in node_ids
        assert "persistence.finalize_turn" in node_ids

    def test_register_node(self) -> None:
        """Test registering a new node."""
        graph = ExecutionTopologyGraph()
        custom_node = ExecutionNode(
            node_id="custom.node",
            node_type=NodeType.ROUTER,
            module_path="custom.module",
            callable_name="custom_func",
        )
        graph.register_node(custom_node)
        retrieved = graph.get_node("custom.node")
        assert retrieved == custom_node

    def test_get_node(self) -> None:
        """Test retrieving registered nodes."""
        graph = ExecutionTopologyGraph()
        entry_node = graph.get_node("control_plane.submit_turn")
        assert entry_node is not None
        assert entry_node.node_type == NodeType.ENTRY

    def test_get_nonexistent_node(self) -> None:
        """Test getting a non-existent node returns None."""
        graph = ExecutionTopologyGraph()
        assert graph.get_node("nonexistent.node") is None

    def test_record_execution(self) -> None:
        """Test recording execution events."""
        graph = ExecutionTopologyGraph()
        graph.record_execution(
            node_id="control_plane.submit_turn",
            trace_id="trace1",
            session_id="sess1",
            timestamp_ms=1000.0,
            metadata={"key": "value"},
        )
        trace = graph.get_execution_trace()
        assert len(trace) == 1
        assert trace[0]["node_id"] == "control_plane.submit_turn"
        assert trace[0]["trace_id"] == "trace1"

    def test_record_execution_at_unknown_node(self) -> None:
        """Test that recording execution at unknown node creates a violation."""
        graph = ExecutionTopologyGraph()
        graph.record_execution(
            node_id="unknown.node",
            trace_id="trace1",
            session_id="sess1",
            timestamp_ms=1000.0,
        )
        violations = graph.get_violations()
        assert len(violations) == 1
        assert violations[0].violation_type == "unknown_node"
        assert violations[0].severity == NodeSeverity.CRITICAL

    def test_validate_canonical_path(self) -> None:
        """Test validation of the canonical execution path."""
        graph = ExecutionTopologyGraph()
        canonical_sequence = [
            "control_plane.submit_turn",
            "kernel_gateway.submit_turn",
            "orchestrator.handle_turn",
            "orchestrator._run_graph_with_trace_binding",
            "graph.execute",
            "response_engine.generate_and_rank",
            "persistence.finalize_turn",
        ]
        is_valid = graph.validate_execution_path(
            trace_id="trace1",
            session_id="sess1",
            execution_sequence=canonical_sequence,
        )
        assert is_valid
        violations = graph.get_violations()
        assert len(violations) == 0

    def test_validate_invalid_entry(self) -> None:
        """Test validation fails for non-entry starting point."""
        graph = ExecutionTopologyGraph()
        invalid_sequence = [
            "graph.execute",  # Should start at entry
            "response_engine.generate_and_rank",
        ]
        is_valid = graph.validate_execution_path(
            trace_id="trace1",
            session_id="sess1",
            execution_sequence=invalid_sequence,
        )
        assert not is_valid
        violations = graph.get_violations()
        assert len(violations) == 1
        assert violations[0].violation_type == "invalid_entry"

    def test_validate_unknown_node_in_sequence(self) -> None:
        """Test validation fails for unknown node in sequence."""
        graph = ExecutionTopologyGraph()
        sequence_with_unknown = [
            "control_plane.submit_turn",
            "unknown.node",
            "graph.execute",
        ]
        is_valid = graph.validate_execution_path(
            trace_id="trace1",
            session_id="sess1",
            execution_sequence=sequence_with_unknown,
        )
        assert not is_valid
        violations = graph.get_violations()
        # Validation should detect either unknown_node or unauthorized_transition
        assert len(violations) > 0
        violation_types = {v.violation_type for v in violations}
        assert "unknown_node" in violation_types or "unauthorized_transition" in violation_types

    def test_validate_unauthorized_transition(self) -> None:
        """Test validation fails for unauthorized node transition."""
        graph = ExecutionTopologyGraph()
        bad_sequence = [
            "control_plane.submit_turn",
            "graph.execute",  # Should not transition directly to graph.execute
        ]
        is_valid = graph.validate_execution_path(
            trace_id="trace1",
            session_id="sess1",
            execution_sequence=bad_sequence,
        )
        assert not is_valid
        violations = graph.get_violations()
        assert any(v.violation_type == "unauthorized_transition" for v in violations)

    def test_clear_violations(self) -> None:
        """Test clearing violation log."""
        graph = ExecutionTopologyGraph()
        # Record a violation
        graph.record_execution(
            node_id="unknown.node",
            trace_id="trace1",
            session_id="sess1",
            timestamp_ms=1000.0,
        )
        assert len(graph.get_violations()) == 1
        graph.clear_violations()
        assert len(graph.get_violations()) == 0

    def test_clear_execution_trace(self) -> None:
        """Test clearing execution trace."""
        graph = ExecutionTopologyGraph()
        graph.record_execution(
            node_id="control_plane.submit_turn",
            trace_id="trace1",
            session_id="sess1",
            timestamp_ms=1000.0,
        )
        assert len(graph.get_execution_trace()) == 1
        graph.clear_execution_trace()
        assert len(graph.get_execution_trace()) == 0

    def test_strict_mode(self) -> None:
        """Test strict mode setting."""
        graph = ExecutionTopologyGraph()
        assert graph._strict_mode is False
        graph.set_strict_mode(True)
        assert graph._strict_mode is True
        graph.set_strict_mode(False)
        assert graph._strict_mode is False

    def test_audit_report(self) -> None:
        """Test audit report generation."""
        graph = ExecutionTopologyGraph()
        # Record some violations
        graph.record_execution(
            node_id="unknown.node",
            trace_id="trace1",
            session_id="sess1",
            timestamp_ms=1000.0,
        )
        graph.record_execution(
            node_id="control_plane.submit_turn",
            trace_id="trace2",
            session_id="sess2",
            timestamp_ms=2000.0,
        )
        report = graph.audit_report()
        assert report["total_nodes"] > 0
        assert report["violations_critical"] >= 1
        # Only the known node gets added to execution trace
        assert report["execution_trace_length"] == 1
        assert len(report["violations"]) >= 1

    def test_singleton_graph(self) -> None:
        """Test that get_execution_topology_graph returns a singleton."""
        graph1 = get_execution_topology_graph()
        graph2 = get_execution_topology_graph()
        assert graph1 is graph2

    def test_canonical_topology_chain(self) -> None:
        """Test that canonical topology forms a proper chain."""
        graph = ExecutionTopologyGraph()
        # Walk from entry to commit
        current_node = graph.get_node("control_plane.submit_turn")
        visited = [current_node.node_id]

        while current_node and current_node.children_node_ids:
            # In canonical path, should have exactly one child
            assert len(current_node.children_node_ids) <= 1
            next_id = current_node.children_node_ids[0]
            current_node = graph.get_node(next_id)
            if current_node is None:
                break
            visited.append(current_node.node_id)

        # Should end at persistence.finalize_turn
        assert visited[-1] == "persistence.finalize_turn"
        # Should have visited all major nodes
        assert len(visited) >= 6

    def test_node_determinism_tracking(self) -> None:
        """Test tracking of deterministic vs non-deterministic nodes."""
        graph = ExecutionTopologyGraph()
        entry_node = graph.get_node("control_plane.submit_turn")
        assert entry_node.is_deterministic is True

        ranker_node = graph.get_node("response_engine.generate_and_rank")
        assert ranker_node.is_deterministic is False

    def test_node_branching_tracking(self) -> None:
        """Test tracking of branching vs non-branching nodes."""
        graph = ExecutionTopologyGraph()
        entry_node = graph.get_node("control_plane.submit_turn")
        assert entry_node.allows_branching is False

        ranker_node = graph.get_node("response_engine.generate_and_rank")
        assert ranker_node.allows_branching is True


class TestTopologyIntegration:
    """Integration tests for execution topology."""

    def test_full_turn_execution_path(self) -> None:
        """Test a full valid turn execution path."""
        graph = ExecutionTopologyGraph()

        # Simulate a complete turn execution
        full_path = [
            "control_plane.submit_turn",
            "kernel_gateway.submit_turn",
            "orchestrator.handle_turn",
            "orchestrator._run_graph_with_trace_binding",
            "graph.execute",
            "response_engine.generate_and_rank",
            "persistence.finalize_turn",
        ]

        for i, node_id in enumerate(full_path):
            graph.record_execution(
                node_id=node_id,
                trace_id="full_turn_test",
                session_id="test_session",
                timestamp_ms=float(i * 100),
            )

        # Validate the path
        is_valid = graph.validate_execution_path(
            trace_id="full_turn_test",
            session_id="test_session",
            execution_sequence=full_path,
        )
        assert is_valid
        assert len(graph.get_violations()) == 0

    def test_multiple_turns_independent_tracking(self) -> None:
        """Test that multiple turns can be tracked independently."""
        graph = ExecutionTopologyGraph()

        # First turn
        for i in range(3):
            graph.record_execution(
                node_id="control_plane.submit_turn" if i == 0 else "graph.execute",
                trace_id="turn1",
                session_id="sess1",
                timestamp_ms=float(i * 100),
            )

        # Second turn
        for i in range(3):
            graph.record_execution(
                node_id="control_plane.submit_turn" if i == 0 else "graph.execute",
                trace_id="turn2",
                session_id="sess2",
                timestamp_ms=float(i * 100),
            )

        trace = graph.get_execution_trace()
        assert len(trace) == 6
        turn1_events = [t for t in trace if t["trace_id"] == "turn1"]
        turn2_events = [t for t in trace if t["trace_id"] == "turn2"]
        assert len(turn1_events) == 3
        assert len(turn2_events) == 3
