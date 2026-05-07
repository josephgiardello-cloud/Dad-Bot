"""Priority 3: Unified Trace Envelope tests.

Tests for canonical TurnTrace object and integration.
"""

import json
import pytest
import time
from unittest.mock import MagicMock, patch

from dadbot.core.turn_trace import (
    TurnTrace,
    ExecutionNode,
    TurnInput,
    TurnOutput,
    NodeType,
    NodeStatus,
    create_turn_trace,
    set_current_trace,
    get_current_trace,
    record_node_to_current_trace,
    record_event_to_current_trace,
    TURN_TRACE_SCHEMA_VERSION,
)


@pytest.mark.unit
class TestExecutionNode:
    """Test ExecutionNode structure."""

    def test_node_creation(self):
        """Verify node can be created."""
        node = ExecutionNode(
            node_type=NodeType.PLANNER,
            node_id="planner-1",
            status=NodeStatus.SUCCESS,
        )
        
        assert node.node_type == NodeType.PLANNER
        assert node.node_id == "planner-1"
        assert node.status == NodeStatus.SUCCESS

    def test_node_serialization(self):
        """Verify node serializes correctly."""
        node = ExecutionNode(
            node_type=NodeType.INFERENCE,
            node_id="inference-1",
            status=NodeStatus.SUCCESS,
            duration_ms=1234.5,
            output="test response",
        )
        
        data = node.to_dict()
        
        assert isinstance(data, dict)
        assert data["node_type"] == "inference"
        assert data["node_id"] == "inference-1"
        assert data["status"] == "success"

    def test_node_with_tools(self):
        """Verify node can track tool invocations."""
        node = ExecutionNode(
            node_type=NodeType.INFERENCE,
            node_id="inference-1",
            tools_invoked=["tool_a", "tool_b"],
        )
        
        assert len(node.tools_invoked) == 2
        assert "tool_a" in node.tools_invoked

    def test_node_with_error(self):
        """Verify node can track errors."""
        node = ExecutionNode(
            node_type=NodeType.SAFETY,
            node_id="safety-1",
            status=NodeStatus.FAILED,
            error="Safety violation detected",
            error_type="SafetyException",
        )
        
        assert node.error == "Safety violation detected"
        assert node.error_type == "SafetyException"


@pytest.mark.unit
class TestTurnInput:
    """Test TurnInput normalization."""

    def test_turn_input_creation(self):
        """Verify turn input can be created."""
        inp = TurnInput(
            text="test input",
            session_id="test-session",
        )
        
        assert inp.text == "test input"
        assert inp.session_id == "test-session"

    def test_turn_input_serialization(self):
        """Verify input serializes."""
        inp = TurnInput(
            text="test",
            attachments=["file1.txt"],
            session_id="session-1",
            metadata={"key": "value"},
        )
        
        data = inp.to_dict()
        
        assert data["text"] == "test"
        assert len(data["attachments"]) == 1
        assert data["metadata"]["key"] == "value"


@pytest.mark.unit
class TestTurnOutput:
    """Test TurnOutput structure."""

    def test_turn_output_creation(self):
        """Verify output can be created."""
        out = TurnOutput(
            response="test response",
            should_end=False,
            confidence=0.95,
        )
        
        assert out.response == "test response"
        assert out.should_end is False
        assert out.confidence == 0.95

    def test_turn_output_fallback_flag(self):
        """Verify output can track fallback recovery."""
        out = TurnOutput(
            response="fallback response",
            should_end=False,
            recovery_fallback=True,
        )
        
        assert out.recovery_fallback is True


@pytest.mark.unit
class TestTurnTrace:
    """Test unified TurnTrace envelope."""

    def test_trace_creation(self):
        """Verify trace can be created."""
        trace = TurnTrace(
            trace_id="tr-123",
            session_id="session-1",
        )
        
        assert trace.trace_id == "tr-123"
        assert trace.session_id == "session-1"
        assert not trace.completed
        assert trace.schema_version == TURN_TRACE_SCHEMA_VERSION

    def test_trace_record_node(self):
        """Verify trace can record nodes."""
        trace = TurnTrace(trace_id="tr-1", session_id="s1")
        
        node1 = ExecutionNode(node_type=NodeType.PLANNER, node_id="p1")
        node2 = ExecutionNode(node_type=NodeType.INFERENCE, node_id="i1")
        
        trace.record_node(node1)
        trace.record_node(node2)
        
        assert len(trace.nodes) == 2
        assert trace.nodes[0].node_id == "p1"
        assert trace.nodes[1].node_id == "i1"

    def test_trace_record_event(self):
        """Verify trace can record events."""
        trace = TurnTrace(trace_id="tr-1")
        
        event1 = {"type": "memory_updated", "key": "test"}
        event2 = {"type": "tool_invoked", "tool": "test_tool"}
        
        trace.record_event(event1)
        trace.record_event(event2)
        
        assert len(trace.trace_events) == 2

    def test_trace_finalize_computes_checksum(self):
        """Verify finalize computes checksum."""
        trace = TurnTrace(
            trace_id="tr-1",
            session_id="s1",
            start_time=time.time(),
        )
        
        node = ExecutionNode(node_type=NodeType.SAVE, node_id="save-1")
        trace.record_node(node)
        trace.output = TurnOutput(response="test", should_end=False)
        
        trace.finalize()
        
        assert trace.completed
        assert trace.checksum.startswith("chk-")
        assert len(trace.checksum) > 10

    def test_trace_validates_commit_boundary(self):
        """Verify trace validates single commit boundary."""
        trace = TurnTrace(trace_id="tr-1")
        
        # Add multiple save nodes
        trace.record_node(ExecutionNode(node_type=NodeType.SAVE, node_id="save-1"))
        trace.record_node(ExecutionNode(node_type=NodeType.SAVE, node_id="save-2"))
        
        trace.finalize()
        
        assert trace.commit_boundary_count == 2
        # Should log warning about invariant violation

    def test_trace_serialization_roundtrip(self):
        """Verify trace can serialize and deserialize."""
        original = TurnTrace(
            trace_id="tr-test",
            session_id="s-test",
            input=TurnInput(text="test input", session_id="s-test"),
            output=TurnOutput(response="test output", should_end=False),
        )
        original.record_node(ExecutionNode(node_type=NodeType.PLANNER, node_id="p1"))
        original.finalize()
        
        data = original.to_dict()
        restored = TurnTrace.from_dict(data)
        
        assert restored.trace_id == original.trace_id
        assert restored.session_id == original.session_id
        assert restored.input.text == original.input.text
        assert len(restored.nodes) == 1
        assert restored.checksum == original.checksum

    def test_trace_json_serialization(self):
        """Verify trace can be converted to JSON."""
        trace = TurnTrace(
            trace_id="tr-json",
            session_id="s-json",
            input=TurnInput(text="test"),
        )
        
        json_str = trace.to_json()
        
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["trace_id"] == "tr-json"

    def test_trace_prevents_modification_after_complete(self):
        """Verify trace is immutable after finalize."""
        trace = TurnTrace(trace_id="tr-1", start_time=time.time())
        trace.finalize()
        
        node = ExecutionNode(node_type=NodeType.PLANNER, node_id="p1")
        
        # Should log warning, not crash
        trace.record_node(node)
        
        # Nodes should not be added
        assert len(trace.nodes) == 0


@pytest.mark.unit
class TestTurnTraceFactory:
    """Test trace creation factory."""

    def test_create_turn_trace(self):
        """Verify factory creates properly initialized trace."""
        trace = create_turn_trace(
            trace_id="tr-factory",
            session_id="s-factory",
            user_input="test input",
            metadata={"key": "value"},
        )
        
        assert trace.trace_id == "tr-factory"
        assert trace.session_id == "s-factory"
        assert trace.input.text == "test input"
        assert trace.input.metadata["key"] == "value"
        assert trace.start_time > 0


@pytest.mark.unit
class TestTraceContextManagement:
    """Test global trace context."""

    def test_set_and_get_current_trace(self):
        """Verify current trace context works."""
        trace = TurnTrace(trace_id="tr-ctx")
        
        set_current_trace(trace)
        retrieved = get_current_trace()
        
        assert retrieved is trace

    def test_record_node_to_current_trace(self):
        """Verify recording to current trace context."""
        trace = TurnTrace(trace_id="tr-ctx")
        set_current_trace(trace)
        
        node = ExecutionNode(node_type=NodeType.PLANNER, node_id="p1")
        record_node_to_current_trace(node)
        
        assert len(trace.nodes) == 1

    def test_record_event_to_current_trace(self):
        """Verify recording events to current trace context."""
        trace = TurnTrace(trace_id="tr-ctx")
        set_current_trace(trace)
        
        event = {"type": "test_event"}
        record_event_to_current_trace(event)
        
        assert len(trace.trace_events) == 1

    def test_record_with_no_current_trace(self):
        """Verify recording when no trace is set doesn't crash."""
        set_current_trace(None)
        
        # Should not raise
        record_node_to_current_trace(ExecutionNode(node_type=NodeType.PLANNER, node_id="p1"))
        record_event_to_current_trace({"type": "test"})


@pytest.mark.unit
class TestTraceIntegrity:
    """Test trace integrity checks."""

    def test_trace_duration_computed(self):
        """Verify trace duration is computed on finalize."""
        start = time.time()
        trace = TurnTrace(trace_id="tr-1", start_time=start)
        
        # Wait a bit
        time.sleep(0.01)
        
        trace.finalize()
        
        assert trace.duration_ms > 0
        assert trace.end_time > trace.start_time

    def test_trace_checksum_changes_with_content(self):
        """Verify checksum changes when content changes."""
        trace1 = TurnTrace(trace_id="tr-1")
        trace1.output = TurnOutput(response="response1", should_end=False)
        trace1.finalize()
        checksum1 = trace1.checksum
        
        trace2 = TurnTrace(trace_id="tr-1")
        trace2.output = TurnOutput(response="response2", should_end=False)
        trace2.finalize()
        checksum2 = trace2.checksum
        
        assert checksum1 != checksum2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
