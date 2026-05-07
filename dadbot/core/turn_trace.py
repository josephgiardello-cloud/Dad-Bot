"""Unified execution trace envelope.

Single canonical trace object for complete turn visibility:
- Input normalization and metadata
- Ordered execution node log
- Tool invocations and side effects
- Memory ledger events
- Invariant gates and transitions
- Output and completion status
- Error context and recovery paths

Design principle: Every turn produces exactly one TurnTrace with
complete ordered record of all execution decisions and effects.

Trace contract:
1. Immutable after turn completion
2. Durable persistence via control plane ledger
3. Indexed for replay and root cause analysis
4. Schema versioned for backward compatibility
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Literal

logger = logging.getLogger(__name__)

TURN_TRACE_SCHEMA_VERSION = "1.0"


class NodeType(str, Enum):
    """Execution graph node types."""
    TEMPORAL = "temporal"
    PREFLIGHT = "preflight"
    PLANNER = "planner"
    INFERENCE = "inference"
    SAFETY = "safety"
    REFLECTION = "reflection"
    SAVE = "save"


class NodeStatus(str, Enum):
    """Node execution status."""
    INIT = "init"
    RUNNING = "running"
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(slots=True)
class ExecutionNode:
    """Single node execution in turn graph."""
    
    node_type: NodeType | str
    node_id: str
    status: NodeStatus | str = "init"
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    
    # Node-specific outputs
    output: Any = None
    
    # Side effects and mutations
    tools_invoked: list[str] = field(default_factory=list)
    memory_events: list[str] = field(default_factory=list)
    
    # Error context
    error: str | None = None
    error_type: str | None = None
    
    # Invariant checks
    pre_invariants: dict[str, Any] = field(default_factory=dict)
    post_invariants: dict[str, Any] = field(default_factory=dict)
    invariant_violations: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionNode:
        """Deserialize from dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass(slots=True)
class TurnInput:
    """Normalized turn input envelope."""
    
    text: str
    attachments: list[str] = field(default_factory=list)
    session_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)


@dataclass(slots=True)
class TurnOutput:
    """Turn execution result."""
    
    response: str | None
    should_end: bool
    confidence: float = 1.0  # 0.0 = uncertain, 1.0 = confident
    recovery_fallback: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)


@dataclass(slots=True)
class TurnTrace:
    """Unified execution trace envelope.
    
    Complete immutable record of a single turn execution.
    Every field is populated and validated before turn completes.
    """
    
    # Identity
    trace_id: str
    turn_id: str = ""
    session_id: str = "default"
    
    # Contract versioning
    schema_version: str = TURN_TRACE_SCHEMA_VERSION
    
    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    
    # Input and output
    input: TurnInput = field(default_factory=lambda: TurnInput(text=""))
    output: TurnOutput = field(default_factory=lambda: TurnOutput(response=None, should_end=False))
    
    # Execution nodes in order
    nodes: list[ExecutionNode] = field(default_factory=list)
    
    # Events recorded during turn execution
    trace_events: list[dict[str, Any]] = field(default_factory=list)
    
    # Trace integrity
    checksum: str = ""
    commit_boundary_count: int = 0
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Status
    completed: bool = False
    error: str | None = None
    
    def record_node(self, node: ExecutionNode) -> None:
        """Record a node execution in trace."""
        if self.completed:
            logger.warning("Cannot record node: trace already completed")
            return
        self.nodes.append(node)
    
    def record_event(self, event: dict[str, Any]) -> None:
        """Record a turn execution event."""
        if self.completed:
            logger.warning("Cannot record event: trace already completed")
            return
        self.trace_events.append(dict(event))
    
    def finalize(self) -> None:
        """Mark trace as complete and compute integrity checks."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        
        # Validate trace integrity
        save_nodes = [n for n in self.nodes if n.node_type == NodeType.SAVE or n.node_type == "save"]
        self.commit_boundary_count = len(save_nodes)
        
        if self.commit_boundary_count != 1:
            logger.warning(
                "Trace invariant violation: expected 1 commit boundary, found %d",
                self.commit_boundary_count,
            )
        
        # Compute trace checksum
        self._compute_checksum()
        self.completed = True
    
    def _compute_checksum(self) -> None:
        """Compute trace integrity checksum."""
        trace_data = {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "ledger_event_count": len(self.trace_events),
            "output": self.output.to_dict(),
        }
        content = json.dumps(trace_data, sort_keys=True, default=str)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        self.checksum = f"chk-{digest[:32]}"
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "trace_id": self.trace_id,
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "schema_version": self.schema_version,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "input": self.input.to_dict(),
            "output": self.output.to_dict(),
            "nodes": [n.to_dict() for n in self.nodes],
            "trace_events": self.trace_events,
            "checksum": self.checksum,
            "commit_boundary_count": self.commit_boundary_count,
            "metadata": self.metadata,
            "completed": self.completed,
            "error": self.error,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurnTrace:
        """Deserialize from dict."""
        trace = cls(
            trace_id=data.get("trace_id", ""),
            turn_id=data.get("turn_id", ""),
            session_id=data.get("session_id", "default"),
            schema_version=data.get("schema_version", TURN_TRACE_SCHEMA_VERSION),
            start_time=float(data.get("start_time", 0.0)),
            end_time=float(data.get("end_time", 0.0)),
            duration_ms=float(data.get("duration_ms", 0.0)),
            checksum=str(data.get("checksum", "")),
            commit_boundary_count=int(data.get("commit_boundary_count", 0)),
            completed=bool(data.get("completed", False)),
            error=data.get("error"),
        )
        
        # Deserialize input
        if "input" in data:
            input_data = data["input"]
            trace.input = TurnInput(
                text=input_data.get("text", ""),
                attachments=input_data.get("attachments", []),
                session_id=input_data.get("session_id", "default"),
                metadata=input_data.get("metadata", {}),
                timestamp=float(input_data.get("timestamp", 0.0)),
            )
        
        # Deserialize output
        if "output" in data:
            output_data = data["output"]
            trace.output = TurnOutput(
                response=output_data.get("response"),
                should_end=bool(output_data.get("should_end", False)),
                confidence=float(output_data.get("confidence", 1.0)),
                recovery_fallback=bool(output_data.get("recovery_fallback", False)),
            )
        
        # Deserialize nodes
        for node_data in data.get("nodes", []):
            trace.nodes.append(ExecutionNode.from_dict(node_data))
        
        # Deserialize ledger events
        trace.trace_events = list(data.get("trace_events", []))
        
        # Deserialize metadata
        trace.metadata = dict(data.get("metadata", {}))
        
        return trace


def create_turn_trace(
    trace_id: str,
    session_id: str = "default",
    user_input: str = "",
    metadata: dict[str, Any] | None = None,
) -> TurnTrace:
    """Factory to create new turn trace."""
    trace = TurnTrace(
        trace_id=trace_id,
        session_id=session_id,
        start_time=time.time(),
        input=TurnInput(
            text=user_input,
            session_id=session_id,
            timestamp=time.time(),
            metadata=dict(metadata or {}),
        ),
    )
    return trace


# Module-level trace registry for current turn (thread-unsafe, for single-threaded context)
_current_trace: TurnTrace | None = None


def set_current_trace(trace: TurnTrace | None) -> None:
    """Set current trace for this execution context."""
    global _current_trace
    _current_trace = trace


def get_current_trace() -> TurnTrace | None:
    """Get current trace for this execution context."""
    return _current_trace


def record_node_to_current_trace(node: ExecutionNode) -> None:
    """Record execution node to current trace."""
    trace = get_current_trace()
    if trace is not None:
        trace.record_node(node)


def record_event_to_current_trace(event: dict[str, Any]) -> None:
    """Record ledger event to current trace."""
    trace = get_current_trace()
    if trace is not None:
        trace.record_event(event)
