"""dadbot/core/execution_topology_graph.py — Execution topology enforcement and validation.

Canonical execution topology:
    control_plane.submit_turn
        ↓ (kernel_gateway delegation)
    kernel_gateway.submit_turn
        ↓ (pre-gates)
    [execution gates]
        ↓
    orchestrator.handle_turn
        ↓ (trace binding)
    orchestrator._run_graph_with_trace_binding
        ↓ (nested async wrapper)
    [nested _run closure]
        ↓
    graph.execute
        ↓ (node traversal)
    [graph nodes: plan, route, execute, rank, commit]
        ↓
    response_engine.generate_and_rank
        ↓
    persistence.finalize_turn (commit boundary)

Enforcement:
- All graph.execute() calls MUST originate from registered topology nodes
- No direct entry to intermediate nodes
- No fallback/bypass execution paths
- Topology violations are hard invariants in strict mode
"""

from __future__ import annotations

import dataclasses
import logging
from enum import StrEnum
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class NodeType(StrEnum):
    """Canonical execution node types."""

    ENTRY = "entry"  # control_plane.submit_turn
    GATE = "gate"  # validation/authorization gates
    PLANNER = "planner"  # cognitive planner
    ROUTER = "router"  # tool/strategy routing
    EXECUTOR = "executor"  # graph execution
    RANKER = "ranker"  # response ranking
    COMMIT = "commit"  # persistence/finalization
    TRACE = "trace"  # trace binding wrapper


class NodeSeverity(StrEnum):
    """Severity of topology violations."""

    CRITICAL = "critical"  # Hard blocker
    HIGH = "high"  # Should not occur
    WARNING = "warning"  # Informational
    INFO = "info"  # Telemetry


@dataclasses.dataclass(frozen=True)
class ExecutionNode:
    """A registered execution node in the canonical topology.

    Attributes:
        node_id: Unique identifier (e.g., "control_plane.submit_turn")
        node_type: Category (entry, gate, planner, etc.)
        module_path: Full Python module path
        callable_name: Function/method name
        parent_node_id: Predecessor in the canonical chain (if applicable)
        children_node_ids: List of allowed successor nodes
        description: Human-readable description
        is_deterministic: Whether execution is deterministic (for replay)
        allows_branching: Whether this node can delegate to multiple branches
    """

    node_id: str
    node_type: NodeType
    module_path: str
    callable_name: str
    parent_node_id: str | None = None
    children_node_ids: list[str] = dataclasses.field(default_factory=list)
    description: str = ""
    is_deterministic: bool = True
    allows_branching: bool = False

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExecutionNode):
            return False
        return self.node_id == other.node_id


@dataclasses.dataclass(frozen=True)
class TopologyViolation:
    """A detected deviation from the canonical execution topology.

    Attributes:
        violation_id: Unique identifier for this violation
        node_id: Which node was involved
        violation_type: What went wrong (unknown_node, unauthorized_parent, etc.)
        severity: Critical/High/Warning/Info
        detail: Human-readable explanation
        context: Additional metadata (trace_id, session_id, etc.)
    """

    violation_id: str
    node_id: str
    violation_type: str
    severity: NodeSeverity
    detail: str
    context: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.violation_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TopologyViolation):
            return False
        return self.violation_id == other.violation_id


class ExecutionTopologyGraph:
    """Registry and validator for the canonical execution topology.

    This class maintains:
    1. The full registry of allowed execution nodes
    2. Parent-child relationships (the topology dag)
    3. Violation tracking and enforcement policies
    4. Audit trail of topology conformance
    """

    def __init__(self) -> None:
        self._nodes: dict[str, ExecutionNode] = {}
        self._violations: list[TopologyViolation] = []
        self._execution_trace: list[dict[str, Any]] = []
        self._strict_mode = False
        self._initialize_canonical_topology()

    def _initialize_canonical_topology(self) -> None:
        """Register all canonical execution nodes in the required order."""
        nodes = [
            # Entry point
            ExecutionNode(
                node_id="control_plane.submit_turn",
                node_type=NodeType.ENTRY,
                module_path="dadbot.core.control_plane",
                callable_name="submit_turn",
                children_node_ids=["kernel_gateway.submit_turn"],
                description="Public API entrypoint for turn submission",
                is_deterministic=True,
                allows_branching=False,
            ),
            # Gateway delegation
            ExecutionNode(
                node_id="kernel_gateway.submit_turn",
                node_type=NodeType.GATE,
                module_path="dadbot.core.kernel_gateway",
                callable_name="submit_turn",
                parent_node_id="control_plane.submit_turn",
                children_node_ids=["orchestrator.handle_turn"],
                description="Kernel gateway turn routing",
                is_deterministic=True,
                allows_branching=False,
            ),
            # Planner stage
            ExecutionNode(
                node_id="orchestrator.handle_turn",
                node_type=NodeType.PLANNER,
                module_path="dadbot.core.orchestrator",
                callable_name="handle_turn",
                parent_node_id="kernel_gateway.submit_turn",
                children_node_ids=["orchestrator._run_graph_with_trace_binding"],
                description="Main turn orchestration and planning",
                is_deterministic=True,
                allows_branching=False,
            ),
            # Trace binding wrapper
            ExecutionNode(
                node_id="orchestrator._run_graph_with_trace_binding",
                node_type=NodeType.TRACE,
                module_path="dadbot.core.orchestrator",
                callable_name="_run_graph_with_trace_binding",
                parent_node_id="orchestrator.handle_turn",
                children_node_ids=["graph.execute"],
                description="Wraps graph execution with trace binding",
                is_deterministic=True,
                allows_branching=False,
            ),
            # Graph execution
            ExecutionNode(
                node_id="graph.execute",
                node_type=NodeType.EXECUTOR,
                module_path="dadbot.core.graph",
                callable_name="execute",
                parent_node_id="orchestrator._run_graph_with_trace_binding",
                children_node_ids=["response_engine.generate_and_rank"],
                description="Core execution graph traversal",
                is_deterministic=True,
                allows_branching=False,
            ),
            # Response generation and ranking
            ExecutionNode(
                node_id="response_engine.generate_and_rank",
                node_type=NodeType.RANKER,
                module_path="dadbot.core.response_engine",
                callable_name="generate_and_rank",
                parent_node_id="graph.execute",
                children_node_ids=["persistence.finalize_turn"],
                description="Multi-candidate generation and ranking with learning",
                is_deterministic=False,  # Learning introduces non-determinism
                allows_branching=True,  # Multiple candidates generated
            ),
            # Commit boundary
            ExecutionNode(
                node_id="persistence.finalize_turn",
                node_type=NodeType.COMMIT,
                module_path="dadbot.core.persistence",
                callable_name="finalize_turn",
                parent_node_id="response_engine.generate_and_rank",
                children_node_ids=[],
                description="Persistence finalization and state commit",
                is_deterministic=True,
                allows_branching=False,
            ),
        ]

        for node in nodes:
            self._nodes[node.node_id] = node

    def register_node(self, node: ExecutionNode) -> None:
        """Register a new execution node (for extensibility)."""
        if node.node_id in self._nodes:
            logger.warning("Node %s already registered; overwriting", node.node_id)
        self._nodes[node.node_id] = node

    def get_node(self, node_id: str) -> ExecutionNode | None:
        """Retrieve a registered node by ID."""
        return self._nodes.get(node_id)

    def get_all_nodes(self) -> list[ExecutionNode]:
        """Get all registered nodes in canonical order."""
        return list(self._nodes.values())

    def record_execution(
        self,
        *,
        node_id: str,
        trace_id: str,
        session_id: str,
        timestamp_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an execution event at a particular node."""
        node = self.get_node(node_id)
        if node is None:
            self._record_violation(
                violation_type="unknown_node",
                node_id=node_id,
                severity=NodeSeverity.CRITICAL,
                detail=f"Execution at unregistered node {node_id}",
                context={
                    "trace_id": str(trace_id or ""),
                    "session_id": str(session_id or ""),
                },
            )
            return

        self._execution_trace.append(
            {
                "node_id": str(node_id),
                "trace_id": str(trace_id or ""),
                "session_id": str(session_id or ""),
                "timestamp_ms": float(timestamp_ms),
                "metadata": dict(metadata or {}),
            },
        )

    def validate_execution_path(
        self,
        *,
        trace_id: str,
        session_id: str,
        execution_sequence: list[str],
    ) -> bool:
        """Validate that an execution sequence follows the canonical topology.

        Args:
            trace_id: Unique trace identifier
            session_id: Session identifier
            execution_sequence: List of node_ids visited in order

        Returns:
            True if sequence is valid, False otherwise
        """
        if not execution_sequence:
            return True

        # First node must be an entry point
        first_node_id = execution_sequence[0]
        first_node = self.get_node(first_node_id)
        if first_node is None or first_node.node_type != NodeType.ENTRY:
            self._record_violation(
                violation_type="invalid_entry",
                node_id=first_node_id,
                severity=NodeSeverity.CRITICAL,
                detail=f"Execution did not start at entry node; started at {first_node_id}",
                context={
                    "trace_id": str(trace_id or ""),
                    "session_id": str(session_id or ""),
                },
            )
            return False

        # Validate each transition
        for i, node_id in enumerate(execution_sequence):
            node = self.get_node(node_id)
            if node is None:
                self._record_violation(
                    violation_type="unknown_node",
                    node_id=node_id,
                    severity=NodeSeverity.CRITICAL,
                    detail=f"Unknown node in execution sequence: {node_id}",
                    context={
                        "trace_id": str(trace_id or ""),
                        "session_id": str(session_id or ""),
                        "position_in_sequence": int(i),
                    },
                )
                return False

            # If not the last node, check that transition is allowed
            if i < len(execution_sequence) - 1:
                next_node_id = execution_sequence[i + 1]
                if next_node_id not in node.children_node_ids:
                    self._record_violation(
                        violation_type="unauthorized_transition",
                        node_id=node_id,
                        severity=NodeSeverity.CRITICAL,
                        detail=f"Transition {node_id} → {next_node_id} not allowed",
                        context={
                            "trace_id": str(trace_id or ""),
                            "session_id": str(session_id or ""),
                            "allowed_next": node.children_node_ids,
                        },
                    )
                    return False

        return True

    def _record_violation(
        self,
        *,
        violation_type: str,
        node_id: str,
        severity: NodeSeverity,
        detail: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record a topology violation."""
        violation = TopologyViolation(
            violation_id=f"topo:{len(self._violations)}:{node_id}",
            node_id=node_id,
            violation_type=violation_type,
            severity=severity,
            detail=detail,
            context=dict(context or {}),
        )
        self._violations.append(violation)
        if self._strict_mode and severity == NodeSeverity.CRITICAL:
            logger.error("Topology violation (critical): %s", detail)

    def get_violations(self) -> list[TopologyViolation]:
        """Get all recorded violations."""
        return list(self._violations)

    def clear_violations(self) -> None:
        """Clear violation log."""
        self._violations.clear()

    def get_execution_trace(self) -> list[dict[str, Any]]:
        """Get the recorded execution trace."""
        return list(self._execution_trace)

    def clear_execution_trace(self) -> None:
        """Clear execution trace log."""
        self._execution_trace.clear()

    def set_strict_mode(self, enabled: bool) -> None:
        """Enable/disable strict mode (treats HIGH violations as errors)."""
        self._strict_mode = bool(enabled)

    def audit_report(self) -> dict[str, Any]:
        """Generate an audit report of topology conformance.

        Returns:
            Dict with:
            - total_nodes: Number of registered nodes
            - violations_critical: Count of critical violations
            - violations_high: Count of high violations
            - violations_warning: Count of warning violations
            - execution_trace_length: Number of recorded execution events
            - violations: Full list of violations
        """
        violations = self.get_violations()
        return {
            "total_nodes": len(self._nodes),
            "violations_critical": len([v for v in violations if v.severity == NodeSeverity.CRITICAL]),
            "violations_high": len([v for v in violations if v.severity == NodeSeverity.HIGH]),
            "violations_warning": len([v for v in violations if v.severity == NodeSeverity.WARNING]),
            "execution_trace_length": len(self._execution_trace),
            "violations": [
                {
                    "violation_id": v.violation_id,
                    "node_id": v.node_id,
                    "violation_type": v.violation_type,
                    "severity": str(v.severity),
                    "detail": v.detail,
                    "context": v.context,
                }
                for v in violations
            ],
        }


# Global singleton topology graph
_global_topology_graph: ExecutionTopologyGraph | None = None


def get_execution_topology_graph() -> ExecutionTopologyGraph:
    """Get the global execution topology graph singleton."""
    global _global_topology_graph
    if _global_topology_graph is None:
        _global_topology_graph = ExecutionTopologyGraph()
    return _global_topology_graph
