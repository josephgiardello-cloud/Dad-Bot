"""dadbot/core/topology_runtime.py — Phase 4 Runtime: Execution Topology Enforcement.

Part of the Phase Closure runtime bundle.

Provides:
- Execution topology validation
- Shadow path detection and enforcement
- Canonical chain enforcement
- Topology conformance audit hooks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from dadbot.core.execution_topology_graph import (
    ExecutionNode,
    NodeSeverity,
    NodeType,
    get_execution_topology_graph,
)

logger = logging.getLogger(__name__)


@dataclass
class TopologyValidationResult:
    """Result of topology validation."""

    passed: bool
    violations_critical: int
    violations_high: int
    violations_total: int
    details: dict[str, Any]


class TopologyRuntime:
    """Phase 4 runtime: enforces canonical execution topology.

    Responsibilities:
    - Register execution events on canonical nodes
    - Validate execution paths against the topology
    - Detect and classify shadow path attempts
    - Produce audit reports for topology conformance
    - Enforce hard invariants in strict mode
    """

    def __init__(self, *, strict_mode: bool = False) -> None:
        self._graph = get_execution_topology_graph()
        self._strict_mode = strict_mode
        self._graph.set_strict_mode(strict_mode)
        self._current_trace_id: str | None = None
        self._current_session_id: str | None = None
        self._execution_sequence: list[str] = []
        self._baseline_violation_count: int = 0

    def _require_registered_node(self, *, node_id: str) -> ExecutionNode | None:
        node = self._graph.get_node(node_id)
        if node is None:
            message = f"Topology runtime violation: unknown node '{node_id}'"
            if self._strict_mode:
                raise RuntimeError(message)
            logger.warning(message)
            self._graph.record_execution(
                node_id=node_id,
                trace_id=self._current_trace_id or "",
                session_id=self._current_session_id or "default",
                timestamp_ms=0.0,
                metadata={"topology_runtime": "unknown_node"},
            )
            return None
        return node

    def _enforce_transition(self, *, node: ExecutionNode) -> None:
        if not self._execution_sequence:
            if node.node_type != NodeType.ENTRY:
                message = (
                    "Topology runtime violation: first node must be entry "
                    f"(got '{node.node_id}')"
                )
                if self._strict_mode:
                    raise RuntimeError(message)
                logger.warning(message)
            return

        previous = self._execution_sequence[-1]
        previous_node = self._graph.get_node(previous)
        if previous_node is None:
            message = f"Topology runtime violation: previous node '{previous}' is unknown"
            if self._strict_mode:
                raise RuntimeError(message)
            logger.warning(message)
            return
        if node.node_id not in previous_node.children_node_ids:
            message = (
                "Topology runtime violation: illegal transition "
                f"{previous_node.node_id} -> {node.node_id}"
            )
            if self._strict_mode:
                raise RuntimeError(message)
            logger.warning(message)

    def begin_turn(self, *, trace_id: str, session_id: str) -> None:
        """Mark the beginning of a turn execution.

        Args:
            trace_id: Unique trace identifier for this turn
            session_id: Session identifier
        """
        self._current_trace_id = str(trace_id or "")
        self._current_session_id = str(session_id or "")
        self._execution_sequence = []
        self._baseline_violation_count = len(self._graph.get_violations())
        logger.debug(
            "Topology runtime: begin_turn trace_id=%s session_id=%s",
            self._current_trace_id,
            self._current_session_id,
        )

    def record_node_entry(
        self,
        *,
        node_id: str,
        timestamp_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record entry into a canonical execution node.

        Args:
            node_id: The node being entered
            timestamp_ms: Optional timestamp in milliseconds
            metadata: Optional metadata about the execution event
        """
        if not self._current_trace_id:
            logger.warning("record_node_entry called without active trace")
            return

        node = self._require_registered_node(node_id=node_id)
        if node is None:
            return
        self._enforce_transition(node=node)

        self._execution_sequence.append(node_id)

        self._graph.record_execution(
            node_id=node_id,
            trace_id=self._current_trace_id,
            session_id=self._current_session_id or "default",
            timestamp_ms=float(timestamp_ms or 0.0),
            metadata=dict(metadata or {}),
        )

        logger.debug(
            "Topology: node entry %s (depth=%d)",
            node_id,
            len(self._execution_sequence),
        )

    def end_turn(self) -> TopologyValidationResult:
        """Finalize and validate the turn's execution path.

        Returns:
            TopologyValidationResult with validation outcome
        """
        if not self._current_trace_id:
            logger.warning("end_turn called without active trace")
            return TopologyValidationResult(
                passed=False,
                violations_critical=0,
                violations_high=0,
                violations_total=0,
                details={"reason": "no_active_trace"},
            )

        is_valid = self._graph.validate_execution_path(
            trace_id=self._current_trace_id,
            session_id=self._current_session_id or "default",
            execution_sequence=self._execution_sequence,
        )

        violations = self._graph.get_violations()
        turn_violations = violations[self._baseline_violation_count :]
        critical = len([v for v in turn_violations if v.severity == NodeSeverity.CRITICAL])
        high = len([v for v in turn_violations if v.severity == NodeSeverity.HIGH])
        passed = bool(is_valid and critical == 0 and high == 0)

        result = TopologyValidationResult(
            passed=passed,
            violations_critical=critical,
            violations_high=high,
            violations_total=len(turn_violations),
            details={
                "trace_id": self._current_trace_id,
                "session_id": self._current_session_id,
                "sequence_length": len(self._execution_sequence),
                "sequence": self._execution_sequence,
                "violations": [
                    {
                        "node_id": v.node_id,
                        "violation_type": v.violation_type,
                        "severity": str(v.severity),
                        "detail": v.detail,
                    }
                    for v in turn_violations
                ],
            },
        )

        logger.info(
            "Topology: end_turn trace_id=%s passed=%s critical=%d high=%d",
            self._current_trace_id,
            passed,
            critical,
            high,
        )

        if not passed and self._strict_mode:
            logger.error(
                "Topology strict mode violation: %d critical violations",
                critical,
            )

        self._current_trace_id = None
        self._current_session_id = None
        self._execution_sequence = []

        return result

    def get_audit_report(self) -> dict[str, Any]:
        """Get comprehensive topology audit report.

        Returns:
            Dict containing:
            - total_nodes: Number of registered nodes
            - violations_critical: Count of critical violations
            - violations_high: Count of high violations
            - violations_warning: Count of warning violations
            - execution_trace_length: Number of recorded execution events
            - violations: Full list of violations
        """
        return self._graph.audit_report()

    def set_strict_mode(self, enabled: bool) -> None:
        """Enable/disable strict mode for topology enforcement."""
        self._strict_mode = bool(enabled)
        self._graph.set_strict_mode(enabled)

    def clear_audit_data(self) -> None:
        """Clear violation and trace logs (for testing/resets)."""
        self._graph.clear_violations()
        self._graph.clear_execution_trace()
        self._execution_sequence = []
