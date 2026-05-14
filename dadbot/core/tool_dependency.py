"""Phase 2.4 — Tool Dependency Semantics.

Provides conditional edges, hard/soft dependency typing, and dependency resolution
for the tool execution DAG.

Formal model:
    - DependencyType: HARD (A must succeed for B) | SOFT (B runs regardless, uses fallback)
    - ConditionalEdge: directed edge from A → B with a predicate on A's output
    - DependencyResolver: given a DAG and observed outputs, returns runnable node_ids

Semantics:
    HARD dependency: if source tool fails or predicate evaluates False → target is BLOCKED
    SOFT dependency: if source fails or predicate is False → target runs with fallback input

This separates structural DAG edges (in tool_dag.py) from behavioral dependency
semantics (this module). The DAG topology is fixed; dependency semantics are runtime.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


# ---------------------------------------------------------------------------
# Dependency type
# ---------------------------------------------------------------------------


class DependencyType(enum.Enum):
    """How a dependency between two tools is enforced.

    HARD:  Target cannot run unless source succeeded AND predicate passed.
           Target is BLOCKED if source fails or predicate is False.
    SOFT:  Target runs regardless of source outcome.
           If source failed or predicate is False, target receives fallback_input.
    """

    HARD = "hard"
    SOFT = "soft"


# ---------------------------------------------------------------------------
# Predicate keys
# ---------------------------------------------------------------------------


# Built-in predicate keys — used by ConditionalEdge.predicate_key.
PREDICATE_ALWAYS_TRUE = "always_true"
PREDICATE_RESULT_NONEMPTY = "result_nonempty"
PREDICATE_STATUS_OK = "status_ok"
PREDICATE_HAS_GOALS = "has_goals"


def _predicate_always_true(_source_output: Any) -> bool:
    return True


def _predicate_result_nonempty(source_output: Any) -> bool:
    if source_output is None:
        return False
    if isinstance(source_output, (list, dict, str, tuple)):
        return bool(source_output)
    return True


def _predicate_status_ok(source_output: Any) -> bool:
    if isinstance(source_output, dict):
        return str(source_output.get("status", "ok")).lower() in _OK_STATUSES
    return source_output is not None


def _predicate_has_goals(source_output: Any) -> bool:
    if isinstance(source_output, dict):
        goals = source_output.get("goals", [])
        return bool(goals)
    if isinstance(source_output, list):
        return bool(source_output)
    return False


_PREDICATE_HANDLERS: dict[str, Any] = {
    PREDICATE_ALWAYS_TRUE: _predicate_always_true,
    PREDICATE_RESULT_NONEMPTY: _predicate_result_nonempty,
    PREDICATE_STATUS_OK: _predicate_status_ok,
    PREDICATE_HAS_GOALS: _predicate_has_goals,
}

_OK_STATUSES: frozenset[str] = frozenset({"ok"})


def _eval_builtin_predicate(predicate_key: str, source_output: Any) -> bool:
    handler = _PREDICATE_HANDLERS.get(str(predicate_key or "").strip())
    if callable(handler):
        return bool(handler(source_output))
    # Unknown predicate key → default True (non-blocking).
    return True


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionalEdge:
    """A behavioral dependency edge from source_node_id → target_node_id.

    Attributes:
        source_node_id:  Node that must produce output first.
        target_node_id:  Node that is conditionally enabled.
        predicate_key:   Predicate evaluated on source's output.
        dependency_type: HARD (blocks target on failure) or SOFT (fallback).
        fallback_input:  Used for SOFT deps when predicate fails. May be None.

    """

    source_node_id: str
    target_node_id: str
    predicate_key: str
    dependency_type: DependencyType
    fallback_input: Any = None

    def evaluate(self, source_output: Any) -> bool:
        """Evaluate the predicate on the source tool's output."""
        return _eval_builtin_predicate(self.predicate_key, source_output)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "predicate_key": self.predicate_key,
            "dependency_type": self.dependency_type.value,
            "fallback_input": self.fallback_input,
        }


# ---------------------------------------------------------------------------
# Resolution result
# ---------------------------------------------------------------------------


class NodeStatus(enum.Enum):
    """Status of a node after dependency resolution."""

    RUNNABLE = "runnable"  # All HARD deps satisfied; may run.
    BLOCKED = "blocked"  # One or more HARD deps not satisfied.
    SOFT_DEGRADED = "soft_degraded"  # SOFT dep failed; runs with fallback.
    PENDING = "pending"  # Dependencies exist but outputs not yet available.


@dataclass(frozen=True)
class NodeResolutionResult:
    """Resolution decision for a single node."""

    node_id: str
    status: NodeStatus
    effective_input: Any  # The input the node should receive (may be fallback).
    blocking_edges: tuple[str, ...]  # IDs of HARD edges that blocked this node.

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "blocking_edges": list(self.blocking_edges),
            "has_fallback": self.status == NodeStatus.SOFT_DEGRADED,
        }


# ---------------------------------------------------------------------------
# Dependency resolver
# ---------------------------------------------------------------------------


class DependencyResolver:
    """Resolves which nodes are runnable given observed tool outputs.

    Usage:
        resolver = DependencyResolver()
        resolver.add_edge(ConditionalEdge(
            source_node_id="node-0001-...",
            target_node_id="node-0002-...",
            predicate_key=PREDICATE_STATUS_OK,
            dependency_type=DependencyType.HARD,
        ))
        results = resolver.resolve(outputs={"node-0001-...": {"status": "ok", "data": [...]}})
        runnable = [r.node_id for r in results if r.status == NodeStatus.RUNNABLE]
    """

    def __init__(self) -> None:
        self._edges: list[ConditionalEdge] = []

    def add_edge(self, edge: ConditionalEdge) -> None:
        self._edges.append(edge)

    def add_edges(self, edges: list[ConditionalEdge]) -> None:
        self._edges.extend(edges)

    def edges_for_target(self, target_node_id: str) -> list[ConditionalEdge]:
        return [e for e in self._edges if e.target_node_id == target_node_id]

    def resolve_node(
        self,
        node_id: str,
        outputs: dict[str, Any],
        *,
        base_input: Any = None,
    ) -> NodeResolutionResult:
        """Resolve a single node given available outputs.

        Args:
            node_id:    The target node to resolve.
            outputs:    Dict of node_id → output for nodes that have already run.
            base_input: The nominal input to pass (used when all HARD deps pass).

        """
        relevant_edges = self.edges_for_target(node_id)
        if not relevant_edges:
            return NodeResolutionResult(
                node_id=node_id,
                status=NodeStatus.RUNNABLE,
                effective_input=base_input,
                blocking_edges=(),
            )

        blocking_edges: list[str] = []
        fallback_inputs: list[Any] = []
        is_soft_degraded = False

        for edge in relevant_edges:
            source_output = outputs.get(edge.source_node_id)
            if source_output is None and edge.source_node_id not in outputs:
                # Source hasn't run yet.
                if edge.dependency_type == DependencyType.HARD:
                    blocking_edges.append(
                        f"{edge.source_node_id}→{edge.target_node_id}",
                    )
                    continue
                # SOFT: no output yet → use fallback.
                if edge.fallback_input is not None:
                    fallback_inputs.append(edge.fallback_input)
                is_soft_degraded = True
                continue

            passed = edge.evaluate(source_output)
            if not passed:
                if edge.dependency_type == DependencyType.HARD:
                    blocking_edges.append(
                        f"{edge.source_node_id}→{edge.target_node_id}",
                    )
                else:
                    # SOFT: predicate failed → use fallback.
                    if edge.fallback_input is not None:
                        fallback_inputs.append(edge.fallback_input)
                    is_soft_degraded = True

        if blocking_edges:
            return NodeResolutionResult(
                node_id=node_id,
                status=NodeStatus.BLOCKED,
                effective_input=None,
                blocking_edges=tuple(blocking_edges),
            )

        if is_soft_degraded:
            effective_input = fallback_inputs[0] if fallback_inputs else base_input
            return NodeResolutionResult(
                node_id=node_id,
                status=NodeStatus.SOFT_DEGRADED,
                effective_input=effective_input,
                blocking_edges=(),
            )

        return NodeResolutionResult(
            node_id=node_id,
            status=NodeStatus.RUNNABLE,
            effective_input=base_input,
            blocking_edges=(),
        )

    def resolve(
        self,
        outputs: dict[str, Any],
        *,
        node_ids: list[str] | None = None,
    ) -> list[NodeResolutionResult]:
        """Resolve all nodes that have at least one edge targeting them.

        Args:
            outputs:  Dict of node_id → output for already-run nodes.
            node_ids: Optional explicit list of nodes to resolve.
                      If None, resolves all nodes that are targets of any edge.

        """
        if node_ids is not None:
            targets = list(node_ids)
        else:
            targets = sorted({e.target_node_id for e in self._edges})

        return [self.resolve_node(nid, outputs) for nid in targets]

    def runnable_node_ids(
        self,
        outputs: dict[str, Any],
        *,
        node_ids: list[str] | None = None,
    ) -> list[str]:
        """Return only the node IDs that are RUNNABLE or SOFT_DEGRADED."""
        results = self.resolve(outputs, node_ids=node_ids)
        return [r.node_id for r in results if r.status in (NodeStatus.RUNNABLE, NodeStatus.SOFT_DEGRADED)]

    def dependency_graph_hash(self) -> str:
        """Stable hash of all registered edges (for audit)."""
        return _sha256(
            {
                "edges": [
                    e.to_dict()
                    for e in sorted(
                        self._edges,
                        key=lambda e: (e.source_node_id, e.target_node_id),
                    )
                ],
            },
        )


__all__ = [
    "PREDICATE_ALWAYS_TRUE",
    "PREDICATE_HAS_GOALS",
    "PREDICATE_RESULT_NONEMPTY",
    "PREDICATE_STATUS_OK",
    "ConditionalEdge",
    "DependencyResolver",
    "DependencyType",
    "NodeResolutionResult",
    "NodeStatus",
]
