"""Tool DAG formalization (Phase 1 + Phase 2).

Phase 1 — ToolDAG: canonical graph spec with:
  - Acyclicity by construction (edge insertion restricted by sequence ordering)
  - Deterministic node IDs and ordering keys
  - Execution order derivable without runtime heuristics (topological sort)

Phase 2 — ToolPlan IR split:
  - ToolPlanIR: planning-layer representation (intent, candidates, constraints)
  - ToolGraph (compiled ToolDAG): executable DAG produced by ToolPlanCompiler
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Phase 1: ToolDAG
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolNode:
    """A single tool invocation vertex in the execution DAG.

    ``sequence`` is the monotonic ordering key assigned at DAG build time.
    ``deterministic_id`` is the stable content-addressed ID derived from
    (tool_name, args) so that identical logical calls produce the same node ID
    regardless of creation order.
    """

    node_id: str
    tool_name: str
    intent: str
    args: dict[str, Any]
    priority: int
    sequence: int
    deterministic_id: str

    @classmethod
    def build(
        cls,
        tool_name: str,
        intent: str,
        args: dict[str, Any],
        priority: int,
        sequence: int,
    ) -> "ToolNode":
        stable = _stable_hash(
            {"tool_name": str(tool_name or "").strip().lower(), "args": dict(args or {})}
        )
        det_id = stable[:24]
        node_id = f"node-{sequence:04d}-{det_id}"
        return cls(
            node_id=node_id,
            tool_name=str(tool_name or "").strip().lower(),
            intent=str(intent or ""),
            args=dict(args or {}),
            priority=int(priority),
            sequence=int(sequence),
            deterministic_id=det_id,
        )

    def ordering_key(self) -> tuple[int, str, str]:
        """Fully deterministic ordering key: (priority, intent, deterministic_id)."""
        return (self.priority, self.intent, self.deterministic_id)


@dataclass(frozen=True)
class ToolEdge:
    """A directed dependency edge in the execution DAG.

    Acyclicity guarantee: edges are only accepted if source.sequence < target.sequence
    (enforced by ToolDAG.add_edge).
    """

    source_id: str
    target_id: str
    edge_type: str = "sequential"  # sequential | parallel | conditional

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
        }


@dataclass
class ToolDAG:
    """Canonical, acyclic tool execution graph.

    Acyclicity by construction: ``add_edge`` refuses any edge where
    source.sequence >= target.sequence, preventing cycles before they form.
    Execution order is fully determined by topological sort (Kahn's algorithm)
    with a stable tiebreaker of ToolNode.ordering_key() — no runtime heuristics.
    """

    nodes: list[ToolNode] = field(default_factory=list)
    edges: list[ToolEdge] = field(default_factory=list)

    # --- node map for fast lookup ---
    _node_by_id: dict[str, ToolNode] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        for n in self.nodes:
            self._node_by_id[n.node_id] = n

    @classmethod
    def from_nodes(cls, nodes: list[ToolNode]) -> "ToolDAG":
        dag = cls(nodes=list(nodes))
        return dag

    def add_node(self, node: ToolNode) -> None:
        if node.node_id not in self._node_by_id:
            self.nodes.append(node)
            self._node_by_id[node.node_id] = node

    def add_edge(self, source_id: str, target_id: str, edge_type: str = "sequential") -> None:
        """Add an edge, enforcing acyclicity by construction.

        Raises ``ValueError`` if source.sequence >= target.sequence or if
        either node is unknown.
        """
        src = self._node_by_id.get(source_id)
        tgt = self._node_by_id.get(target_id)
        if src is None:
            raise ValueError(f"ToolDAG.add_edge: unknown source node {source_id!r}")
        if tgt is None:
            raise ValueError(f"ToolDAG.add_edge: unknown target node {target_id!r}")
        if src.sequence >= tgt.sequence:
            raise ValueError(
                f"ToolDAG.add_edge: source.sequence ({src.sequence}) must be strictly less than "
                f"target.sequence ({tgt.sequence}) to guarantee acyclicity — "
                f"rejected edge {source_id!r} → {target_id!r}"
            )
        edge = ToolEdge(source_id=source_id, target_id=target_id, edge_type=edge_type)
        if edge not in self.edges:
            self.edges.append(edge)

    @property
    def root_nodes(self) -> list[ToolNode]:
        """Nodes with no incoming edges (execution entry points)."""
        target_ids = {e.target_id for e in self.edges}
        return [n for n in self.nodes if n.node_id not in target_ids]

    @property
    def terminal_nodes(self) -> list[ToolNode]:
        """Nodes with no outgoing edges (execution exit points)."""
        source_ids = {e.source_id for e in self.edges}
        return [n for n in self.nodes if n.node_id not in source_ids]

    def execution_order(self) -> list[ToolNode]:
        """Topologically sorted execution order with a deterministic tiebreaker.

        Uses Kahn's algorithm.  Nodes at the same topological level are sorted
        by ``ToolNode.ordering_key()`` so the result is fully reproducible.
        Raises ``RuntimeError`` if the graph contains a cycle (should not occur
        when ``add_edge`` is used, but guards against direct attribute mutation).
        """
        in_degree: dict[str, int] = {n.node_id: 0 for n in self.nodes}
        adjacency: dict[str, list[str]] = {n.node_id: [] for n in self.nodes}
        for edge in self.edges:
            in_degree[edge.target_id] = in_degree.get(edge.target_id, 0) + 1
            adjacency.setdefault(edge.source_id, []).append(edge.target_id)

        node_by_id = {n.node_id: n for n in self.nodes}
        queue: list[ToolNode] = sorted(
            [n for n in self.nodes if in_degree[n.node_id] == 0],
            key=lambda n: n.ordering_key(),
        )
        result: list[ToolNode] = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            neighbours = sorted(
                [node_by_id[nid] for nid in adjacency.get(node.node_id, []) if nid in node_by_id],
                key=lambda n: n.ordering_key(),
            )
            for neighbour in neighbours:
                in_degree[neighbour.node_id] -= 1
                if in_degree[neighbour.node_id] == 0:
                    # Insert in sorted order.
                    import bisect
                    keys = [n.ordering_key() for n in queue]
                    idx = bisect.bisect_left(keys, neighbour.ordering_key())
                    queue.insert(idx, neighbour)
        if len(result) != len(self.nodes):
            raise RuntimeError(
                "ToolDAG.execution_order: cycle detected — topological sort incomplete. "
                f"Processed {len(result)} of {len(self.nodes)} nodes."
            )
        return result

    def is_acyclic(self) -> bool:
        """True iff the graph has no cycles.  Should always be True when using add_edge."""
        try:
            order = self.execution_order()
            return len(order) == len(self.nodes)
        except RuntimeError:
            return False

    def deterministic_hash(self) -> str:
        """Content-addressed hash of the DAG structure (nodes + edges in execution order)."""
        try:
            ordered = self.execution_order()
        except RuntimeError:
            ordered = sorted(self.nodes, key=lambda n: n.ordering_key())
        payload = {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "tool_name": n.tool_name,
                    "intent": n.intent,
                    "priority": n.priority,
                    "sequence": n.sequence,
                    "deterministic_id": n.deterministic_id,
                }
                for n in ordered
            ],
            "edges": sorted(
                [e.to_dict() for e in self.edges],
                key=lambda e: (e["source_id"], e["target_id"]),
            ),
        }
        return _stable_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        try:
            exec_order = [n.node_id for n in self.execution_order()]
        except RuntimeError:
            exec_order = []
        return {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "tool_name": n.tool_name,
                    "intent": n.intent,
                    "args": n.args,
                    "priority": n.priority,
                    "sequence": n.sequence,
                    "deterministic_id": n.deterministic_id,
                }
                for n in self.nodes
            ],
            "edges": [e.to_dict() for e in self.edges],
            "root_nodes": [n.node_id for n in self.root_nodes],
            "terminal_nodes": [n.node_id for n in self.terminal_nodes],
            "execution_order": exec_order,
            "dag_hash": self.deterministic_hash(),
        }


def build_dag_from_execution_plan(execution_plan: list[dict[str, Any]]) -> ToolDAG:
    """Construct a ToolDAG from a ToolRouter-produced execution_plan list.

    Nodes are ordered by (priority, intent, deterministic_id).  Sequential edges
    are added between adjacent nodes in execution order — guaranteeing acyclicity
    because each target has a strictly greater sequence than its source.
    """
    if not execution_plan:
        return ToolDAG()

    nodes: list[ToolNode] = []
    for item in execution_plan:
        node = ToolNode.build(
            tool_name=str(item.get("tool_name") or ""),
            intent=str(item.get("intent") or ""),
            args=dict(item.get("args") or {}),
            priority=int(item.get("priority") or 100),
            sequence=int(item.get("sequence") or 0),
        )
        nodes.append(node)

    # Stable sort by ordering_key so DAG topology is reproducible.
    nodes.sort(key=lambda n: n.ordering_key())
    # Re-assign sequence numbers post-sort to ensure strict monotonicity.
    reassigned: list[ToolNode] = []
    for idx, n in enumerate(nodes):
        reassigned.append(
            ToolNode(
                node_id=f"node-{idx:04d}-{n.deterministic_id}",
                tool_name=n.tool_name,
                intent=n.intent,
                args=n.args,
                priority=n.priority,
                sequence=idx,
                deterministic_id=n.deterministic_id,
            )
        )

    dag = ToolDAG.from_nodes(reassigned)

    # Add sequential edges so downstream nodes can declare explicit ordering.
    for i in range(len(reassigned) - 1):
        dag.add_edge(
            reassigned[i].node_id,
            reassigned[i + 1].node_id,
            edge_type="sequential",
        )

    return dag


# ---------------------------------------------------------------------------
# Phase 2: ToolPlan IR → ToolGraph (compiled)
# ---------------------------------------------------------------------------


@dataclass
class ToolPlanIR:
    """Planning-layer representation of tool intent before compilation.

    Separates planning concerns (what to achieve) from execution concerns
    (how to execute it deterministically).
    """

    intent_summary: str
    tool_candidates: list[dict[str, Any]]
    constraints: dict[str, Any]
    optimization_mode: str  # "sequential" | "parallel" | "priority"

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_summary": self.intent_summary,
            "tool_candidates": list(self.tool_candidates),
            "constraints": dict(self.constraints),
            "optimization_mode": self.optimization_mode,
        }

    def plan_hash(self) -> str:
        return _stable_hash(self.to_dict())


class ToolPlanCompiler:
    """Compiles a ToolPlanIR into an executable ToolDAG.

    Determinism guarantee: same ToolPlanIR → same ToolDAG (by content hash).
    Different ToolPlanIR → structurally different ToolDAG (by node/edge set).
    No hidden nondeterminism in compilation.
    """

    _VALID_TOOLS = frozenset({"memory_lookup"})
    _VALID_INTENTS = frozenset({"goal_lookup", "session_memory_fetch"})

    def compile(self, plan: ToolPlanIR) -> ToolDAG:
        """Compile ToolPlanIR → ToolDAG.

        - Validates each candidate against allowed tools + intents.
        - Applies constraints (max_nodes).
        - Deduplicates by deterministic_id.
        - Sorts and sequences nodes.
        - Wires sequential edges.
        """
        max_nodes = int((plan.constraints or {}).get("max_nodes") or 16)
        candidates = list(plan.tool_candidates or [])

        accepted: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for cand in candidates[:max_nodes]:
            tool_name = str(cand.get("tool_name") or "").strip().lower()
            intent = str(cand.get("intent") or "").strip().lower()
            args = cand.get("args")
            if tool_name not in self._VALID_TOOLS:
                continue
            if intent not in self._VALID_INTENTS:
                continue
            if not isinstance(args, dict):
                continue
            det_id = _stable_hash(
                {"tool_name": tool_name, "args": dict(args)}
            )[:24]
            if det_id in seen_ids:
                continue
            seen_ids.add(det_id)
            accepted.append({
                "tool_name": tool_name,
                "intent": intent,
                "args": dict(args),
                "priority": int(cand.get("priority") or 100),
                "sequence": len(accepted),
                "deterministic_id": det_id,
            })

        accepted.sort(
            key=lambda c: (int(c.get("priority") or 100), str(c.get("intent") or ""), str(c.get("deterministic_id") or ""))
        )
        # Re-sequence after sort.
        for idx, c in enumerate(accepted):
            c["sequence"] = idx

        return build_dag_from_execution_plan(accepted)


__all__ = [
    "ToolDAG",
    "ToolEdge",
    "ToolNode",
    "ToolPlanCompiler",
    "ToolPlanIR",
    "build_dag_from_execution_plan",
]
