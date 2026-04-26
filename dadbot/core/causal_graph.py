"""Phase 3 — Causal Execution Graph Engine.

Provides full causal explainability for the dynamic routing + fallback layer
introduced in the external tool runtime.

3.1  CausalGraph
     Unified graph of execution events.  Nodes represent causal actors:
       - planner_decision  — a goal/plan choice made by the planner
       - tool_selection    — a tool was chosen for an intent/slot
       - tool_outcome      — the result of a tool execution attempt
       - fallback_activation — a fallback was triggered after a failure
       - retry_event       — a retry attempt was initiated

     Edges carry a CausalEdgeKind label (caused, triggered, selected_over,
     replaced_by, influenced).

3.2  CausalReconstructionAPI
     Answers «why did we choose tool B instead of A?» by tracing the
     causal path through the graph.

3.3  InfluenceTracer
     Traces the influence of a routing decision on final answer quality,
     expressed as a weighted influence score.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------


class CausalNodeKind(str, Enum):
    PLANNER_DECISION = "planner_decision"
    TOOL_SELECTION = "tool_selection"
    TOOL_OUTCOME = "tool_outcome"
    FALLBACK_ACTIVATION = "fallback_activation"
    RETRY_EVENT = "retry_event"


class CausalEdgeKind(str, Enum):
    CAUSED = "caused"
    TRIGGERED = "triggered"
    SELECTED_OVER = "selected_over"       # tool A was selected over tool B
    REPLACED_BY = "replaced_by"           # tool A was replaced by B (fallback)
    INFLUENCED = "influenced"             # routing decision influenced outcome quality


# ---------------------------------------------------------------------------
# Graph elements
# ---------------------------------------------------------------------------


@dataclass
class CausalNode:
    node_id: str
    kind: CausalNodeKind
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "label": self.label,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass
class CausalEdge:
    edge_id: str
    source_id: str
    target_id: str
    kind: CausalEdgeKind
    weight: float = 1.0
    reason: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "kind": self.kind.value,
            "weight": round(self.weight, 4),
            "reason": self.reason,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# 3.1 — CausalGraph
# ---------------------------------------------------------------------------


def _edge_id(source: str, target: str, kind: CausalEdgeKind) -> str:
    return hashlib.sha1(f"{source}|{kind.value}|{target}".encode()).hexdigest()[:16]


class CausalGraph:
    """Append-only causal execution graph.

    Thread-safe; all mutations acquire the internal lock.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._nodes: dict[str, CausalNode] = {}
        self._edges: dict[str, CausalEdge] = {}
        # adjacency: node_id → list[edge_id]
        self._out_edges: dict[str, list[str]] = {}
        self._in_edges: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        kind: CausalNodeKind,
        label: str,
        metadata: dict[str, Any] | None = None,
    ) -> CausalNode:
        with self._lock:
            if node_id not in self._nodes:
                node = CausalNode(
                    node_id=node_id,
                    kind=kind,
                    label=label,
                    metadata=dict(metadata or {}),
                )
                self._nodes[node_id] = node
            return self._nodes[node_id]

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        kind: CausalEdgeKind,
        weight: float = 1.0,
        reason: str = "",
    ) -> CausalEdge:
        eid = _edge_id(source_id, target_id, kind)
        with self._lock:
            if eid not in self._edges:
                edge = CausalEdge(
                    edge_id=eid,
                    source_id=source_id,
                    target_id=target_id,
                    kind=kind,
                    weight=round(float(weight), 4),
                    reason=str(reason or ""),
                )
                self._edges[eid] = edge
                self._out_edges.setdefault(source_id, []).append(eid)
                self._in_edges.setdefault(target_id, []).append(eid)
            return self._edges[eid]

    # ------------------------------------------------------------------
    # Convenience recording methods
    # ------------------------------------------------------------------

    def record_planner_decision(
        self, decision_id: str, intent: str, selected_plan: str, alternatives: list[str] | None = None
    ) -> CausalNode:
        return self.add_node(
            decision_id,
            CausalNodeKind.PLANNER_DECISION,
            label=f"plan:{selected_plan}",
            metadata={"intent": intent, "selected_plan": selected_plan, "alternatives": list(alternatives or [])},
        )

    def record_tool_selection(
        self,
        selection_id: str,
        tool_name: str,
        intent: str,
        score: float = 1.0,
        rejected_tools: list[str] | None = None,
        decision_id: str | None = None,
    ) -> CausalNode:
        node = self.add_node(
            selection_id,
            CausalNodeKind.TOOL_SELECTION,
            label=f"select:{tool_name}",
            metadata={"tool_name": tool_name, "intent": intent, "score": score, "rejected": list(rejected_tools or [])},
        )
        if decision_id:
            self.add_edge(decision_id, selection_id, CausalEdgeKind.CAUSED, weight=1.0)
        # Create SELECTED_OVER edges for rejected alternatives
        for rejected in (rejected_tools or []):
            rejected_node_id = f"rejected:{rejected}:{selection_id}"
            self.add_node(rejected_node_id, CausalNodeKind.TOOL_SELECTION, label=f"rejected:{rejected}", metadata={"tool_name": rejected, "rejected": True})
            self.add_edge(selection_id, rejected_node_id, CausalEdgeKind.SELECTED_OVER, weight=score, reason=f"{tool_name} scored higher than {rejected}")
        return node

    def record_tool_outcome(
        self,
        outcome_id: str,
        tool_name: str,
        status: str,
        latency_ms: float = 0.0,
        confidence: float = 1.0,
        selection_id: str | None = None,
    ) -> CausalNode:
        node = self.add_node(
            outcome_id,
            CausalNodeKind.TOOL_OUTCOME,
            label=f"outcome:{tool_name}:{status}",
            metadata={"tool_name": tool_name, "status": status, "latency_ms": latency_ms, "confidence": confidence},
        )
        if selection_id:
            self.add_edge(selection_id, outcome_id, CausalEdgeKind.CAUSED, weight=1.0)
        return node

    def record_fallback_activation(
        self,
        fallback_id: str,
        from_tool: str,
        to_tool: str,
        reason: str = "",
        outcome_id: str | None = None,
    ) -> CausalNode:
        node = self.add_node(
            fallback_id,
            CausalNodeKind.FALLBACK_ACTIVATION,
            label=f"fallback:{from_tool}→{to_tool}",
            metadata={"from_tool": from_tool, "to_tool": to_tool, "reason": reason},
        )
        if outcome_id:
            self.add_edge(outcome_id, fallback_id, CausalEdgeKind.TRIGGERED, weight=1.0, reason=reason)
        return node

    def record_retry_event(
        self,
        retry_id: str,
        tool_name: str,
        attempt: int,
        reason: str = "",
        outcome_id: str | None = None,
    ) -> CausalNode:
        node = self.add_node(
            retry_id,
            CausalNodeKind.RETRY_EVENT,
            label=f"retry:{tool_name}:attempt_{attempt}",
            metadata={"tool_name": tool_name, "attempt": attempt, "reason": reason},
        )
        if outcome_id:
            self.add_edge(outcome_id, retry_id, CausalEdgeKind.TRIGGERED, weight=1.0, reason=reason)
        return node

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> Optional[CausalNode]:
        with self._lock:
            return self._nodes.get(node_id)

    def get_nodes_by_kind(self, kind: CausalNodeKind) -> list[CausalNode]:
        with self._lock:
            return [n for n in self._nodes.values() if n.kind == kind]

    def out_edges(self, node_id: str) -> list[CausalEdge]:
        with self._lock:
            return [self._edges[eid] for eid in self._out_edges.get(node_id, [])]

    def in_edges(self, node_id: str) -> list[CausalEdge]:
        with self._lock:
            return [self._edges[eid] for eid in self._in_edges.get(node_id, [])]

    def node_count(self) -> int:
        with self._lock:
            return len(self._nodes)

    def edge_count(self) -> int:
        with self._lock:
            return len(self._edges)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "nodes": [n.to_dict() for n in self._nodes.values()],
                "edges": [e.to_dict() for e in self._edges.values()],
            }


# ---------------------------------------------------------------------------
# 3.2 — Causal Reconstruction API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionRationale:
    """Why tool B was chosen instead of A."""
    selected_tool: str
    rejected_tools: list[str]
    reason_chain: list[str]          # ordered explanation steps
    selection_score: float
    fallback_depth: int              # 0 = primary selection, >0 = fallback Nth


class CausalReconstructionAPI:
    """Answers causal questions about execution decisions.

    Given a ``CausalGraph``, this API can explain:
    - Why a specific tool was selected
    - What caused a fallback to activate
    - What alternatives were considered and why they were rejected
    """

    def __init__(self, graph: CausalGraph) -> None:
        self._graph = graph

    def why_tool_selected(self, selection_id: str) -> Optional[SelectionRationale]:
        """Return a human-readable rationale for why a tool selection was made."""
        node = self._graph.get_node(selection_id)
        if node is None or node.kind != CausalNodeKind.TOOL_SELECTION:
            return None

        meta = node.metadata
        selected_tool = str(meta.get("tool_name", "unknown"))
        rejected_tools = list(meta.get("rejected", []))
        score = float(meta.get("score", 0.0))

        reason_chain: list[str] = []

        # Walk incoming edges to find planner decision
        for edge in self._graph.in_edges(selection_id):
            src = self._graph.get_node(edge.source_id)
            if src and src.kind == CausalNodeKind.PLANNER_DECISION:
                reason_chain.append(
                    f"Planner chose plan '{src.metadata.get('selected_plan', '?')}' "
                    f"for intent '{src.metadata.get('intent', '?')}'."
                )

        if rejected_tools:
            reason_chain.append(
                f"'{selected_tool}' scored {score:.3f}, outscoring: {', '.join(rejected_tools)}."
            )
        else:
            reason_chain.append(f"'{selected_tool}' was the only candidate for this intent.")

        # Determine fallback depth by walking REPLACED_BY chains
        fallback_depth = self._fallback_depth(selection_id)

        return SelectionRationale(
            selected_tool=selected_tool,
            rejected_tools=rejected_tools,
            reason_chain=reason_chain,
            selection_score=score,
            fallback_depth=fallback_depth,
        )

    def why_fallback_activated(self, fallback_id: str) -> list[str]:
        """Return an ordered explanation of why a fallback was activated."""
        node = self._graph.get_node(fallback_id)
        if node is None or node.kind != CausalNodeKind.FALLBACK_ACTIVATION:
            return []

        meta = node.metadata
        steps: list[str] = [
            f"Tool '{meta.get('from_tool', '?')}' failed; "
            f"fallback to '{meta.get('to_tool', '?')}' activated.",
        ]
        if meta.get("reason"):
            steps.append(f"Failure reason: {meta['reason']}.")

        # Walk incoming edges to find the failed outcome
        for edge in self._graph.in_edges(fallback_id):
            src = self._graph.get_node(edge.source_id)
            if src and src.kind == CausalNodeKind.TOOL_OUTCOME:
                steps.append(
                    f"'{src.metadata.get('tool_name', '?')}' returned status "
                    f"'{src.metadata.get('status', '?')}' "
                    f"(latency={src.metadata.get('latency_ms', 0):.0f}ms)."
                )
        return steps

    def causal_path(self, from_id: str, to_id: str, max_depth: int = 10) -> list[str]:
        """Return the shortest causal path from one node to another as a list of node IDs."""
        # BFS
        from collections import deque
        visited: set[str] = set()
        queue: deque[list[str]] = deque([[from_id]])
        while queue:
            path = queue.popleft()
            current = path[-1]
            if current == to_id:
                return path
            if current in visited or len(path) > max_depth:
                continue
            visited.add(current)
            for edge in self._graph.out_edges(current):
                queue.append(path + [edge.target_id])
        return []

    def _fallback_depth(self, node_id: str) -> int:
        """Count how many REPLACED_BY hops precede this selection."""
        depth = 0
        visited: set[str] = set()
        current = node_id
        while current and current not in visited:
            visited.add(current)
            for edge in self._graph.in_edges(current):
                src_node = self._graph.get_node(edge.source_id)
                if src_node and src_node.kind == CausalNodeKind.FALLBACK_ACTIVATION:
                    depth += 1
                    current = edge.source_id
                    break
            else:
                break
        return depth


# ---------------------------------------------------------------------------
# 3.3 — Influence Tracer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InfluenceTrace:
    """How much a routing decision influenced the final answer quality."""
    source_node_id: str
    target_node_id: str
    influence_score: float          # 0.0 – 1.0; higher = more influential
    path: list[str]                 # node IDs on the influence path
    explanation: str


class InfluenceTracer:
    """Traces how routing decisions propagate influence to outcome quality nodes.

    Influence is computed as the product of edge weights along the causal
    path from a routing node to a terminal outcome node, normalized by path
    length (to penalize long, attenuated chains).
    """

    def __init__(self, graph: CausalGraph) -> None:
        self._graph = graph

    def trace(
        self,
        source_id: str,
        target_id: str,
        decay_per_hop: float = 0.85,
    ) -> Optional[InfluenceTrace]:
        """Compute the influence of source_id on target_id.

        Returns None if no causal path exists.
        """
        path = CausalReconstructionAPI(self._graph).causal_path(source_id, target_id)
        if not path:
            return None

        # Multiply edge weights along the path, apply per-hop decay
        score = 1.0
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            edge_weight = self._edge_weight(s, t)
            score *= edge_weight * decay_per_hop

        score = max(0.0, min(1.0, round(score, 4)))

        src_node = self._graph.get_node(source_id)
        tgt_node = self._graph.get_node(target_id)
        explanation = (
            f"Routing decision '{src_node.label if src_node else source_id}' "
            f"influenced outcome '{tgt_node.label if tgt_node else target_id}' "
            f"with score {score:.3f} over {len(path) - 1} hop(s)."
        )

        return InfluenceTrace(
            source_node_id=source_id,
            target_node_id=target_id,
            influence_score=score,
            path=path,
            explanation=explanation,
        )

    def top_influences(
        self,
        target_id: str,
        source_kind: CausalNodeKind = CausalNodeKind.TOOL_SELECTION,
        top_k: int = 5,
        decay_per_hop: float = 0.85,
    ) -> list[InfluenceTrace]:
        """Find the top-k most influential source nodes for a given target."""
        candidates = self._graph.get_nodes_by_kind(source_kind)
        traces = []
        for node in candidates:
            if node.node_id == target_id:
                continue
            tr = self.trace(node.node_id, target_id, decay_per_hop=decay_per_hop)
            if tr is not None:
                traces.append(tr)
        return sorted(traces, key=lambda t: t.influence_score, reverse=True)[:max(1, top_k)]

    def _edge_weight(self, source_id: str, target_id: str) -> float:
        for edge in self._graph.out_edges(source_id):
            if edge.target_id == target_id:
                return max(0.0, float(edge.weight))
        return 0.0
