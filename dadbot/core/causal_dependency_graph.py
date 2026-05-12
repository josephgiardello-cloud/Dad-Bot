"""Cross-Tool Causal Dependency Graph.

Gap 3 of the causal loop: execution was linear (tool result → memory entry)
but lacked any model of:

  - Tool A producing output that influences Tool B
  - Memory entries feeding back into which tool gets chosen
  - Chained causality across multiple hops (A → B → C)

This module introduces a lightweight, directed acyclic graph (DAG) built on
top of CausalMemoryEntry objects.  No external graph library is required.

Core types
----------
CausalNode      — Entry + graph metadata (in/out edge sets)
CausalEdge      — Directed dependency: source → target with a reason label
CausalDepGraph  — Mutable graph: add_node / add_edge / query operations

Key queries
-----------
ancestors_of(key)        — All entries that causally influenced a given entry
descendants_of(key)      — All entries causally triggered by a given entry
causal_chain(key)        — Ordered list from root cause to given entry
detect_cycles()          — Returns any cycle paths (safety check; real DAGs have none)

Usage
-----
    graph = CausalDepGraph()
    graph.add_node(entry_a)
    graph.add_node(entry_b)
    graph.add_edge(entry_a.causal_key, entry_b.causal_key,
                   reason="tool_a output fed tool_b input")

    chain = graph.causal_chain(entry_b.causal_key)
    # [entry_a, entry_b]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

from dadbot.core.tool_memory_causal_contract import CausalMemoryEntry


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CausalEdge:
    """Directed dependency between two CausalMemoryEntry nodes.

    Attributes
    ----------
    source_key:
        causal_key of the upstream (causing) entry.
    target_key:
        causal_key of the downstream (caused) entry.
    reason:
        Human-readable label for this dependency.
    edge_type:
        Categorical tag for the kind of causality being modelled.
    """

    source_key: str
    target_key: str
    reason: str = ""
    edge_type: str = "data_flow"   # e.g. data_flow | policy_override | fallback_trigger


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


@dataclass
class CausalNode:
    """A memory entry placed inside the causal graph.

    Attributes
    ----------
    entry:
        The wrapped CausalMemoryEntry.
    in_keys:
        causal_keys of all nodes that have an edge pointing *to* this node.
    out_keys:
        causal_keys of all nodes this node has an edge pointing *to*.
    """

    entry: CausalMemoryEntry
    in_keys: set[str] = field(default_factory=set)
    out_keys: set[str] = field(default_factory=set)

    @property
    def key(self) -> str:
        return self.entry.causal_key

    def is_root(self) -> bool:
        """True if no entries caused this one (it's a causal origin)."""
        return len(self.in_keys) == 0

    def is_leaf(self) -> bool:
        """True if this entry causes nothing else in the current graph."""
        return len(self.out_keys) == 0


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


class CausalDepGraph:
    """Mutable, directed causal dependency graph over CausalMemoryEntry objects.

    All operations are O(n) or O(n + e) in the number of nodes (n) and
    edges (e).  For practical tool-execution graphs these are small (< 10^3).

    Raises
    ------
    KeyError
        If add_edge references a causal_key that was not registered via add_node.
    ValueError
        If an edge would create a self-loop (source_key == target_key).
    """

    def __init__(self) -> None:
        self._nodes: dict[str, CausalNode] = {}  # causal_key → node
        self._edges: list[CausalEdge] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self, entry: CausalMemoryEntry) -> CausalNode:
        """Register an entry.  If the key already exists the entry is replaced."""
        node = CausalNode(entry=entry)
        existing = self._nodes.get(entry.causal_key)
        if existing is not None:
            # Preserve edge connectivity when updating an existing node
            node.in_keys = existing.in_keys
            node.out_keys = existing.out_keys
        self._nodes[entry.causal_key] = node
        return node

    def add_edge(
        self,
        source_key: str,
        target_key: str,
        *,
        reason: str = "",
        edge_type: str = "data_flow",
    ) -> CausalEdge:
        """Add a directed dependency: source caused / influenced target.

        Raises
        ------
        ValueError
            If source_key == target_key.
        KeyError
            If either key is not registered in the graph.
        """
        if source_key == target_key:
            raise ValueError(f"Self-loop not allowed: {source_key!r}")
        if source_key not in self._nodes:
            raise KeyError(f"Source node not found: {source_key!r}")
        if target_key not in self._nodes:
            raise KeyError(f"Target node not found: {target_key!r}")

        edge = CausalEdge(
            source_key=source_key,
            target_key=target_key,
            reason=reason,
            edge_type=edge_type,
        )
        self._edges.append(edge)
        self._nodes[source_key].out_keys.add(target_key)
        self._nodes[target_key].in_keys.add(source_key)
        return edge

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    def node(self, key: str) -> CausalNode | None:
        return self._nodes.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    @property
    def nodes(self) -> list[CausalNode]:
        return list(self._nodes.values())

    @property
    def edges(self) -> list[CausalEdge]:
        return list(self._edges)

    def roots(self) -> list[CausalNode]:
        """Return all nodes with no incoming edges (causal origins)."""
        return [n for n in self._nodes.values() if n.is_root()]

    def leaves(self) -> list[CausalNode]:
        """Return all nodes with no outgoing edges (causal endpoints)."""
        return [n for n in self._nodes.values() if n.is_leaf()]

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def ancestors_of(self, key: str) -> list[CausalMemoryEntry]:
        """Return all entries that transitively caused the given entry.

        The result is in breadth-first order from root toward `key`.
        The entry itself is not included.
        """
        if key not in self._nodes:
            return []
        visited: set[str] = set()
        queue = list(self._nodes[key].in_keys)
        order: list[str] = []
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            order.append(current)
            node = self._nodes.get(current)
            if node:
                queue.extend(k for k in node.in_keys if k not in visited)
        # Reverse so ancestors come before descendants in the returned list
        order.reverse()
        return [self._nodes[k].entry for k in order if k in self._nodes]

    def descendants_of(self, key: str) -> list[CausalMemoryEntry]:
        """Return all entries transitively caused by the given entry.

        The result is in breadth-first order from `key` toward leaves.
        The entry itself is not included.
        """
        if key not in self._nodes:
            return []
        visited: set[str] = set()
        queue = list(self._nodes[key].out_keys)
        order: list[str] = []
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            order.append(current)
            node = self._nodes.get(current)
            if node:
                queue.extend(k for k in node.out_keys if k not in visited)
        return [self._nodes[k].entry for k in order if k in self._nodes]

    def causal_chain(self, key: str) -> list[CausalMemoryEntry]:
        """Return the complete chain from root cause to `key` (inclusive).

        When multiple root paths exist, all ancestors are included, sorted by
        timestamp_ms ascending (oldest root first).
        """
        if key not in self._nodes:
            return []
        ancestors = self.ancestors_of(key)
        current = self._nodes[key].entry
        # Sort ancestors by timestamp so root comes first
        ancestors_sorted = sorted(ancestors, key=lambda e: e.timestamp_ms)
        return ancestors_sorted + [current]

    def detect_cycles(self) -> list[list[str]]:
        """Return any cycle paths found in the graph.

        Returns an empty list if the graph is a proper DAG.
        Uses DFS with a recursion stack.
        """
        visited: set[str] = set()
        rec_stack: set[str] = set()
        cycles: list[list[str]] = []

        def _dfs(key: str, path: list[str]) -> None:
            visited.add(key)
            rec_stack.add(key)
            node = self._nodes.get(key)
            if node is None:
                rec_stack.discard(key)
                return
            for child_key in node.out_keys:
                if child_key not in visited:
                    _dfs(child_key, path + [child_key])
                elif child_key in rec_stack:
                    cycle_start = path.index(child_key) if child_key in path else 0
                    cycles.append(path[cycle_start:] + [child_key])
            rec_stack.discard(key)

        for start_key in list(self._nodes.keys()):
            if start_key not in visited:
                _dfs(start_key, [start_key])

        return cycles

    def subgraph_for_tool(self, tool_name: str) -> "CausalDepGraph":
        """Return a new CausalDepGraph containing only nodes whose tool_name matches."""
        sub = CausalDepGraph()
        for node in self._nodes.values():
            if node.entry.tool_name == tool_name:
                sub.add_node(node.entry)
        # Add edges where both endpoints are in the subgraph
        for edge in self._edges:
            if edge.source_key in sub._nodes and edge.target_key in sub._nodes:
                try:
                    sub.add_edge(
                        edge.source_key,
                        edge.target_key,
                        reason=edge.reason,
                        edge_type=edge.edge_type,
                    )
                except (KeyError, ValueError):
                    pass
        return sub

    def __iter__(self) -> Iterator[CausalNode]:
        return iter(self._nodes.values())


# ---------------------------------------------------------------------------
# Convenience: auto-link from a temporal sequence
# ---------------------------------------------------------------------------


def build_temporal_graph(
    entries: Sequence[CausalMemoryEntry],
    *,
    same_tool_only: bool = False,
) -> CausalDepGraph:
    """Build a graph by linking consecutive entries in timestamp order.

    Each entry is assumed to be causally downstream of the previous one.
    If ``same_tool_only`` is True, edges are only created between entries
    from the same tool (modelling intra-tool dependency chains).

    This is a convenient heuristic for when explicit edge annotations are not
    available.  For precise graph construction, use add_edge() directly.
    """
    graph = CausalDepGraph()
    sorted_entries = sorted(entries, key=lambda e: e.timestamp_ms)
    for entry in sorted_entries:
        graph.add_node(entry)

    for i in range(1, len(sorted_entries)):
        prev = sorted_entries[i - 1]
        curr = sorted_entries[i]
        if same_tool_only and prev.tool_name != curr.tool_name:
            continue
        try:
            graph.add_edge(
                prev.causal_key,
                curr.causal_key,
                reason="temporal_sequence",
                edge_type="temporal",
            )
        except (KeyError, ValueError):
            pass

    return graph


__all__ = [
    "CausalEdge",
    "CausalNode",
    "CausalDepGraph",
    "build_temporal_graph",
]
