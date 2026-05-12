from __future__ import annotations

from typing import Any


class TurnGraphTopology:
    """Owns graph shape (nodes and edges) without execution behavior."""

    def __init__(self, *, nodes: list[Any]) -> None:
        self.nodes = list(nodes)
        self.node_map: dict[str, Any] = {}
        self.edges: dict[str, str] = {}
        self.entry_node: str | None = None

        for node in self.nodes:
            node_name = str(getattr(node, "name", type(node).__name__) or type(node).__name__).strip().lower()
            if node_name and node_name not in self.node_map:
                self.node_map[node_name] = node

    def pipeline_items(self) -> list[tuple[str, Any]]:
        if self.node_map and self.entry_node is not None:
            items: list[tuple[str, Any]] = []
            node_name = self.entry_node
            while node_name is not None:
                items.append((node_name, self.node_map[node_name]))
                node_name = self.edges.get(node_name)
            return items
        return [
            (
                str(getattr(node, "name", type(node).__name__) or type(node).__name__),
                node,
            )
            for node in self.nodes
        ]

    def add_node(self, name: str, node: Any) -> None:
        if self.entry_node is None:
            self.entry_node = name
        self.node_map[name] = node

    def set_edge(self, source: str, target: str) -> None:
        self.edges[source] = target
