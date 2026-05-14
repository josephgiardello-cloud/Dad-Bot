"""Memory graph node/edge accumulation helpers.

These are thin re-exports of the canonical implementations in
:class:`dadbot.managers.long_term.LongTermSignalsManager`.  Import directly
from here to avoid going through the DadBot facade::

    from dadbot.utils.graph import accumulate_memory_graph_node, build_memory_graph_nodes
"""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module


@lru_cache(maxsize=1)
def _lts_cls():
    return import_module("dadbot.managers.long_term").LongTermSignalsManager


def accumulate_memory_graph_node(
    node_weights,
    node_types,
    label,
    node_type,
    weight: int = 1,
):
    """Add or increment a node in the in-progress graph accumulation dicts."""
    return _lts_cls().accumulate_memory_graph_node(
        node_weights,
        node_types,
        label,
        node_type,
        weight=weight,
    )


def accumulate_memory_graph_edge(edge_weights, left, right, weight: int = 1):
    """Add or increment an edge in the in-progress graph accumulation dict."""
    return _lts_cls().accumulate_memory_graph_edge(
        edge_weights,
        left,
        right,
        weight=weight,
    )


def build_memory_graph_nodes(node_weights, node_types) -> list:
    """Convert accumulation dicts into a list of node objects."""
    return _lts_cls().build_memory_graph_nodes(node_weights, node_types)


def build_memory_graph_edges(edge_weights) -> list:
    """Convert an edge accumulation dict into a list of edge objects."""
    return _lts_cls().build_memory_graph_edges(edge_weights)


def persona_announcement(trait: str, reason: str) -> str:
    """Format a persona evolution announcement for display."""
    return _lts_cls().persona_announcement(trait, reason)


def pattern_identity(pattern) -> str:
    """Return a stable string identity for a life-pattern entry."""
    return _lts_cls().pattern_identity(pattern)


__all__ = [
    "accumulate_memory_graph_edge",
    "accumulate_memory_graph_node",
    "build_memory_graph_edges",
    "build_memory_graph_nodes",
    "pattern_identity",
    "persona_announcement",
]
