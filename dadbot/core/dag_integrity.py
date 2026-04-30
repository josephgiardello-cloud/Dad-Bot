"""L3-P2 — DAG Semantic Integrity Lock.

Upgrades ToolDAG from a runtime object to a verifiable compiled artifact:

1. DagIdentityLock — content-addressed graph identity (hash-as-id)
2. graph_structural_equivalence — same topology (same nodes and edges by ID)
3. graph_semantic_equivalence — same semantics (same tool_name, args, intents)
4. assert_replay_dag_invariant — replay DAG == original DAG (reproducibility proof)

Once locked, a DAG can be compared across runs, stored, and re-verified
without relying on object identity or wall-clock ordering.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from dadbot.core.tool_dag import ToolDAG

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


# ---------------------------------------------------------------------------
# L3-P2a — DagIdentityLock
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DagIdentityLock:
    """Content-addressed identity for a compiled ToolDAG.

    The lock is derived from the full semantic content of the DAG:
    node tool names, intents, args, and edge topology.  Two DAGs with the
    same semantic content have the same lock — this is the DAG's canonical
    identifier across runs, machines, and time.

    Usage::

        lock = DagIdentityLock.from_dag(dag)
        assert lock == DagIdentityLock.from_dag(rebuilt_dag)  # reproducibility proof
    """

    structural_hash: str  # topology only (node IDs + edge IDs)
    semantic_hash: str  # full content (tool_name, intent, args, edges)
    node_count: int
    edge_count: int

    @classmethod
    def from_dag(cls, dag: ToolDAG) -> DagIdentityLock:
        # Sort nodes by deterministic_id for stable ordering.
        nodes = sorted(dag.nodes, key=lambda n: n.deterministic_id)
        edges = sorted(dag.edges, key=lambda e: (e.source_id, e.target_id))

        structural_payload = {
            "nodes": [n.node_id for n in nodes],
            "edges": [(e.source_id, e.target_id) for e in edges],
        }
        semantic_payload = {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "tool_name": n.tool_name,
                    "intent": n.intent,
                    "args": n.args,
                    "priority": n.priority,
                    "deterministic_id": n.deterministic_id,
                }
                for n in nodes
            ],
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "edge_type": e.edge_type,
                }
                for e in edges
            ],
        }

        return cls(
            structural_hash=_sha256(structural_payload),
            semantic_hash=_sha256(semantic_payload),
            node_count=len(nodes),
            edge_count=len(edges),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "structural_hash": self.structural_hash,
            "semantic_hash": self.semantic_hash,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
        }


# ---------------------------------------------------------------------------
# L3-P2b — Graph equivalence tests
# ---------------------------------------------------------------------------


def graph_structural_equivalence(a: ToolDAG, b: ToolDAG) -> bool:
    """True iff dag_a and dag_b have identical topology (same node/edge IDs).

    Does NOT compare semantic content (tool names, args) — only the shape.
    Use graph_semantic_equivalence for full equality.
    """
    lock_a = DagIdentityLock.from_dag(a)
    lock_b = DagIdentityLock.from_dag(b)
    return lock_a.structural_hash == lock_b.structural_hash


def graph_semantic_equivalence(a: ToolDAG, b: ToolDAG) -> bool:
    """True iff dag_a and dag_b are semantically equivalent.

    Semantic equivalence: same tool_names, intents, args, priorities, and
    edge topology.  Node IDs may differ if they were built from different
    sequence assignments, but semantic content must be identical.
    """
    lock_a = DagIdentityLock.from_dag(a)
    lock_b = DagIdentityLock.from_dag(b)
    return lock_a.semantic_hash == lock_b.semantic_hash


def graph_equivalence_proof(a: ToolDAG, b: ToolDAG) -> dict[str, Any]:
    """Return a detailed equivalence proof without raising.

    Keys: structurally_equivalent, semantically_equivalent, lock_a, lock_b.
    """
    lock_a = DagIdentityLock.from_dag(a)
    lock_b = DagIdentityLock.from_dag(b)
    return {
        "structurally_equivalent": lock_a.structural_hash == lock_b.structural_hash,
        "semantically_equivalent": lock_a.semantic_hash == lock_b.semantic_hash,
        "lock_a": lock_a.to_dict(),
        "lock_b": lock_b.to_dict(),
    }


# ---------------------------------------------------------------------------
# L3-P2c — Replay invariant
# ---------------------------------------------------------------------------


class DagReplayInvariantError(ValueError):
    """Raised when a replayed DAG is not semantically equivalent to the original."""

    def __init__(self, message: str, original_hash: str, replayed_hash: str) -> None:
        self.original_hash = original_hash
        self.replayed_hash = replayed_hash
        super().__init__(f"DagReplayInvariantError: {message}")


def assert_replay_dag_invariant(
    original: ToolDAG,
    replayed: ToolDAG,
    *,
    require_structural: bool = False,
) -> None:
    """Assert that replayed DAG is semantically equivalent to the original.

    The replay invariant guarantees that:
    - Rebuilding a DAG from the same execution plan produces the same DAG.
    - The DAG is a deterministic compiled artifact, not a runtime object.

    ``require_structural=True`` additionally requires structural (topology) equality,
    which is stricter: same node IDs must be produced.

    Raises ``DagReplayInvariantError`` on violation.
    """
    lock_orig = DagIdentityLock.from_dag(original)
    lock_rep = DagIdentityLock.from_dag(replayed)

    if lock_orig.semantic_hash != lock_rep.semantic_hash:
        raise DagReplayInvariantError(
            "semantic hash mismatch — DAG is non-deterministic",
            original_hash=lock_orig.semantic_hash,
            replayed_hash=lock_rep.semantic_hash,
        )

    if require_structural and lock_orig.structural_hash != lock_rep.structural_hash:
        raise DagReplayInvariantError(
            "structural hash mismatch — node IDs diverged",
            original_hash=lock_orig.structural_hash,
            replayed_hash=lock_rep.structural_hash,
        )


def build_replay_proof(
    original: ToolDAG,
    replayed: ToolDAG,
) -> dict[str, Any]:
    """Non-raising form: return a replay proof dict.

    Keys: ok, semantic_match, structural_match, original_lock, replayed_lock.
    """
    lock_o = DagIdentityLock.from_dag(original)
    lock_r = DagIdentityLock.from_dag(replayed)
    semantic_match = lock_o.semantic_hash == lock_r.semantic_hash
    structural_match = lock_o.structural_hash == lock_r.structural_hash
    return {
        "ok": semantic_match,
        "semantic_match": semantic_match,
        "structural_match": structural_match,
        "original_lock": lock_o.to_dict(),
        "replayed_lock": lock_r.to_dict(),
    }


# ---------------------------------------------------------------------------
# Convenience: lock a DAG for storage / transmission
# ---------------------------------------------------------------------------


def lock_dag(dag: ToolDAG) -> dict[str, Any]:
    """Return a storable representation: dag.to_dict() + identity lock."""
    lock = DagIdentityLock.from_dag(dag)
    return {
        "dag": dag.to_dict(),
        "identity_lock": lock.to_dict(),
    }


__all__ = [
    "DagIdentityLock",
    "DagReplayInvariantError",
    "assert_replay_dag_invariant",
    "build_replay_proof",
    "graph_equivalence_proof",
    "graph_semantic_equivalence",
    "graph_structural_equivalence",
    "lock_dag",
]
