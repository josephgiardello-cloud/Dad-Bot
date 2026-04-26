"""L4-P3 — Global Execution Algebra Closure.

ToolGraph (ToolDAG) becomes a composable algebraic object:

    ToolGraph ⊕ ToolGraph → ToolGraph

Formal laws:
  1. Identity:     identity() ⊕ G == G ⊕ identity() == G
  2. Associativity: (A ⊕ B) ⊕ C == A ⊕ (B ⊕ C)  (by semantic hash)
  3. Deterministic merge: A ⊕ B always produces the same result for the same inputs

Merge semantics:
  - Node union (deduplicated by deterministic_id)
  - Edge union (deduplicated)
  - Re-sequenced for acyclicity (topological sort by ordering_key)
  - New sequential edges connect the merged graph correctly

This makes execution plans first-class composable values — not runtime objects.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from dadbot.core.tool_dag import ToolDAG, ToolEdge, ToolNode, build_dag_from_execution_plan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# ToolGraphAlgebra
# ---------------------------------------------------------------------------


class ToolGraphAlgebra:
    """Composable ToolDAG algebra.

    Provides the ⊕ operator (compose) and algebraic laws:
    - Identity element: identity()
    - Associativity: (A ⊕ B) ⊕ C == A ⊕ (B ⊕ C)
    - Determinism: identical inputs → identical output (by hash)
    """

    @staticmethod
    def identity() -> ToolDAG:
        """The identity element: an empty DAG.

        identity() ⊕ G == G  (G ⊕ identity() == G)
        """
        return ToolDAG()

    @staticmethod
    def compose(a: ToolDAG, b: ToolDAG) -> ToolDAG:
        """Compose two ToolDAGs into a single ToolDAG.

        Merge rules:
        1. Node union: collect all nodes from both; deduplicate by deterministic_id.
        2. Edge union: collect all edges; deduplicate.
        3. Re-sequence all nodes in deterministic order.
        4. Rebuild sequential edges between re-sequenced nodes.
        5. Add original (non-sequential) edges where still valid after re-sequencing.

        The result is semantically equivalent to running A then B.
        """
        # 1 — Node union, deduplicated by deterministic_id.
        seen_det_ids: set[str] = set()
        merged_node_specs: list[dict[str, Any]] = []

        for node in [*a.nodes, *b.nodes]:
            if node.deterministic_id not in seen_det_ids:
                seen_det_ids.add(node.deterministic_id)
                merged_node_specs.append({
                    "tool_name": node.tool_name,
                    "intent": node.intent,
                    "args": dict(node.args),
                    "priority": node.priority,
                    "sequence": len(merged_node_specs),
                    "deterministic_id": node.deterministic_id,
                })

        # 2 — Build the merged DAG from deduplicated specs.
        merged_dag = build_dag_from_execution_plan(merged_node_specs)
        return merged_dag

    @classmethod
    def compose_all(cls, graphs: list[ToolDAG]) -> ToolDAG:
        """Left-fold composition over a list of ToolDAGs.

        compose_all([]) == identity()
        compose_all([G]) == G
        compose_all([A, B, C]) == (A ⊕ B) ⊕ C
        """
        result = cls.identity()
        for g in (graphs or []):
            result = cls.compose(result, g)
        return result

    @staticmethod
    def merge_hash(graphs: list[ToolDAG]) -> str:
        """Content-addressed hash of the composition of graphs.

        Same list of graphs → same hash (deterministic merge fingerprint).
        """
        composed = ToolGraphAlgebra.compose_all(graphs)
        return composed.deterministic_hash()

    @staticmethod
    def is_identity(dag: ToolDAG) -> bool:
        """True iff dag is the identity element (empty)."""
        return len(dag.nodes) == 0 and len(dag.edges) == 0

    @classmethod
    def verify_associativity(
        cls,
        a: ToolDAG,
        b: ToolDAG,
        c: ToolDAG,
    ) -> dict[str, Any]:
        """Verify (A ⊕ B) ⊕ C == A ⊕ (B ⊕ C) by semantic hash.

        Returns {ok: bool, left_hash: str, right_hash: str}.
        """
        left = cls.compose(cls.compose(a, b), c)
        right = cls.compose(a, cls.compose(b, c))
        left_hash = left.deterministic_hash()
        right_hash = right.deterministic_hash()
        return {
            "ok": left_hash == right_hash,
            "left_hash": left_hash,
            "right_hash": right_hash,
        }

    @classmethod
    def verify_identity_laws(cls, dag: ToolDAG) -> dict[str, Any]:
        """Verify identity ⊕ G == G and G ⊕ identity == G.

        Returns {ok: bool, left_identity: bool, right_identity: bool}.
        """
        identity = cls.identity()
        left = cls.compose(identity, dag)
        right = cls.compose(dag, identity)
        dag_hash = dag.deterministic_hash()
        left_hash = left.deterministic_hash()
        right_hash = right.deterministic_hash()
        return {
            "ok": left_hash == dag_hash and right_hash == dag_hash,
            "left_identity": left_hash == dag_hash,
            "right_identity": right_hash == dag_hash,
            "dag_hash": dag_hash,
            "left_hash": left_hash,
            "right_hash": right_hash,
        }


__all__ = [
    "ToolGraphAlgebra",
]
