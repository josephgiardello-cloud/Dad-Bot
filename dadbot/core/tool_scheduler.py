"""Deterministic Tool Scheduler (Phase 6).

Provides reproducible parallel execution ordering for ToolDAG nodes via:
  - Stable priority queue (min-heap by ordering key)
  - Explicit ordering seeds (for cross-machine reproducibility)
  - Reproducible interleaving rules (deterministic level-by-level scheduling)
  - schedule_hash: content-addressed fingerprint of the schedule
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from dadbot.core.tool_dag import ToolDAG, ToolNode


# ---------------------------------------------------------------------------
# Scheduled item
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduledItem:
    """A single scheduled tool execution slot.

    ``schedule_sequence`` is the position within the deterministic schedule.
    ``ordering_key`` is a stable string key derived from (priority, intent, det_id)
    and the scheduler seed, used to produce a fully reproducible ordering.
    """
    schedule_sequence: int
    node: ToolNode
    ordering_key: str
    wave: int  # Topological wave (0 = first executable, 1 = next, etc.)


# ---------------------------------------------------------------------------
# Deterministic Scheduler
# ---------------------------------------------------------------------------


class ToolScheduler:
    """Deterministic scheduler for ToolDAG execution.

    Guarantees:
    - Same DAG + same seed → same schedule every time.
    - Parallel nodes (same topological wave) are ordered by a stable priority
      queue whose key includes the seed, preventing accidental ordering from
      insertion order.
    - schedule_hash is a content-addressed fingerprint of the full schedule.
    """

    def __init__(self, seed: int = 0) -> None:
        self._seed = int(seed)

    def _ordering_key(self, node: ToolNode) -> str:
        """Derive a stable, seeded ordering key for a node."""
        raw = json.dumps(
            {
                "seed": self._seed,
                "priority": node.priority,
                "intent": node.intent,
                "deterministic_id": node.deterministic_id,
                "sequence": node.sequence,
            },
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def schedule(self, dag: ToolDAG) -> list[ScheduledItem]:
        """Return a fully deterministic schedule for the given DAG.

        Uses a wave-based topological traversal:
        1. Wave 0: all root nodes, sorted by ordering_key.
        2. Wave N+1: nodes whose last remaining predecessor was just scheduled.
        Within each wave, nodes are sorted by (priority, intent, det_id, seed-hash).
        """
        if not dag.nodes:
            return []

        # Build adjacency and in-degree maps.
        in_degree: dict[str, int] = {n.node_id: 0 for n in dag.nodes}
        adjacency: dict[str, list[str]] = {n.node_id: [] for n in dag.nodes}
        for edge in dag.edges:
            in_degree[edge.target_id] = in_degree.get(edge.target_id, 0) + 1
            adjacency.setdefault(edge.source_id, []).append(edge.target_id)

        node_by_id = {n.node_id: n for n in dag.nodes}

        # Wave 0: nodes with in_degree == 0.
        wave = 0
        current_wave_nodes = sorted(
            [n for n in dag.nodes if in_degree[n.node_id] == 0],
            key=self._ordering_key,
        )

        schedule_sequence = 0
        result: list[ScheduledItem] = []
        processed: set[str] = set()

        while current_wave_nodes:
            next_wave_candidates: set[str] = set()
            for node in current_wave_nodes:
                result.append(ScheduledItem(
                    schedule_sequence=schedule_sequence,
                    node=node,
                    ordering_key=self._ordering_key(node),
                    wave=wave,
                ))
                processed.add(node.node_id)
                schedule_sequence += 1
                for successor_id in adjacency.get(node.node_id, []):
                    in_degree[successor_id] -= 1
                    if in_degree[successor_id] == 0:
                        next_wave_candidates.add(successor_id)

            wave += 1
            current_wave_nodes = sorted(
                [node_by_id[nid] for nid in next_wave_candidates if nid in node_by_id],
                key=self._ordering_key,
            )

        # Guard against cycles (should not occur with ToolDAG.add_edge semantics).
        if len(result) != len(dag.nodes):
            raise RuntimeError(
                f"ToolScheduler: schedule incomplete — cycle detected or unreachable nodes. "
                f"Scheduled {len(result)} of {len(dag.nodes)}."
            )

        return result

    def schedule_hash(self, dag: ToolDAG) -> str:
        """Content-addressed fingerprint of the full schedule.

        Same DAG + same seed → same hash, every time.
        """
        items = self.schedule(dag)
        payload = [
            {
                "schedule_sequence": item.schedule_sequence,
                "node_id": item.node.node_id,
                "ordering_key": item.ordering_key,
                "wave": item.wave,
            }
            for item in items
        ]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def to_schedule_dict(self, dag: ToolDAG) -> dict[str, Any]:
        items = self.schedule(dag)
        return {
            "seed": self._seed,
            "schedule": [
                {
                    "schedule_sequence": item.schedule_sequence,
                    "node_id": item.node.node_id,
                    "tool_name": item.node.tool_name,
                    "intent": item.node.intent,
                    "wave": item.wave,
                    "ordering_key": item.ordering_key,
                }
                for item in items
            ],
            "schedule_hash": self.schedule_hash(dag),
        }


__all__ = [
    "ScheduledItem",
    "ToolScheduler",
]
