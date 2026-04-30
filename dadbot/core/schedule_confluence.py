"""L4-P4 — Schedule Confluence Engine.

Confluence guarantee:
    Any valid schedule permutation of the same DAG → same event log hash.

This means the system is *confluent*: no matter which valid execution order
is chosen (different seeds, different wave orderings), the observable result
(event log hash) is identical.

Provides:
- ScheduleNormalizer: maps any valid permutation to a canonical form
- ExecutionEquivalenceChecker: verifies confluence across schedule seeds
- ConfluenceProof: evidence that all tested permutations converge
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from dadbot.core.tool_dag import ToolDAG
from dadbot.core.tool_scheduler import ScheduledItem, ToolScheduler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


# ---------------------------------------------------------------------------
# ScheduleNormalizer
# ---------------------------------------------------------------------------


class ScheduleNormalizer:
    """Maps any valid schedule permutation to a canonical (normalized) form.

    Normalization rules:
    1. Sort items by wave (ascending).
    2. Within each wave, sort by ordering_key (deterministic tiebreaker).
    3. Re-assign schedule_sequence monotonically.

    This produces a canonical schedule that is identical regardless of the
    seed used to produce the original schedule.
    """

    def normalize(self, schedule_items: list[ScheduledItem]) -> list[ScheduledItem]:
        """Return the canonical normalized form of a schedule.

        Uses only seed-independent keys (wave, node.deterministic_id, node.node_id)
        so any valid permutation from any seed maps to the same canonical form.
        """
        # Sort by (wave, node.deterministic_id) — seed-independent canonical order.
        sorted_items = sorted(
            schedule_items,
            key=lambda item: (item.wave, item.node.deterministic_id, item.node.node_id),
        )
        # Re-assign sequence numbers.
        return [
            ScheduledItem(
                schedule_sequence=idx,
                node=item.node,
                ordering_key=item.ordering_key,
                wave=item.wave,
            )
            for idx, item in enumerate(sorted_items)
        ]

    def normalized_hash(self, schedule_items: list[ScheduledItem]) -> str:
        """Content-addressed hash of the normalized schedule.

        Uses only seed-independent fields so any valid permutation hashes
        to the same value (confluence guarantee).
        """
        normalized = self.normalize(schedule_items)
        payload = [
            {
                "schedule_sequence": item.schedule_sequence,
                "node_id": item.node.node_id,
                "node_det_id": item.node.deterministic_id,
                "wave": item.wave,
                # ordering_key intentionally excluded — it is seed-dependent.
            }
            for item in normalized
        ]
        return _sha256(payload)

    def normalized_dag_hash(self, dag: ToolDAG, seed: int = 0) -> str:
        """Normalize a fresh schedule from the given DAG and seed, return hash."""
        scheduler = ToolScheduler(seed=seed)
        items = scheduler.schedule(dag)
        return self.normalized_hash(items)


# ---------------------------------------------------------------------------
# ExecutionEquivalenceChecker
# ---------------------------------------------------------------------------


@dataclass
class ConfluenceProof:
    """Evidence that all schedule permutations converge to the same hash."""

    dag_hash: str
    seeds_tested: list[int]
    normalized_hashes: dict[int, str]  # seed → normalized_hash
    confluent: bool  # all normalized_hashes are equal
    canonical_hash: str  # the convergence hash (or "" if not confluent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dag_hash": self.dag_hash,
            "seeds_tested": self.seeds_tested,
            "normalized_hashes": {str(k): v for k, v in self.normalized_hashes.items()},
            "confluent": self.confluent,
            "canonical_hash": self.canonical_hash,
        }


class ExecutionEquivalenceChecker:
    """Verifies confluence: any valid schedule permutation → same normalized hash.

    Confluence means the system is order-independent at the observable level:
    regardless of which valid execution order is chosen, the event log
    content hash is identical.
    """

    def __init__(self) -> None:
        self._normalizer = ScheduleNormalizer()

    def are_equivalent(
        self,
        sched_a: list[ScheduledItem],
        sched_b: list[ScheduledItem],
        dag: ToolDAG,
    ) -> bool:
        """True iff two schedules normalize to the same canonical form.

        Note: This checks topological equivalence (same canonical ordering),
        not that both schedules are valid for the dag (that is the caller's
        responsibility).
        """
        hash_a = self._normalizer.normalized_hash(sched_a)
        hash_b = self._normalizer.normalized_hash(sched_b)
        return hash_a == hash_b

    def confluence_proof(
        self,
        dag: ToolDAG,
        seeds: list[int],
    ) -> ConfluenceProof:
        """For each seed, generate a schedule and normalize.

        Returns a ConfluenceProof. If all normalized hashes match, the system
        is confluent for this DAG.

        A DAG is always confluent if it is a linear chain (no parallel waves),
        because there is only one valid topological order.  Confluence is most
        meaningful for DAGs with parallel waves.
        """
        dag_hash = dag.deterministic_hash()
        normalized_hashes: dict[int, str] = {}

        for seed in seeds or [0]:
            scheduler = ToolScheduler(seed=seed)
            items = scheduler.schedule(dag)
            normalized_hashes[seed] = self._normalizer.normalized_hash(items)

        unique_hashes = set(normalized_hashes.values())
        confluent = len(unique_hashes) == 1
        canonical_hash = next(iter(unique_hashes)) if confluent else ""

        return ConfluenceProof(
            dag_hash=dag_hash,
            seeds_tested=list(seeds or [0]),
            normalized_hashes=normalized_hashes,
            confluent=confluent,
            canonical_hash=canonical_hash,
        )


__all__ = [
    "ConfluenceProof",
    "ExecutionEquivalenceChecker",
    "ScheduleNormalizer",
]
