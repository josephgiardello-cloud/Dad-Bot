"""L3-P4 — Memory as Rankable State Space.

Formalizes the memory system as an ordered state vector space:

- MemoryStateVector: ordered state vector — each memory entry is a
  positional element in the space, with a stable hash for equality
- MemoryRankerOperator: transformation operator T: V → V (re-orders by score)
- GoalWeightingFunction: w(entry, goals) → float, weights entries by relevance
  to active goals

Design principle:
    memory = ordered state vector space
    ranker = transformation operator
    goal influence = weighting function

These are pure mathematical objects — no I/O, no DB access, no LLM calls.
"""

from __future__ import annotations

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
# MemoryStateVector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryStateVector:
    """Ordered state vector over memory entries.

    Formal properties:
    - Entries are positionally indexed (ordered).
    - Two vectors with identical entry content have the same space_hash.
    - Projection reduces dimensionality: project([0,2]) → sub-vector.
    - Dimension == len(entries).
    """

    entries: tuple[dict[str, Any], ...]
    space_hash: str

    @classmethod
    def from_memories(cls, memories: list[dict[str, Any]]) -> MemoryStateVector:
        """Construct a MemoryStateVector from a list of memory dicts."""
        frozen = tuple(dict(m) for m in (memories or []))
        return cls(
            entries=frozen,
            space_hash=_sha256({"entries": [dict(e) for e in frozen]}),
        )

    @property
    def dimension(self) -> int:
        return len(self.entries)

    def project(self, indices: list[int]) -> MemoryStateVector:
        """Return a sub-vector containing only the entries at given indices."""
        projected = tuple(self.entries[i] for i in sorted(set(indices)) if 0 <= i < len(self.entries))
        return MemoryStateVector(
            entries=projected,
            space_hash=_sha256({"entries": [dict(e) for e in projected]}),
        )

    def to_list(self) -> list[dict[str, Any]]:
        return [dict(e) for e in self.entries]

    def __len__(self) -> int:
        return len(self.entries)


# ---------------------------------------------------------------------------
# GoalWeightingFunction
# ---------------------------------------------------------------------------


class GoalWeightingFunction:
    """Weighting function w: (entry, active_goals) → float.

    For each memory entry, computes a goal-relevance weight by measuring
    token overlap between the entry's content and the active goals'
    descriptions.  This is a pure functional operator — no side effects.

    Weight formula:
        overlap_count / max(len(goal_tokens), 1) for each goal;
        take max across all goals;
        clamp to [0.0, 1.0].
    """

    def __init__(self, active_goals: list[dict[str, Any]]) -> None:
        self._goals = list(active_goals or [])
        self._goal_token_sets = [self._tokenize(str(g.get("description") or "")) for g in self._goals]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {t.lower() for t in text.split() if len(t) > 2}

    def weight(self, entry: dict[str, Any]) -> float:
        """Compute goal-relevance weight for a single memory entry."""
        if not self._goals:
            return 1.0  # identity weight when no goals

        content = " ".join(
            [
                str(entry.get("content") or ""),
                str(entry.get("text") or ""),
                str(entry.get("summary") or ""),
                str(entry.get("description") or ""),
            ],
        )
        entry_tokens = self._tokenize(content)
        if not entry_tokens:
            return 0.0

        best = 0.0
        for goal_tokens in self._goal_token_sets:
            if not goal_tokens:
                continue
            overlap = len(entry_tokens & goal_tokens)
            ratio = overlap / len(goal_tokens)
            best = max(best, ratio)

        return min(1.0, max(0.0, best))

    def apply(
        self,
        vector: MemoryStateVector,
    ) -> list[tuple[float, dict[str, Any]]]:
        """Apply weighting to all entries in the vector.

        Returns a list of (weight, entry) pairs, unsorted.
        """
        return [(self.weight(entry), dict(entry)) for entry in vector.entries]


# ---------------------------------------------------------------------------
# MemoryRankerOperator
# ---------------------------------------------------------------------------


class MemoryRankerOperator:
    """Transformation operator T: V → V.

    Applies a GoalWeightingFunction to a MemoryStateVector and returns a new
    MemoryStateVector whose entries are sorted by descending weight.

    This is a pure mathematical transformation: no side effects, no state.
    The output dimension equals the input dimension (no entries removed).
    """

    def apply(
        self,
        vector: MemoryStateVector,
        weighting: GoalWeightingFunction,
    ) -> MemoryStateVector:
        """Apply goal weighting and return a ranked MemoryStateVector.

        Entries are sorted by descending goal-relevance weight.
        Ties are broken by original index (stable sort).
        """
        weighted = weighting.apply(vector)
        # Sort descending by weight, stable for ties.
        sorted_entries = [
            entry
            for _, entry in sorted(
                enumerate(weighted),
                key=lambda x: (-x[1][0], x[0]),
            )
        ]
        # Rebuild: sorted_entries is list of (weight, entry) → just entries.
        ranked_entries = [entry for (weight, entry) in sorted(weighted, key=lambda we: -we[0])]
        return MemoryStateVector.from_memories(ranked_entries)

    def rank_with_scores(
        self,
        vector: MemoryStateVector,
        weighting: GoalWeightingFunction,
    ) -> list[tuple[float, dict[str, Any]]]:
        """Return (score, entry) pairs sorted by descending score."""
        weighted = weighting.apply(vector)
        return sorted(weighted, key=lambda we: -we[0])


__all__ = [
    "GoalWeightingFunction",
    "MemoryRankerOperator",
    "MemoryStateVector",
]
