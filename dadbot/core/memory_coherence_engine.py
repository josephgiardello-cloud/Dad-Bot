"""Global Memory Coherence Engine.

Gap 2 of the causal loop: memory entries are structured but not yet
self-consistent over time.  This module adds:

  - Decay weighting (time-based weight reduction; older entries matter less)
  - Contradiction detection (same tool, same failure class, conflicting status)
  - Conflict resolution (deterministic strategy for choosing one winner)
  - Priority weighting (severity, policy action, recency all influence final weight)

The engine never mutates stored entries — it returns *views* (CoherentMemoryView)
that carry an assigned weight and an optional ConflictRecord when a contradiction
was resolved.

Usage
-----
    engine = MemoryCoherenceEngine(
        decay=DecayRule(half_life_seconds=3600 * 24),   # 1-day half-life
        conflict_strategy=ConflictResolutionStrategy.HIGHEST_SEVERITY,
    )
    view = engine.process(entries)
    # view.weighted — list of (entry, weight) sorted by descending weight
    # view.conflicts — list of ConflictRecord detailing each resolved clash
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from dadbot.core.tool_memory_causal_contract import CausalMemoryEntry


# ---------------------------------------------------------------------------
# Decay rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecayRule:
    """Controls exponential time-based decay of memory entry weight.

    weight_at_age = base_weight * exp(-ln(2) * age_seconds / half_life_seconds)

    Attributes
    ----------
    half_life_seconds:
        Time (in seconds) after which an entry's weight is halved.
        0 disables decay — all entries get weight 1.0 from age alone.
    min_weight:
        Entries with computed weight below this are considered "forgotten"
        and excluded from the coherent view.
    """

    half_life_seconds: float = 0.0    # 0 = no decay
    min_weight: float = 0.01

    def __post_init__(self) -> None:
        if self.half_life_seconds < 0:
            raise ValueError("half_life_seconds must be >= 0")
        if not (0.0 < self.min_weight <= 1.0):
            raise ValueError("min_weight must be in (0, 1]")

    def weight_for(self, entry: CausalMemoryEntry, *, now_ms: int | None = None) -> float:
        """Compute time-decay weight for entry.  Returns value in (0, 1]."""
        if self.half_life_seconds == 0.0:
            return 1.0
        now = (now_ms if now_ms is not None else int(time.time() * 1000))
        age_seconds = max(0.0, (now - entry.timestamp_ms) / 1000.0)
        decay = math.exp(-math.log(2) * age_seconds / self.half_life_seconds)
        return max(self.min_weight, min(1.0, decay))


# ---------------------------------------------------------------------------
# Conflict resolution strategy
# ---------------------------------------------------------------------------


class ConflictResolutionStrategy(Enum):
    """How to resolve two entries with conflicting status for the same tool."""

    LATEST_WINS = "latest_wins"
    """Discard the older entry; keep the most-recently written one."""

    HIGHEST_SEVERITY = "highest_severity"
    """Keep the entry with the higher failure severity (CRITICAL > HIGH > MEDIUM > LOW).
    Ties are broken by recency."""

    MOST_RETRIES = "most_retries"
    """Keep the entry with the most execution attempts.
    Ties are broken by recency."""

    POLICY_ESCALATE_WINS = "policy_escalate_wins"
    """Prefer entries with policy_action=ESCALATE or ABORT over others.
    Ties are broken by highest severity, then recency."""


# ---------------------------------------------------------------------------
# Internal severity rank (shared with injector but kept local to avoid coupling)
# ---------------------------------------------------------------------------

_SEVERITY_RANK: dict[str | None, int] = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
    None: 0,
    "": 0,
    "unknown": 0,
}

_POLICY_URGENT: frozenset[str] = frozenset({"escalate", "abort"})


def _severity_rank(entry: CausalMemoryEntry) -> int:
    return _SEVERITY_RANK.get((entry.failure_severity or "").lower(), 0)


def _is_urgent_policy(entry: CausalMemoryEntry) -> bool:
    return (entry.policy_action or "").lower() in _POLICY_URGENT


def _resolve_conflict(
    a: CausalMemoryEntry,
    b: CausalMemoryEntry,
    strategy: ConflictResolutionStrategy,
) -> tuple[CausalMemoryEntry, CausalMemoryEntry]:
    """Return (winner, loser) given strategy."""
    if strategy == ConflictResolutionStrategy.LATEST_WINS:
        if a.timestamp_ms >= b.timestamp_ms:
            return a, b
        return b, a

    if strategy == ConflictResolutionStrategy.HIGHEST_SEVERITY:
        ra, rb = _severity_rank(a), _severity_rank(b)
        if ra > rb:
            return a, b
        if rb > ra:
            return b, a
        # tie-break by recency
        return (a, b) if a.timestamp_ms >= b.timestamp_ms else (b, a)

    if strategy == ConflictResolutionStrategy.MOST_RETRIES:
        if a.attempt > b.attempt:
            return a, b
        if b.attempt > a.attempt:
            return b, a
        return (a, b) if a.timestamp_ms >= b.timestamp_ms else (b, a)

    if strategy == ConflictResolutionStrategy.POLICY_ESCALATE_WINS:
        ua, ub = _is_urgent_policy(a), _is_urgent_policy(b)
        if ua and not ub:
            return a, b
        if ub and not ua:
            return b, a
        # Both urgent or neither — fall back to severity then recency
        ra, rb = _severity_rank(a), _severity_rank(b)
        if ra > rb:
            return a, b
        if rb > ra:
            return b, a
        return (a, b) if a.timestamp_ms >= b.timestamp_ms else (b, a)

    # Default: latest wins
    return (a, b) if a.timestamp_ms >= b.timestamp_ms else (b, a)


# ---------------------------------------------------------------------------
# Conflict record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConflictRecord:
    """Documents a detected contradiction and its resolution.

    Attributes
    ----------
    conflict_key:
        The grouping key on which entries conflicted (e.g. "tool_name:failure_class").
    winner:
        The entry that was retained.
    loser:
        The entry that was discarded.
    strategy:
        The resolution strategy that selected the winner.
    reason:
        Human-readable explanation.
    """

    conflict_key: str
    winner: CausalMemoryEntry
    loser: CausalMemoryEntry
    strategy: ConflictResolutionStrategy
    reason: str


# ---------------------------------------------------------------------------
# Priority weighting
# ---------------------------------------------------------------------------


def _priority_weight(entry: CausalMemoryEntry) -> float:
    """Compute a priority multiplier in [0.1, 2.0] based on severity + policy action.

    This is layered on top of the decay weight so important failures stay visible
    even as they age.
    """
    sev = _SEVERITY_RANK.get((entry.failure_severity or "").lower(), 0)
    base = 1.0 + sev * 0.25   # up to +1.0 for critical

    if _is_urgent_policy(entry):
        base = min(2.0, base + 0.5)

    # Successful entries get a slight boost (useful signal)
    if entry.status.lower() == "ok":
        base = max(base, 1.0)

    return min(2.0, max(0.1, base))


# ---------------------------------------------------------------------------
# Coherent view
# ---------------------------------------------------------------------------


@dataclass
class CoherentMemoryView:
    """Output of MemoryCoherenceEngine.process().

    Attributes
    ----------
    weighted:
        List of (entry, final_weight) sorted by descending weight.
        Entries below DecayRule.min_weight are excluded.
    conflicts:
        List of ConflictRecord documenting each resolved contradiction.
    forgotten:
        Entries excluded because their computed weight fell below min_weight.
    """

    weighted: list[tuple[CausalMemoryEntry, float]] = field(default_factory=list)
    conflicts: list[ConflictRecord] = field(default_factory=list)
    forgotten: list[CausalMemoryEntry] = field(default_factory=list)

    @property
    def entries(self) -> list[CausalMemoryEntry]:
        """Just the surviving entries, sorted by descending weight."""
        return [e for e, _w in self.weighted]

    def top(self, n: int) -> list[CausalMemoryEntry]:
        """Return the top-n highest-weight entries."""
        return self.entries[:max(0, n)]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _conflict_key(entry: CausalMemoryEntry) -> str:
    """Stable grouping key for detecting contradictions.

    Two entries conflict when they share the same tool_name + failure_class
    but have different statuses (or one succeeded and one failed).
    """
    fc = entry.failure_class or "ok"
    return f"{entry.tool_name}:{fc}"


class MemoryCoherenceEngine:
    """Applies decay, contradiction detection, and priority weighting to memory entries.

    Parameters
    ----------
    decay:
        Time-decay rule.  Defaults to no decay (half_life_seconds=0).
    conflict_strategy:
        How to resolve contradicting entries.
    detect_contradictions:
        If False, skip conflict detection (useful for pure decay/priority use).
    """

    def __init__(
        self,
        *,
        decay: DecayRule | None = None,
        conflict_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.LATEST_WINS,
        detect_contradictions: bool = True,
    ) -> None:
        self._decay = decay or DecayRule()
        self._conflict_strategy = conflict_strategy
        self._detect_contradictions = detect_contradictions

    @property
    def decay(self) -> DecayRule:
        return self._decay

    @property
    def conflict_strategy(self) -> ConflictResolutionStrategy:
        return self._conflict_strategy

    def process(
        self,
        entries: Sequence[CausalMemoryEntry],
        *,
        now_ms: int | None = None,
    ) -> CoherentMemoryView:
        """Apply full coherence pipeline to a list of entries.

        Returns
        -------
        CoherentMemoryView
            Surviving entries with weights, plus conflict and forgotten lists.
        """
        if not entries:
            return CoherentMemoryView()

        now = now_ms if now_ms is not None else int(time.time() * 1000)

        # Step 1: Compute decay weights and filter forgotten
        decay_weighted: list[tuple[CausalMemoryEntry, float]] = []
        forgotten: list[CausalMemoryEntry] = []
        for entry in entries:
            w = self._decay.weight_for(entry, now_ms=now)
            if w <= self._decay.min_weight and self._decay.half_life_seconds > 0:
                forgotten.append(entry)
            else:
                decay_weighted.append((entry, w))

        # Step 2: Contradiction detection and resolution
        conflicts: list[ConflictRecord] = []
        if self._detect_contradictions:
            # Group by conflict_key; resolve pairwise
            groups: dict[str, list[tuple[CausalMemoryEntry, float]]] = {}
            for entry, w in decay_weighted:
                key = _conflict_key(entry)
                groups.setdefault(key, []).append((entry, w))

            resolved_pairs: list[tuple[CausalMemoryEntry, float]] = []
            for key, group in groups.items():
                if len(group) == 1:
                    resolved_pairs.append(group[0])
                    continue
                # Reduce by pairwise conflict resolution
                winner_entry, winner_w = group[0]
                for other_entry, other_w in group[1:]:
                    # Only flag as conflict if statuses differ
                    if winner_entry.status != other_entry.status:
                        resolved_winner, loser = _resolve_conflict(
                            winner_entry, other_entry, self._conflict_strategy
                        )
                        lost_entry = other_entry if resolved_winner is winner_entry else winner_entry
                        conflicts.append(ConflictRecord(
                            conflict_key=key,
                            winner=resolved_winner,
                            loser=lost_entry,
                            strategy=self._conflict_strategy,
                            reason=(
                                f"status conflict: {winner_entry.status!r} vs "
                                f"{other_entry.status!r}; "
                                f"strategy={self._conflict_strategy.value}"
                            ),
                        ))
                        winner_entry = resolved_winner
                        winner_w = winner_w if resolved_winner is winner_entry else other_w
                    else:
                        # Same status — keep the one with higher weight
                        if other_w > winner_w:
                            winner_entry, winner_w = other_entry, other_w
                resolved_pairs.append((winner_entry, winner_w))
            decay_weighted = resolved_pairs

        # Step 3: Apply priority multiplier
        final: list[tuple[CausalMemoryEntry, float]] = []
        for entry, base_w in decay_weighted:
            priority = _priority_weight(entry)
            final_w = min(2.0, base_w * priority)
            final.append((entry, final_w))

        # Step 4: Sort by descending weight
        final.sort(key=lambda item: item[1], reverse=True)

        return CoherentMemoryView(
            weighted=final,
            conflicts=conflicts,
            forgotten=forgotten,
        )


__all__ = [
    "DecayRule",
    "ConflictResolutionStrategy",
    "ConflictRecord",
    "CoherentMemoryView",
    "MemoryCoherenceEngine",
]
