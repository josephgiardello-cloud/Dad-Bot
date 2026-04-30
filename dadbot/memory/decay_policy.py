"""Deterministic memory decay policy.

Scores and classifies consolidated memory entries using *only*:
  - TurnTemporalAxis.epoch_seconds  (canonical per-turn time)
  - entry fields: last_reinforced_at / updated_at, source_count,
                  confidence, importance_score

No datetime.now().  No external clocks.  No hidden randomness.
Every decay decision is replayable given the same inputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Default thresholds â€” change via constructor, never in-place mutation.
_PRUNE_THRESHOLD: float = 0.15
_WEAKEN_THRESHOLD: float = 0.30
_WEAKEN_FACTOR: float = 0.80  # multiply importance_score for weakened entries


@dataclass
class DecayResult:
    """Output contract of MemoryDecayPolicy.apply."""

    pruned: list[str] = field(default_factory=list)
    weakened: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    total_score_map: dict[str, float] = field(default_factory=dict)


def _parse_epoch(ts_str: str | None) -> float | None:
    """Parse an ISO timestamp string to epoch seconds.  No external clocks."""
    if not ts_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(str(ts_str), fmt).replace(tzinfo=UTC)
            return dt.timestamp()
        except ValueError:
            continue
    return None


def _recency_weight(last_epoch: float | None, turn_epoch: float) -> float:
    """1.0 at t=0 (same moment), decays linearly to 0.0 at 365 days."""
    if last_epoch is None:
        return 0.5  # unknown age â†’ neutral
    age_days = max(0.0, turn_epoch - last_epoch) / 86400.0
    return max(0.0, 1.0 - age_days / 365.0)


def _reinforcement_weight(source_count: int) -> float:
    """0.0 at 1 source, 1.0 at 5+ sources (linear ramp)."""
    return min(1.0, max(0.0, (source_count - 1) / 4.0))


def _score_entry(entry: dict, turn_epoch: float) -> float:
    """Compute a deterministic decay score in [0.0, 1.0].

    score = (recency + reinforcement + quality + relationship_affinity) / 4
    """
    last_ts = entry.get("last_reinforced_at") or entry.get("updated_at")
    recency = _recency_weight(_parse_epoch(last_ts), turn_epoch)

    try:
        src = max(1, int(entry.get("source_count", 1) or 1))
    except (TypeError, ValueError):
        src = 1
    reinforcement = _reinforcement_weight(src)

    try:
        quality = float(entry.get("confidence", 0.5) or 0.5)
    except (TypeError, ValueError):
        quality = 0.5
    quality = max(0.0, min(1.0, quality))

    try:
        relationship_affinity = float(entry.get("importance_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        relationship_affinity = 0.0
    relationship_affinity = max(0.0, min(1.0, relationship_affinity))

    return (recency + reinforcement + quality + relationship_affinity) / 4.0


class MemoryDecayPolicy:
    """Pure deterministic scoring and categorization of consolidated memory entries.

    Instantiate once per turn (or share a singleton â€” it holds no mutable state).
    Call ``apply(entries, turn_context)`` to get a ``DecayResult``.

    The caller is responsible for acting on the result (pruning / weakening
    entries in the memory store).  This class only decides *what* to do.
    """

    def __init__(
        self,
        prune_threshold: float = _PRUNE_THRESHOLD,
        weaken_threshold: float = _WEAKEN_THRESHOLD,
        weaken_factor: float = _WEAKEN_FACTOR,
    ) -> None:
        self.prune_threshold = prune_threshold
        self.weaken_threshold = weaken_threshold
        self.weaken_factor = weaken_factor

    def apply(self, entries: list[dict], turn_context: Any) -> DecayResult:
        """Score and classify each entry.  Returns a ``DecayResult`` with memory ids.

        If no temporal axis is available on ``turn_context`` the method returns
        a safe no-op result (all entries unchanged) rather than failing.
        """
        temporal = getattr(turn_context, "temporal", None)
        turn_epoch: float | None = getattr(temporal, "epoch_seconds", None)

        if turn_epoch is None:
            # No temporal axis â€” safe no-op pass-through
            unchanged = [str(e.get("id", "")) for e in entries if e.get("id")]
            return DecayResult(unchanged=unchanged)

        result = DecayResult()
        for entry in entries:
            entry_id = str(entry.get("id", ""))
            if not entry_id:
                continue

            # Pinned entries are immune from decay
            if bool(entry.get("pinned", False)):
                result.unchanged.append(entry_id)
                result.total_score_map[entry_id] = 1.0
                continue

            score = _score_entry(entry, turn_epoch)
            result.total_score_map[entry_id] = round(score, 4)

            if score < self.prune_threshold:
                result.pruned.append(entry_id)
            elif score < self.weaken_threshold:
                result.weakened.append(entry_id)
            else:
                result.unchanged.append(entry_id)

        if result.pruned or result.weakened:
            logger.info(
                "MemoryDecayPolicy: pruned=%d weakened=%d unchanged=%d",
                len(result.pruned),
                len(result.weakened),
                len(result.unchanged),
            )
        return result
