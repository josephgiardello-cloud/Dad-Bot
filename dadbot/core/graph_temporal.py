"""Temporal data types for the TurnGraph pipeline.

Extracted from graph.py to keep that module below 1800 lines.
All types remain re-exported from ``dadbot.core.graph`` for backward compatibility.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


@dataclass
class VirtualClock:
    """Deterministic seeded clock for eliminating wall-time nondeterminism in replay/tests.

    Assign to ``TurnContext.virtual_clock`` before turn execution.  ``TemporalNode``
    will call ``tick()`` and derive the turn's ``TurnTemporalAxis`` from the virtual
    timestamp instead of the real wall clock, making temporal fields 100% reproducible
    across replay runs.

    Usage::

        vc = VirtualClock(base_epoch=1_700_000_000.0, step_size_seconds=30.0)
        ctx.virtual_clock = vc
        # First tick => epoch=1_700_000_030.0 (base + 1 step)
    """

    base_epoch: float = field(default_factory=time.time)
    step_size_seconds: float = 1.0
    _step: int = field(default=0, init=False, repr=False)

    def now(self) -> float:
        """Current virtual epoch (does NOT advance the clock)."""
        return self.base_epoch + self._step * self.step_size_seconds

    def tick(self) -> float:
        """Advance clock by one step and return the new epoch."""
        self._step += 1
        return self.now()

    def to_datetime(self) -> datetime:
        """Convert current virtual epoch to a timezone-aware datetime."""
        return datetime.fromtimestamp(self.now()).astimezone().replace(microsecond=0)


@dataclass(frozen=True)
class TurnTemporalAxis:
    """Frozen temporal base shared by every stage in a single turn."""

    turn_started_at: str
    wall_time: str
    wall_date: str
    timezone: str
    utc_offset_minutes: int
    epoch_seconds: float

    @classmethod
    def from_now(cls) -> TurnTemporalAxis:
        now = datetime.now().astimezone().replace(microsecond=0)
        offset = now.utcoffset()
        offset_minutes = int(offset.total_seconds() // 60) if offset is not None else 0
        wall_time = now.isoformat(timespec="seconds")
        return cls(
            turn_started_at=wall_time,
            wall_time=wall_time,
            wall_date=now.date().isoformat(),
            timezone=str(now.tzname() or "local").strip() or "local",
            utc_offset_minutes=offset_minutes,
            epoch_seconds=now.timestamp(),
        )

    @classmethod
    def from_lock_hash(cls, lock_hash: str) -> TurnTemporalAxis:
        """Derive a deterministic temporal axis from a lock hash.

        This is used by strict replay mode so identical lock payloads produce
        identical turn timestamps and event ordering metadata.
        """
        seed = str(lock_hash or "").strip().lower()
        if not seed:
            return cls.from_now()
        try:
            seed_int = int(seed[:16], 16)
        except ValueError:
            return cls.from_now()
        # Base at 2024-01-01T00:00:00+00:00 and keep deterministic offsets.
        base_epoch = 1704067200
        # Keep offset bounded to avoid far-future drift while remaining stable.
        offset_seconds = seed_int % (365 * 24 * 60 * 60)
        epoch = float(base_epoch + offset_seconds)
        dt = datetime.fromtimestamp(epoch).astimezone().replace(microsecond=0)
        offset = dt.utcoffset()
        offset_minutes = int(offset.total_seconds() // 60) if offset is not None else 0
        wall_time = dt.isoformat(timespec="seconds")
        return cls(
            turn_started_at=wall_time,
            wall_time=wall_time,
            wall_date=dt.date().isoformat(),
            timezone=str(dt.tzname() or "local").strip() or "local",
            utc_offset_minutes=offset_minutes,
            epoch_seconds=epoch,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_started_at": self.turn_started_at,
            "wall_time": self.wall_time,
            "wall_date": self.wall_date,
            "timezone": self.timezone,
            "utc_offset_minutes": self.utc_offset_minutes,
            "epoch_seconds": self.epoch_seconds,
        }


class TurnPhase(str, Enum):
    PLAN = "PLAN"
    ACT = "ACT"
    OBSERVE = "OBSERVE"
    RESPOND = "RESPOND"


_PHASE_ORDER: tuple[TurnPhase, ...] = (
    TurnPhase.PLAN,
    TurnPhase.ACT,
    TurnPhase.OBSERVE,
    TurnPhase.RESPOND,
)
