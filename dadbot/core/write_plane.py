"""dadbot/core/write_plane.py — Write interception layer (Phase 2, Step 2.0).

PURPOSE
-------
All state mutations in the system route through WritePlane.write() so that:

  1. Every write is logged (source, path, value_repr, turn_id, timestamp)
  2. Determinism can be validated via mutation_log comparison across replays
  3. Single-writer authority can be enforced in a future step (Phase 2.3)

CURRENT STATUS: Phase 2.0 — OBSERVATIONAL ONLY
-----------------------------------------------
No write is blocked.  No behavior is changed.  Instrumented systems call
``get_write_plane().write(...)`` alongside their existing write, producing a
complete mutation log per turn.

INSTRUMENTED WRITE SYSTEMS (all 5):
  - InternalStateManager   → memory.internal_state
  - BeliefStateEngine      → memory.belief_state
  - TurnStateMutator       → bot._last_turn_pipeline
  - InteractionStateEngine → interaction_state._state
  - RelationshipState      → memory.relationship_state (via lifecycle.py)

USAGE (write site)
------------------
    from dadbot.core.write_plane import get_write_plane

    plane = get_write_plane()
    plane.write("TurnStateMutator", "bot._last_turn_pipeline", value)
    self._bot._last_turn_pipeline = value   # original write is unchanged

USAGE (turn boundary)
---------------------
    plane.begin_turn(turn_id)
    ...execute turn...
    log = plane.drain_log()   # or snapshot_log() for non-destructive read
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any


@dataclasses.dataclass(slots=True)
class MutationRecord:
    """Single write event captured at the point of mutation."""

    source: str          # which system is writing (class name or label)
    path: str            # logical path e.g. "memory.internal_state" or "bot._last_turn_pipeline"
    value_repr: str      # repr() of value, truncated — for diffing, not replay
    turn_id: str | None  # active turn when this write occurred, or None if outside a turn
    timestamp: float     # monotonic time of the write


def _safe_repr(value: Any, maxlen: int = 256) -> str:
    try:
        r = repr(value)
    except Exception:
        r = "<repr-error>"
    return r[:maxlen] if len(r) > maxlen else r


class WritePlane:
    """Intercept point for all state mutations.

    Phase 2.0: logs every write without blocking it.
    Phase 2.3: will raise WriteAuthorityError for unauthorized sources.
    """

    def __init__(self) -> None:
        self._log: list[MutationRecord] = []
        self._turn_id: str | None = None

    # ------------------------------------------------------------------
    # Turn lifecycle
    # ------------------------------------------------------------------

    def begin_turn(self, turn_id: str) -> None:
        """Mark the start of a turn boundary for log correlation."""
        self._turn_id = str(turn_id or "")

    def end_turn(self) -> str | None:
        """Clear the active turn boundary.  Returns the turn_id that ended."""
        tid = self._turn_id
        self._turn_id = None
        return tid

    # ------------------------------------------------------------------
    # Core write method — transparent pass-through in Phase 2.0
    # ------------------------------------------------------------------

    def write(
        self,
        source: str,
        path: str,
        value: Any,
        *,
        metadata: dict[str, Any] | None = None,  # reserved for Phase 2.3 authority checks
    ) -> Any:
        """Log a state write and return ``value`` unchanged.

        Callers pattern-match this as a transparent pass-through::

            result = plane.write("MySystem", "some.path", computed_value)
            self._field = result   # original write still happens
        """
        record = MutationRecord(
            source=str(source or "unknown"),
            path=str(path or "?"),
            value_repr=_safe_repr(value),
            turn_id=self._turn_id,
            timestamp=time.monotonic(),
        )
        self._log.append(record)
        return value

    # ------------------------------------------------------------------
    # Log access
    # ------------------------------------------------------------------

    def drain_log(self) -> list[MutationRecord]:
        """Remove and return all accumulated records (destructive)."""
        log = list(self._log)
        self._log.clear()
        return log

    def snapshot_log(self) -> list[dict[str, Any]]:
        """Return a serialisable snapshot without clearing the log (non-destructive)."""
        return [dataclasses.asdict(r) for r in self._log]

    def drain_turn_log(self, turn_id: str) -> list[MutationRecord]:
        """Remove and return only records matching a specific ``turn_id``."""
        matching = [r for r in self._log if r.turn_id == turn_id]
        self._log = [r for r in self._log if r.turn_id != turn_id]
        return matching

    def paths_written(self) -> list[str]:
        """Return deduplicated list of paths in log order (for assertions)."""
        seen: set[str] = set()
        result: list[str] = []
        for r in self._log:
            if r.path not in seen:
                seen.add(r.path)
                result.append(r.path)
        return result

    def __len__(self) -> int:
        return len(self._log)

    def __repr__(self) -> str:
        return f"WritePlane(turn_id={self._turn_id!r}, log_size={len(self._log)})"


# ---------------------------------------------------------------------------
# Module-level singleton — one write plane for the process lifetime.
# Tests must call reset_write_plane() in setUp/tearDown for isolation.
# ---------------------------------------------------------------------------

_GLOBAL_WRITE_PLANE: WritePlane = WritePlane()


def get_write_plane() -> WritePlane:
    """Return the process-global WritePlane instance."""
    return _GLOBAL_WRITE_PLANE


def reset_write_plane() -> WritePlane:
    """Replace the global instance with a fresh one and return it.

    Intended for test isolation::

        def setUp(self):
            self.plane = reset_write_plane()
    """
    global _GLOBAL_WRITE_PLANE
    _GLOBAL_WRITE_PLANE = WritePlane()
    return _GLOBAL_WRITE_PLANE


__all__ = [
    "MutationRecord",
    "WritePlane",
    "get_write_plane",
    "reset_write_plane",
]
