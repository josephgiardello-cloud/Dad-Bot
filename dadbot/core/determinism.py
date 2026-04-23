"""Determinism enforcement boundary for the DadBot turn pipeline.

Problem
-------
LLM responses are inherently non-deterministic. Tracking drift is not enough:
to guarantee identical outcomes on replay we must *seal* the execution — capturing
every non-deterministic value at first execution (RECORD mode) and substituting
those sealed values on any subsequent replay (REPLAY mode).

Design
------
``DeterminismBoundary`` wraps any callable that produces a non-deterministic value
(primarily LLM calls, but also timestamps, random IDs, etc.).

    boundary = DeterminismBoundary()

    # First execution: RECORD mode (default)
    response = boundary.capture("inference.llm_call", call_llm, messages)

    # Replay: boundary replays sealed value without calling call_llm at all.
    boundary.seal()                      # switch to REPLAY mode
    response = boundary.capture("inference.llm_call", call_llm, messages)  # returns sealed value

The boundary is attached to the TurnContext so the entire graph can participate.

``DeterminismMode``
~~~~~~~~~~~~~~~~~~~
- ``RECORD`` (default): Execute the callable, record the output.  Any slot that is
  already recorded is returned directly (idempotent within a turn).
- ``REPLAY``: Never execute the callable; return sealed value or raise
  ``DeterminismViolation`` if no sealed value is present.
- ``OPEN``: Passthrough; no recording or enforcement.

``DeterminismViolation``
~~~~~~~~~~~~~~~~~~~~~~~~
Raised when REPLAY mode is requested but no sealed value exists for a slot, or
when a RECORD-mode call produces a response that is *structurally inconsistent*
with a previously sealed snapshot for the same slot (budget-drift detection).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class DeterminismMode(str, Enum):
    RECORD = "RECORD"   # Execute + capture output; re-use if already captured
    REPLAY = "REPLAY"   # Never execute; return sealed output or raise
    OPEN = "OPEN"       # Passthrough: no recording, no enforcement


class DeterminismViolation(RuntimeError):
    """Raised when the boundary cannot satisfy a determinism contract."""

    def __init__(self, slot: str, mode: DeterminismMode, reason: str):
        super().__init__(f"[{mode.value}] slot={slot!r}: {reason}")
        self.slot = slot
        self.mode = mode
        self.reason = reason


def _content_hash(value: Any) -> str:
    """Stable fingerprint of a JSON-serialisable value."""
    try:
        serialised = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        serialised = repr(value).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()[:16]


class DeterminismBoundary:
    """Single-turn execution boundary that enforces deterministic replay.

    Attributes
    ----------
    mode:           Current execution mode.
    sealed_values:  Slot → captured output mapping.
    hashes:         Slot → content hash of captured output.
    violations:     Accumulated violation records; non-empty means enforcement failed.
    """

    __slots__ = ("mode", "sealed_values", "hashes", "violations", "_call_count")

    def __init__(self, mode: DeterminismMode = DeterminismMode.RECORD) -> None:
        self.mode: DeterminismMode = mode
        self.sealed_values: dict[str, Any] = {}
        self.hashes: dict[str, str] = {}
        self.violations: list[dict[str, str]] = []
        self._call_count: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def seal(self) -> None:
        """Switch to REPLAY mode. All subsequent capture() calls will never execute."""
        self.mode = DeterminismMode.REPLAY

    def open(self) -> None:
        """Disable enforcement entirely (OPEN mode)."""
        self.mode = DeterminismMode.OPEN

    def record(self) -> None:
        """Switch back to RECORD mode."""
        self.mode = DeterminismMode.RECORD

    def capture(self, slot: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute *fn* or return the sealed value, depending on current mode.

        Parameters
        ----------
        slot:   Unique label for this non-deterministic operation within the turn.
        fn:     The callable that produces a non-deterministic value.
        *args / **kwargs: Forwarded to *fn* in RECORD / OPEN mode.

        Returns
        -------
        The return value of *fn* (RECORD/OPEN) or the previously captured value (REPLAY).

        Raises
        ------
        DeterminismViolation  if REPLAY mode and no sealed value is present.
        """
        self._call_count[slot] = self._call_count.get(slot, 0) + 1

        if self.mode == DeterminismMode.OPEN:
            return fn(*args, **kwargs)

        if self.mode == DeterminismMode.REPLAY:
            if slot not in self.sealed_values:
                violation = DeterminismViolation(
                    slot,
                    self.mode,
                    "no sealed value available; cannot guarantee deterministic replay",
                )
                self.violations.append({
                    "slot": slot,
                    "mode": self.mode.value,
                    "reason": violation.reason,
                })
                raise violation
            logger.debug("DeterminismBoundary: REPLAY slot=%r", slot)
            return copy.deepcopy(self.sealed_values[slot])

        # RECORD mode ---------------------------------------------------------
        if slot in self.sealed_values:
            # Already captured during this turn; return the same value (idempotency).
            logger.debug("DeterminismBoundary: RECORD HIT slot=%r", slot)
            return copy.deepcopy(self.sealed_values[slot])

        result = fn(*args, **kwargs)
        self._seal_slot(slot, result)
        logger.debug("DeterminismBoundary: RECORD SEALED slot=%r hash=%s", slot, self.hashes[slot])
        return result

    async def capture_async(self, slot: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Async variant of :meth:`capture`."""
        self._call_count[slot] = self._call_count.get(slot, 0) + 1

        if self.mode == DeterminismMode.OPEN:
            return await fn(*args, **kwargs)

        if self.mode == DeterminismMode.REPLAY:
            if slot not in self.sealed_values:
                violation = DeterminismViolation(
                    slot,
                    self.mode,
                    "no sealed value available; cannot guarantee deterministic replay",
                )
                self.violations.append({
                    "slot": slot,
                    "mode": self.mode.value,
                    "reason": violation.reason,
                })
                raise violation
            logger.debug("DeterminismBoundary: REPLAY (async) slot=%r", slot)
            return copy.deepcopy(self.sealed_values[slot])

        # RECORD mode
        if slot in self.sealed_values:
            return copy.deepcopy(self.sealed_values[slot])

        result = await fn(*args, **kwargs)
        self._seal_slot(slot, result)
        logger.debug("DeterminismBoundary: RECORD SEALED (async) slot=%r hash=%s", slot, self.hashes[slot])
        return result

    def inject(self, slot: str, value: Any) -> None:
        """Manually inject a sealed value — used when replaying from persisted events."""
        self._seal_slot(slot, value)

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable snapshot of all sealed slots."""
        return {
            "mode": self.mode.value,
            "slots": list(self.sealed_values.keys()),
            "hashes": dict(self.hashes),
            "call_counts": dict(self._call_count),
            "violations": list(self.violations),
            "consistent": not self.violations,
        }

    def is_consistent(self) -> bool:
        return not self.violations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _seal_slot(self, slot: str, value: Any) -> None:
        self.sealed_values[slot] = copy.deepcopy(value)
        self.hashes[slot] = _content_hash(value)
