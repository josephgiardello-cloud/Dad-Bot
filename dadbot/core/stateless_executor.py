"""L4-P5 — Stateless Execution Mode.

Formalizes the execution layer as a pure function:

    execute(user_input, event_log, config) → StatelessExecutionResult

Key invariants:
- No runtime memory: the executor does not own any mutable state.
- All state is derived from the event log (input parameter).
- The executor is bootstrapped iff the event log contains at least one
  SESSION_STATE_UPDATED event.
- Same (input, event_log) → same output, always (deterministic).

Design principle:
    A stateless executor is a reduction machine:
    Given an event log, it reconstructs state, processes the input against
    that state, and produces a result + a new event to append.

The executor does NOT call LLMs or tools — it operates on event evidence
already in the log.  This makes it suitable for replay, verification,
and deterministic audit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from dadbot.core.event_authority import rebuild_state_from_events

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatelessExecutionResult:
    """Result of a stateless execution step.

    Attributes:
        state:          Derived state at the time of execution.
        event_log:      The full event log used as input (immutable copy).
        execution_hash: Deterministic hash of (user_input, event_log, config).
        reconstructed:  True iff state was derived from the log (always True here).
        bootstrapped:   True iff the log contained a SESSION_STATE_UPDATED event.
        session_id:     Session derived from the event log (or "").
        notes:          Any runtime notes from the execution step (informational).

    """

    state: dict[str, Any]
    event_log: tuple[dict[str, Any], ...]
    execution_hash: str
    reconstructed: bool
    bootstrapped: bool
    session_id: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": dict(self.state),
            "event_count": len(self.event_log),
            "execution_hash": self.execution_hash,
            "reconstructed": self.reconstructed,
            "bootstrapped": self.bootstrapped,
            "session_id": self.session_id,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Bootstrap check
# ---------------------------------------------------------------------------

_BOOTSTRAP_EVENT_TYPES = frozenset(
    {
        "SESSION_STATE_UPDATED",
        "session_state_updated",
        "session_start",
        "SESSION_START",
        "TURN_COMPLETED",
        "turn_completed",
    },
)


def is_bootstrapped(event_log: list[dict[str, Any]]) -> bool:
    """True iff the event log contains at least one bootstrap event.

    A bootstrapped log has SESSION_STATE_UPDATED (or equivalent) — meaning
    at least one full session turn has completed and the system is defined.

    An empty log or a log with only TOOL_EXECUTED events is not bootstrapped.
    """
    return any(str(e.get("type") or "").strip() in _BOOTSTRAP_EVENT_TYPES for e in (event_log or []))


# ---------------------------------------------------------------------------
# Stateless executor
# ---------------------------------------------------------------------------


def _execution_hash(
    user_input: str,
    event_log: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:
    payload = {
        "user_input": str(user_input or ""),
        "event_count": len(event_log),
        "event_log": event_log,
        "config_keys": sorted((config or {}).keys()),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


class StatelessExecutor:
    """Pure-function execution machine.

    execute(user_input, event_log, config) → StatelessExecutionResult

    - No owned mutable state.
    - All state derived from event_log.
    - Deterministic: same inputs → same result.
    - Bootstrappable check: logs without SESSION_STATE_UPDATED are flagged.
    """

    def __init__(
        self,
        reducer: Any | None = None,
    ) -> None:
        """Args:
        reducer: Optional CanonicalEventReducer.  If None, uses
                 rebuild_state_from_events (which creates its own).

        """
        self._reducer = reducer

    def execute(
        self,
        user_input: str,
        event_log: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> StatelessExecutionResult:
        """Execute a single cognitive step as a pure function.

        This does NOT run any LLM or tool.  It:
        1. Reconstructs state from the event log.
        2. Derives session_id, bootstrapped status.
        3. Returns a fully-described result without mutating anything.

        For replay / audit use: identical inputs always produce identical
        execution_hash.
        """
        log = list(event_log or [])
        cfg = dict(config or {})
        inp = str(user_input or "")

        # Reconstruct state purely.
        if self._reducer is not None:
            state = self._reducer.reduce(log)
        else:
            state = rebuild_state_from_events(log)

        # Derive session context from state.
        sessions = dict(state.get("sessions") or {})
        session_id = str(
            state.get("last_session_id") or (next(iter(sessions)) if sessions else ""),
        )
        bootstrapped = is_bootstrapped(log)

        e_hash = _execution_hash(inp, log, cfg)

        notes = ""
        if not bootstrapped:
            notes = "Log is not bootstrapped — no SESSION_STATE_UPDATED found."
        elif not log:
            notes = "Empty event log."

        return StatelessExecutionResult(
            state=state,
            event_log=tuple(dict(e) for e in log),
            execution_hash=e_hash,
            reconstructed=True,
            bootstrapped=bootstrapped,
            session_id=session_id,
            notes=notes,
        )

    def is_bootstrapped(self, event_log: list[dict[str, Any]]) -> bool:
        """Convenience wrapper around module-level is_bootstrapped."""
        return is_bootstrapped(event_log)

    def replay_execution(
        self,
        original_result: StatelessExecutionResult,
        user_input: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Re-execute with the same event_log and verify determinism.

        Returns::
            {
                "ok": bool,
                "original_hash": str,
                "replayed_hash": str,
                "hash_matches": bool,
            }
        """
        replayed = self.execute(user_input, list(original_result.event_log), config)
        return {
            "ok": replayed.execution_hash == original_result.execution_hash,
            "original_hash": original_result.execution_hash,
            "replayed_hash": replayed.execution_hash,
            "hash_matches": replayed.execution_hash == original_result.execution_hash,
        }


__all__ = [
    "StatelessExecutionResult",
    "StatelessExecutor",
    "is_bootstrapped",
]
