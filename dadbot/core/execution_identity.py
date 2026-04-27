"""Canonical execution identity model for turn execution.

ExecutionIdentity is the SINGLE authoritative fingerprint for whether a turn
execution is deterministically equivalent to a prior execution.  It unifies:

  - trace_hash          (ordered execution event stream hash)
  - lock_hash           (determinism lock seeded at turn start)
  - checkpoint_chain_hash (last node in the durable checkpoint chain)
  - mutation_tx_count   (number of committed mutation transactions)
  - event_count         (total execution trace events emitted)

into one canonical ``fingerprint`` value that replay validators can compare.

The hard runtime contract
~~~~~~~~~~~~~~~~~~~~~~~~~
``seal_from_context`` builds the identity and immediately calls
``raise_if_mismatch`` when an expected fingerprint is present in turn metadata
(``metadata["expected_execution_fingerprint"]``).  This runs unconditionally at
every turn exit — not only during test assertions.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


class ExecutionIdentityViolation(RuntimeError):
    """Raised when the execution identity fingerprint does not match expectation."""

    def __init__(self, *, expected: str, actual: str) -> None:
        super().__init__(
            f"Execution identity fingerprint mismatch: "
            f"expected={expected!r}, actual={actual!r}"
        )
        self.expected = expected
        self.actual = actual


@dataclass
class ExecutionIdentity:
    """Canonical single-source execution fingerprint for a completed turn.

    Build via ``ExecutionIdentity.from_turn_context(turn_context)`` after the
    execution trace contract has been finalized.
    """

    trace_id: str
    trace_hash: str
    lock_hash: str
    checkpoint_chain_hash: str
    mutation_tx_count: int
    event_count: int

    @property
    def fingerprint(self) -> str:
        """Canonical SHA-256 fingerprint combining all identity components."""
        canonical = {
            "trace_id": str(self.trace_id or ""),
            "trace_hash": str(self.trace_hash or ""),
            "lock_hash": str(self.lock_hash or ""),
            "checkpoint_chain_hash": str(self.checkpoint_chain_hash or ""),
            "mutation_tx_count": int(self.mutation_tx_count),
            "event_count": int(self.event_count),
        }
        return hashlib.sha256(
            json.dumps(canonical, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": str(self.trace_id or ""),
            "trace_hash": str(self.trace_hash or ""),
            "lock_hash": str(self.lock_hash or ""),
            "checkpoint_chain_hash": str(self.checkpoint_chain_hash or ""),
            "mutation_tx_count": int(self.mutation_tx_count),
            "event_count": int(self.event_count),
            "fingerprint": self.fingerprint,
        }

    def verify_against(self, expected_fingerprint: str) -> bool:
        """Return True if expected is empty (no contract) or matches self.fingerprint."""
        expected = str(expected_fingerprint or "").strip()
        if not expected:
            return True
        return self.fingerprint == expected

    def raise_if_mismatch(self, expected_fingerprint: str) -> None:
        """Raise ExecutionIdentityViolation when expected is set and does not match."""
        expected = str(expected_fingerprint or "").strip()
        if expected and not self.verify_against(expected):
            raise ExecutionIdentityViolation(expected=expected, actual=self.fingerprint)

    @classmethod
    def from_turn_context(cls, turn_context: Any) -> "ExecutionIdentity":
        """Build ExecutionIdentity from a completed TurnContext.

        Must be called after ``_finalize_execution_trace_contract`` so that
        ``state["execution_trace_contract"]`` is populated.
        """
        state = getattr(turn_context, "state", None) or {}
        metadata = getattr(turn_context, "metadata", None) or {}
        trace_contract = dict(state.get("execution_trace_contract") or {})
        determinism = dict(metadata.get("determinism") or {})
        mutation_queue = getattr(turn_context, "mutation_queue", None)
        mutation_snapshot = (
            mutation_queue.snapshot()
            if hasattr(mutation_queue, "snapshot")
            else {}
        )
        return cls(
            trace_id=str(getattr(turn_context, "trace_id", "") or ""),
            trace_hash=str(trace_contract.get("trace_hash") or ""),
            lock_hash=str(determinism.get("lock_hash") or ""),
            checkpoint_chain_hash=str(getattr(turn_context, "last_checkpoint_hash", "") or ""),
            mutation_tx_count=int(mutation_snapshot.get("transactions", 0)),
            event_count=int(trace_contract.get("event_count", 0)),
        )
