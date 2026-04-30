"""graph_mutation — turn-scoped mutation management primitives.

Extracted from graph.py to reduce TurnGraph god-class surface area.
All names are re-exported from dadbot.core.graph for backward compatibility.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dadbot.core.contracts.mutation import coerce_mutation_kind, ensure_valid_mutation_op
from dadbot.core.execution_policy import FatalTurnError
from dadbot.core.graph_types import (
    GoalMutationOp,
    LedgerMutationOp,
    MemoryMutationOp,
    MutationKind,
    MutationTransactionRecord,
    MutationTransactionStatus,
    RelationshipMutationOp,
)

logger = logging.getLogger(__name__)

# Canonical durability boundary contract for the execution graph.
SAVE_NODE_COMMIT_CONTRACT = (
    "All durable mutations MUST go through SaveNode. The graph guarantees speculative execution until that boundary."
)

# ---------------------------------------------------------------------------
# Mutation intent payload schema (pre-commit validation gate)
# ---------------------------------------------------------------------------

# Required top-level keys per mutation kind (runtime-enforced)
_INTENT_REQUIRED_PAYLOAD_KEYS: dict[str, list[str]] = {
    "memory": ["temporal"],
    "relationship": ["temporal"],
    "graph": [],
    "ledger": ["temporal"],
    "goal": [],
}


def _validate_mutation_intent_payload(intent: MutationIntent) -> None:
    """Enforce per-kind required-field schema on a MutationIntent payload.

    Raises ``ValueError`` when a required field is absent, converting silent
    data corruption into an explicit pre-commit schema failure.
    """
    kind = str(intent.type).lower()
    required = _INTENT_REQUIRED_PAYLOAD_KEYS.get(kind, [])
    # Only validate temporal when requires_temporal=True — matches existing guard.
    if not bool(intent.requires_temporal) and "temporal" in required:
        required = [k for k in required if k != "temporal"]
    for key in required:
        if key not in intent.payload:
            raise ValueError(
                f"MutationIntent({kind!r}) is missing required payload key: {key!r}",
            )


@dataclass
class MutationIntent:
    """A single deferred persistent mutation to be committed at SaveNode."""

    type: MutationKind
    payload: dict[str, Any]
    requires_temporal: bool = True
    source: str = ""  # caller tag for audit/replay
    priority: int = 100
    turn_index: int = 0
    sequence_id: int = 0
    # Optional compensator used by transactional drain rollback.
    compensator: Callable[[], None] | None = field(default=None, repr=False)
    # Content hash computed in __post_init__; not part of __init__ signature.
    payload_hash: str = field(default="", init=False, repr=False)

    # ------------------------------------------------------------------
    # __post_init__ helpers (static, so they can be called before self is valid)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_temporal_payload(payload: dict) -> None:
        temporal = payload.get("temporal")
        if not isinstance(temporal, dict):
            raise RuntimeError("TemporalNode required — execution invalid")
        wall_time = str(temporal.get("wall_time") or "").strip()
        wall_date = str(temporal.get("wall_date") or "").strip()
        if not wall_time or not wall_date:
            raise RuntimeError("TemporalNode required — execution invalid")

    @staticmethod
    def _validate_ordering_fields(priority: Any, turn_index: Any, sequence_id: Any) -> tuple[int, int, int]:
        try:
            return int(priority), int(turn_index), int(sequence_id)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("MutationIntent ordering fields must be integers") from exc

    @staticmethod
    def _validate_op_for_kind(kind: MutationKind, op: str) -> None:
        # Canonical taxonomy check (single shared vocabulary contract).
        ensure_valid_mutation_op(kind, op)
        if kind is MutationKind.MEMORY:
            if op and op not in {item.value for item in MemoryMutationOp}:
                raise RuntimeError(f"Unsupported memory mutation op: {op!r}")
        elif kind is MutationKind.RELATIONSHIP:
            if op and op not in {item.value for item in RelationshipMutationOp}:
                raise RuntimeError(f"Unsupported relationship mutation op: {op!r}")
        elif kind is MutationKind.LEDGER:
            if op and op not in {item.value for item in LedgerMutationOp}:
                raise RuntimeError(f"Unsupported ledger mutation op: {op!r}")
        elif kind is MutationKind.GOAL:
            if op and op not in {item.value for item in GoalMutationOp}:
                raise RuntimeError(f"Unsupported goal mutation op: {op!r}")

    def __post_init__(self) -> None:
        self.type = coerce_mutation_kind(self.type)
        if not isinstance(self.payload, dict):
            raise RuntimeError("MutationIntent.payload must be a dict")
        if self.requires_temporal:
            self._validate_temporal_payload(self.payload)
        self.priority, self.turn_index, self.sequence_id = self._validate_ordering_fields(
            self.priority,
            self.turn_index,
            self.sequence_id,
        )
        op = str(self.payload.get("op") or "").strip().lower()
        self._validate_op_for_kind(self.type, op)
        _validate_mutation_intent_payload(self)
        self.payload_hash = hashlib.sha256(
            json.dumps(self.payload, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()[:24]


class MutationQueue:
    """Turn-scoped queue for all pending persistent mutations.

    Rules:
    - Inside SaveNode (commit_active=True): execute immediately or drain
    - Outside SaveNode: queue for SaveNode to drain
    - If queue cannot be drained at SaveNode: hard fail (nothing dropped silently)
    """

    def __init__(self) -> None:
        self._queue: list[MutationIntent] = []
        self._drained: list[MutationIntent] = []
        self._failed: list[tuple[MutationIntent, str]] = []
        self._transactions: list[MutationTransactionRecord] = []
        self._owner_trace_id: str = ""
        self._sequence_counter: int = 0
        self._mutations_locked: bool = False

    def lock(self) -> None:
        """Lock the queue to prevent mutations outside SaveNode."""
        self._mutations_locked = True

    def unlock(self) -> None:
        """Unlock the queue to allow mutations."""
        self._mutations_locked = False

    def bind_owner(self, owner_id: str) -> None:
        owner = str(owner_id or "").strip()
        if not owner:
            raise RuntimeError(
                "MutationQueue owner binding requires non-empty trace_id",
            )
        if self._owner_trace_id and self._owner_trace_id != owner:
            raise RuntimeError(
                f"MutationQueue cross-turn reuse detected: owner={self._owner_trace_id!r}, new_owner={owner!r}",
            )
        self._owner_trace_id = owner

    def _assert_owner(self) -> None:
        if not self._owner_trace_id:
            raise RuntimeError("MutationQueue is not bound to a turn trace_id")

    def queue(self, intent: MutationIntent) -> None:
        """Add a mutation intent to the pending queue."""
        self._assert_owner()
        if self._mutations_locked:
            raise RuntimeError(
                f"MutationGuard violation: mutation attempted outside SaveNode (type={getattr(intent, 'type', '?')!r})",
            )
        if int(getattr(intent, "sequence_id", 0) or 0) <= 0:
            self._sequence_counter += 1
            intent.sequence_id = self._sequence_counter
        self._queue.append(intent)

    def pending(self) -> list[MutationIntent]:
        self._assert_owner()
        return list(self._queue)

    def is_empty(self) -> bool:
        self._assert_owner()
        return len(self._queue) == 0

    def size(self) -> int:
        self._assert_owner()
        return len(self._queue)

    def drain(
        self,
        executor: Callable[[MutationIntent], Any],
        *,
        hard_fail_on_error: bool = True,
        transactional: bool = True,
    ) -> list[tuple[MutationIntent, str]]:
        """Execute all queued intents through ``executor``. Returns failures.

        If ``hard_fail_on_error=True`` (default for SaveNode), failures raise
        ``FatalTurnError``. With ``transactional=True``, hard failures attempt
        rollback and re-queue the full transaction for replay-safe retries.
        """
        self._assert_owner()
        to_drain = sorted(
            list(self._queue),
            key=lambda m: (
                int(getattr(m, "priority", 0) or 0),
                int(getattr(m, "turn_index", 0) or 0),
                int(getattr(m, "sequence_id", 0) or 0),
            ),
        )
        self._queue.clear()

        tx_id = uuid.uuid4().hex
        applied: list[tuple[MutationIntent, Any]] = []
        rollback_count = 0
        rollback_failures = 0

        for index, intent in enumerate(to_drain):
            try:
                result = executor(intent)
                compensator = result if callable(result) else getattr(intent, "compensator", None)
                if compensator is not None and not callable(compensator):
                    compensator = None
                applied.append((intent, compensator))
                self._drained.append(intent)
            except Exception as exc:
                failure = (intent, str(exc))
                self._failed.append(failure)

                if not hard_fail_on_error:
                    self._queue = [intent, *to_drain[index + 1 :], *self._queue]
                    continue

                if transactional:
                    for applied_intent, compensator in reversed(applied):
                        if callable(compensator):
                            try:
                                compensator()
                                rollback_count += 1
                            except Exception as exc:  # noqa: BLE001 — must not let rollback abort the loop
                                logger.warning("Rollback compensator raised: %s", exc)
                                rollback_failures += 1

                    self._queue = [*to_drain, *self._queue]
                    for applied_intent, _ in applied:
                        with contextlib.suppress(ValueError):
                            self._drained.remove(applied_intent)
                    status = (
                        MutationTransactionStatus.ROLLBACK_FAILED
                        if rollback_failures
                        else MutationTransactionStatus.ROLLED_BACK
                    )
                    self._transactions.append(
                        MutationTransactionRecord(
                            transaction_id=tx_id,
                            status=status,
                            applied_count=len(applied),
                            failed_count=1,
                            rollback_count=rollback_count,
                            rollback_failures=rollback_failures,
                            trace_id=self._owner_trace_id,
                            error=str(exc),
                        ),
                    )
                else:
                    self._queue = [intent, *to_drain[index + 1 :], *self._queue]

                raise FatalTurnError(
                    f"MutationQueue drain failed at SaveNode - type={intent.type!r} source={intent.source!r}: {exc}",
                ) from exc

        if to_drain:
            self._transactions.append(
                MutationTransactionRecord(
                    transaction_id=tx_id,
                    status=MutationTransactionStatus.COMMITTED,
                    applied_count=len(applied),
                    failed_count=0,
                    rollback_count=0,
                    rollback_failures=0,
                    trace_id=self._owner_trace_id,
                ),
            )
        return list(self._failed)

    def snapshot(self) -> dict[str, Any]:
        self._assert_owner()
        pending_ledger = sum(1 for intent in self._queue if intent.type is MutationKind.LEDGER)
        drained_ledger = sum(1 for intent in self._drained if intent.type is MutationKind.LEDGER)
        failed_ledger = sum(1 for intent, _ in self._failed if intent.type is MutationKind.LEDGER)
        # Strip created_at and transaction_id: both are non-deterministic across
        # separate executions (wall-clock time and UUID4 respectively) and must
        # not appear in determinism-audit snapshots compared across runs.
        latest_tx_raw = self._transactions[-1].to_dict() if self._transactions else {}
        latest_tx = {k: v for k, v in latest_tx_raw.items() if k not in ("created_at", "transaction_id")}
        return {
            "owner_trace_id": self._owner_trace_id,
            # Contract counters keep non-ledger mutation accounting stable for
            # restart-boundary audits that verify durable mutation invariants.
            "pending": len(self._queue) - pending_ledger,
            "drained": len(self._drained) - drained_ledger,
            "failed": len(self._failed) - failed_ledger,
            # Ledger counters remain visible for observability.
            "ledger_pending": pending_ledger,
            "ledger_drained": drained_ledger,
            "ledger_failed": failed_ledger,
            "transactions": len(self._transactions),
            "latest_transaction": latest_tx,
        }


class MutationGuard:
    """Context manager that blocks mutation queueing outside SaveNode.

    Wrap every non-SaveNode stage execution with this guard to enforce the
    invariant that only SaveNode may commit mutations at runtime. Any attempt
    to call ``MutationQueue.queue()`` while the guard is active raises
    ``RuntimeError``, converting a convention into an enforced runtime contract.
    """

    def __init__(self, mutation_queue: MutationQueue) -> None:
        self._queue = mutation_queue

    def __enter__(self) -> MutationGuard:
        self._queue.lock()
        return self

    def __exit__(self, *_: object) -> None:
        self._queue.unlock()
