from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class TransactionParticipant(Protocol):
    """Participant in a coordinated transaction commit."""

    def prepare(self) -> None:
        """Validate and stage resources for commit.

        Must not make durable external changes.
        """

    def commit(self) -> None:
        """Apply durable changes."""

    def rollback(self) -> None:
        """Attempt to restore pre-transaction state."""


@dataclass
class TransactionReport:
    """Result metadata for transaction execution."""

    ok: bool
    committed_participants: int
    rolled_back_participants: int
    error: str = ""


@dataclass
class TransactionContext:
    """Container for transaction-scoped metadata and participants."""

    transaction_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


class TransactionCoordinator:
    """Single authority for all-or-nothing participant commits.

    Flow:
    1. register(participant)
    2. execute() -> prepare all, commit all, rollback on first failure
    """

    def __init__(self, context: TransactionContext) -> None:
        self._context = context
        self._participants: list[TransactionParticipant] = []

    @property
    def context(self) -> TransactionContext:
        return self._context

    def register(self, participant: TransactionParticipant) -> None:
        self._participants.append(participant)

    def execute(self) -> TransactionReport:
        prepared: list[TransactionParticipant] = []
        committed: list[TransactionParticipant] = []
        try:
            for participant in self._participants:
                participant.prepare()
                prepared.append(participant)

            for participant in prepared:
                participant.commit()
                committed.append(participant)

            return TransactionReport(
                ok=True,
                committed_participants=len(committed),
                rolled_back_participants=0,
                error="",
            )
        except Exception as exc:
            rolled_back = 0
            for participant in reversed(committed):
                try:
                    participant.rollback()
                    rolled_back += 1
                except Exception:
                    # Best-effort rollback: continue to restore remaining participants.
                    pass

            # Rollback non-committed participants too if they can restore staged state.
            for participant in reversed(prepared):
                if participant in committed:
                    continue
                try:
                    participant.rollback()
                    rolled_back += 1
                except Exception:
                    pass

            return TransactionReport(
                ok=False,
                committed_participants=len(committed),
                rolled_back_participants=rolled_back,
                error=str(exc),
            )


__all__ = [
    "TransactionContext",
    "TransactionCoordinator",
    "TransactionParticipant",
    "TransactionReport",
]
