from __future__ import annotations

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_backend import InMemoryLedgerBackend


class InMemoryExecutionLedger(ExecutionLedger):
    """Concrete in-memory execution ledger used by runtime control-plane wiring."""

    def __init__(self, *, strict_writes: bool = False) -> None:
        super().__init__(backend=InMemoryLedgerBackend(), strict_writes=strict_writes)


__all__ = ["InMemoryExecutionLedger"]
