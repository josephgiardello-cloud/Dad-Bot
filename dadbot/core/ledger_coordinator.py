from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dadbot.core.effect_journal import EffectJournal
from dadbot.core.ledger_index import LedgerIndex
from dadbot.core.ledger_reader import LedgerReader
from dadbot.core.ledger_writer_adapter import LedgerWriterAdapter


@dataclass(frozen=True)
class LedgerRuntime:
    writer: LedgerWriterAdapter
    index: LedgerIndex
    effect_journal: EffectJournal
    reader: LedgerReader
    reconcile_queue: Any


class LedgerCoordinator:
    """Builds ledger read/write services used by the execution control plane."""

    def __init__(self, *, ledger: Any, scope_validator: Callable[[Any], None]) -> None:
        self._ledger = ledger
        self._scope_validator = scope_validator

    def build_runtime(
        self,
        *,
        reconcile_queue_factory: Callable[[LedgerWriterAdapter, EffectJournal], Any],
        writer: LedgerWriterAdapter | None = None,
    ) -> LedgerRuntime:
        writer = writer or LedgerWriterAdapter(self._ledger, scope_validator=self._scope_validator)
        index = LedgerIndex(self._ledger)
        effect_journal = EffectJournal(writer=writer, index=index)
        reader = LedgerReader(self._ledger)
        reconcile_queue = reconcile_queue_factory(writer, effect_journal)
        return LedgerRuntime(
            writer=writer,
            index=index,
            effect_journal=effect_journal,
            reader=reader,
            reconcile_queue=reconcile_queue,
        )
