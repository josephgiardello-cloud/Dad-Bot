from __future__ import annotations

import pytest

from dadbot.core.policy_store import InMemoryAsyncPolicyPersistence
from dadbot.memory.ledger import MemoryLedger


@pytest.mark.asyncio
async def test_memory_ledger_appends_and_verifies_chain() -> None:
    persistence = InMemoryAsyncPolicyPersistence()
    ledger = MemoryLedger(persistence)

    first = await ledger.append_memory_event({"type": "memory_update", "topic": "family"})
    second = await ledger.append_memory_event({"type": "relationship_shift", "delta": 0.2})

    assert first["entry_hash"]
    assert second["prev_hash"] == first["entry_hash"]
    assert await ledger.verify_chain() is True


@pytest.mark.asyncio
async def test_memory_ledger_detects_tamper() -> None:
    persistence = InMemoryAsyncPolicyPersistence()
    ledger = MemoryLedger(persistence)

    await ledger.append_memory_event({"type": "memory_update", "value": "x"})
    await ledger.append_memory_event({"type": "memory_update", "value": "y"})

    rows = await persistence.lrange("memory:ledger", 0, -1)
    assert len(rows) == 2

    tampered = rows[1].replace('"value":"y"', '"value":"z"')
    persistence._lists["memory:ledger"][1] = tampered

    assert await ledger.verify_chain() is False
