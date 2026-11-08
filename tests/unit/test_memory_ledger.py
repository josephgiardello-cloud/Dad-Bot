from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from dadbot.core.policy_store import InMemoryAsyncPolicyPersistence
from dadbot.memory.ledger import MemoryLedger, SQLiteAsyncMemoryLedgerPersistence


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


@pytest.mark.asyncio
async def test_memory_ledger_sqlite_persists_across_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    ledger_one = MemoryLedger(SQLiteAsyncMemoryLedgerPersistence(str(db_path)))
    first = await ledger_one.append_memory_event({"type": "memory_update", "value": "first"})

    ledger_two = MemoryLedger(SQLiteAsyncMemoryLedgerPersistence(str(db_path)))
    second = await ledger_two.append_memory_event({"type": "memory_update", "value": "second"})

    assert second["prev_hash"] == first["entry_hash"]
    assert await ledger_two.verify_chain() is True


@pytest.mark.asyncio
async def test_sqlite_ledger_concurrent_appends_share_one_chain(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    ledgers = [
        MemoryLedger(SQLiteAsyncMemoryLedgerPersistence(str(db_path)))
        for _ in range(3)
    ]

    entries = await asyncio.gather(
        *(ledgers[index % len(ledgers)].append_memory_event({"index": index}) for index in range(40)),
    )

    assert len({entry["entry_id"] for entry in entries}) == 40
    assert len(await ledgers[0].entries()) == 40
    assert await ledgers[0].verify_chain()


@pytest.mark.asyncio
async def test_sqlite_ledger_head_failure_rolls_back_event(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    ledger = MemoryLedger(SQLiteAsyncMemoryLedgerPersistence(str(db_path)))
    first = await ledger.append_memory_event({"value": "first"})
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "CREATE TRIGGER reject_ledger_head BEFORE UPDATE ON dadbot_memory_kv "
            "BEGIN SELECT RAISE(ABORT, 'head update failed'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="head update failed"):
        await ledger.append_memory_event({"value": "second"})

    assert await ledger.entries() == [first]
    assert await ledger.verify_chain()


@pytest.mark.asyncio
async def test_memory_ledger_rejects_malformed_rows_and_missing_tail(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    ledger = MemoryLedger(SQLiteAsyncMemoryLedgerPersistence(str(db_path)))
    await ledger.append_memory_event({"value": "first"})
    await ledger.append_memory_event({"value": "second"})

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("UPDATE dadbot_memory_list SET value = 'not-json' WHERE seq = 2")
    assert not await ledger.verify_chain()
    with pytest.raises(ValueError, match="Malformed memory ledger"):
        await ledger.entries()

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("DELETE FROM dadbot_memory_list WHERE seq = 2")
    assert not await ledger.verify_chain()
    with pytest.raises(ValueError, match="head does not match"):
        await ledger.append_memory_event({"value": "third"})


@pytest.mark.asyncio
async def test_sqlite_ledger_rejects_append_after_tail_tampering(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    ledger = MemoryLedger(SQLiteAsyncMemoryLedgerPersistence(str(db_path)))
    await ledger.append_memory_event({"value": "first"})
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "UPDATE dadbot_memory_list SET value = replace(value, ?, ?)",
            ('"value":"first"', '"value":"tampered"'),
        )

    assert not await ledger.verify_chain()
    with pytest.raises(ValueError, match="invalid hash"):
        await ledger.append_memory_event({"value": "second"})
