"""Tests for tool IO recording infrastructure.

Validates ToolIORecord and ToolIOLedger data structures:
- Immutability of ToolIORecord
- Fast lookup by (tool_name, input_hash)
- Checkpoint serialization round-trip
- Determinism of hashes
"""

from __future__ import annotations

import pytest

from dadbot.core.tool_recording import ToolIOLedger, ToolIORecord

pytestmark = pytest.mark.unit


class TestToolIORecord:
    """ToolIORecord immutability and basic operations."""

    def test_creation_basic(self) -> None:
        record = ToolIORecord(
            sequence=1,
            tool_name="set_reminder",
            input_hash="abc123",
            input_payload={"title": "Test reminder", "due_text": "tomorrow"},
            output_payload={"id": "r1", "title": "Test reminder"},
            output_hash="def456",
            status="ok",
            latency_ms=42.5,
        )
        assert record.sequence == 1
        assert record.tool_name == "set_reminder"
        assert record.status == "ok"
        assert record.latency_ms == 42.5

    def test_is_frozen(self) -> None:
        record = ToolIORecord(
            sequence=1,
            tool_name="web_search",
            input_hash="abc123",
            input_payload={"query": "coffee"},
            output_payload={"results": []},
            output_hash="def456",
            status="ok",
            latency_ms=100.0,
        )
        # Attempt to mutate should fail
        with pytest.raises(AttributeError):
            record.status = "error"  # type: ignore

    def test_is_success(self) -> None:
        success_statuses = ["ok", "partial", "cached", "replayed"]
        for status in success_statuses:
            record = ToolIORecord(
                sequence=1,
                tool_name="set_reminder",
                input_hash="x",
                input_payload={},
                output_payload={},
                output_hash="y",
                status=status,
                latency_ms=1.0,
            )
            assert record.is_success(), f"Status '{status}' should be success"

        error_statuses = ["error", "timeout", "degraded"]
        for status in error_statuses:
            record = ToolIORecord(
                sequence=1,
                tool_name="set_reminder",
                input_hash="x",
                input_payload={},
                output_payload={},
                output_hash="y",
                status=status,
                latency_ms=1.0,
            )
            assert not record.is_success(), f"Status '{status}' should not be success"

    def test_is_replay_hit(self) -> None:
        live_record = ToolIORecord(
            sequence=1,
            tool_name="web_search",
            input_hash="x",
            input_payload={},
            output_payload={},
            output_hash="y",
            status="ok",
            latency_ms=50.0,
        )
        assert not live_record.is_replay_hit()

        replay_record = ToolIORecord(
            sequence=1,
            tool_name="web_search",
            input_hash="x",
            input_payload={},
            output_payload={},
            output_hash="y",
            status="replayed",
            latency_ms=50.0,
        )
        assert replay_record.is_replay_hit()

    def test_matches_request(self) -> None:
        record = ToolIORecord(
            sequence=1,
            tool_name="set_reminder",
            input_hash="hash123",
            input_payload={"title": "Test"},
            output_payload={"id": "r1"},
            output_hash="out_hash",
            status="ok",
            latency_ms=1.0,
        )

        # Exact match
        assert record.matches_request("set_reminder", "hash123")

        # Name mismatch
        assert not record.matches_request("web_search", "hash123")

        # Hash mismatch
        assert not record.matches_request("set_reminder", "different_hash")

        # Both mismatch
        assert not record.matches_request("web_search", "different_hash")

    def test_lookup_key(self) -> None:
        record = ToolIORecord(
            sequence=1,
            tool_name="set_reminder",
            input_hash="abc123def456",
            input_payload={},
            output_payload={},
            output_hash="xyz",
            status="ok",
            latency_ms=1.0,
        )
        assert record.lookup_key() == "set_reminder|abc123def456||"

    def test_with_metadata(self) -> None:
        record = ToolIORecord(
            sequence=1,
            tool_name="set_reminder",
            input_hash="x",
            input_payload={},
            output_payload={},
            output_hash="y",
            status="ok",
            latency_ms=1.0,
            error="",
            metadata={"retry_count": 1, "permission": "granted"},
        )
        assert record.metadata["retry_count"] == 1
        assert record.metadata["permission"] == "granted"


class TestToolIOLedger:
    """ToolIOLedger sequential storage and fast lookup."""

    def test_empty_ledger(self) -> None:
        ledger = ToolIOLedger()
        assert ledger.is_empty()
        assert len(ledger.records) == 0
        assert ledger.last_record() is None
        assert ledger.successful_calls() == 0
        assert ledger.failed_calls() == 0

    def test_append_single(self) -> None:
        ledger = ToolIOLedger()
        record = ToolIORecord(
            sequence=1,
            tool_name="set_reminder",
            input_hash="h1",
            input_payload={"title": "Test"},
            output_payload={"id": "r1"},
            output_hash="out1",
            status="ok",
            latency_ms=10.0,
        )
        ledger.append(record)

        assert not ledger.is_empty()
        assert len(ledger.records) == 1
        assert ledger.last_record() == record
        assert ledger.successful_calls() == 1
        assert ledger.failed_calls() == 0

    def test_append_multiple(self) -> None:
        ledger = ToolIOLedger()
        records = [
            ToolIORecord(
                sequence=i,
                tool_name="set_reminder" if i % 2 == 0 else "web_search",
                input_hash=f"h{i}",
                input_payload={},
                output_payload={},
                output_hash=f"o{i}",
                status="ok" if i % 3 != 0 else "error",
                latency_ms=float(i * 10),
            )
            for i in range(1, 6)
        ]
        for rec in records:
            ledger.append(rec)

        assert len(ledger.records) == 5
        assert ledger.records == records
        assert ledger.successful_calls() == 4  # i=1,2,4,5 succeed
        assert ledger.failed_calls() == 1  # i=3 fails

    def test_lookup_after_append(self) -> None:
        ledger = ToolIOLedger()
        record1 = ToolIORecord(
            sequence=1,
            tool_name="set_reminder",
            input_hash="reminder_h1",
            input_payload={"title": "Reminder 1"},
            output_payload={"id": "r1"},
            output_hash="o1",
            status="ok",
            latency_ms=10.0,
        )
        record2 = ToolIORecord(
            sequence=2,
            tool_name="web_search",
            input_hash="search_h1",
            input_payload={"query": "coffee"},
            output_payload={"results": 42},
            output_hash="o2",
            status="ok",
            latency_ms=50.0,
        )
        ledger.append(record1)
        ledger.append(record2)

        # Lookup by tool_name + input_hash
        found1 = ledger.lookup("set_reminder", "reminder_h1")
        assert found1 == record1

        found2 = ledger.lookup("web_search", "search_h1")
        assert found2 == record2

        # Miss: wrong tool_name
        assert ledger.lookup("set_reminder", "search_h1") is None

        # Miss: wrong input_hash
        assert ledger.lookup("set_reminder", "wrong_hash") is None

        # Miss: tool not in ledger
        assert ledger.lookup("unknown_tool", "reminder_h1") is None

    def test_append_invalid_type(self) -> None:
        ledger = ToolIOLedger()
        with pytest.raises(TypeError):
            ledger.append({"not": "a record"})  # type: ignore

    def test_all_records_returns_copy_order(self) -> None:
        ledger = ToolIOLedger()
        records = [
            ToolIORecord(
                sequence=i,
                tool_name="tool",
                input_hash=f"h{i}",
                input_payload={},
                output_payload={},
                output_hash=f"o{i}",
                status="ok",
                latency_ms=1.0,
            )
            for i in range(1, 4)
        ]
        for rec in records:
            ledger.append(rec)

        all_recs = ledger.all_records()
        assert all_recs == records
        assert all_recs is not records  # Should be a copy/list, not the original


class TestToolIOLedgerSerialization:
    """Checkpoint persistence (serialization/deserialization)."""

    def test_to_dict_empty(self) -> None:
        ledger = ToolIOLedger()
        data = ledger.to_dict()
        assert data["records"] == []
        assert data["sequence_count"] == 0
        assert data["successful_count"] == 0
        assert data["failed_count"] == 0

    def test_to_dict_with_records(self) -> None:
        ledger = ToolIOLedger()
        record = ToolIORecord(
            sequence=1,
            tool_name="set_reminder",
            input_hash="h1",
            input_payload={"title": "Test"},
            output_payload={"id": "r1"},
            output_hash="o1",
            status="ok",
            latency_ms=10.0,
            metadata={"retry": 1},
        )
        ledger.append(record)

        data = ledger.to_dict()
        assert len(data["records"]) == 1
        assert data["records"][0]["tool_name"] == "set_reminder"
        assert data["records"][0]["status"] == "ok"
        assert data["records"][0]["metadata"]["retry"] == 1
        assert data["sequence_count"] == 1
        assert data["successful_count"] == 1
        assert data["failed_count"] == 0

    def test_from_dict_empty(self) -> None:
        ledger = ToolIOLedger.from_dict({})
        assert ledger.is_empty()

        ledger2 = ToolIOLedger.from_dict({"records": []})
        assert ledger2.is_empty()

    def test_from_dict_with_records(self) -> None:
        data = {
            "records": [
                {
                    "sequence": 1,
                    "tool_name": "set_reminder",
                    "input_hash": "h1",
                    "input_payload": {"title": "Test"},
                    "output_payload": {"id": "r1"},
                    "output_hash": "o1",
                    "status": "ok",
                    "latency_ms": 10.0,
                    "error": "",
                    "metadata": {},
                },
            ],
            "sequence_count": 1,
            "successful_count": 1,
            "failed_count": 0,
        }
        ledger = ToolIOLedger.from_dict(data)

        assert not ledger.is_empty()
        assert len(ledger.records) == 1
        assert ledger.last_record().tool_name == "set_reminder"
        assert ledger.lookup("set_reminder", "h1") is not None

    def test_round_trip_serialization(self) -> None:
        # Create original ledger
        original_ledger = ToolIOLedger()
        records = [
            ToolIORecord(
                sequence=1,
                tool_name="set_reminder",
                input_hash="h1",
                input_payload={"title": "Test 1"},
                output_payload={"id": "r1"},
                output_hash="o1",
                status="ok",
                latency_ms=10.0,
            ),
            ToolIORecord(
                sequence=2,
                tool_name="web_search",
                input_hash="h2",
                input_payload={"query": "coffee"},
                output_payload={"results": []},
                output_hash="o2",
                status="ok",
                latency_ms=50.0,
            ),
        ]
        for rec in records:
            original_ledger.append(rec)

        # Serialize
        data = original_ledger.to_dict()

        # Deserialize
        restored_ledger = ToolIOLedger.from_dict(data)

        # Verify
        assert len(restored_ledger.records) == 2
        assert restored_ledger.successful_calls() == 2
        assert restored_ledger.lookup("set_reminder", "h1") is not None
        assert restored_ledger.lookup("web_search", "h2") is not None
        assert restored_ledger.records[0].tool_name == "set_reminder"
        assert restored_ledger.records[1].tool_name == "web_search"

    def test_from_dict_skips_bad_records(self) -> None:
        data = {
            "records": [
                {
                    "sequence": 1,
                    "tool_name": "set_reminder",
                    "input_hash": "h1",
                    "input_payload": {},
                    "output_payload": {},
                    "output_hash": "o1",
                    "status": "ok",
                    "latency_ms": 10.0,
                    # Missing error, metadata
                },
                {
                    # Missing required fields
                    "sequence": 2,
                    "tool_name": "web_search",
                },
            ],
        }
        ledger = ToolIOLedger.from_dict(data)
        # Should only have the valid record
        assert len(ledger.records) == 1
        assert ledger.records[0].tool_name == "set_reminder"

    def test_from_dict_handles_non_dict_input(self) -> None:
        ledger = ToolIOLedger.from_dict(None)  # type: ignore
        assert ledger.is_empty()

        ledger2 = ToolIOLedger.from_dict([])  # type: ignore
        assert ledger2.is_empty()

    def test_from_dict_handles_bad_records_list(self) -> None:
        data = {
            "records": "not_a_list",  # Invalid
        }
        ledger = ToolIOLedger.from_dict(data)
        assert ledger.is_empty()


class TestToolIOLedgerClear:
    """Clear operation for resetting between turns."""

    def test_clear_empties_ledger(self) -> None:
        ledger = ToolIOLedger()
        for i in range(3):
            record = ToolIORecord(
                sequence=i + 1,
                tool_name="tool",
                input_hash=f"h{i}",
                input_payload={},
                output_payload={},
                output_hash=f"o{i}",
                status="ok",
                latency_ms=1.0,
            )
            ledger.append(record)

        assert len(ledger.records) == 3
        ledger.clear()
        assert len(ledger.records) == 0
        assert ledger.is_empty()
        assert ledger._by_request_hash == {}


class TestToolIOLedgerStatistics:
    """Statistics and counting methods."""

    def test_statistics_with_mixed_statuses(self) -> None:
        ledger = ToolIOLedger()
        statuses = ["ok", "ok", "error", "partial", "replayed", "error"]
        for i, status in enumerate(statuses):
            record = ToolIORecord(
                sequence=i + 1,
                tool_name="tool",
                input_hash=f"h{i}",
                input_payload={},
                output_payload={},
                output_hash=f"o{i}",
                status=status,
                latency_ms=1.0,
            )
            ledger.append(record)

        assert ledger.successful_calls() == 4  # ok, ok, partial, replayed
        assert ledger.failed_calls() == 2  # error, error
        assert len(ledger.all_records()) == 6
