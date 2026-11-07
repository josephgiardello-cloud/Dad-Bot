from __future__ import annotations

import json
from typing import Any

import pytest

from dadbot.services.vector_memory import SovereignMemory

pytestmark = pytest.mark.unit


class _CollectionStub:
    def __init__(self) -> None:
        self.query_kwargs: dict[str, Any] = {}
        self.get_kwargs: dict[str, Any] = {}
        self.deleted_ids: list[str] = []
        self.query_error: Exception | None = None
        self.query_result: dict[str, Any] | None = None

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.query_kwargs = dict(kwargs)
        if self.query_error is not None:
            raise self.query_error
        if self.query_result is not None:
            return self.query_result
        return {
            "documents": [["The dog's name is Juniper."]],
            "metadatas": [[
                {
                    "event_id": "fact-pet",
                    "event_type": "FACT",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                },
            ]],
            "distances": [[0.1]],
        }

    def get(self, **kwargs: Any) -> dict[str, list[str]]:
        self.get_kwargs = dict(kwargs)
        return {"ids": ["old-event-1", "old-event-2"]}

    def delete(self, *, ids: list[str]) -> None:
        self.deleted_ids = list(ids)


def _memory_with_collection(collection: _CollectionStub) -> SovereignMemory:
    memory = SovereignMemory.__new__(SovereignMemory)
    memory._initialized = True
    memory._collection = collection
    return memory


def test_retrieve_context_builds_valid_combined_metadata_filter():
    collection = _CollectionStub()
    memory = _memory_with_collection(collection)

    results = memory.retrieve_context(
        "What is my dog's name?",
        limit=2,
        time_window_days=30,
        event_type_filter="FACT",
    )

    where = collection.query_kwargs["where"]
    assert set(where) == {"$and"}
    assert isinstance(where["$and"][0]["timestamp_epoch"]["$gte"], float)
    assert where["$and"][1] == {"event_type": {"$eq": "FACT"}}
    assert results[0].event_id == "fact-pet"
    assert "Juniper" in results[0].content


def test_retrieve_context_reranks_by_fact_topic_over_entity_name():
    collection = _CollectionStub()
    collection.query_result = {
        "documents": [[
            json.dumps({"summary": "My dog's name is Juniper."}),
            json.dumps({"summary": "My dog's breed is a beagle."}),
        ]],
        "metadatas": [[
            {"event_id": "fact-name", "event_type": "FACT", "timestamp": ""},
            {"event_id": "fact-breed", "event_type": "FACT", "timestamp": ""},
        ]],
        "distances": [[1.3582, 1.7288]],
    }
    memory = _memory_with_collection(collection)

    results = memory.retrieve_context("What breed is Juniper?", limit=1)

    assert results[0].event_id == "fact-breed"
    assert collection.query_kwargs["n_results"] > 1


def test_retrieve_context_includes_relevant_results_above_old_distance_cutoff():
    collection = _CollectionStub()
    collection.query_result = {
        "documents": [[json.dumps({"summary": "My cat's favorite nap spot is the sunny window."})]],
        "metadatas": [[{"event_id": "fact-nap", "event_type": "FACT", "timestamp": ""}]],
        "distances": [[1.5537]],
    }
    memory = _memory_with_collection(collection)

    results = memory.retrieve_context("Where does Luna like to nap?", limit=1)

    assert results[0].event_id == "fact-nap"


def test_retrieve_context_builds_time_only_filter_without_empty_clause():
    collection = _CollectionStub()
    memory = _memory_with_collection(collection)

    memory.retrieve_context("What is my dog's name?", time_window_days=30)

    where = collection.query_kwargs["where"]
    assert set(where) == {"timestamp_epoch"}
    assert isinstance(where["timestamp_epoch"]["$gte"], float)


def test_retrieve_context_rejects_negative_time_window():
    memory = _memory_with_collection(_CollectionStub())

    with pytest.raises(ValueError, match="time_window_days"):
        memory.retrieve_context("Recall the fact", time_window_days=-1)


def test_retrieve_context_surfaces_backend_errors_in_production(monkeypatch):
    collection = _CollectionStub()
    collection.query_error = RuntimeError("vector index unavailable")
    memory = _memory_with_collection(collection)
    monkeypatch.setenv("DADBOT_ENVIRONMENT", "production")

    with pytest.raises(RuntimeError, match="vector index unavailable"):
        memory.retrieve_context("Recall the fact")


def test_delete_old_events_deletes_records_before_cutoff():
    collection = _CollectionStub()
    memory = _memory_with_collection(collection)

    deleted = memory.delete_old_events(before_days=30)

    assert deleted == 2
    assert isinstance(collection.get_kwargs["where"]["timestamp_epoch"]["$lt"], float)
    assert collection.deleted_ids == ["old-event-1", "old-event-2"]


def test_delete_old_events_rejects_negative_retention():
    memory = _memory_with_collection(_CollectionStub())

    with pytest.raises(ValueError, match="before_days"):
        memory.delete_old_events(before_days=-1)


def test_backfill_adds_numeric_timestamp_to_legacy_records():
    class _LegacyCollection:
        def __init__(self) -> None:
            self.updated: tuple[list[str], list[dict[str, Any]]] | None = None

        def count(self) -> int:
            return 1

        def get(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ids": ["legacy-event"],
                "metadatas": [{"timestamp": "2026-01-02T03:04:05+00:00", "event_type": "FACT"}],
            }

        def update(self, *, ids: list[str], metadatas: list[dict[str, Any]]) -> None:
            self.updated = (ids, metadatas)

    memory = SovereignMemory.__new__(SovereignMemory)
    collection = _LegacyCollection()
    memory._collection = collection

    memory._backfill_timestamp_metadata()

    assert collection.updated is not None
    ids, metadatas = collection.updated
    assert ids == ["legacy-event"]
    assert metadatas[0]["timestamp_epoch"] == 1767323045.0
