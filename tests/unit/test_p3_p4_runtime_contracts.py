from __future__ import annotations

import pytest

from dadbot.core.canonical_event import canonicalize_event_payload
from dadbot.core.determinism import DeterminismBoundary, DeterminismMode, DeterminismViolation
from dadbot.core.execution_context import (
    build_external_system_call_graph,
    ensure_execution_trace_root,
    record_external_system_call,
)
from dadbot.core.state_lineage import canonical_state_hash

pytestmark = pytest.mark.unit


def test_memory_confluence_same_entries_different_order_same_canonical_hash(bot) -> None:
    entry_a = {
        "summary": "Tony has been using direct planning checklists.",
        "category": "preferences",
        "mood": "positive",
        "updated_at": "2026-05-10",
    }
    entry_b = {
        "summary": "Tony is saving for emergency expenses.",
        "category": "finance",
        "mood": "neutral",
        "updated_at": "2026-05-10",
    }

    bot.MEMORY_STORE = bot.default_memory_store()
    bot.mutate_memory_store(memories=[entry_a, entry_b])
    hash_ab = canonical_state_hash(dict(bot.memory.memory_store or {}))

    bot.MEMORY_STORE = bot.default_memory_store()
    bot.mutate_memory_store(memories=[entry_b, entry_a])
    hash_ba = canonical_state_hash(dict(bot.memory.memory_store or {}))

    assert hash_ab == hash_ba


def test_memory_mutation_records_canonical_state_hash_observability(bot) -> None:
    bot.MEMORY_STORE = bot.default_memory_store()
    bot.mutate_memory_store(
        memories=[
            {
                "summary": "Tony has been tracking recurring expenses.",
                "category": "finance",
                "mood": "neutral",
                "updated_at": "2026-05-10",
            }
        ],
    )

    assert isinstance(getattr(bot, "_last_memory_state_hash", ""), str)
    assert len(str(bot._last_memory_state_hash)) == 64


def test_memory_confluence_total_order_on_sort_key_ties(bot) -> None:
    # Same updated_at/created_at/summary; category/mood/hash must break ties stably.
    entry_a = {
        "summary": "Tony has been planning meals.",
        "category": "home",
        "mood": "neutral",
        "updated_at": "2026-05-10",
        "created_at": "2026-05-10",
    }
    entry_b = {
        "summary": "Tony has been planning meals.",
        "category": "health",
        "mood": "positive",
        "updated_at": "2026-05-10",
        "created_at": "2026-05-10",
    }

    bot.MEMORY_STORE = bot.default_memory_store()
    bot.mutate_memory_store(memories=[entry_a, entry_b])
    store_ab = dict(bot.memory.memory_store or {})
    hash_ab = canonical_state_hash(store_ab)

    bot.MEMORY_STORE = bot.default_memory_store()
    bot.mutate_memory_store(memories=[entry_b, entry_a])
    store_ba = dict(bot.memory.memory_store or {})
    hash_ba = canonical_state_hash(store_ba)

    assert hash_ab == hash_ba
    assert list(store_ab.get("memories") or []) == list(store_ba.get("memories") or [])


def test_memory_confluence_same_logical_turn_repeats_with_identical_hash(bot) -> None:
    entry_a = {
        "summary": "Tony has been planning meals.",
        "category": "home",
        "mood": "neutral",
        "updated_at": "2026-05-10",
        "created_at": "2026-05-10",
    }
    entry_b = {
        "summary": "Tony has been planning meals.",
        "category": "health",
        "mood": "positive",
        "updated_at": "2026-05-10",
        "created_at": "2026-05-10",
    }

    observed_hashes: list[str] = []
    observed_memories: list[list[dict]] = []

    for _ in range(10):
        bot.MEMORY_STORE = bot.default_memory_store()
        # Same logical input each run; determinism requires identical output hash.
        bot.mutate_memory_store(memories=[entry_b, entry_a])
        snapshot = dict(bot.memory.memory_store or {})
        observed_hashes.append(canonical_state_hash(snapshot))
        observed_memories.append(list(snapshot.get("memories") or []))

    assert len(set(observed_hashes)) == 1
    assert all(memories == observed_memories[0] for memories in observed_memories)


def test_determinism_boundary_replay_requires_sealed_value_failure_semantics() -> None:
    boundary = DeterminismBoundary(mode=DeterminismMode.REPLAY)

    with pytest.raises(DeterminismViolation):
        boundary.capture("external.http.get", lambda: {"ok": True})

    assert boundary.violations
    assert boundary.violations[-1]["slot"] == "external.http.get"


def test_external_system_call_graph_is_stable_and_observable() -> None:
    with ensure_execution_trace_root(
        operation="p3_p4_external_call_graph_test",
        prompt="[unit-test]",
        metadata={"source": "tests.unit.test_p3_p4_runtime_contracts"},
        required=True,
    ):
        step1 = record_external_system_call(
            operation="http_get",
            system="weather_api",
            request_payload={"city": "Austin", "units": "metric"},
            response_payload={"temp_c": 24, "condition": "clear"},
            deterministic_id="weather:city:austin",
        )
        step2 = record_external_system_call(
            operation="http_get",
            system="weather_api",
            request_payload={"city": "Austin", "units": "metric"},
            response_payload={"temp_c": 24, "condition": "clear"},
            deterministic_id="weather:city:austin",
        )

    assert step1 is not None and step2 is not None
    p1 = dict(step1.get("payload") or {})
    p2 = dict(step2.get("payload") or {})

    # Same canonical request should produce stable request hash/time token.
    assert p1.get("request_hash") == p2.get("request_hash")
    assert p1.get("time_token") == p2.get("time_token")

    graph1 = build_external_system_call_graph([step1, step2])
    graph2 = build_external_system_call_graph([step1, step2])
    assert graph1.get("graph_hash") == graph2.get("graph_hash")
    assert len(list(graph1.get("nodes") or [])) == 2


def test_memory_state_canonicalized_event_payload_matches_before_after_hashes(bot) -> None:
    bot.MEMORY_STORE = bot.default_memory_store()
    before_store = dict(bot.memory.memory_store or {})
    expected_before_hash = canonical_state_hash(before_store)

    with ensure_execution_trace_root(
        operation="p3_p4_memory_hash_payload",
        prompt="[unit-test]",
        metadata={"source": "tests.unit.test_p3_p4_runtime_contracts"},
        required=True,
    ) as recorder:
        bot.mutate_memory_store(
            memories=[
                {
                    "summary": "Tony has been comparing grocery prices.",
                    "category": "finance",
                    "mood": "neutral",
                    "updated_at": "2026-05-10",
                }
            ],
        )
        steps = list(recorder.steps)

    event_steps = [s for s in steps if str(s.get("operation") or "") == "memory_state_canonicalized"]
    assert event_steps
    payload = dict(event_steps[-1].get("payload") or {})

    expected_after_hash = canonical_state_hash(dict(bot.memory.memory_store or {}))
    assert payload.get("before_hash") == expected_before_hash
    assert payload.get("after_hash") == expected_after_hash
    assert payload.get("changed") is True


def test_external_system_call_payload_contains_normalized_deterministic_metadata() -> None:
    with ensure_execution_trace_root(
        operation="p3_p4_external_metadata",
        prompt="[unit-test]",
        metadata={"source": "tests.unit.test_p3_p4_runtime_contracts"},
        required=True,
    ):
        step = record_external_system_call(
            operation="HTTP_GET",
            system="Weather_API",
            request_payload={"city": "Austin", "units": "metric"},
            response_payload={"temp_c": 24, "condition": "clear"},
            status="OK",
            deterministic_id="det:austin:weather",
        )

    assert step is not None
    payload = dict(step.get("payload") or {})
    assert payload.get("operation") == "http_get"
    assert payload.get("system") == "weather_api"
    assert payload.get("status") == "ok"
    assert payload.get("deterministic_id") == "det:austin:weather"


def test_canonical_payload_strips_non_deterministic_fields() -> None:
    payload = {
        "summary": "stable",
        "submitted_at": "2026-05-10T12:00:00",
        "request_id": "abc-123",
        "nested": {
            "updated_at": "2026-05-10T12:00:01",
            "value": 1,
        },
    }
    canonical = canonicalize_event_payload(payload)
    assert "submitted_at" not in canonical
    assert "request_id" not in canonical
    assert "updated_at" not in dict(canonical.get("nested") or {})
    assert canonical.get("summary") == "stable"
