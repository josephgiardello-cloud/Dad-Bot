from __future__ import annotations

import pytest

from dadbot.core.execution_memory_view import merge_memory_retrieval_sets

pytestmark = pytest.mark.unit


def test_merge_memory_retrieval_sets_prefers_later_source_for_same_memory_key() -> None:
    pre_turn = [
        {"id": "m1", "summary": "remember the blue binder", "source": "pre_turn"},
        {"id": "m2", "summary": "keep the invoice", "source": "pre_turn"},
    ]
    post_tool = [
        {"id": "m1", "summary": "remember the red binder", "source": "post_tool"},
    ]

    merged, reconciliation = merge_memory_retrieval_sets(
        pre_turn,
        post_tool,
        source_labels=["pre_turn", "post_tool"],
    )

    assert [item["id"] for item in merged] == ["m1", "m2"]
    assert merged[0]["summary"] == "remember the red binder"
    assert merged[0]["memory_merge_source"] == "post_tool"
    assert int(reconciliation["conflict_count"]) == 1
    assert reconciliation["sources"][0]["label"] == "pre_turn"
    assert reconciliation["sources"][1]["label"] == "post_tool"


def test_merge_memory_retrieval_sets_keeps_distinct_entries_in_order() -> None:
    pre_turn = [{"summary": "first"}]
    scheduled = [{"summary": "second"}]

    merged, reconciliation = merge_memory_retrieval_sets(
        pre_turn,
        scheduled,
        source_labels=["pre_turn", "scheduled"],
    )

    assert [item["summary"] for item in merged] == ["first", "second"]
    assert int(reconciliation["merged_count"]) == 2
    assert int(reconciliation["conflict_count"]) == 0
