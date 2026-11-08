from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from tests.eval.memory_benchmark import MemoryBenchmarkCase
from tests.eval.qwen_reranker import gated_candidates, reranked_candidates
from tests.eval.reranker_experiment import evaluate_case

pytestmark = pytest.mark.unit


def test_gate_uses_undiversified_score_margin_and_five_distinct_candidates():
    bot = SimpleNamespace(memory_query=SimpleNamespace(memory_diversity_key=lambda memory: memory["summary"]))
    rows = [
        (score, 0.0, {"summary": f"Work memory {index}", "category": "work"}, 0.1)
        for index, score in enumerate((1.0, 0.9, 0.8, 0.7, 0.6, 0.5))
    ]
    rows.insert(1, rows[0])
    fired, gap, candidates = gated_candidates(bot, rows)
    assert fired
    assert gap == pytest.approx(0.075)
    assert [memory["summary"] for _, memory in candidates] == [f"Work memory {index}" for index in range(5)]


@pytest.mark.parametrize("dense_second,fired", [(0.801, True), (0.8, False), (0.7, False)])
def test_gate_boundary(dense_second, fired):
    bot = SimpleNamespace(memory_query=SimpleNamespace(memory_diversity_key=lambda memory: memory["summary"]))
    rows = [(1.0, 0.0, {"summary": "first"}, 0.2), (dense_second, 0.0, {"summary": "second"}, 0.99)]
    assert gated_candidates(bot, rows)[0] is fired


def test_reranker_stable_ties_and_invalid_output():
    candidates = [(0.9, {"summary": "A"}), (0.8, {"summary": "B"})]
    assert reranked_candidates(candidates, [0.4, 0.4]) == candidates
    assert reranked_candidates(candidates, [0.4, 0.9]) == list(reversed(candidates))
    for scores in ([math.nan, 0.3], [1.1, 0.3], [0.2], [math.inf, 0.3]):
        with pytest.raises(ValueError, match="finite probability"):
            reranked_candidates(candidates, scores)


def test_skipped_gate_does_not_call_model_or_change_baseline(bot):
    case = MemoryBenchmarkCase("dog's name", "My dog's name is Juniper.")
    rows = [
        (1.0, 1.0, {"summary": case.expected_summary}, 0.1),
        (0.1, 0.0, {"summary": "Other memory"}, 0.99),
    ]

    def fail(*_args):
        pytest.fail("Model must not run when gate is closed.")

    baseline = [case.expected_summary, "Other memory"]
    record = evaluate_case(bot, SimpleNamespace(score=fail), case, rows, baseline)
    assert not record["gate_fired"]
    assert record["added_seconds"] == 0.0
    assert record["reranked_top3"] == baseline


def test_model_can_promote_fifth_same_category_candidate(bot):
    case = MemoryBenchmarkCase("dog's name", "My dog's name is Juniper.")
    memories = [{"summary": f"Other pet {index}", "category": "pets"} for index in range(4)]
    memories.append({"summary": case.expected_summary, "category": "pets"})
    rows = [(1 - index * 0.01, 0.0, memory, 0.5) for index, memory in enumerate(memories)]
    received = []

    def scores(_query, documents):
        received.extend(documents)
        return [0.01, 0.02, 0.03, 0.04, 0.9]

    record = evaluate_case(
        bot, SimpleNamespace(score=scores), case, rows, [memory["summary"] for memory in memories[:3]]
    )
    assert len(received) == 5
    assert record["reranked_hits"] == (1, 1)
    assert record["reranked_top3"][0] == case.expected_summary
