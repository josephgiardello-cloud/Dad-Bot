from __future__ import annotations

import hashlib
import json
import math
import os
from statistics import mean, median
from time import perf_counter

from tests.eval.memory_benchmark import MEMORY_BENCHMARK_FACTS
from tests.eval.qwen_reranker import (
    CANDIDATE_LIMIT,
    MARGIN,
    MODEL_ID,
    MODEL_REVISION,
    LocalQwenReranker,
    gated_candidates,
    reranked_candidates,
)


def hit_counts(case, summaries):
    matches = [case.expected_summary.casefold() in summary.casefold() for summary in summaries[:3]]
    return int(bool(matches and matches[0])), int(any(matches))


def run_reranker_experiment(bot, cases, baseline_rankings):
    path = os.environ.get("DADBOT_QWEN_RERANKER_PATH")
    if not path:
        raise ValueError("Set DADBOT_QWEN_RERANKER_PATH to the downloaded local Qwen model.")
    reranker = LocalQwenReranker(path)
    start = perf_counter()
    reranker.score("Where is the telescope?", ["The telescope is in the cupboard."] * CANDIDATE_LIMIT)
    warmup_seconds = perf_counter() - start
    records = [
        evaluate_case(bot, reranker, case, rows, baseline)
        for (case, rows), baseline in zip(cases, baseline_rankings, strict=True)
    ]
    report = summarize_records(records)
    report.update(
        {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "device": "cpu",
            "dtype": "float32",
            "threads": 4,
            "margin": MARGIN,
            "candidate_limit": CANDIDATE_LIMIT,
            "load_seconds": reranker.load_seconds,
            "five_pair_warmup_seconds": warmup_seconds,
        }
    )
    print("Qwen reranker experiment: " + json.dumps(report, sort_keys=True))
    measure_confidence_gate(records)
    for record in records:
        if record["reranked_hits"][0] == 0 or record["baseline_hits"] != record["reranked_hits"]:
            print("Qwen query result: " + json.dumps(record, sort_keys=True))
    return report


def evaluate_case(bot, reranker, case, rows, baseline):
    fired, gap, candidates = gated_candidates(bot, rows)
    summaries = baseline
    elapsed = 0.0
    scores = []
    if fired:
        start = perf_counter()
        scores = reranker.score(case.question, [memory["summary"] for _, memory in candidates])
        ranked = reranked_candidates(candidates, scores)
        summaries = [memory["summary"] for _, memory in bot.memory_query.select_diverse_ranked_memories(ranked, 3)]
        elapsed = perf_counter() - start
    topics = {fact.summary: index for index, fact in enumerate(MEMORY_BENCHMARK_FACTS)}
    return {
        "query": case.question,
        "expected": case.expected_summary,
        "split": "held_out" if topics[case.expected_summary] % 5 == 0 else "train",
        "gate_fired": fired,
        "fused_top1": candidates[0][0] if candidates else 0.0,
        "fused_margin": gap,
        "added_seconds": elapsed,
        "baseline_hits": hit_counts(case, baseline),
        "reranked_hits": hit_counts(case, summaries),
        "reranked_top3": summaries,
        "expected_in_top5": any(
            case.expected_summary.casefold() in memory["summary"].casefold() for _, memory in candidates
        ),
        "candidate_scores": [[memory["summary"], score] for (_, memory), score in zip(candidates, scores, strict=True)]
        if fired
        else [],
    }


def summarize_records(records):
    added = [record["added_seconds"] for record in records]
    fired = [record["added_seconds"] for record in records if record["gate_fired"]]
    report = {
        "questions": len(records),
        "baseline_top1": sum(record["baseline_hits"][0] for record in records),
        "baseline_top3": sum(record["baseline_hits"][1] for record in records),
        "reranked_top1": sum(record["reranked_hits"][0] for record in records),
        "reranked_top3": sum(record["reranked_hits"][1] for record in records),
        "gate_fires": len(fired),
        "gate_rate": len(fired) / len(records),
        "mean_added_seconds": mean(added),
        "median_added_seconds": median(added),
        "p95_added_seconds": sorted(added)[math.ceil(0.95 * len(added)) - 1],
        "median_gate_seconds": median(fired) if fired else 0.0,
        "p95_gate_seconds": sorted(fired)[math.ceil(0.95 * len(fired)) - 1] if fired else 0.0,
        "top1_improvements": sum(record["reranked_hits"][0] > record["baseline_hits"][0] for record in records),
        "top1_regressions": sum(record["reranked_hits"][0] < record["baseline_hits"][0] for record in records),
        "ranking_fingerprint": hashlib.sha256(
            json.dumps([record["reranked_top3"] for record in records], ensure_ascii=True).encode("ascii")
        ).hexdigest(),
        "meets_top1_target": sum(record["reranked_hits"][0] for record in records) >= 45,
    }
    for split in ("train", "held_out"):
        subset = [record for record in records if record["split"] == split]
        report[split] = {
            "questions": len(subset),
            "top1": sum(record["reranked_hits"][0] for record in subset),
            "top3": sum(record["reranked_hits"][1] for record in subset),
        }
    return report


def measure_confidence_gate(records):
    correct = [record["fused_top1"] for record in records if record["baseline_hits"][0]]
    print("Correct-case fused top1 scores: " + json.dumps(sorted(correct)))
    for threshold in (0.8, 0.85, 0.9, 0.95, 1.0):
        hits = [
            record["reranked_hits"]
            if record["gate_fired"] and record["fused_top1"] < threshold
            else record["baseline_hits"]
            for record in records
        ]
        fired = [record for record in records if record["gate_fired"] and record["fused_top1"] < threshold]
        print(
            "Two-dimensional gate: "
            + json.dumps(
                {
                    "top1_threshold": threshold,
                    "top1": sum(row[0] for row in hits),
                    "top3": sum(row[1] for row in hits),
                    "fires": len(fired),
                    "saved_model_calls": sum(record["gate_fired"] for record in records) - len(fired),
                    "mean_added_seconds": sum(record["added_seconds"] for record in fired) / len(records),
                    "train_top1": sum(
                        row[0] for row, record in zip(hits, records, strict=True) if record["split"] == "train"
                    ),
                    "held_out_top1": sum(
                        row[0] for row, record in zip(hits, records, strict=True) if record["split"] == "held_out"
                    ),
                },
                sort_keys=True,
            )
        )
