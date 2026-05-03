#!/usr/bin/env python3
"""
Shadow Mode Report
==================
Reads session_logs/shadow_mode.jsonl and produces a minimal
divergence analysis report.

Usage:
    python tools/shadow_mode_report.py
    python tools/shadow_mode_report.py --top 20
    python tools/shadow_mode_report.py --log path/to/shadow_mode.jsonl

Output goes to stdout.  No external dependencies — stdlib only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "session_logs" / "shadow_mode.jsonl"


def _load_records(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    records = []
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100 * numerator / denominator:.1f}%"


def report(log_path: Path, top_n: int) -> str:
    records = _load_records(log_path)
    total = len(records)

    if total == 0:
        return (
            "================================================================\n"
            "  SHADOW MODE REPORT\n"
            "================================================================\n"
            "  No records found.\n"
            f"  Expected log: {log_path}\n"
            "================================================================\n"
        )

    # --- core counters -------------------------------------------------------
    pass_count = sum(1 for r in records if r.get("result") == "pass")
    fail_count = total - pass_count

    # primary bucket counts (diverged records only)
    bucket_counts: Counter[str] = Counter()
    detail_counts: Counter[str] = Counter()
    unknown_count = 0  # diverged but detail is None/missing

    for r in records:
        dtype = r.get("divergence_type") or "unknown"
        detail = r.get("divergence_detail")
        if dtype == "none":
            continue
        bucket_counts[dtype] += 1
        if detail:
            detail_counts[f"{dtype}:{detail}"] += 1
        else:
            unknown_count += 1

    # --- input_hash clustering -----------------------------------------------
    # For each input_hash, count total appearances and fail appearances
    hash_total: Counter[str] = Counter()
    hash_fail: Counter[str] = Counter()
    for r in records:
        ih = str(r.get("input_hash") or "missing")
        hash_total[ih] += 1
        if r.get("result") != "pass":
            hash_fail[ih] += 1

    # Top N input hashes by absolute divergence count (only hashes with ≥1 fail)
    top_hashes = sorted(
        [(h, hash_fail[h], hash_total[h]) for h in hash_fail if hash_fail[h] > 0],
        key=lambda x: (-x[1], -x[2]),
    )[:top_n]

    # --- format ---------------------------------------------------------------
    lines = [
        "================================================================",
        "  SHADOW MODE REPORT",
        f"  Log: {log_path}",
        "================================================================",
        "",
        "SAMPLE OVERVIEW",
        "----------------------------------------------------------------",
        f"  total_samples    : {total}",
        f"  match_rate       : {_pct(pass_count, total)}  ({pass_count}/{total})",
        f"  divergence_rate  : {_pct(fail_count, total)}  ({fail_count}/{total})",
        "",
        "DIVERGENCE BY PRIMARY BUCKET",
        "----------------------------------------------------------------",
    ]

    if fail_count == 0:
        lines.append("  (no divergences recorded)")
    else:
        for bucket, count in bucket_counts.most_common():
            lines.append(f"  {bucket:<20} {_pct(count, fail_count):>6}  ({count})")

    lines += [
        "",
        "DIVERGENCE BY DETAIL TAG",
        "----------------------------------------------------------------",
    ]

    if not detail_counts:
        lines.append("  (no detail tags recorded yet)")
    else:
        for tag, count in detail_counts.most_common():
            lines.append(f"  {tag:<36} {_pct(count, fail_count):>6}  ({count})")

    lines += [
        "",
        "UNKNOWN DIVERGENCE RATE",
        "----------------------------------------------------------------",
        f"  diverged with no detail tag : {_pct(unknown_count, fail_count)}  ({unknown_count}/{fail_count})",
        "  (high unknown = classifier lacks visibility, not system failure)",
        "",
        f"TOP {top_n} INPUT HASHES BY DIVERGENCE COUNT",
        "----------------------------------------------------------------",
    ]

    if not top_hashes:
        lines.append("  (no divergences to cluster yet)")
    else:
        lines.append(f"  {'input_hash':<18}  {'fails':>5}  {'total':>5}  {'fail%':>6}")
        lines.append(f"  {'-'*18}  {'-'*5}  {'-'*5}  {'-'*6}")
        for ih, fails, tot in top_hashes:
            lines.append(
                f"  {ih[:18]:<18}  {fails:>5}  {tot:>5}  {_pct(fails, tot):>6}"
            )

    lines += [
        "",
        "================================================================",
        "  Recommended thresholds:",
        "    ≥99%  match  → extremely stable",
        "    97-99% match → healthy",
        "    94-97% match → acceptable, classify causes",
        "    <94%  match  → real issues exist",
        "  Do NOT add normalization rules until ≥300 samples collected.",
        "================================================================",
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow Mode divergence report")
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help=f"Path to shadow_mode.jsonl (default: {DEFAULT_LOG})",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top input hashes to show (default: 10)",
    )
    args = parser.parse_args()
    print(report(args.log, args.top))


if __name__ == "__main__":
    main()
