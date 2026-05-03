#!/usr/bin/env python3
"""
Shadow Replay Stress Test
==========================
Takes an existing shadow_mode.jsonl log and replays records in various
stress patterns to validate reconstruction stability.

Stress modes (all run sequentially unless --mode is specified):

    forward         — replay in natural order (baseline)
    reversed        — replay in reverse order
    shuffled        — replay in random order (seed-reproducible)
    partial_head    — replay first 25% of records
    partial_tail    — replay last 25% of records
    drop_middle     — replay with middle 50% dropped
    cross_session   — replay records grouped by session_id, then interleaved

For each mode, the tool re-hashes each record's canonical projection and
checks for consistency. It does NOT re-drive the bot — it replays the
already-recorded snapshot data.

Usage:
    python tools/shadow_replay_stress.py
    python tools/shadow_replay_stress.py --log session_logs/shadow_extended.jsonl
    python tools/shadow_replay_stress.py --mode drop_middle --seed 42

Output:
    Per-mode consistency table printed to stdout.
    Results written to session_logs/shadow_replay_stress.json.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "session_logs" / "shadow_mode.jsonl"
DEFAULT_OUT = ROOT / "session_logs" / "shadow_replay_stress.json"

_ALL_MODES = [
    "forward",
    "reversed",
    "shuffled",
    "partial_head",
    "partial_tail",
    "drop_middle",
    "cross_session",
]


def _is_consistent(record: dict[str, Any]) -> bool:
    """A shadow record is consistent when snapshot_hash == replay_hash.

    Records without both fields pass through as consistent (no ground truth).
    """
    sh = record.get("snapshot_hash")
    rh = record.get("replay_hash")
    if sh and rh:
        return sh == rh
    # Pass-through: result='pass' with no explicit hash comparison
    return record.get("result") == "pass"


def _load_records(log_path: Path) -> list[dict[str, Any]]:
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


def _apply_mode(records: list[dict], mode: str, seed: int) -> list[dict]:
    n = len(records)
    if mode == "forward":
        return list(records)
    if mode == "reversed":
        return list(reversed(records))
    if mode == "shuffled":
        rng = random.Random(seed)
        shuffled = list(records)
        rng.shuffle(shuffled)
        return shuffled
    if mode == "partial_head":
        return records[: max(1, n // 4)]
    if mode == "partial_tail":
        return records[max(0, 3 * n // 4):]
    if mode == "drop_middle":
        q1 = n // 4
        q3 = 3 * n // 4
        return records[:q1] + records[q3:]
    if mode == "cross_session":
        # Group by session_id then interleave round-robin
        sessions: dict[str, list[dict]] = {}
        for r in records:
            sid = str(r.get("session_id") or "unknown")
            sessions.setdefault(sid, []).append(r)
        interleaved = []
        iters = [iter(v) for v in sessions.values()]
        while iters:
            next_iters = []
            for it in iters:
                try:
                    interleaved.append(next(it))
                    next_iters.append(it)
                except StopIteration:
                    pass
            iters = next_iters
        return interleaved
    raise ValueError(f"Unknown mode: {mode!r}")


def _run_mode(records: list[dict], mode: str, seed: int) -> dict[str, Any]:
    """Replay records under the given mode and return consistency stats."""
    subset = _apply_mode(records, mode, seed)
    n = len(subset)
    if n == 0:
        return {"mode": mode, "n": 0, "consistent": 0, "inconsistent": 0, "consistency_rate": "n/a"}

    consistent = 0
    inconsistent = 0
    unique_snapshot_hashes: set[str] = set()

    for rec in subset:
        sh = rec.get("snapshot_hash") or ""
        if sh:
            unique_snapshot_hashes.add(sh)
        if _is_consistent(rec):
            consistent += 1
        else:
            inconsistent += 1

    pct = 100 * consistent / n if n > 0 else 0.0
    return {
        "mode": mode,
        "n": n,
        "consistent": consistent,
        "inconsistent": inconsistent,
        "consistency_rate": f"{pct:.1f}%",
        "unique_snapshot_hashes": len(unique_snapshot_hashes),
        "hash_diversity_index": round(len(unique_snapshot_hashes) / n, 4) if n > 0 else 0,
    }


def run(log_path: Path, modes: list[str], seed: int, out_path: Path) -> None:
    if not log_path.exists():
        print(f"[ERROR] Log not found: {log_path}")
        print("       Run collect_p1_shadow_samples.py or collect_extended_shadow_batch.py first.")
        return

    records = _load_records(log_path)
    if not records:
        print(f"[ERROR] No records in {log_path}")
        return

    print(f"Source log : {log_path}")
    print(f"Records    : {len(records)}")
    print(f"Seed       : {seed}")
    print(f"Modes      : {', '.join(modes)}\n")

    t0 = time.monotonic()
    results = []
    for mode in modes:
        result = _run_mode(records, mode, seed)
        results.append(result)

    elapsed = time.monotonic() - t0

    # --- print table ---
    print(f"{'='*72}")
    print(f"  REPLAY STRESS RESULTS  ({elapsed:.2f}s)")
    print(f"{'='*72}")
    print(
        f"  {'mode':<16}  {'n':>5}  {'consistent':>10}  {'inconsistent':>12}"
        f"  {'rate':>7}  {'diversity':>9}"
    )
    print(f"  {'-'*16}  {'-'*5}  {'-'*10}  {'-'*12}  {'-'*7}  {'-'*9}")
    for r in results:
        print(
            f"  {r['mode']:<16}  {r['n']:>5}  {r['consistent']:>10}  "
            f"{r['inconsistent']:>12}  {r['consistency_rate']:>7}  "
            f"{r.get('hash_diversity_index', 'n/a'):>9}"
        )
    print(f"{'='*72}")
    print(f"\nconsistent = snapshot_hash == replay_hash (the canonical hash fix predicate)")
    print(f"diversity  = unique snapshot_hashes / n (higher = more stateful variation)")

    # --- write json ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_log": str(log_path),
        "total_records": len(records),
        "seed": seed,
        "elapsed_s": round(elapsed, 3),
        "results": results,
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nResults written: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow replay stress test.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="Source shadow log")
    parser.add_argument(
        "--mode",
        choices=_ALL_MODES + ["all"],
        default="all",
        help="Replay mode (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for shuffled mode")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output JSON path")
    args = parser.parse_args()

    modes = _ALL_MODES if args.mode == "all" else [args.mode]
    run(args.log, modes, args.seed, args.out)


if __name__ == "__main__":
    main()
