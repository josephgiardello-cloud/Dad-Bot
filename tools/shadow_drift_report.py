#!/usr/bin/env python3
"""
Shadow Drift Report
====================
Measures three explicit drift signals from a shadow_mode.jsonl log:

    1. HASH VARIANCE
       Under identical semantics (same input → same pipeline), snapshot_hash
       should be constant. This tool measures the variance coefficient of hashes
       across runs with the same input_hash.
       Target: 0.0 variance (zero drift).

    2. REPLAY DIVERGENCE RATE
       Of all records that have both a snapshot_hash and a replay_hash,
       what fraction diverge? After the fix this should be 0%.

    3. KERNEL REJECTION DRIFT
       Track whether the fraction of records with result="fail" changes over
       time (rolling 50-record window). Rising rejection drift = regression.

Usage:
    python tools/shadow_drift_report.py
    python tools/shadow_drift_report.py --log session_logs/shadow_extended.jsonl
    python tools/shadow_drift_report.py --window 100 --top 15

Output:
    Drift table printed to stdout.
    Optional --out writes a JSON summary.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "session_logs" / "shadow_mode.jsonl"


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


def _pct(num: int, den: int) -> str:
    if den == 0:
        return "n/a"
    return f"{100 * num / den:.2f}%"


def _hash_variance_score(records: list[dict]) -> dict[str, Any]:
    """
    For each unique input_hash, collect all observed snapshot_hashes.

    NOTE: For a stateful conversational bot, the same input at different
    positions in the conversation will always produce different snapshot_hashes
    (because memory, context, and prior turns change).  High snapshot diversity
    per input_hash is therefore EXPECTED and is NOT a signal of non-determinism.

    What this metric actually tells us:
      - variants_per_input > 1: the bot is stateful (expected)
      - variants_per_input == 1: the bot ignored all context (unexpected)

    True non-determinism is measured by metric 2 (replay divergence): if
    snapshot_hash != replay_hash, the canonical hash computation is unstable.
    """
    by_input: dict[str, set[str]] = defaultdict(set)
    for r in records:
        ih = str(r.get("input_hash") or "")
        sh = str(r.get("snapshot_hash") or "")
        if ih and sh:
            by_input[ih].add(sh)

    total = len(by_input)
    single_variant = sum(1 for v in by_input.values() if len(v) == 1)
    multi_variant = total - single_variant
    max_variants = max((len(v) for v in by_input.values()), default=0)
    avg_variants = round(sum(len(v) for v in by_input.values()) / total, 2) if total > 0 else 0.0

    return {
        "total_input_hashes": total,
        "single_variant_inputs": single_variant,
        "multi_variant_inputs": multi_variant,
        "max_variants_per_input": max_variants,
        "avg_variants_per_input": avg_variants,
        "note": "multi-variant is EXPECTED for stateful bot; see metric 2 for true drift",
    }


def _replay_divergence(records: list[dict]) -> dict[str, Any]:
    """
    Of records that have both snapshot_hash and replay_hash, count matches vs mismatches.
    After the canonical hash fix, this should be 0% divergence.
    """
    has_both = [r for r in records if r.get("snapshot_hash") and r.get("replay_hash")]
    if not has_both:
        return {"eligible": 0, "match": 0, "mismatch": 0, "divergence_rate": "n/a"}

    match = sum(1 for r in has_both if r["snapshot_hash"] == r["replay_hash"])
    mismatch = len(has_both) - match
    return {
        "eligible": len(has_both),
        "match": match,
        "mismatch": mismatch,
        "divergence_rate": _pct(mismatch, len(has_both)),
    }


def _kernel_rejection_drift(records: list[dict], window: int) -> dict[str, Any]:
    """
    Compute rolling rejection rate (result != 'pass') over a sliding window.
    Drift = whether the rate increases in the second half vs first half.
    """
    if len(records) < 2:
        return {"drift_detected": False, "first_half_rate": "n/a", "second_half_rate": "n/a"}

    outcomes = [0 if r.get("result") == "pass" else 1 for r in records]
    n = len(outcomes)
    mid = n // 2

    first_half_rate = sum(outcomes[:mid]) / mid if mid > 0 else 0.0
    second_half_rate = sum(outcomes[mid:]) / (n - mid) if (n - mid) > 0 else 0.0
    drift_detected = second_half_rate > first_half_rate + 0.02  # >2% increase = drift

    # Rolling windows
    rolling = []
    for i in range(0, n - window + 1, max(1, window // 4)):
        chunk = outcomes[i: i + window]
        rolling.append(round(sum(chunk) / len(chunk), 4))

    return {
        "drift_detected": drift_detected,
        "first_half_rejection_rate": f"{100*first_half_rate:.2f}%",
        "second_half_rejection_rate": f"{100*second_half_rate:.2f}%",
        "rate_delta": f"{100*(second_half_rate - first_half_rate):+.2f}%",
        "rolling_windows": rolling,
        "window_size": window,
    }


def report(log_path: Path, window: int, top_n: int, out_path: Path | None) -> str:
    if not log_path.exists():
        return f"[ERROR] Log not found: {log_path}\n"

    records = _load_records(log_path)
    if not records:
        return f"[ERROR] No records in {log_path}\n"

    hv = _hash_variance_score(records)
    rd = _replay_divergence(records)
    kd = _kernel_rejection_drift(records, window)

    lines = [
        "================================================================",
        "  SHADOW DRIFT REPORT",
        f"  Log   : {log_path}",
        f"  Sample: {len(records)} records",
        "================================================================",
        "",
        "1. SNAPSHOT HASH DIVERSITY  (informational — statefulness indicator)",
        "----------------------------------------------------------------",
        f"  total_input_hashes       : {hv['total_input_hashes']}",
        f"  single_variant_inputs    : {hv['single_variant_inputs']}  (same input always → same snapshot)",
        f"  multi_variant_inputs     : {hv['multi_variant_inputs']}  (same input → different snapshots per context)",
        f"  max_variants_per_input   : {hv['max_variants_per_input']}",
        f"  avg_variants_per_input   : {hv['avg_variants_per_input']}",
        f"  NOTE: multi-variant is EXPECTED for a stateful bot. True drift = metric 2.",
    ]
    lines += [
        "",
        "2. REPLAY DIVERGENCE RATE  (0% = canonical hash fix holding)",
        "----------------------------------------------------------------",
        f"  eligible (have both hashes) : {rd['eligible']}",
        f"  match                       : {rd.get('match', 'n/a')}",
        f"  mismatch                    : {rd.get('mismatch', 'n/a')}",
        f"  divergence_rate             : {rd['divergence_rate']}  (target: 0.0%)",
        "",
        "3. KERNEL REJECTION DRIFT",
        "----------------------------------------------------------------",
        f"  drift_detected               : {kd['drift_detected']}",
        f"  first_half_rejection_rate    : {kd['first_half_rejection_rate']}",
        f"  second_half_rejection_rate   : {kd['second_half_rejection_rate']}",
        f"  rate_delta                   : {kd['rate_delta']}  (target: ≤0%)",
    ]
    if kd.get("rolling_windows"):
        lines.append(f"  rolling windows ({kd['window_size']}-record): {kd['rolling_windows']}")
    lines += [
        "",
        "================================================================",
        "  VERDICT",
        "----------------------------------------------------------------",
    ]

    ok = (
        rd.get("mismatch", 0) == 0
        and not kd["drift_detected"]
    )
    if ok:
        lines.append("  ALL DRIFT SIGNALS CLEAN — deterministic-core-v1 is stable")
    else:
        if rd.get("mismatch", 0) > 0:
            lines.append(f"  [WARN] Replay divergence: {rd['divergence_rate']}")
        if kd["drift_detected"]:
            lines.append(f"  [WARN] Kernel rejection drift: {kd['rate_delta']}")
    lines.append("================================================================")

    output = "\n".join(lines)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "log": str(log_path),
            "total_records": len(records),
            "snapshot_diversity": hv,
            "replay_divergence": rd,
            "kernel_rejection_drift": kd,
            "verdict": "clean" if ok else "drift_detected",
        }
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        output += f"\n\nJSON written: {out_path}"

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow drift analysis report.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--window", type=int, default=50, help="Rolling window size (default 50)")
    parser.add_argument("--top", type=int, default=10, help="Top N drifted inputs to show")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()
    print(report(args.log, args.window, args.top, args.out))


if __name__ == "__main__":
    main()
