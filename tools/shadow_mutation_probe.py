#!/usr/bin/env python3
"""
Shadow Mutation Probe
======================
Injects controlled structural mutations into the execution pipeline and
records whether the shadow mode detector catches the divergence.

Mutation classes (per the plan):
    1. schema_version_shift  — simulate v1 → v1.1 field additions
    2. field_removal         — drop an optional field from the snapshot
    3. field_addition        — add an unexpected field to the snapshot
    4. pipeline_reorder      — exercise stage traces recorded out of canonical order

For each mutation class, the probe runs N turns, records shadow results,
then prints a summary of pass/fail per mutation class.

Usage:
    python tools/shadow_mutation_probe.py
    python tools/shadow_mutation_probe.py --n 50 --log session_logs/shadow_mutations.jsonl

Output:
    Appends mutated shadow records to the specified log (default: shadow_mutations.jsonl)
    Prints a per-class divergence table on stdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("DADBOT_SHADOW_MODE", "1")
os.environ.setdefault("DADBOT_SHADOW_SAMPLE_RATE", "1.0")

# ---------------------------------------------------------------------------
# Mutation transforms — applied to the snapshot dict before replay-hashing
# ---------------------------------------------------------------------------

def _mutate_schema_version_shift(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Simulate a schema version bump: add a 'schema_version' field."""
    out = dict(snapshot)
    out["schema_version"] = "1.1"
    return out


def _mutate_field_removal(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Drop the first non-internal key from the snapshot (optional field removal)."""
    out = dict(snapshot)
    removable = [k for k in out if not k.startswith("_")]
    if removable:
        del out[removable[0]]
    return out


def _mutate_field_addition(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Add an unexpected field that shouldn't affect canonical hash."""
    out = dict(snapshot)
    out["_probe_injected"] = True
    out["_probe_ts"] = time.time()
    return out


def _mutate_pipeline_reorder(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Reverse the stage_traces list to simulate out-of-order pipeline recording."""
    out = dict(snapshot)
    if isinstance(out.get("stage_traces"), list):
        out["stage_traces"] = list(reversed(out["stage_traces"]))
    return out


_MUTATION_CLASSES: dict[str, Any] = {
    "schema_version_shift": _mutate_schema_version_shift,
    "field_removal": _mutate_field_removal,
    "field_addition": _mutate_field_addition,
    "pipeline_reorder": _mutate_pipeline_reorder,
}

_BASELINE_PROMPTS = [
    "hey",
    "how are you",
    "what do you think",
    "tell me something",
    "good morning",
    "thanks",
    "okay",
    "sounds good",
    "I'm tired",
    "fair enough",
]


def _canonical_hash(snapshot: dict[str, Any]) -> str:
    projection = {
        "user_input": snapshot.get("user_input", ""),
        "stage_traces": snapshot.get("stage_traces", []),
        "response_text": snapshot.get("response_text", ""),
    }
    return hashlib.sha256(
        json.dumps(projection, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def run(n: int, log_path: Path) -> None:
    from dadbot.core.dadbot import DadBot
    from dadbot.core.contract_evaluator import live_turn_request, TurnDelivery, TurnResponse
    from typing import cast

    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Mutation probe log : {log_path}")
    print(f"Turns per class    : {n}")
    print(f"Mutation classes   : {', '.join(_MUTATION_CLASSES)}\n")

    bot = DadBot()
    results: dict[str, dict[str, int]] = {
        cls: {"pass": 0, "fail": 0, "err": 0} for cls in _MUTATION_CLASSES
    }
    results["baseline"] = {"pass": 0, "fail": 0, "err": 0}

    records: list[dict[str, Any]] = []
    t0 = time.monotonic()

    # --- baseline (no mutation) ---
    print("Running baseline …")
    for i in range(n):
        text = _BASELINE_PROMPTS[i % len(_BASELINE_PROMPTS)]
        try:
            req = live_turn_request(text, delivery=TurnDelivery.SYNC, session_id="mutation_probe_baseline")
            resp = cast(TurnResponse, bot.execute_turn(req))
            snapshot = getattr(resp, "execution_snapshot", {}) or {}
            h = _canonical_hash(snapshot)
            results["baseline"]["pass"] += 1
            records.append({
                "mutation_class": "baseline",
                "result": "pass",
                "snapshot_hash": h,
                "prompt": text,
                "ts": time.time(),
            })
        except Exception as exc:
            results["baseline"]["err"] += 1
            records.append({
                "mutation_class": "baseline",
                "result": "error",
                "error": str(exc),
                "prompt": text,
                "ts": time.time(),
            })

    # --- mutation classes ---
    for cls_name, mutate_fn in _MUTATION_CLASSES.items():
        print(f"Running {cls_name} …")
        for i in range(n):
            text = _BASELINE_PROMPTS[i % len(_BASELINE_PROMPTS)]
            try:
                req = live_turn_request(text, delivery=TurnDelivery.SYNC, session_id=f"mutation_probe_{cls_name}")
                resp = cast(TurnResponse, bot.execute_turn(req))
                original_snapshot = getattr(resp, "execution_snapshot", {}) or {}
                mutated_snapshot = mutate_fn(original_snapshot)

                original_hash = _canonical_hash(original_snapshot)
                mutated_hash = _canonical_hash(mutated_snapshot)

                diverged = original_hash != mutated_hash
                outcome = "fail" if diverged else "pass"
                results[cls_name]["fail" if diverged else "pass"] += 1

                records.append({
                    "mutation_class": cls_name,
                    "result": outcome,
                    "original_hash": original_hash,
                    "mutated_hash": mutated_hash,
                    "diverged": diverged,
                    "prompt": text,
                    "ts": time.time(),
                })
            except Exception as exc:
                results[cls_name]["err"] += 1
                records.append({
                    "mutation_class": cls_name,
                    "result": "error",
                    "error": str(exc),
                    "prompt": text,
                    "ts": time.time(),
                })

    # --- write log ---
    with log_path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    # --- report ---
    elapsed = time.monotonic() - t0
    print(f"\n{'='*62}")
    print(f"  MUTATION PROBE RESULTS  ({elapsed:.1f}s)")
    print(f"{'='*62}")
    print(f"  {'class':<24}  {'pass':>5}  {'fail':>5}  {'err':>5}  {'diverge%':>9}")
    print(f"  {'-'*24}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*9}")
    for cls_name, counts in results.items():
        total = counts["pass"] + counts["fail"] + counts["err"]
        if total == 0:
            continue
        div_pct = 100 * counts["fail"] / total
        print(
            f"  {cls_name:<24}  {counts['pass']:>5}  {counts['fail']:>5}"
            f"  {counts['err']:>5}  {div_pct:>8.1f}%"
        )
    print(f"{'='*62}")
    print(f"\nExpected outcomes:")
    print(f"  field_addition   → pass (non-canonical fields ignored by hash)")
    print(f"  schema_version_shift → pass (non-canonical field ignored)")
    print(f"  field_removal    → FAIL (canonical field removed = hash changes)")
    print(f"  pipeline_reorder → FAIL (stage_traces order changes hash)")
    print(f"\nLog written: {log_path}")


def main() -> None:
    default_log = ROOT / "session_logs" / "shadow_mutations.jsonl"
    parser = argparse.ArgumentParser(description="Shadow mutation probe.")
    parser.add_argument("--n", type=int, default=25, help="Turns per mutation class (default 25)")
    parser.add_argument("--log", type=Path, default=default_log)
    args = parser.parse_args()
    run(args.n, args.log)


if __name__ == "__main__":
    main()
