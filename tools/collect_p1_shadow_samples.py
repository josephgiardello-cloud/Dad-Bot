#!/usr/bin/env python3
"""
P1 Shadow Sample Collector
===========================
Drives 300 turns through the live bot to populate session_logs/shadow_mode.jsonl.

Usage:
    python tools/collect_p1_shadow_samples.py            # default 300 turns
    python tools/collect_p1_shadow_samples.py --n 50     # quick smoke test

Requirements:
    - Bot must NOT already be running (this script boots its own instance).
    - Ollama must be reachable on port 11434.
    - DADBOT_SHADOW_MODE=1 is set automatically by this script.

Output:
    Appends records to session_logs/shadow_mode.jsonl.
    Prints a one-line progress counter.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Force shadow mode on before importing the bot
os.environ.setdefault("DADBOT_SHADOW_MODE", "1")
os.environ.setdefault("DADBOT_SHADOW_SAMPLE_RATE", "1.0")

# ---------------------------------------------------------------------------
# Prompts — varied enough to exercise memory, mood, and temporal paths
# ---------------------------------------------------------------------------
_PROMPTS = [
    "hey",
    "how are you",
    "what do you think about that",
    "tell me something interesting",
    "okay",
    "yeah",
    "thanks",
    "sounds good",
    "I'm good, how about you",
    "what's new",
    "anything interesting happen today",
    "good morning",
    "good night",
    "I'm tired",
    "that's funny",
    "not bad",
    "I agree",
    "fair enough",
    "I see",
    "sure",
]


def _build_prompts(n: int) -> list[str]:
    """Return n prompts, cycling through the pool."""
    result = []
    pool = _PROMPTS
    for i in range(n):
        result.append(pool[i % len(pool)])
    return result


def _count_existing_samples(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    count = 0
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def run(n: int) -> None:
    from dadbot.core.dadbot import DadBot
    from dadbot.core.contract_evaluator import live_turn_request, TurnDelivery, TurnResponse
    from typing import cast

    log_path = ROOT / "session_logs" / "shadow_mode.jsonl"
    before = _count_existing_samples(log_path)

    print(f"Shadow log: {log_path}")
    print(f"Existing samples: {before}")
    print(f"Collecting {n} new turns …\n")

    bot = DadBot()

    prompts = _build_prompts(n)
    ok = 0
    err = 0
    t0 = time.monotonic()

    for i, text in enumerate(prompts, 1):
        try:
            req = live_turn_request(text, delivery=TurnDelivery.SYNC, session_id="p1_shadow_collection")
            resp = cast(TurnResponse, bot.execute_turn(req))
            # execute_turn with SYNC returns TurnResponse directly (not a coroutine)
            ok += 1
        except Exception as exc:
            err += 1
            print(f"  [WARN] turn {i} failed: {type(exc).__name__}: {exc}")

        elapsed = time.monotonic() - t0
        rate = ok / elapsed if elapsed > 0 else 0
        print(
            f"\r  {i}/{n}  ok={ok}  err={err}  {rate:.1f} turns/s   ",
            end="",
            flush=True,
        )

    print()  # newline after progress line

    after = _count_existing_samples(log_path)
    new_samples = after - before

    print(f"\n{'='*60}")
    print(f"  Turns attempted : {n}")
    print(f"  Turns ok        : {ok}")
    print(f"  Turns error     : {err}")
    print(f"  New samples     : {new_samples}  (before={before}, after={after})")
    print(f"  Elapsed         : {time.monotonic() - t0:.1f}s")
    print(f"{'='*60}")

    if new_samples < ok:
        print(
            f"\n[WARN] Fewer samples logged than turns completed ({new_samples} < {ok})."
            "\n       Check that DADBOT_SHADOW_MODE=1 is active in the running process."
        )

    print("\nNext: python tools/shadow_mode_report.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect P1 shadow samples.")
    parser.add_argument("--n", type=int, default=300, help="Number of turns to drive (default 300)")
    args = parser.parse_args()
    run(args.n)


if __name__ == "__main__":
    main()
