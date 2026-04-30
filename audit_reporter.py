"""audit_reporter.py — Audit result formatting and output.

Receives result dicts produced by audit_runners; produces human-readable output.
No execution logic, no state reads, no side effects beyond stdout.
"""

from __future__ import annotations

from typing import Any


def print_results(results: dict[str, dict[str, Any]]) -> None:
    print("\n=== SYSTEM AUDIT RESULTS ===")
    for key in sorted(results.keys()):
        payload = results[key]
        elapsed = payload.get("elapsed_s")
        elapsed_txt = f", elapsed={elapsed}s" if elapsed is not None else ""
        print(f"{key}: {payload['status']} ({payload.get('detail', '')}{elapsed_txt})")
