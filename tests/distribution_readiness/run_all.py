"""Distribution Readiness Gate — master CI entrypoint.

Tier 1: HARD FAIL (exits 1 if any test fails)
Tier 2: WARN ONLY (non-blocking, results reported)
Tier 3: ADVISORY ONLY (results reported, no fail)
"""
from __future__ import annotations

import sys

import pytest


def run() -> None:
    # Tier 1 — HARD FAIL: kernel correctness
    tier1 = pytest.main([
        "tests/distribution_readiness/tier1_kernel",
        "-q",
        "--tb=short",
    ])

    if tier1 != 0:
        print("\n\u274c Tier 1 FAILED: kernel not distribution-ready")
        sys.exit(1)

    # Tier 2 — WARN: reliability (non-blocking)
    tier2 = pytest.main([
        "tests/distribution_readiness/tier2_reliability",
        "-q",
        "--tb=short",
    ])

    # Tier 3 — ADVISORY: long-term evolution signals
    tier3 = pytest.main([
        "tests/distribution_readiness/tier3_advisory",
        "-q",
        "--tb=short",
    ])

    print("\n=== DISTRIBUTION READINESS REPORT ===")
    print("Tier 1 (kernel):      PASS")
    print(f"Tier 2 (reliability): {'PASS' if tier2 == 0 else 'WARN'} (exit={tier2})")
    print(f"Tier 3 (advisory):    {'PASS' if tier3 == 0 else 'ADVISORY'} (exit={tier3})")
    sys.exit(0)


if __name__ == "__main__":
    run()
