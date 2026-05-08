"""Distribution Readiness Gate — master CI entrypoint.

Tier 1: HARD FAIL (exits 1 if any test fails)
Tier 2: HARD FAIL (exits 1 if any test fails)
Tier 3: HARD FAIL (exits 1 if any test fails)
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

    # Tier 2 — HARD FAIL: reliability
    tier2 = pytest.main([
        "tests/distribution_readiness/tier2_reliability",
        "-q",
        "--tb=short",
    ])
    if tier2 != 0:
        print("\n❌ Tier 2 FAILED: reliability gate not distribution-ready")
        sys.exit(1)

    # Tier 3 — HARD FAIL: long-term evolution signals
    tier3 = pytest.main([
        "tests/distribution_readiness/tier3_advisory",
        "-q",
        "--tb=short",
    ])
    if tier3 != 0:
        print("\n❌ Tier 3 FAILED: advisory gate is now blocking")
        sys.exit(1)

    print("\n=== DISTRIBUTION READINESS REPORT ===")
    print("Tier 1 (kernel):      PASS")
    print("Tier 2 (reliability): PASS")
    print("Tier 3 (advisory):    PASS")
    sys.exit(0)


if __name__ == "__main__":
    run()
