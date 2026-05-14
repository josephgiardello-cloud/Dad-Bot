#!/usr/bin/env python3
"""Phase 4A: Orchestrator Integration - Real Execution Demo

This script demonstrates how to:
1. Initialize a DadBot with real orchestrator
2. Run scenarios through the actual agent pipeline
3. Capture real execution traces
4. Generate baseline intelligence profile

Run this script to transition from synthetic (Phase 1) to real measurements.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.benchmark_runner import BenchmarkRunner, print_benchmark_results
from tests.scenario_suite import SCENARIOS

logger = logging.getLogger(__name__)


async def run_phase4a_real():
    """Run Phase 4A with real orchestrator (if available)."""
    print("\n" + "=" * 80)
    print("PHASE 4A: ORCHESTRATOR INTEGRATION")
    print("Attempting to initialize real DadBot orchestrator...")
    print("=" * 80 + "\n")

    try:
        # Try to import and initialize real bot
        from dadbot.core.dadbot import DadBot

        print("✓ Importing DadBot...")
        bot = DadBot()
        print("✓ DadBot initialized")

        orchestrator = getattr(bot, "turn_orchestrator", None)
        if not orchestrator:
            print("✗ DadBot has no turn_orchestrator attribute")
            print("  Falling back to mock execution...")
            return None

        print("✓ Orchestrator acquired")
        print(f"  Type: {type(orchestrator).__name__}")

        # Create runner with real orchestrator
        runner = BenchmarkRunner(
            strict=False,
            mode="orchestrator",
            orchestrator=orchestrator,
        )

        print(f"\n📋 Executing {len(SCENARIOS)} scenarios through real agent pipeline...")
        print("-" * 80)

        results = runner.run_all_scenarios(SCENARIOS)

        print("-" * 80)
        print_benchmark_results(results, mode="orchestrator")

        # Compute summary statistics
        successes = sum(1 for r in results if r["execution"].get("completed"))
        total = len(results)
        print("\n📊 PHASE 4A SUMMARY:")
        print(f"  Success Rate: {successes}/{total} ({100 * successes / total:.1f}%)")

        # By category
        by_cat = {}
        for r in results:
            cat = r["category"]
            if cat not in by_cat:
                by_cat[cat] = {"success": 0, "total": 0}
            by_cat[cat]["total"] += 1
            if r["execution"].get("completed"):
                by_cat[cat]["success"] += 1

        print("\n  By Category:")
        for cat in sorted(by_cat.keys()):
            s = by_cat[cat]["success"]
            t = by_cat[cat]["total"]
            pct = 100 * s / t if t > 0 else 0
            print(f"    {cat:12s}: {s}/{t} ({pct:5.1f}%)")

        return results

    except ImportError as e:
        print(f"✗ Cannot import DadBot: {e}")
        print("  This is expected if the runtime environment is not properly configured.")
        return None
    except Exception as e:
        logger.exception(f"Error during Phase 4A: {e}")
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        return None


def run_phase1_mock():
    """Run Phase 1 with mock execution (always works)."""
    print("\n" + "=" * 80)
    print("PHASE 1: MOCK EXECUTION")
    print("Running scenario validation with mock backend...")
    print("=" * 80 + "\n")

    runner = BenchmarkRunner(strict=False, mode="mock")
    results = runner.run_all_scenarios(SCENARIOS)

    print_benchmark_results(results, mode="mock")
    return results


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )

    print("\n" + "=" * 80)
    print("BENCHMARK RUNNER: PHASE 1 → PHASE 4A PROGRESSION")
    print("=" * 80)
    print("\nThis tool demonstrates the scenario-based capability measurement framework:")
    print("  • Phase 1: Mock execution (validates scenario structure)")
    print("  • Phase 4A: Real orchestrator execution (measures actual capability)")
    print()

    # Try Phase 4A first (real execution)
    results_4a = asyncio.run(run_phase4a_real())

    if results_4a is None:
        print("\n" + "-" * 80)
        print("Phase 4A unavailable. Running Phase 1 fallback...")
        print("-" * 80)
        results_1 = run_phase1_mock()
    else:
        print("\n" + "-" * 80)
        print("Phase 4A completed successfully!")
        print("-" * 80)
        print("\nNext steps:")
        print("  1. Analyze results to identify capability gaps")
        print("  2. Expand scoring with Phase 4B (planner diagnostics)")
        print("  3. Add tool intelligence metrics (Phase 4E)")
        print("  4. Generate maturity report (Phase 4G)")


if __name__ == "__main__":
    main()
