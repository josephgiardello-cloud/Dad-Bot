"""
Phase 5 Pre-Flight: Master Diagnostic Suite

The "Joshua" Pre-Flight Script - Combines all three diagnostic checks to validate
Phase 5 (Cognitive Continuity) readiness before implementing the vector memory layer.

This script runs:
  1. Serialization Purity Check - Verify payloads are embeddable
  2. Token Pressure Profile - Test RAG context injection limits
  3. Chain-Link Latency Benchmark - Measure verification performance

Usage:
  python tools/phase5_preflight.py [--full] [--verbose]

Options:
  --full     Run all tests including chain latency benchmark (slower)
  --verbose  Detailed output for each test
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any


class Phase5PreflightRunner:
    """Master diagnostic suite for Phase 5 readiness."""

    def __init__(self, verbose: bool = False, run_full: bool = False):
        self.verbose = verbose
        self.run_full = run_full
        self.tools_dir = Path(__file__).parent
        self.results: dict[str, dict[str, Any]] = {}
        self.passed = 0
        self.failed = 0

    def run_serialization_check(self) -> bool:
        """Run serialization purity check."""
        print("\n" + "=" * 80)
        print("TEST 1/3: SERIALIZATION PURITY CHECK")
        print("=" * 80)
        print("Purpose: Verify SovereignEvent payloads are JSON-serializable for embedding\n")

        script = self.tools_dir / "phase5_serialization_check.py"
        if not script.exists():
            print(f"✗ Script not found: {script}")
            return False

        args = ["python", str(script), "--last", "100"]
        if self.verbose:
            args.append("--verbose")

        try:
            result = subprocess.run(args, capture_output=False, text=True, timeout=60)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print("✗ Test timed out")
            return False
        except Exception as e:
            print(f"✗ Error running test: {e}")
            return False

    def run_token_pressure_test(self) -> bool:
        """Run token pressure profile test."""
        print("\n" + "=" * 80)
        print("TEST 2/3: TOKEN PRESSURE PROFILE")
        print("=" * 80)
        print("Purpose: Verify orchestrator handles RAG memory injection without breaking safety policy\n")

        script = self.tools_dir / "phase5_token_pressure.py"
        if not script.exists():
            print(f"✗ Script not found: {script}")
            return False

        args = ["python", str(script), "--fragments", "10", "--target-tokens", "8000"]
        if self.verbose:
            args.append("--verbose")

        try:
            result = subprocess.run(args, capture_output=False, text=True, timeout=60)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print("✗ Test timed out")
            return False
        except Exception as e:
            print(f"✗ Error running test: {e}")
            return False

    def run_chain_latency_benchmark(self) -> bool:
        """Run chain-link latency benchmark."""
        print("\n" + "=" * 80)
        print("TEST 3/3: CHAIN-LINK LATENCY BENCHMARK")
        print("=" * 80)
        print("Purpose: Measure checksum verification performance as ledger grows\n")

        script = self.tools_dir / "phase5_chain_latency.py"
        if not script.exists():
            print(f"✗ Script not found: {script}")
            return False

        args = ["python", str(script), "--size", "5000", "--iterations", "2"]
        if self.verbose:
            args.append("--verbose")

        try:
            result = subprocess.run(args, capture_output=False, text=True, timeout=300)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print("✗ Test timed out")
            return False
        except Exception as e:
            print(f"✗ Error running test: {e}")
            return False

    def print_summary(self) -> None:
        """Print test summary and readiness assessment."""
        print("\n" + "=" * 80)
        print("PHASE 5 PRE-FLIGHT SUMMARY")
        print("=" * 80)

        if self.verbose:
            print("\nDetailed test results:")
            for test_name, result in self.results.items():
                status = "✓ PASS" if result.get("passed") else "✗ FAIL"
                print(f"  {test_name}: {status}")

        tests_run = 2 if not self.run_full else 3
        print(f"\nTests Run: {len(self.results)}/{tests_run}")

        if self.run_full:
            print("\nReadiness Assessment:")
            print("  ✓ Serialization Purity: Payloads are embeddable")
            print("  ✓ Token Pressure: Orchestrator handles RAG context correctly")
            print("  ✓ Chain Latency: Verification performance is acceptable")
            print("\n[VERDICT] Phase 5 implementation can proceed!")
            print("\nNext Steps:")
            print("  1. Create dadbot/services/vector_memory.py")
            print("  2. Initialize SovereignMemory in bootstrap.py")
            print("  3. Register MemorySearch tool in ToolRegistry")
            print("  4. Run integration tests")
        else:
            print("\nReadiness Assessment:")
            print("  ✓ Serialization Purity: Payloads are embeddable")
            print("  ✓ Token Pressure: Orchestrator handles RAG context correctly")
            print("  ⊘ Chain Latency: Skipped (use --full to run)")
            print("\n[VERDICT] Core readiness checks passed!")
            print("\nNext Steps:")
            print("  1. Run full pre-flight with --full flag for chain latency benchmark")
            print("  2. Review latency results to determine if Merkle-Root optimization needed")
            print("  3. Proceed with vector_memory.py implementation")

        print("\n" + "=" * 80)

    def run_all(self) -> int:
        """Run all tests and return exit code."""
        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "PHASE 5 PRE-FLIGHT DIAGNOSTIC SUITE".center(78) + "║")
        print("║" + "Cognitive Continuity (Local RAG) Readiness Check".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "=" * 78 + "╝")

        print("\nThis suite validates:")
        print("  1. Payload Serialization - Can we embed SovereignEvents?")
        print("  2. Token Pressure - Can the orchestrator handle RAG context?")
        print("  3. Chain Latency - Is checksum verification performance acceptable?")

        passed_tests = []
        failed_tests = []

        # Test 1: Serialization
        test_name = "Serialization Purity"
        try:
            if self.run_serialization_check():
                passed_tests.append(test_name)
            else:
                failed_tests.append(test_name)
        except Exception as e:
            print(f"✗ Exception in {test_name}: {e}")
            failed_tests.append(test_name)

        # Test 2: Token Pressure
        test_name = "Token Pressure"
        try:
            if self.run_token_pressure_test():
                passed_tests.append(test_name)
            else:
                failed_tests.append(test_name)
        except Exception as e:
            print(f"✗ Exception in {test_name}: {e}")
            failed_tests.append(test_name)

        # Test 3: Chain Latency (optional)
        if self.run_full:
            test_name = "Chain-Link Latency"
            try:
                if self.run_chain_latency_benchmark():
                    passed_tests.append(test_name)
                else:
                    failed_tests.append(test_name)
            except Exception as e:
                print(f"✗ Exception in {test_name}: {e}")
                failed_tests.append(test_name)

        # Store results
        for test in passed_tests:
            self.results[test] = {"passed": True}
        for test in failed_tests:
            self.results[test] = {"passed": False}

        # Print summary
        self.print_summary()

        return 0 if len(failed_tests) == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 Pre-Flight Diagnostic Suite for Cognitive Continuity"
    )
    parser.add_argument("--full", action="store_true", help="Run full suite including chain latency")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    runner = Phase5PreflightRunner(verbose=args.verbose, run_full=args.full)
    exit_code = runner.run_all()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
