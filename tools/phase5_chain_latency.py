"""
Phase 5 Pre-Flight: Chain-Link Latency Benchmark

Measures the performance of checksum chain verification as the ledger grows.
As the ledger grows to thousands of events (a year of life with a companion),
that O(N) verification will slow down the response time.

The Check:
  - Generate a dummy ledger with N chained events
  - Time the execution_ledger.load() and verify_current_chain() calls
  - Result: If it takes >500ms, we need Merkle-Root Checkpoint optimization

Usage:
  python tools/phase5_chain_latency.py [--size N] [--iterations I] [--verbose]

Options:
  --size N       Number of events to test (default: 5000)
  --iterations I Number of repeated measurements (default: 3)
  --verbose      Show detailed timing info
"""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _event_sha256(event: dict[str, Any]) -> str:
    """Compute SHA256 of event (mimics execution_ledger logic)."""
    canonical = {
        "type": str(event.get("type") or ""),
        "session_id": str(event.get("session_id") or ""),
        "payload": event.get("payload", {}),
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


def _chain_hash(prev_chain_hash: str, event_sha256: str) -> str:
    """Compute chain hash (mimics execution_ledger logic)."""
    return hashlib.sha256(
        f"{prev_chain_hash or ''!s}:{event_sha256 or ''!s}".encode(),
    ).hexdigest()


def generate_chained_ledger(size: int, output_path: Path | None = None) -> tuple[list[dict], Path]:
    """
    Generate a ledger file with N chained events.
    
    Returns: (events_list, ledger_path)
    """
    if output_path is None:
        output_path = Path("bench_ledger_tmp.jsonl")

    events = []
    prev_chain = ""

    print(f"[GENERATING] {size} chained events...")

    for i in range(size):
        event = {
            "type": f"BENCH_EVENT_{i % 5}",
            "session_id": "bench_session",
            "session_index": i,
            "event_id": f"evt_{i:08d}",
            "kernel_step_id": f"step_{i}",
            "payload": {
                "kind": "GENERIC",
                "data": {
                    "index": i,
                    "sequence": f"event_sequence_{i}",
                    "metadata": {"iteration": i, "timestamp": f"2026-05-08T10:{i%60:02d}:00Z"},
                },
            },
        }

        # Compute chain hashes
        event_sha = _event_sha256(event)
        chain_hash = _chain_hash(prev_chain, event_sha)

        event["event_sha256"] = event_sha
        event["prev_chain_hash"] = prev_chain
        event["chain_hash"] = chain_hash

        events.append(event)
        prev_chain = chain_hash

        if (i + 1) % (size // 10) == 0:
            print(f"  Generated {i + 1}/{size} events...")

    # Write to JSONL
    with open(output_path, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    print(f"✓ Ledger generated: {output_path} ({size} events)")
    return events, output_path


def _verify_persisted_chain_benchmark(events: list[dict[str, Any]]) -> tuple[int, list[str]]:
    """
    Verify chain integrity (mimics execution_ledger logic).
    
    Returns: (violations_count, violation_details)
    """
    violations: list[str] = []
    prev_chain = ""

    for idx, event in enumerate(events):
        expected_event_sha = _event_sha256(event)
        expected_prev = prev_chain
        expected_chain = _chain_hash(expected_prev, expected_event_sha)

        actual_event_sha = str(event.get("event_sha256") or "")
        actual_prev = str(event.get("prev_chain_hash") or "")
        actual_chain = str(event.get("chain_hash") or "")

        if actual_event_sha and actual_event_sha != expected_event_sha:
            violations.append(f"event[{idx}].event_sha256_mismatch")
        if actual_prev and actual_prev != expected_prev:
            violations.append(f"event[{idx}].prev_chain_hash_mismatch")
        if actual_chain and actual_chain != expected_chain:
            violations.append(f"event[{idx}].chain_hash_mismatch")

        prev_chain = expected_chain

    return len(violations), violations


def load_ledger_from_file(ledger_path: Path) -> list[dict[str, Any]]:
    """Load ledger from JSONL file."""
    events = []
    with open(ledger_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    ledger_size: int
    load_time_ms: float
    verify_time_ms: float
    total_time_ms: float
    events_per_second: float
    violations_count: int

    def report(self, verbose: bool = False) -> str:
        """Format as report string."""
        lines = [
            f"Ledger Size:      {self.ledger_size:>6} events",
            f"Load Time:        {self.load_time_ms:>8.2f} ms",
            f"Verify Time:      {self.verify_time_ms:>8.2f} ms",
            f"Total Time:       {self.total_time_ms:>8.2f} ms",
            f"Events/Second:    {self.events_per_second:>8.0f} ev/s",
            f"Violations:       {self.violations_count:>6}",
        ]
        return "\n".join(lines)


def run_benchmark(ledger_path: Path, iterations: int = 3, verbose: bool = False) -> list[BenchmarkResult]:
    """
    Run latency benchmark with multiple iterations.
    
    Returns: list of BenchmarkResult objects
    """
    results = []

    for iteration in range(iterations):
        print(f"\n[BENCHMARK] Iteration {iteration + 1}/{iterations}")

        # Measure load time
        start = time.perf_counter()
        events = load_ledger_from_file(ledger_path)
        load_time = (time.perf_counter() - start) * 1000  # Convert to ms

        # Measure verification time
        start = time.perf_counter()
        violations_count, violations = _verify_persisted_chain_benchmark(events)
        verify_time = (time.perf_counter() - start) * 1000  # Convert to ms

        total_time = load_time + verify_time
        events_per_second = (len(events) / (total_time / 1000)) if total_time > 0 else 0

        result = BenchmarkResult(
            ledger_size=len(events),
            load_time_ms=load_time,
            verify_time_ms=verify_time,
            total_time_ms=total_time,
            events_per_second=events_per_second,
            violations_count=violations_count,
        )
        results.append(result)

        if verbose:
            print(result.report(verbose=True))

    return results


def print_benchmark_report(results: list[BenchmarkResult]) -> None:
    """Pretty-print benchmark results with analysis."""
    print("\n" + "=" * 80)
    print("CHAIN-LINK LATENCY BENCHMARK - PHASE 5 PRE-FLIGHT")
    print("=" * 80)

    if not results:
        print("No results to report.")
        return

    # Calculate statistics
    avg_total = sum(r.total_time_ms for r in results) / len(results)
    avg_verify = sum(r.verify_time_ms for r in results) / len(results)
    avg_load = sum(r.load_time_ms for r in results) / len(results)
    avg_eps = sum(r.events_per_second for r in results) / len(results)

    print(f"\nRuns: {len(results)}")
    print(f"Ledger Size: {results[0].ledger_size:,} events")

    print(f"\n[TIMING AVERAGES]")
    print(f"  Load Time:     {avg_load:>8.2f} ms")
    print(f"  Verify Time:   {avg_verify:>8.2f} ms")
    print(f"  Total Time:    {avg_total:>8.2f} ms")
    print(f"  Throughput:    {avg_eps:>8.0f} events/sec")

    # Performance assessment
    print(f"\n[PERFORMANCE ASSESSMENT]")
    
    LOAD_THRESHOLD = 100  # ms
    VERIFY_THRESHOLD = 500  # ms
    TOTAL_THRESHOLD = 600  # ms

    load_status = "✓ OK" if avg_load < LOAD_THRESHOLD else "✗ SLOW"
    verify_status = "✓ OK" if avg_verify < VERIFY_THRESHOLD else "✗ SLOW"
    total_status = "✓ OK" if avg_total < TOTAL_THRESHOLD else "✗ SLOW"

    print(f"  Load:  {load_status:20} (target: <{LOAD_THRESHOLD}ms, actual: {avg_load:.1f}ms)")
    print(f"  Verify: {verify_status:20} (target: <{VERIFY_THRESHOLD}ms, actual: {avg_verify:.1f}ms)")
    print(f"  Total: {total_status:20} (target: <{TOTAL_THRESHOLD}ms, actual: {avg_total:.1f}ms)")

    # Recommendations
    print(f"\n[RECOMMENDATIONS]")
    if avg_verify > VERIFY_THRESHOLD:
        print(f"  ⚠ Checksum verification is slow for {results[0].ledger_size:,} events.")
        print(f"    Consider implementing Merkle-Root Checkpoint optimization:")
        print(f"    - Only verify the last 10 hashes + a 'State Root'")
        print(f"    - Reduces O(N) verification to O(1) for large ledgers")
        print(f"    - Full chain verification can still be run periodically (e.g., hourly)")
    else:
        print(f"  ✓ Performance is acceptable for current ledger size.")
        print(f"    If ledger grows to 20,000+ events, revisit checkpoint strategy.")

    # Violation check
    if any(r.violations_count > 0 for r in results):
        print(f"\n[CHAIN INTEGRITY]")
        print(f"  ✗ Violations detected during verification!")
        print(f"    The chain may be corrupted.")
    else:
        print(f"\n[CHAIN INTEGRITY]")
        print(f"  ✓ All chain hashes verified successfully!")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark checksum chain verification latency"
    )
    parser.add_argument("--size", type=int, default=5000, help="Number of events to generate")
    parser.add_argument("--iterations", type=int, default=3, help="Number of benchmark iterations")
    parser.add_argument("--keep-ledger", action="store_true", help="Keep the temporary ledger file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    ledger_path = Path("bench_ledger_tmp.jsonl")

    # Generate ledger
    _, ledger_path = generate_chained_ledger(args.size, ledger_path)

    try:
        # Run benchmark
        results = run_benchmark(ledger_path, iterations=args.iterations, verbose=args.verbose)

        # Print report
        print_benchmark_report(results)

        # Determine exit code
        avg_total = sum(r.total_time_ms for r in results) / len(results)
        sys.exit(0 if avg_total < 600 else 1)

    finally:
        # Clean up
        if not args.keep_ledger and ledger_path.exists():
            ledger_path.unlink()
            print(f"\n[CLEANUP] Removed {ledger_path}")


if __name__ == "__main__":
    main()
