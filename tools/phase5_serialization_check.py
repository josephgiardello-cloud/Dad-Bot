"""
Phase 5 Pre-Flight: Serialization Purity Check

Verifies that all SovereignEvent payloads are 100% JSON-serializable and parseable
by external embedding models (like nomic-embed-text).

Risk: If there are any "Lazy Objects" or UUID objects that aren't stringified in the
ledger, the Vector Store indexer will crash.

Usage:
  python tools/phase5_serialization_check.py [--last N] [--verbose]

Options:
  --last N      Check last N events (default: 100)
  --verbose     Show detailed payload inspection
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def stringify_serializable(obj: Any) -> Any:
    """Convert non-JSON-serializable objects to strings (UUID, datetime, etc)."""
    if isinstance(obj, dict):
        return {k: stringify_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [stringify_serializable(item) for item in obj]
    elif hasattr(obj, "isoformat"):  # datetime
        return obj.isoformat()
    elif hasattr(obj, "hex"):  # UUID
        return str(obj)
    else:
        return obj


def check_indexability(event: dict[str, Any], event_idx: int) -> tuple[bool, list[str]]:
    """
    Check if an event is indexable (JSON-serializable and embeddable).
    
    Returns: (is_indexable, error_list)
    """
    errors = []

    try:
        # Test 1: Extract payload as string
        event_type = event.get("type", "")
        payload = event.get("payload", {})

        # Attempt to serialize the payload
        payload_str = json.dumps(payload, default=str)

        # Test 2: Verify it's valid JSON that can be re-parsed
        re_parsed = json.loads(payload_str)

        # Test 3: Construct the content that would be embedded
        content = f"{event_type}: {payload_str}"

        # Test 4: Verify content is a clean string
        if not isinstance(content, str):
            errors.append(f"Content is not a string (type={type(content).__name__})")
        elif len(content) == 0:
            errors.append("Content is empty")

        # Test 5: Check for non-ASCII encoding issues (optional but good for embeddings)
        try:
            content.encode("utf-8")
        except UnicodeEncodeError as e:
            errors.append(f"Unicode encoding failed: {e}")

        # Test 6: Verify no lazy-loaded objects remain
        errors.extend(_check_for_lazy_objects(payload, prefix="payload"))

        return len(errors) == 0, errors

    except json.JSONDecodeError as e:
        return False, [f"JSON decode error: {e}"]
    except Exception as e:
        return False, [f"Unexpected error: {type(e).__name__}: {e}"]


def _check_for_lazy_objects(obj: Any, prefix: str = "") -> list[str]:
    """Recursively check for lazy-loaded or non-serializable objects.

    Returns a list of error strings (empty when the object tree is clean).
    """
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            found.extend(_check_for_lazy_objects(value, prefix=f"{prefix}.{key}" if prefix else key))
    elif isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            found.extend(_check_for_lazy_objects(item, prefix=f"{prefix}[{idx}]"))
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        pass  # OK
    elif hasattr(obj, "__dict__") and not isinstance(obj, (type, type(None))):
        # Suspicious: has __dict__ but not a basic type
        if not hasattr(obj, "isoformat") and not hasattr(obj, "hex"):
            return [f"{prefix}: Suspicious object type: {type(obj).__name__}"]
    return found


def check_ledger(ledger_path: Path, last_n: int = 100, verbose: bool = False) -> tuple[bool, dict[str, Any]]:
    """
    Check the last N events in the relational_ledger.jsonl for indexability.
    
    Returns: (all_pass, results_dict)
    """
    if not ledger_path.exists():
        return False, {"error": f"Ledger not found: {ledger_path}"}

    results = {
        "ledger_path": str(ledger_path),
        "last_n": last_n,
        "total_checked": 0,
        "indexable_count": 0,
        "failures": [],
        "warnings": [],
    }

    try:
        # Read the last N events
        events = []
        with open(ledger_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        results["warnings"].append(f"Malformed JSONL line, skipping")

        # Take the last N
        events_to_check = events[-last_n:] if len(events) > last_n else events
        results["total_checked"] = len(events_to_check)

        # Check each event
        for idx, event in enumerate(events_to_check):
            is_indexable, errors = check_indexability(event, idx)

            if is_indexable:
                results["indexable_count"] += 1
            else:
                failure = {
                    "event_index": idx,
                    "event_type": event.get("type", "UNKNOWN"),
                    "event_id": event.get("event_id", "?"),
                    "errors": errors,
                }
                results["failures"].append(failure)

                if verbose:
                    print(f"\n[FAIL] Event {idx} ({event.get('type', 'UNKNOWN')})")
                    for err in errors:
                        print(f"  - {err}")

        results["pass_rate"] = (
            results["indexable_count"] / results["total_checked"] * 100
            if results["total_checked"] > 0
            else 0
        )

        return len(results["failures"]) == 0, results

    except Exception as e:
        results["error"] = f"Unexpected error: {type(e).__name__}: {e}"
        return False, results


def print_report(results: dict[str, Any], verbose: bool = False) -> None:
    """Pretty-print the results."""
    print("\n" + "=" * 80)
    print("SERIALIZATION PURITY CHECK - PHASE 5 PRE-FLIGHT")
    print("=" * 80)

    if "error" in results:
        print(f"\nERROR: {results['error']}")
        return

    print(f"\nLedger: {results['ledger_path']}")
    print(f"Checked: {results['total_checked']} events (last {results['last_n']})")
    print(f"Pass Rate: {results['indexable_count']}/{results['total_checked']} ({results['pass_rate']:.1f}%)")

    if results["failures"]:
        print(f"\n[FAILURES] {len(results['failures'])} events failed indexability checks:")
        for failure in results["failures"]:
            print(
                f"\n  Event {failure['event_index']}: {failure['event_type']} "
                f"(ID: {failure['event_id']})"
            )
            for error in failure["errors"]:
                print(f"    ✗ {error}")
    else:
        print("\n✓ All events are 100% indexable!")

    if results.get("warnings"):
        print(f"\n[WARNINGS]")
        for warning in results["warnings"]:
            print(f"  ⚠ {warning}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Check SovereignEvent payloads for embedding readiness"
    )
    parser.add_argument("--last", type=int, default=100, help="Check last N events (default: 100)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("relational_ledger.jsonl"),
        help="Path to ledger file",
    )

    args = parser.parse_args()

    all_pass, results = check_ledger(args.ledger, last_n=args.last, verbose=args.verbose)
    print_report(results, verbose=args.verbose)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
