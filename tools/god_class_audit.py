"""God-Class Growth Gate.

Detects god-class growth by comparing current metrics against a baseline snapshot.

FAIL conditions:
  - LOC increases AND CC does not decrease
  - decision_points increases

Usage:
  python tools/god_class_audit.py                    # compare against baseline
  python tools/god_class_audit.py --update-baseline  # write new baseline
  python tools/god_class_audit.py --baseline path/to/baseline_metrics.json
  python tools/god_class_audit.py --files dadbot/core/graph.py dadbot/core/nodes.py

Exit codes:
  0 — all clear
  1 — one or more violations detected
  2 — baseline file not found (run with --update-baseline to create it)
"""

from __future__ import annotations

import ast
import argparse
import json
import sys
from pathlib import Path

try:
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit
except ImportError:
    print("ERROR: radon is required. Install with: pip install radon", file=sys.stderr)
    sys.exit(1)

# ── Hot-file targets ──────────────────────────────────────────────────────────
DEFAULT_HOT_FILES = [
    "dadbot/core/graph.py",
    "dadbot/core/nodes.py",
    "dadbot/memory/graph_manager.py",
    "dadbot/core/orchestrator.py",
    "dadbot/core/graph_mutation.py",
    "dadbot/core/graph_context.py",
    "dadbot/core/graph_pipeline_nodes.py",
]

DEFAULT_BASELINE = Path("tools/baseline_metrics.json")
DEFAULT_LOCK = Path("tools/architecture_baseline_lock.json")

# Decision-point AST node types
_DECISION_NODES = (
    ast.If,
    ast.For,
    ast.While,
    ast.ExceptHandler,
    ast.With,
    ast.Assert,
    ast.Raise,
    ast.Match,  # Python 3.10+
)


# ── Metric collection ─────────────────────────────────────────────────────────

def _count_loc(source: str) -> int:
    """Non-blank, non-comment source lines."""
    count = 0
    for line in source.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return count


def _count_decision_points(source: str) -> int:
    """Count branch/decision AST nodes."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, _DECISION_NODES):
            count += 1
        # also count elif branches (each is a nested If inside an orelse)
        if isinstance(node, ast.If) and node.orelse:
            # orelse with a single If is an elif — already counted as the child If
            pass
    return count


def _cc_total(source: str, filename: str = "<unknown>") -> int:
    """Sum of cyclomatic complexity across all functions/methods."""
    try:
        results = cc_visit(source)
    except Exception:
        return 0
    return sum(int(getattr(r, "complexity", 0)) for r in results)


def _cc_max(source: str) -> int:
    """Max cyclomatic complexity of any single function/method."""
    try:
        results = cc_visit(source)
    except Exception:
        return 0
    if not results:
        return 0
    return max(int(getattr(r, "complexity", 0)) for r in results)


def measure_file(filepath: Path) -> dict:
    """Return metric dict for a single file."""
    if not filepath.exists():
        return {"exists": False, "loc": 0, "cc_total": 0, "cc_max": 0, "decision_points": 0}
    source = filepath.read_text(encoding="utf-8")
    return {
        "exists": True,
        "loc": _count_loc(source),
        "cc_total": _cc_total(source, str(filepath)),
        "cc_max": _cc_max(source),
        "decision_points": _count_decision_points(source),
    }


# ── Comparison logic ──────────────────────────────────────────────────────────

def _check_file(name: str, current: dict, baseline: dict) -> list[str]:
    """Return list of violation strings for one file (empty = clean)."""
    violations: list[str] = []

    if not current.get("exists"):
        # File was deleted — not a god-class growth violation
        return violations

    b_loc = baseline.get("loc", 0)
    b_cc = baseline.get("cc_total", 0)
    b_dp = baseline.get("decision_points", 0)

    c_loc = current["loc"]
    c_cc = current["cc_total"]
    c_dp = current["decision_points"]

    # Rule 1: LOC grew AND CC didn't shrink → god-class absorption
    if c_loc > b_loc and c_cc >= b_cc:
        violations.append(
            f"  LOC grew {b_loc}→{c_loc} (+{c_loc - b_loc}) "
            f"but CC did not decrease ({b_cc}→{c_cc})"
        )

    # Rule 2: Decision points increased → branching complexity absorbed
    if c_dp > b_dp:
        violations.append(
            f"  decision_points grew {b_dp}→{c_dp} (+{c_dp - b_dp})"
        )

    return violations


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="God-Class Growth Gate — detect responsibility absorption",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        metavar="FILE",
        help=f"Path to baseline JSON (default: {DEFAULT_BASELINE})",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write current metrics as the new baseline and exit 0",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        metavar="FILE",
        help="Files to audit (default: built-in hot-file list)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit JSON report to stdout",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=DEFAULT_LOCK,
        metavar="FILE",
        help=f"Path to architecture baseline lock (default: {DEFAULT_LOCK})",
    )
    parser.add_argument(
        "--ignore-lock",
        action="store_true",
        help="Ignore migration-inflation allowlist from lock file",
    )
    return parser.parse_args(argv)


def _load_migration_allowlist(lock_path: Path) -> set[str]:
    if not lock_path.exists():
        return set()
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    allowlist = payload.get("reconciliation", {}).get("migration_inflation_allowlist", [])
    approved: set[str] = set()
    for item in allowlist:
        if not isinstance(item, dict):
            continue
        if item.get("source") != "god_class_audit":
            continue
        file_path = item.get("file")
        if isinstance(file_path, str) and file_path:
            approved.add(file_path)
    return approved


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(__file__).parent.parent
    lock_path = args.lock if args.lock.is_absolute() else root / args.lock
    migration_allowlist = set() if args.ignore_lock else _load_migration_allowlist(lock_path)

    target_files: list[Path] = []
    if args.files:
        target_files = [Path(f) for f in args.files]
    else:
        target_files = [root / f for f in DEFAULT_HOT_FILES]

    # ── Measure current state ─────────────────────────────────────────────────
    current: dict[str, dict] = {}
    for filepath in target_files:
        rel = (filepath.relative_to(root).as_posix() if filepath.is_absolute() else str(filepath).replace("\\", "/"))
        current[rel] = measure_file(filepath if filepath.is_absolute() else root / filepath)

    # ── Update-baseline mode ──────────────────────────────────────────────────
    if args.update_baseline:
        baseline_path = args.baseline if args.baseline.is_absolute() else root / args.baseline
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: {kk: vv for kk, vv in v.items() if kk != "exists"} for k, v in current.items()}
        baseline_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[god_class_audit] Baseline written → {baseline_path}")
        for name, metrics in payload.items():
            print(f"  {name}: loc={metrics['loc']} cc_total={metrics['cc_total']} dp={metrics['decision_points']}")
        return 0

    # ── Load baseline ─────────────────────────────────────────────────────────
    baseline_path = args.baseline if args.baseline.is_absolute() else root / args.baseline
    if not baseline_path.exists():
        print(
            f"[god_class_audit] ERROR: baseline not found at {baseline_path}\n"
            "  Run with --update-baseline to create it.",
            file=sys.stderr,
        )
        return 2

    raw_baseline: dict[str, dict] = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline: dict[str, dict] = {
        str(name).replace("\\", "/"): metrics for name, metrics in raw_baseline.items()
    }

    # ── Compare ───────────────────────────────────────────────────────────────
    all_violations: dict[str, list[str]] = {}
    suppressed_violations: dict[str, list[str]] = {}
    report_rows: list[dict] = []

    for name, metrics in current.items():
        b = baseline.get(name, {})
        violations = _check_file(name, metrics, b)
        if violations and name in migration_allowlist:
            suppressed_violations[name] = violations
            violations = []
        if violations:
            all_violations[name] = violations
        report_rows.append({
            "file": name,
            "loc": metrics.get("loc", 0),
            "cc_total": metrics.get("cc_total", 0),
            "cc_max": metrics.get("cc_max", 0),
            "decision_points": metrics.get("decision_points", 0),
            "baseline_loc": b.get("loc", "—"),
            "baseline_cc_total": b.get("cc_total", "—"),
            "baseline_dp": b.get("decision_points", "—"),
            "violations": violations,
        })

    if args.json_output:
        print(
            json.dumps(
                {
                    "violations": all_violations,
                    "suppressed": suppressed_violations,
                    "report": report_rows,
                },
                indent=2,
            )
        )
    else:
        # Human-readable table
        print("\n[god_class_audit] Metrics delta report")
        print(f"  {'File':<50} {'LOC':>6} {'dLOC':>6} {'CC':>6} {'dCC':>6} {'DP':>5} {'dDP':>5}")
        print("  " + "-" * 92)
        for row in report_rows:
            b = baseline.get(row["file"], {})
            d_loc = row["loc"] - b.get("loc", row["loc"])
            d_cc = row["cc_total"] - b.get("cc_total", row["cc_total"])
            d_dp = row["decision_points"] - b.get("decision_points", row["decision_points"])
            flag = " [!]" if row["violations"] else ""
            print(
                f"  {row['file']:<50} {row['loc']:>6} {d_loc:>+5} "
                f"{row['cc_total']:>6} {d_cc:>+5} {row['decision_points']:>5} {d_dp:>+5}{flag}"
            )

        if all_violations:
            print("\n[god_class_audit] VIOLATIONS DETECTED:")
            for fname, msgs in all_violations.items():
                print(f"  {fname}")
                for msg in msgs:
                    print(msg)
            print(f"\n  → {len(all_violations)} file(s) failed the god-class growth gate\n")
            return 1
        else:
            if suppressed_violations:
                print("\n[god_class_audit] migration inflation suppressed by baseline lock:")
                for fname, msgs in suppressed_violations.items():
                    print(f"  {fname}")
                    for msg in msgs:
                        print(msg)
            print("\n[god_class_audit] OK - All files within baseline, no god-class growth detected\n")
            return 0


if __name__ == "__main__":
    sys.exit(main())
