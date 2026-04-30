"""Complexity diff gate for hot-path files.

Compares current metrics to tools/baseline_metrics.json and fails when any file
regresses in both LOC and CC totals, or when decision points increase.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

try:
    from radon.complexity import cc_visit
except ImportError:
    print("ERROR: radon is required. Install with: pip install radon", file=sys.stderr)
    raise SystemExit(1)

DEFAULT_BASELINE = Path("tools/baseline_metrics.json")
DEFAULT_LOCK = Path("tools/architecture_baseline_lock.json")


def _loc(source: str) -> int:
    return sum(1 for line in source.splitlines() if line.strip() and not line.strip().startswith("#"))


def _decision_points(source: str) -> int:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    branch_nodes = (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.Match)
    return sum(1 for node in ast.walk(tree) if isinstance(node, branch_nodes))


def _metrics(pyfile: Path) -> dict[str, int]:
    source = pyfile.read_text(encoding="utf-8")
    blocks = cc_visit(source)
    return {
        "loc": _loc(source),
        "cc_total": sum(int(getattr(block, "complexity", 0)) for block in blocks),
        "cc_max": max((int(getattr(block, "complexity", 0)) for block in blocks), default=0),
        "decision_points": _decision_points(source),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complexity diff gate")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--ignore-lock", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
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
        if item.get("source") != "complexity_diff_gate":
            continue
        file_path = item.get("file")
        if isinstance(file_path, str) and file_path:
            approved.add(file_path)
    return approved


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    baseline_path = args.baseline if args.baseline.is_absolute() else root / args.baseline
    lock_path = args.lock if args.lock.is_absolute() else root / args.lock

    if not baseline_path.exists():
        print(f"[complexity_diff_gate] baseline missing: {baseline_path}", file=sys.stderr)
        return 2

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    migration_allowlist = set() if args.ignore_lock else _load_migration_allowlist(lock_path)
    violations: list[str] = []
    suppressed: list[str] = []
    rows: dict[str, dict[str, int]] = {}

    for rel_file, baseline_metrics in baseline.items():
        rel_norm = str(rel_file).replace("\\", "/")
        candidate = root / rel_file
        if not candidate.exists():
            continue
        current = _metrics(candidate)
        rows[rel_norm] = current

        loc_up = int(current.get("loc", 0)) > int(baseline_metrics.get("loc", 0))
        cc_not_down = int(current.get("cc_total", 0)) >= int(baseline_metrics.get("cc_total", 0))
        dp_up = int(current.get("decision_points", 0)) > int(baseline_metrics.get("decision_points", 0))

        file_violations: list[str] = []
        if loc_up and cc_not_down:
            file_violations.append(
                f"{rel_norm}: LOC {baseline_metrics.get('loc')}->{current.get('loc')} and CC {baseline_metrics.get('cc_total')}->{current.get('cc_total')}",
            )
        if dp_up:
            file_violations.append(
                f"{rel_norm}: decision_points {baseline_metrics.get('decision_points')}->{current.get('decision_points')}",
            )

        if rel_norm in migration_allowlist:
            suppressed.extend(file_violations)
        else:
            violations.extend(file_violations)

    if args.json_output:
        print(json.dumps({"violations": violations, "suppressed": suppressed, "current": rows}, indent=2))
    else:
        print(
            f"[complexity_diff_gate] files_checked={len(rows)} "
            f"violations={len(violations)} suppressed={len(suppressed)}"
        )
        for item in violations:
            print(f"  {item}")
        if suppressed:
            print("[complexity_diff_gate] migration inflation suppressed by baseline lock:")
            for item in suppressed:
                print(f"  {item}")

    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
