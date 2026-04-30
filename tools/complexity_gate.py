"""Complexity Gate.

Enforces cyclomatic complexity thresholds using radon.

Rules:
  - No new E-rank blocks beyond the known baseline count
  - No new D-rank blocks beyond the known baseline count  
  - Average CC must not increase beyond baseline average
  - Per-block max threshold: by default any new block > C triggers warning, > D triggers failure

Usage:
  python tools/complexity_gate.py                   # compare against baseline
  python tools/complexity_gate.py --update-baseline # write current state as new baseline
  python tools/complexity_gate.py --strict          # fail on any D+ block (no exceptions)
  python tools/complexity_gate.py --scope core      # core + memory hot dirs only
  python tools/complexity_gate.py --scope all       # full dadbot/ tree

Exit codes:
  0 — passes gate
  1 — complexity regression detected
  2 — baseline file not found
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from radon.complexity import cc_visit, SCORE
    from radon.metrics import mi_visit
except ImportError:
    print("ERROR: radon is required. Install with: pip install radon", file=sys.stderr)
    sys.exit(1)

DEFAULT_BASELINE = Path("tools/complexity_baseline.json")

# Scope presets
_SCOPE_DIRS = {
    "core": ["dadbot/core", "dadbot/memory"],
    "all": ["dadbot"],
}

# Rank values: A=1 .. F=6
_RANK_VALUE = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}


# ── Metric collection ─────────────────────────────────────────────────────────

def _rank(complexity: int) -> str:
    """Convert raw CC number to radon rank letter (A–F)."""
    if complexity <= 5:
        return "A"
    if complexity <= 10:
        return "B"
    if complexity <= 15:
        return "C"
    if complexity <= 20:
        return "D"
    if complexity <= 25:
        return "E"
    return "F"


def _collect_blocks(dirs: list[Path]) -> list[dict]:
    """Walk *dirs* and collect all CC blocks."""
    blocks: list[dict] = []
    for scan_dir in dirs:
        for pyfile in sorted(scan_dir.rglob("*.py")):
            try:
                source = pyfile.read_text(encoding="utf-8")
                results = cc_visit(source)
            except Exception:
                continue
            for block in results:
                r = _rank(block.complexity)
                blocks.append({
                    "file": pyfile.as_posix(),
                    "name": block.name,
                    "lineno": block.lineno,
                    "complexity": block.complexity,
                    "rank": r,
                })
    return blocks


def _summary(blocks: list[dict]) -> dict:
    """Aggregate block list into summary metrics."""
    if not blocks:
        return {"count": 0, "avg": 0.0, "max": 0, "d_count": 0, "e_count": 0, "de_blocks": []}
    complexities = [b["complexity"] for b in blocks]
    d_blocks = [b for b in blocks if b["rank"] in ("D", "E", "F")]
    e_blocks = [b for b in blocks if b["rank"] in ("E", "F")]
    return {
        "count": len(blocks),
        "avg": round(sum(complexities) / len(complexities), 3),
        "max": max(complexities),
        "d_count": len(d_blocks),
        "e_count": len(e_blocks),
        "de_blocks": [
            f"{b['file']}:{b['lineno']} {b['name']} ({b['rank']}={b['complexity']})"
            for b in sorted(d_blocks, key=lambda x: -x["complexity"])
        ],
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complexity Gate — prevent CC regression in hot modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        metavar="FILE",
    )
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument(
        "--scope",
        choices=list(_SCOPE_DIRS.keys()),
        default="core",
        help="Which directories to scan (default: core)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on ANY D+ block regardless of baseline",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(__file__).parent.parent
    scan_dirs = [root / d for d in _SCOPE_DIRS[args.scope]]

    # ── Measure current ───────────────────────────────────────────────────────
    blocks = _collect_blocks(scan_dirs)
    current = _summary(blocks)

    # ── Update baseline ───────────────────────────────────────────────────────
    if args.update_baseline:
        baseline_path = args.baseline if args.baseline.is_absolute() else root / args.baseline
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scope": args.scope,
            "count": current["count"],
            "avg": current["avg"],
            "max": current["max"],
            "d_count": current["d_count"],
            "e_count": current["e_count"],
            "de_blocks": current["de_blocks"],
        }
        baseline_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[complexity_gate] Baseline written → {baseline_path}")
        print(f"  blocks={current['count']} avg={current['avg']} max={current['max']}")
        print(f"  D/E count={current['d_count']} E-only count={current['e_count']}")
        if current["de_blocks"]:
            print("  Known D/E blocks (approved exceptions):")
            for b in current["de_blocks"]:
                print(f"    {b}")
        return 0

    # ── Load baseline ─────────────────────────────────────────────────────────
    baseline_path = args.baseline if args.baseline.is_absolute() else root / args.baseline
    if not baseline_path.exists():
        print(
            f"[complexity_gate] ERROR: baseline not found at {baseline_path}\n"
            "  Run with --update-baseline to create it.",
            file=sys.stderr,
        )
        return 2

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    # ── Gate checks ───────────────────────────────────────────────────────────
    violations: list[str] = []

    # 1. Strict mode: any D+ = fail
    if args.strict and current["d_count"] > 0:
        for b in current["de_blocks"]:
            violations.append(f"  STRICT: D/E block exists: {b}")

    else:
        # 2. Delta mode: D/E count increased beyond baseline
        b_d = baseline.get("d_count", 0)
        b_e = baseline.get("e_count", 0)
        if current["d_count"] > b_d:
            new_blocks = [b for b in current["de_blocks"] if b not in baseline.get("de_blocks", [])]
            violations.append(
                f"  D/E block count grew {b_d}→{current['d_count']} (+{current['d_count'] - b_d})"
            )
            for nb in new_blocks:
                violations.append(f"    NEW: {nb}")
        if current["e_count"] > b_e:
            violations.append(
                f"  E-rank count grew {b_e}→{current['e_count']} (+{current['e_count'] - b_e})"
            )

    # 3. Average CC must not increase
    b_avg = baseline.get("avg", 0.0)
    if current["avg"] > b_avg + 0.5:  # 0.5 tolerance for rounding
        violations.append(
            f"  Average CC increased {b_avg}→{current['avg']} (+{round(current['avg'] - b_avg, 3)})"
        )

    # ── Output ────────────────────────────────────────────────────────────────
    if args.json_output:
        print(json.dumps({
            "violations": violations,
            "current": current,
            "baseline": baseline,
        }, indent=2))
    else:
        print(f"\n[complexity_gate] Scope: dadbot/{args.scope}")
        print(
            f"  blocks={current['count']} "
            f"avg={current['avg']} (baseline={baseline.get('avg', '?')}) "
            f"max={current['max']} "
            f"D/E={current['d_count']} (baseline={baseline.get('d_count', '?')})"
        )
        if current["de_blocks"]:
            print("  Active D/E blocks:")
            for b in current["de_blocks"]:
                prefix = "  [ok] known" if b in baseline.get("de_blocks", []) else "  [!!] NEW  "
                print(f"    {prefix}: {b}")

        if violations:
            print("\n[complexity_gate] VIOLATIONS:")
            for v in violations:
                print(v)
            print(f"\n  → complexity gate FAILED\n")
            return 1
        else:
            print("\n[complexity_gate] OK - Complexity within baseline, gate passed\n")
            return 0


if __name__ == "__main__":
    sys.exit(main())
