"""Responsibility Ownership Drift Scanner.

Detects when a module absorbs responsibility that belongs to another layer.
Uses AST-level keyword density analysis to enforce architecture ownership rules.

Ownership model:
  graph.py         → routing/dispatch only; MUST NOT contain mutation/IO/policy decisions
  orchestrator.py  → orchestration only; MUST NOT accumulate inline policy decisions
  nodes.py         → node execution; MUST NOT contain direct persistence IO
  graph_manager.py → memory graph; MUST NOT contain turn-execution logic
  turn_service*    → turn boundary; capped side-effect call density

Usage:
  python tools/ownership_drift.py                  # scan with default rules
  python tools/ownership_drift.py --strict         # tighter thresholds
  python tools/ownership_drift.py --json           # JSON output
  python tools/ownership_drift.py --update-policy  # write current counts as new policy baseline

Exit codes:
  0 — all clear
  1 — one or more ownership violations
"""

from __future__ import annotations

import ast
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple


# ── Responsibility keyword taxonomy ──────────────────────────────────────────

# "Mutation" keywords: calls whose name contains any of these substrings
_MUTATION_KEYWORDS = frozenset({
    "write", "update", "commit", "rollback", "queue", "enqueue",
    "drain", "snapshot", "persist", "save", "flush",
})

# "Policy/Decision" keywords: function/method names that signal policy logic
_POLICY_KEYWORDS = frozenset({
    "retry", "backoff", "ratelimit", "rate_limit", "throttle",
    "authorize", "authorize", "validate", "enforce", "classify",
    "reject", "approve", "gate", "check_policy", "decide",
})

# "IO" keywords: imports or calls that indicate direct external I/O
_IO_IMPORTS = frozenset({
    "requests", "httpx", "aiohttp", "urllib", "socket",
    "subprocess", "os.system", "shutil",
})
_IO_CALL_KEYWORDS = frozenset({
    "open", "read_text", "write_text", "read_bytes", "write_bytes",
    "connect", "send", "recv", "fetch",
})

# "Branch density" (policy decisions inline): each If/elif/ExceptHandler adds 1
_BRANCH_NODES = (ast.If, ast.ExceptHandler, ast.Match)


# ── AST visitor ──────────────────────────────────────────────────────────────

class _Counts(NamedTuple):
    mutation_calls: int
    policy_calls: int
    io_calls: int
    branch_density: int
    io_imports: int


def _name_contains(name: str, keywords: frozenset[str]) -> bool:
    name_lower = name.lower()
    return any(kw in name_lower for kw in keywords)


def _extract_call_name(node: ast.expr) -> str:
    """Best-effort: extract the function name from a Call node's func."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def scan_file(filepath: Path) -> _Counts:
    """Parse *filepath* and return responsibility counts."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (OSError, SyntaxError):
        return _Counts(0, 0, 0, 0, 0)

    mutation_calls = 0
    policy_calls = 0
    io_calls = 0
    branch_density = 0
    io_imports = 0

    for node in ast.walk(tree):
        # Call analysis
        if isinstance(node, ast.Call):
            name = _extract_call_name(node.func)
            if _name_contains(name, _MUTATION_KEYWORDS):
                mutation_calls += 1
            if _name_contains(name, _POLICY_KEYWORDS):
                policy_calls += 1
            if _name_contains(name, _IO_CALL_KEYWORDS):
                io_calls += 1

        # Import analysis (detects direct IO dependencies)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                names = [node.module or ""]
            for mod in names:
                if any(io_mod in (mod or "") for io_mod in _IO_IMPORTS):
                    io_imports += 1

        # Branch density
        elif isinstance(node, _BRANCH_NODES):
            branch_density += 1

    return _Counts(
        mutation_calls=mutation_calls,
        policy_calls=policy_calls,
        io_calls=io_calls,
        branch_density=branch_density,
        io_imports=io_imports,
    )


# ── Ownership rules ───────────────────────────────────────────────────────────

class OwnershipRule(NamedTuple):
    file_pattern: str        # suffix/glob-style filename substring
    max_mutation_calls: int  # -1 = no limit
    max_policy_calls: int
    max_io_calls: int
    max_branch_density: int
    max_io_imports: int
    rationale: str


# Strict thresholds for architecture enforcement
_STRICT_RULES: list[OwnershipRule] = [
    OwnershipRule(
        file_pattern="dadbot/core/graph.py",
        max_mutation_calls=15,    # graph dispatches to mutation layer, some calls expected
        max_policy_calls=25,      # graph legitimately invokes policy methods (classify_failure, StagePhaseMappingPolicy, etc.)
        max_io_calls=5,
        max_branch_density=80,    # graph has routing branches
        max_io_imports=0,
        rationale="graph.py must stay a dispatcher; mutation/policy/IO must be delegated",
    ),
    OwnershipRule(
        file_pattern="orchestrator.py",
        max_mutation_calls=20,
        max_policy_calls=10,      # orchestrators may invoke policy but not own it
        max_io_calls=10,
        max_branch_density=120,
        max_io_imports=2,
        rationale="orchestrator.py must not accumulate inline policy decisions",
    ),
    OwnershipRule(
        file_pattern="dadbot/core/nodes.py",
        max_mutation_calls=40,    # nodes execute mutations
        max_policy_calls=15,
        max_io_calls=20,          # inference node makes IO calls
        max_branch_density=200,
        max_io_imports=5,
        rationale="nodes.py must not perform direct persistence IO outside SaveNode boundary",
    ),
    OwnershipRule(
        file_pattern="graph_manager.py",
        max_mutation_calls=60,    # memory manager owns its own mutations
        max_policy_calls=10,
        max_io_calls=40,          # graph manager does disk/db IO
        max_branch_density=250,
        max_io_imports=5,
        rationale="graph_manager.py must not contain turn-execution dispatch logic",
    ),
    OwnershipRule(
        file_pattern="graph_mutation.py",
        max_mutation_calls=30,    # mutation module — owns mutations
        max_policy_calls=5,
        max_io_calls=3,
        max_branch_density=60,
        max_io_imports=0,
        rationale="graph_mutation.py is the mutation authority; must not acquire IO/policy",
    ),
    OwnershipRule(
        file_pattern="graph_context.py",
        max_mutation_calls=10,
        max_policy_calls=5,
        max_io_calls=2,
        max_branch_density=40,
        max_io_imports=0,
        rationale="graph_context.py is a data container; must stay side-effect free",
    ),
]

# Loose thresholds (default mode — same ratios, more headroom)
_LOOSE_MULTIPLIER = 1.5

_LOOSE_RULES: list[OwnershipRule] = [
    OwnershipRule(
        r.file_pattern,
        int(r.max_mutation_calls * _LOOSE_MULTIPLIER),
        int(r.max_policy_calls * _LOOSE_MULTIPLIER),
        int(r.max_io_calls * _LOOSE_MULTIPLIER),
        int(r.max_branch_density * _LOOSE_MULTIPLIER),
        r.max_io_imports,  # IO imports don't get multiplier — hard architectural boundary
        r.rationale,
    )
    for r in _STRICT_RULES
]


# ── Violation checking ────────────────────────────────────────────────────────

def _check_rule(
    filepath: Path,
    counts: _Counts,
    rule: OwnershipRule,
) -> list[str]:
    violations: list[str] = []

    def _check(label: str, actual: int, limit: int) -> None:
        if limit >= 0 and actual > limit:
            violations.append(f"  {label}: {actual} > limit {limit}")

    _check("mutation_calls", counts.mutation_calls, rule.max_mutation_calls)
    _check("policy_calls", counts.policy_calls, rule.max_policy_calls)
    _check("io_calls", counts.io_calls, rule.max_io_calls)
    _check("branch_density", counts.branch_density, rule.max_branch_density)
    _check("io_imports", counts.io_imports, rule.max_io_imports)

    return violations


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Responsibility Ownership Drift Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use tighter ownership thresholds",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit JSON report to stdout",
    )
    parser.add_argument(
        "--update-policy",
        action="store_true",
        help="Print current counts as a suggested policy baseline and exit 0",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        metavar="FILE",
        help="Override file list to scan",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(__file__).parent.parent
    rules = _STRICT_RULES if args.strict else _LOOSE_RULES

    # Resolve files to scan from rules (or explicit override)
    if args.files:
        scan_targets = [(Path(f), None) for f in args.files]
    else:
        scan_targets = []
        for rule in rules:
            # Find all files matching the pattern under root (exclude .venv and site-packages)
            pattern = rule.file_pattern.replace("\\", "/")
            for candidate in root.rglob("*.py"):
                rel = candidate.relative_to(root).as_posix()
                # Skip virtualenv and third-party packages
                if any(skip in rel for skip in (".venv/", "site-packages/", "/dist-packages/")):
                    continue
                if rel == pattern or rel.endswith("/" + pattern.split("/")[-1]):
                    scan_targets.append((candidate, rule))

    # Deduplicate (same file may match multiple rules — take first)
    seen: set[Path] = set()
    deduped: list[tuple[Path, OwnershipRule | None]] = []
    for fp, rule in scan_targets:
        if fp not in seen:
            seen.add(fp)
            deduped.append((fp, rule))

    # ── Scan + report ─────────────────────────────────────────────────────────
    all_violations: dict[str, list[str]] = {}
    report_rows: list[dict] = []

    for filepath, rule in deduped:
        if not filepath.exists():
            continue
        counts = scan_file(filepath)
        rel = filepath.relative_to(root).as_posix()

        violations: list[str] = []
        if rule is not None:
            violations = _check_rule(filepath, counts, rule)
            if violations:
                all_violations[rel] = violations

        report_rows.append({
            "file": rel,
            "mutation_calls": counts.mutation_calls,
            "policy_calls": counts.policy_calls,
            "io_calls": counts.io_calls,
            "branch_density": counts.branch_density,
            "io_imports": counts.io_imports,
            "violations": violations,
            "rationale": rule.rationale if rule else "",
        })

    # ── Update-policy mode ────────────────────────────────────────────────────
    if args.update_policy:
        print("[ownership_drift] Current responsibility counts (use as policy baseline):")
        for row in report_rows:
            print(
                f"  {row['file']}: "
                f"mutation={row['mutation_calls']} "
                f"policy={row['policy_calls']} "
                f"io={row['io_calls']} "
                f"branches={row['branch_density']} "
                f"io_imports={row['io_imports']}"
            )
        return 0

    # ── Output ────────────────────────────────────────────────────────────────
    if args.json_output:
        print(json.dumps({"violations": all_violations, "report": report_rows}, indent=2))
    else:
        mode_label = "STRICT" if args.strict else "STANDARD"
        print(f"\n[ownership_drift] Responsibility scan ({mode_label} mode)")
        print(f"  {'File':<50} {'MUT':>5} {'POL':>5} {'IO':>5} {'BR':>5} {'IMP':>5}")
        print("  " + "-" * 80)
        for row in report_rows:
            flag = " [!]" if row["violations"] else ""
            print(
                f"  {row['file']:<50} "
                f"{row['mutation_calls']:>5} "
                f"{row['policy_calls']:>5} "
                f"{row['io_calls']:>5} "
                f"{row['branch_density']:>5} "
                f"{row['io_imports']:>5}"
                f"{flag}"
            )

        if all_violations:
            print("\n[ownership_drift] OWNERSHIP VIOLATIONS:")
            for fname, msgs in all_violations.items():
                print(f"  {fname}")
                for msg in msgs:
                    print(msg)
            print(f"\n  → {len(all_violations)} file(s) failed the ownership drift gate\n")
            return 1
        else:
            print("\n[ownership_drift] OK - All files within ownership boundaries\n")
            return 0


if __name__ == "__main__":
    sys.exit(main())
