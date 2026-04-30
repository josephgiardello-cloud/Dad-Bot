"""
CI-style enforcement gate for the staged architectural refactor pipeline.

Usage:
  python _ci_gate_check.py --pre-module  <module.name>     # before starting a module
  python _ci_gate_check.py --post-module <module.name>     # after completing a module
  python _ci_gate_check.py --group-gate  <group_number>    # full hard-gate evaluation
  python _ci_gate_check.py --cycle-check                   # import cycle delta only
  python _ci_gate_check.py --no-parallel-groups            # parallel branch check only

Exit codes:
  0 — all checks passed (PROCEED)
  1 — one or more checks failed (BLOCK)
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import subprocess
import sys

import yaml

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.resolve()
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
GROUP1_MANIFEST = ROOT / "refactor_group_1.yaml"
BACKLOG = ROOT / "refactor_backlog.yaml"
BASELINE_CYCLE_FILE = ROOT / "_baseline_cycles.txt"

# Group-prefix patterns for refactor branches (e.g. refactor/group-02-*)
GROUP_BRANCH_PATTERN = re.compile(r"refactor/group-(\d+)-")

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _run(cmd: list[str], *, capture: bool = True) -> tuple[int, str]:
    """Run a subprocess and return (returncode, combined_output)."""
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        cwd=str(ROOT),
    )
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode, output.strip()


def _pass(msg: str) -> None:
    print(f"  PASS  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def _warn(msg: str) -> None:
    print(f"  WARN  {msg}")


# ─── Check: no parallel refactor group branches ───────────────────────────────


def check_no_parallel_groups(active_group: int | None = None) -> bool:
    """Fail if refactor branches from more than one group are open simultaneously."""
    rc, output = _run(["git", "branch", "--list", "refactor/group-*"])
    if rc != 0:
        _fail(f"git branch listing failed: {output}")
        return False

    branches = [line.strip().lstrip("* ") for line in output.splitlines() if line.strip()]
    groups_found: set[int] = set()
    for branch in branches:
        m = GROUP_BRANCH_PATTERN.search(branch)
        if m:
            groups_found.add(int(m.group(1)))

    if len(groups_found) <= 1:
        _pass(f"No parallel group branches detected (found: {sorted(groups_found) or 'none'})")
        return True

    if active_group is not None and groups_found == {active_group}:
        _pass(f"Only group {active_group} branches open — OK")
        return True

    _fail(
        f"PARALLEL GROUP BRANCHES DETECTED — groups {sorted(groups_found)} are simultaneously open. "
        "Only one group may be active at a time."
    )
    return False


# ─── Check: group 1 all modules complete ─────────────────────────────────────


def check_group1_all_complete() -> bool:
    """Fail unless all 4 Group 1 modules have status=complete in the manifest."""
    if not GROUP1_MANIFEST.exists():
        _fail(f"Group 1 manifest not found: {GROUP1_MANIFEST}")
        return False

    with GROUP1_MANIFEST.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)

    modules = manifest.get("modules", [])
    if not modules:
        _fail("Group 1 manifest has no modules listed.")
        return False

    incomplete = [m["name"] for m in modules if str(m.get("status", "") or "").strip().upper() != "COMPLETE"]
    if incomplete:
        _fail(
            f"Group 1 modules not yet complete: {incomplete}. "
            "Mark each module status=complete in refactor_group_1.yaml before proceeding."
        )
        return False

    _pass(f"All {len(modules)} Group 1 modules are status=complete.")
    return True


# ─── Check: module position gate (pre-module) ────────────────────────────────


def check_pre_module_gate(module_name: str) -> bool:
    """Fail if the previous module in Group 1 is not yet complete."""
    if not GROUP1_MANIFEST.exists():
        _fail(f"Group 1 manifest not found: {GROUP1_MANIFEST}")
        return False

    with GROUP1_MANIFEST.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)

    modules = manifest.get("modules", [])
    position = None
    for m in modules:
        if m["name"] == module_name:
            position = m["position"]
            break

    if position is None:
        # Not a Group 1 module — skip gate
        _warn(f"{module_name} is not in Group 1; skipping position gate.")
        return True

    if position == 1:
        _pass(f"{module_name} is position 1 — no predecessor gate.")
        return True

    predecessor = next((m for m in modules if m["position"] == position - 1), None)
    if predecessor is None:
        _fail(f"Cannot find predecessor for position {position}.")
        return False

    if str(predecessor.get("status", "pending") or "").strip().upper() != "COMPLETE":
        _fail(
            f"Predecessor module '{predecessor['name']}' (position {position - 1}) "
            f"is status='{predecessor.get('status', 'pending')}'. "
            f"Complete position {position - 1} before starting {module_name}."
        )
        return False

    _pass(f"Predecessor '{predecessor['name']}' is complete — {module_name} may start.")
    return True


# ─── Check: import cycles ─────────────────────────────────────────────────────


def _collect_import_cycles() -> set[frozenset[str]]:
    """
    Walk every Python file under dadbot/ and dadbot_system/ and detect
    simple bidirectional import cycles using AST parsing only.

    Returns a set of frozensets, each being a pair (or group) of module names
    that form a cycle.  Only 2-node mutual cycles are detected here; larger
    SCCs require a full graph traversal, but catching mutual imports covers
    the most common regression.
    """
    imports: dict[str, set[str]] = {}
    search_roots = [ROOT / "dadbot", ROOT / "dadbot_system"]

    for search_root in search_roots:
        if not search_root.exists():
            continue
        for py_file in search_root.rglob("*.py"):
            rel = py_file.relative_to(ROOT)
            # Convert path to dotted module name
            parts = rel.with_suffix("").parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            module_name = ".".join(parts)

            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue

            module_imports: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_imports.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_imports.add(node.module.split(".")[0])
            imports[module_name] = module_imports

    cycles: set[frozenset[str]] = set()
    module_names = list(imports.keys())
    for i, mod_a in enumerate(module_names):
        for mod_b in module_names[i + 1 :]:
            # Simplified: check top-level mutual imports
            top_a = mod_a.split(".")[0]
            top_b = mod_b.split(".")[0]
            if top_a == top_b:
                continue
            if top_b in imports.get(mod_a, set()) and top_a in imports.get(mod_b, set()):
                cycles.add(frozenset((mod_a, mod_b)))
    return cycles


def _load_baseline_cycles() -> set[frozenset[str]] | None:
    if not BASELINE_CYCLE_FILE.exists():
        return None
    lines = BASELINE_CYCLE_FILE.read_text(encoding="utf-8").splitlines()
    result: set[frozenset[str]] = set()
    for line in lines:
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 2:
            result.add(frozenset(parts))
    return result


def _save_baseline_cycles(cycles: set[frozenset[str]]) -> None:
    lines = [" | ".join(sorted(c)) for c in sorted(cycles, key=lambda c: sorted(c))]
    BASELINE_CYCLE_FILE.write_text("\n".join(lines), encoding="utf-8")


def check_no_new_cycles() -> bool:
    """Fail if the current import graph has more cycles than the saved baseline."""
    current_cycles = _collect_import_cycles()
    baseline_cycles = _load_baseline_cycles()

    if baseline_cycles is None:
        # First run: save baseline and pass
        _save_baseline_cycles(current_cycles)
        _pass(
            f"Baseline cycle snapshot created with {len(current_cycles)} cycles — future runs will diff against this."
        )
        return True

    new_cycles = current_cycles - baseline_cycles
    resolved_cycles = baseline_cycles - current_cycles

    if new_cycles:
        _fail(
            f"NEW IMPORT CYCLES DETECTED ({len(new_cycles)}):\n"
            + "\n".join(f"    {' ↔ '.join(sorted(c))}" for c in sorted(new_cycles, key=lambda c: sorted(c)))
        )
        return False

    if resolved_cycles:
        _pass(f"No new cycles. {len(resolved_cycles)} baseline cycle(s) resolved. {len(current_cycles)} remaining.")
    else:
        _pass(f"No new import cycles. {len(current_cycles)} cycle(s) match baseline.")
    return True


# ─── Check: DEV lane ─────────────────────────────────────────────────────────


def check_dev_lane(baseline_count: int = 216) -> bool:
    """Run DEV lane (unit tests) and fail if any test fails or count drops below baseline."""
    print("  Running DEV lane (unit tests)...")
    rc, output = _run(
        [str(VENV_PYTHON), "-m", "pytest", "-m", "unit", "--tb=line", "--no-header", "-p", "no:warnings"],
        capture=True,
    )

    # Parse pytest summary line
    summary_match = re.search(r"(\d+) passed", output)
    failed_match = re.search(r"(\d+) failed", output)

    passed = int(summary_match.group(1)) if summary_match else 0
    failed = int(failed_match.group(1)) if failed_match else 0

    if failed > 0:
        _fail(f"DEV lane: {failed} test(s) failed. Output tail:\n{output[-800:]}")
        return False

    if passed < baseline_count:
        _fail(f"DEV lane: only {passed} passed (baseline: {baseline_count}). Test count regression detected.")
        return False

    _pass(f"DEV lane: {passed} passed, 0 failed.")
    return True


# ─── Check: integration lane ─────────────────────────────────────────────────


def check_integration_lane() -> bool:
    """Run integration tests and fail if any fail."""
    print("  Running integration lane...")
    rc, output = _run(
        [str(VENV_PYTHON), "-m", "pytest", "-m", "integration", "-q", "--tb=line", "--no-header"],
        capture=True,
    )

    failed_match = re.search(r"(\d+) failed", output)
    failed = int(failed_match.group(1)) if failed_match else 0

    if failed > 0:
        _fail(f"Integration lane: {failed} test(s) failed. Output tail:\n{output[-800:]}")
        return False

    passed_match = re.search(r"(\d+) passed", output)
    passed = int(passed_match.group(1)) if passed_match else 0
    _pass(f"Integration lane: {passed} passed, 0 failed.")
    return True


# ─── Full hard gate — Group N ─────────────────────────────────────────────────


def run_group_gate(group_number: int) -> bool:
    """Evaluate the hard gate before Group N+1 can start."""
    print(f"\n{'=' * 70}")
    print(f"  HARD GATE — Group {group_number} completion check")
    print(f"{'=' * 70}\n")

    results: list[bool] = []

    if group_number == 1:
        print("[ 1/5 ] All Group 1 modules complete?")
        results.append(check_group1_all_complete())

        print("\n[ 2/5 ] DEV lane green?")
        results.append(check_dev_lane())

        print("\n[ 3/5 ] Integration lane green?")
        results.append(check_integration_lane())

        print("\n[ 4/5 ] No new import cycles?")
        results.append(check_no_new_cycles())

        print("\n[ 5/5 ] No parallel group branches?")
        results.append(check_no_parallel_groups())
    else:
        _warn(f"No gate specification for group {group_number}. Add it to _ci_gate_check.py.")
        return False

    print()
    if all(results):
        print(f"{'=' * 70}")
        print(f"  GATE PASSED — Group {group_number} complete. Group {group_number + 1} may begin.")
        print("  MANUAL CONFIRMATION REQUIRED: verify no LangGraph boundary work has")
        print("  been started on any open branch before proceeding.")
        print(f"{'=' * 70}\n")
        return True
    else:
        failed_count = sum(1 for r in results if not r)
        print(f"{'=' * 70}")
        print(f"  GATE FAILED — {failed_count} check(s) did not pass.")
        print(f"  Group {group_number + 1} is BLOCKED until all conditions are met.")
        print(f"{'=' * 70}\n")
        return False


# ─── Pre-module check ─────────────────────────────────────────────────────────


def run_pre_module(module_name: str) -> bool:
    print(f"\n{'=' * 70}")
    print(f"  PRE-MODULE CHECK — {module_name}")
    print(f"{'=' * 70}\n")

    results: list[bool] = []

    print("[ 1/2 ] Predecessor gate (previous module complete)?")
    results.append(check_pre_module_gate(module_name))

    print("\n[ 2/2 ] No parallel group branches?")
    results.append(check_no_parallel_groups())

    print()
    if all(results):
        print(f"  PRE-MODULE PASSED — {module_name} may start.\n")
        return True
    else:
        print(f"  PRE-MODULE BLOCKED — resolve failures before starting {module_name}.\n")
        return False


# ─── Post-module check ────────────────────────────────────────────────────────


def run_post_module(module_name: str) -> bool:
    print(f"\n{'=' * 70}")
    print(f"  POST-MODULE CHECK — {module_name}")
    print(f"{'=' * 70}\n")

    results: list[bool] = []

    print("[ 1/2 ] No new import cycles?")
    results.append(check_no_new_cycles())

    print("\n[ 2/2 ] No parallel group branches?")
    results.append(check_no_parallel_groups())

    print()
    if all(results):
        print(f"  POST-MODULE PASSED — mark {module_name} status=complete in refactor_group_1.yaml.\n")
        return True
    else:
        print(f"  POST-MODULE FAILED — do not advance; fix issues in {module_name} first.\n")
        return False


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CI enforcement gate for the staged refactor pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pre-module", metavar="MODULE", help="Run pre-start checks for a module")
    parser.add_argument("--post-module", metavar="MODULE", help="Run post-completion checks for a module")
    parser.add_argument("--group-gate", metavar="N", type=int, help="Run full hard gate for group N")
    parser.add_argument("--cycle-check", action="store_true", help="Import cycle delta check only")
    parser.add_argument("--no-parallel-groups", action="store_true", help="Parallel branch check only")
    parser.add_argument(
        "--save-baseline", action="store_true", help="Overwrite baseline cycle snapshot with current state"
    )

    args = parser.parse_args()

    if args.save_baseline:
        cycles = _collect_import_cycles()
        _save_baseline_cycles(cycles)
        print(f"Baseline saved: {len(cycles)} cycles written to {BASELINE_CYCLE_FILE}")
        sys.exit(0)

    if args.pre_module:
        ok = run_pre_module(args.pre_module)
        sys.exit(0 if ok else 1)

    if args.post_module:
        ok = run_post_module(args.post_module)
        sys.exit(0 if ok else 1)

    if args.group_gate is not None:
        ok = run_group_gate(args.group_gate)
        sys.exit(0 if ok else 1)

    if args.cycle_check:
        print("\n[ cycle check ]")
        ok = check_no_new_cycles()
        sys.exit(0 if ok else 1)

    if args.no_parallel_groups:
        print("\n[ parallel-group branch check ]")
        ok = check_no_parallel_groups()
        sys.exit(0 if ok else 1)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
