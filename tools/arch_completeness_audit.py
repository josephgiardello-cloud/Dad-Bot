"""Architectural Completeness Audit.

Implements all 6 completeness checks against the architecture manifest:

  Step 7.1  Load architecture_manifest.json (the "should exist" contract)
  Step 7.2  Module presence check   -> missing_modules
  Step 7.3  Subsystem coverage      -> partial_subsystems
  Step 7.4  Contract propagation    -> missing_parameters
  Step 7.5  Interface completeness  -> interface_drift
  Step 7.6  Orphan expectation      -> orphan_expectations

Produces: audit/architectural_completeness_report.json

Usage:
  python tools/arch_completeness_audit.py
  python tools/arch_completeness_audit.py --manifest audit/spec/architecture_manifest.json
  python tools/arch_completeness_audit.py --output audit/architectural_completeness_report.json
  python tools/arch_completeness_audit.py --verbose
  python tools/arch_completeness_audit.py --json     # emit JSON only (for CI piping)

Exit codes:
  0  completeness_score >= 80 (healthy)
  1  completeness_score < 80 (gaps detected)
  2  manifest not found
"""

from __future__ import annotations

import ast
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parent.parent
DEFAULT_MANIFEST = ROOT / "audit" / "spec" / "architecture_manifest.json"
DEFAULT_OUTPUT   = ROOT / "audit" / "architectural_completeness_report.json"

# Execution-path heuristic: function names that indicate turn/stage execution paths
_EXECUTION_PATH_PATTERNS = re.compile(
    r"^(execute|run|dispatch|process_turn|handle_turn|_run_stage|ainvoke|submit_turn|"
    r"invoke|call|perform|start_turn|end_turn|do_execute|pipeline_step)$",
    re.IGNORECASE,
)

# Required cross-cutting fields to look for as parameter names or attribute access
_CROSS_CUTTING_FIELDS = [
    "trace_id",
    "kernel_step_id",
    "determinism_manifest",
    "ledger_entry",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _read_source(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _parse_tree(source: str, filename: str = "<unknown>") -> ast.Module | None:
    try:
        return ast.parse(source, filename=filename)
    except SyntaxError:
        return None


def _defined_names(tree: ast.Module) -> set[str]:
    """All top-level class/function/variable names defined in a module."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


def _method_names_of_class(tree: ast.Module, class_name: str) -> set[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                n.name for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return set()


def _is_execution_path(func_name: str) -> bool:
    return bool(_EXECUTION_PATH_PATTERNS.match(func_name))


def _func_param_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for arg in func.args.args + func.args.kwonlyargs + func.args.posonlyargs:
        names.add(arg.arg)
    return names


def _attribute_accesses(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Attribute names accessed inside a function (self.foo -> 'foo')."""
    names: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def _call_names(tree: ast.Module) -> list[str]:
    """All function/method names called anywhere in the module."""
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.append(node.func.attr)
    return names


def _grep_identifier(text: str, identifier: str) -> bool:
    """Return True if *identifier* appears as a whole word in *text*."""
    return bool(re.search(r"\b" + re.escape(identifier) + r"\b", text))


# ── Step 7.2 — Module presence check ─────────────────────────────────────────

def check_missing_modules(manifest: dict) -> dict:
    missing: list[dict] = []
    present: list[str] = []

    for logical_name, rel_path in manifest.get("core_modules", {}).items():
        full_path = ROOT / rel_path
        if not full_path.exists():
            missing.append({"module": logical_name, "expected_path": rel_path})
        else:
            present.append(logical_name)

    return {
        "missing": missing,
        "present_count": len(present),
        "missing_count": len(missing),
    }


# ── Step 7.3 — Subsystem coverage check ──────────────────────────────────────

def _cohesion_score(files: list[Path]) -> float:
    """
    Cohesion heuristic: fraction of files that share at least one imported name
    with the entrypoint.  Range 0.0–1.0.
    """
    if len(files) <= 1:
        return 1.0

    entry_source = _read_source(files[0])
    if not entry_source:
        return 0.0
    entry_tree = _parse_tree(entry_source)
    if entry_tree is None:
        return 0.0
    entry_names = _defined_names(entry_tree)

    shared = 0
    for f in files[1:]:
        src = _read_source(f)
        if not src:
            continue
        # Check if this file imports anything from entrypoint's names
        if any(name in src for name in entry_names if len(name) > 3):
            shared += 1

    return round(shared / len(files[1:]), 3)


def check_subsystem_coverage(manifest: dict) -> dict:
    results: dict[str, dict] = {}

    for name, spec in manifest.get("required_subsystems", {}).items():
        entry_path = ROOT / spec.get("entrypoint", "")
        supporting_paths = [ROOT / p for p in spec.get("supporting", [])]
        required_symbols = spec.get("required_symbols", [])

        all_files = [entry_path] + supporting_paths
        existing_files = [f for f in all_files if f.exists()]
        missing_files = [_rel(f) for f in all_files if not f.exists()]

        # Symbol coverage
        missing_symbols: list[str] = []
        if entry_path.exists() and required_symbols:
            src = _read_source(entry_path)
            tree = _parse_tree(src or "", str(entry_path))
            if tree:
                defined = _defined_names(tree)
                missing_symbols = [s for s in required_symbols if s not in defined]
            else:
                missing_symbols = list(required_symbols)

        cohesion = _cohesion_score(existing_files) if len(existing_files) > 1 else 1.0
        is_fragmented = (
            len(existing_files) > 3
            and cohesion < 0.4
        )
        is_partial = (
            len(missing_files) > 0
            or len(missing_symbols) > 0
        )

        results[name] = {
            "exists": entry_path.exists(),
            "file_count": len(all_files),
            "missing_files": missing_files,
            "missing_symbols": missing_symbols,
            "coverage_score": round(len(existing_files) / max(len(all_files), 1), 3),
            "cohesion_score": cohesion,
            "is_fragmented": is_fragmented,
            "is_partial": is_partial,
        }

    partial = [k for k, v in results.items() if v["is_partial"]]
    fragmented = [k for k, v in results.items() if v["is_fragmented"]]

    return {
        "subsystems": results,
        "partial_count": len(partial),
        "fragmented_count": len(fragmented),
        "partial": partial,
        "fragmented": fragmented,
    }


# ── Step 7.4 — Contract propagation check ────────────────────────────────────

def check_param_propagation(manifest: dict) -> dict:
    """
    Scan execution-path functions for presence of required cross-cutting fields.
    A field is "present" if it appears as a parameter name OR as an attribute access
    on self (self.trace_id), OR on any local variable named 'context'/'ctx'/'turn_context'.
    """
    violations: list[dict] = []
    scanned_functions = 0

    cross_cutting = manifest.get("required_cross_cutting", {})
    required_fields = list(cross_cutting.keys())

    # Files to scan: entrypoint + supporting of every subsystem, plus hot files
    scan_files: set[Path] = set()
    for spec in manifest.get("required_subsystems", {}).values():
        scan_files.add(ROOT / spec.get("entrypoint", ""))
        for p in spec.get("supporting", []):
            scan_files.add(ROOT / p)
    for rel in manifest.get("core_modules", {}).values():
        scan_files.add(ROOT / rel)

    for filepath in sorted(scan_files):
        if not filepath.exists():
            continue
        src = _read_source(filepath)
        if not src:
            continue
        tree = _parse_tree(src, str(filepath))
        if not tree:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _is_execution_path(node.name):
                continue

            scanned_functions += 1
            params = _func_param_names(node)
            attrs = _attribute_accesses(node)
            all_names = params | attrs

            missing = []
            for field in required_fields:
                spec = cross_cutting[field]
                scan_as = spec.get("scan_as", "parameter_or_attribute")
                required_in = spec.get("required_in_patterns", [])

                # Only enforce in files that match the required_in_patterns
                rel = _rel(filepath)
                if required_in and not any(pat in rel for pat in required_in):
                    continue

                if scan_as in ("parameter_or_attribute",):
                    if field not in all_names:
                        missing.append(field)
                elif scan_as == "call_or_import":
                    if not _grep_identifier(src, field):
                        missing.append(field)

            if missing:
                violations.append({
                    "file": _rel(filepath),
                    "function": node.name,
                    "missing_fields": missing,
                    "lineno": node.lineno,
                })

    return {
        "violations": violations,
        "scanned_functions": scanned_functions,
        "violation_count": len(violations),
    }


# ── Step 7.5 — Interface completeness check ───────────────────────────────────

def check_interface_completeness(manifest: dict) -> dict:
    drift: list[dict] = []

    for interface_name, spec in manifest.get("required_interfaces", {}).items():
        filepath = ROOT / spec.get("file", "")
        required_methods = spec.get("required_methods", [])

        if not filepath.exists():
            drift.append({
                "interface": interface_name,
                "file": spec.get("file"),
                "issue": "file_missing",
                "missing_methods": required_methods,
            })
            continue

        src = _read_source(filepath)
        tree = _parse_tree(src or "", str(filepath))
        if not tree:
            drift.append({
                "interface": interface_name,
                "file": _rel(filepath),
                "issue": "parse_error",
                "missing_methods": required_methods,
            })
            continue

        # Find the interface class (Protocol or ABC)
        actual_methods = _method_names_of_class(tree, interface_name)

        # Also accept if the methods exist as module-level functions
        if not actual_methods:
            all_funcs = {
                n.name for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            actual_methods = all_funcs

        missing = [m for m in required_methods if m not in actual_methods]

        if missing:
            drift.append({
                "interface": interface_name,
                "file": _rel(filepath),
                "issue": "missing_methods",
                "missing_methods": missing,
                "defined_methods": sorted(actual_methods),
            })

    return {
        "drift": drift,
        "drift_count": len(drift),
    }


# ── Step 7.6 — Orphan expectation check ──────────────────────────────────────

def check_orphan_expectations(manifest: dict) -> dict:
    targets = manifest.get("orphan_search_targets", [])
    orphans: list[dict] = []

    # Collect all non-venv Python source
    all_py: list[Path] = []
    for p in ROOT.rglob("*.py"):
        rel = _rel(p)
        if any(skip in rel for skip in (".venv/", "site-packages/")):
            continue
        all_py.append(p)

    for target in targets:
        occurrences: list[dict] = []
        for filepath in all_py:
            src = _read_source(filepath)
            if not src:
                continue
            if _grep_identifier(src, target):
                # Find which lines
                lines = [
                    i + 1
                    for i, line in enumerate(src.splitlines())
                    if re.search(r"\b" + re.escape(target) + r"\b", line)
                ]
                occurrences.append({
                    "file": _rel(filepath),
                    "lines": lines[:5],  # cap at 5 for readability
                })

        if occurrences:
            # Check if the symbol is actually defined somewhere
            is_defined = any(
                any(
                    isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name == target
                    for n in ast.walk(_parse_tree(_read_source(p) or "", str(p)) or ast.parse(""))
                )
                for p in all_py
                if p.exists()
            )
            orphans.append({
                "symbol": target,
                "referenced_in": occurrences,
                "is_defined": is_defined,
                "is_orphan": not is_defined,
            })

    actual_orphans = [o for o in orphans if o["is_orphan"]]
    found_defined = [o for o in orphans if o["is_defined"]]

    return {
        "searched": targets,
        "found_referenced_and_undefined": actual_orphans,
        "found_referenced_and_defined": [o["symbol"] for o in found_defined],
        "not_referenced": [t for t in targets if not any(o["symbol"] == t for o in orphans)],
        "orphan_count": len(actual_orphans),
    }


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_completeness_score(
    missing_modules: dict,
    subsystems: dict,
    propagation: dict,
    interfaces: dict,
    orphans: dict,
    total_modules: int,
) -> int:
    """Weighted 0-100 completeness score."""

    # Module presence: 35 points
    module_score = 35.0
    if total_modules > 0:
        pct_missing = missing_modules["missing_count"] / total_modules
        module_score = round(35.0 * (1.0 - pct_missing), 1)

    # Subsystem coverage: 25 points
    total_sub = len(subsystems.get("subsystems", {}))
    partial = subsystems.get("partial_count", 0)
    fragmented = subsystems.get("fragmented_count", 0)
    sub_penalty = (partial * 1.5 + fragmented * 1.0) / max(total_sub, 1)
    sub_score = round(max(0.0, 25.0 * (1.0 - sub_penalty)), 1)

    # Contract propagation: 20 points
    prop_violations = propagation.get("violation_count", 0)
    scanned = max(propagation.get("scanned_functions", 1), 1)
    prop_score = round(max(0.0, 20.0 * (1.0 - prop_violations / scanned)), 1)

    # Interface completeness: 10 points
    drift_count = interfaces.get("drift_count", 0)
    total_ifaces = 4  # from manifest
    iface_score = round(max(0.0, 10.0 * (1.0 - drift_count / max(total_ifaces, 1))), 1)

    # Orphan expectations: 10 points
    orphan_count = orphans.get("orphan_count", 0)
    total_targets = max(len(orphans.get("searched", [])), 1)
    orphan_score = round(max(0.0, 10.0 * (1.0 - orphan_count / total_targets)), 1)

    total = int(module_score + sub_score + prop_score + iface_score + orphan_score)
    return min(100, max(0, total))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Architectural Completeness Audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_only",
                        help="Emit JSON report to stdout only")
    parser.add_argument("--no-write", action="store_true",
                        help="Do not write the output file")
    return parser.parse_args(argv)


def _print_section(title: str, data: dict, verbose: bool) -> None:
    print(f"\n[arch_audit] {title}")
    if isinstance(data, dict):
        for k, v in data.items():
            if k in ("subsystems", "violations", "drift", "orphans"):
                continue  # detailed only in verbose
            print(f"  {k}: {v}")
    if verbose:
        print("  detail:", json.dumps(data, indent=4, default=str))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # ── Load manifest ──────────────────────────────────────────────────────────
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    if not manifest_path.exists():
        print(f"[arch_audit] ERROR: manifest not found at {manifest_path}", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # ── Run all 6 steps ────────────────────────────────────────────────────────
    if not args.json_only:
        print("[arch_audit] Running architectural completeness audit...")
        print(f"  manifest: {_rel(manifest_path)}")
        print(f"  repo root: {ROOT}")

    total_modules = len(manifest.get("core_modules", {}))

    step72 = check_missing_modules(manifest)
    step73 = check_subsystem_coverage(manifest)
    step74 = check_param_propagation(manifest)
    step75 = check_interface_completeness(manifest)
    step76 = check_orphan_expectations(manifest)

    score = compute_completeness_score(
        step72, step73, step74, step75, step76, total_modules
    )

    # ── Assemble report ────────────────────────────────────────────────────────
    report: dict[str, Any] = {
        "completeness_score": score,
        "missing_modules": step72.get("missing", []),
        "partial_subsystems": step73.get("partial", []),
        "fragmented_subsystems": step73.get("fragmented", []),
        "subsystem_detail": step73.get("subsystems", {}),
        "contract_drift": step75.get("drift", []),
        "missing_parameters": step74.get("violations", []),
        "orphan_expectations": step76.get("found_referenced_and_undefined", []),
        "summary": {
            "total_modules_checked": total_modules,
            "missing_modules": step72.get("missing_count", 0),
            "partial_subsystems": step73.get("partial_count", 0),
            "fragmented_subsystems": step73.get("fragmented_count", 0),
            "interface_drift_count": step75.get("drift_count", 0),
            "param_propagation_violations": step74.get("violation_count", 0),
            "scanned_exec_functions": step74.get("scanned_functions", 0),
            "orphan_symbols": step76.get("orphan_count", 0),
            "not_referenced_orphan_targets": step76.get("not_referenced", []),
        },
    }

    # ── Output ─────────────────────────────────────────────────────────────────
    if args.json_only:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_section("7.2 Missing Modules", step72, args.verbose)
        _print_section("7.3 Subsystem Coverage", step73, args.verbose)
        _print_section("7.4 Contract Propagation", step74, args.verbose)
        _print_section("7.5 Interface Completeness", step75, args.verbose)
        _print_section("7.6 Orphan Expectations", step76, args.verbose)

        print(f"\n[arch_audit] Completeness score: {score}/100")
        if score >= 80:
            print("[arch_audit] PASS - Architecture health within acceptable bounds")
        else:
            print("[arch_audit] FAIL - Architecture gaps detected (score < 80)")

        # Print key violations
        if step72.get("missing"):
            print("\n  Missing core modules:")
            for m in step72["missing"]:
                print(f"    {m['module']} -> expected at {m['expected_path']}")

        if step73.get("partial"):
            print("\n  Partial subsystems:")
            for s in step73["partial"]:
                detail = step73["subsystems"].get(s, {})
                print(f"    {s}: missing_files={detail.get('missing_files', [])} missing_symbols={detail.get('missing_symbols', [])}")

        if step74.get("violations"):
            print(f"\n  Param propagation violations ({step74['violation_count']}):")
            for v in step74["violations"][:10]:
                print(f"    {v['file']}:{v['lineno']} {v['function']}() missing {v['missing_fields']}")
            if step74["violation_count"] > 10:
                print(f"    ... and {step74['violation_count'] - 10} more")

        if step75.get("drift"):
            print("\n  Interface drift:")
            for d in step75["drift"]:
                print(f"    {d['interface']} ({d['file']}): {d['issue']} missing={d.get('missing_methods', [])}")

        if step76.get("found_referenced_and_undefined"):
            print("\n  Orphan symbols (referenced but not defined):")
            for o in step76["found_referenced_and_undefined"]:
                locs = ", ".join(r["file"] for r in o["referenced_in"][:3])
                print(f"    {o['symbol']}: found in {locs}")

    # ── Write report ────────────────────────────────────────────────────────────
    if not args.no_write:
        output_path = args.output if args.output.is_absolute() else ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        if not args.json_only:
            print(f"\n[arch_audit] Report written -> {_rel(output_path)}")

    return 0 if score >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())
