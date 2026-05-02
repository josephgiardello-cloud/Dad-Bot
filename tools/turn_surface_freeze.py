"""Strict freeze gate for turn-entrypoint surface.

This gate enforces an exact frozen surface for turn entrypoints. Unlike
baseline-aware AST checks that only block *new violations* in specific rules,
this script blocks any external contract drift (addition/removal/rename) in
the frozen turn API surface.

Usage:
  python tools/turn_surface_freeze.py
  python tools/turn_surface_freeze.py --update-baseline
  python tools/turn_surface_freeze.py --json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_BASELINE = Path("tools/turn_surface_freeze_baseline.json")
CONTRACT_MODULE = Path("dadbot/core/_contract_freeze.py")
TURN_MIXIN_MODULE = Path("dadbot/core/turn_mixin.py")
TURN_SERVICE_MODULE = Path("dadbot/services/turn_service.py")

_MIXIN_ENTRYPOINT_PREFIXES: tuple[str, ...] = (
    "execute_turn",
    "process_user_message",
    "handle_turn",
    "run_turn",
)

_SERVICE_ENTRYPOINT_PREFIXES: tuple[str, ...] = ("process_user_message",)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict turn surface freeze gate")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args(argv)


def _read_ast(path: Path) -> ast.Module:
    source = path.read_text(encoding="utf-8-sig").replace("\ufeff", "")
    return ast.parse(source, filename=path.as_posix())


def _extract_literal_tuple_strings(module: ast.Module, name: str) -> list[str]:
    for node in module.body:
        target_name = ""
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            target_name = node.targets[0].id
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value = node.value
        else:
            continue
        if target_name != name or value is None:
            continue
        if isinstance(value, (ast.Tuple, ast.List)):
            items: list[str] = []
            for elt in value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    items.append(elt.value)
            return items
    return []


def _extract_literal_string(module: ast.Module, name: str) -> str:
    for node in module.body:
        target_name = ""
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            target_name = node.targets[0].id
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value = node.value
        else:
            continue
        if target_name != name or value is None:
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return ""


def _class_method_names(module: ast.Module, class_name: str) -> list[str]:
    for node in module.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        names: list[str] = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.append(item.name)
        return names
    return []


def _candidate_entrypoints(method_names: list[str], prefixes: tuple[str, ...]) -> list[str]:
    return sorted(
        name
        for name in method_names
        if not name.startswith("_") and any(name.startswith(prefix) for prefix in prefixes)
    )


def _normalize_service_entries(entries: list[str]) -> list[str]:
    names: list[str] = []
    for item in entries:
        stripped = str(item or "").strip()
        if not stripped:
            continue
        names.append(stripped.split(".")[-1])
    return sorted(set(names))


def collect_current_surface(repo_root: Path) -> dict[str, Any]:
    contract_ast = _read_ast(repo_root / CONTRACT_MODULE)
    mixin_ast = _read_ast(repo_root / TURN_MIXIN_MODULE)
    service_ast = _read_ast(repo_root / TURN_SERVICE_MODULE)

    canonical = _extract_literal_string(contract_ast, "CANONICAL_TURN_ENTRYPOINT")
    legacy_facade = sorted(_extract_literal_tuple_strings(contract_ast, "LEGACY_FACADE_ENTRYPOINTS"))
    legacy_service = sorted(_extract_literal_tuple_strings(contract_ast, "LEGACY_SERVICE_ENTRYPOINTS"))

    mixin_methods = _candidate_entrypoints(
        _class_method_names(mixin_ast, "DadBotTurnMixin"),
        _MIXIN_ENTRYPOINT_PREFIXES,
    )
    service_methods = _candidate_entrypoints(
        _class_method_names(service_ast, "TurnService"),
        _SERVICE_ENTRYPOINT_PREFIXES,
    )

    return {
        "canonical_turn_entrypoint": canonical,
        "legacy_facade_entrypoints": legacy_facade,
        "legacy_service_entrypoints": legacy_service,
        "frozen_runtime_surface": {
            "dadbot/core/turn_mixin.py::DadBotTurnMixin": mixin_methods,
            "dadbot/services/turn_service.py::TurnService": service_methods,
        },
    }


def _list_diff(expected: list[str], actual: list[str]) -> dict[str, list[str]]:
    expected_set = set(expected)
    actual_set = set(actual)
    return {
        "added": sorted(actual_set - expected_set),
        "removed": sorted(expected_set - actual_set),
    }


def _drift_for_key(baseline: dict[str, Any], current: dict[str, Any], key: str) -> dict[str, Any]:
    if key == "canonical_turn_entrypoint":
        expected = str(baseline.get(key) or "")
        actual = str(current.get(key) or "")
        return {
            "changed": expected != actual,
            "expected": expected,
            "actual": actual,
        }
    return _list_diff(
        list(baseline.get(key) or []),
        list(current.get(key) or []),
    )


def evaluate_freeze(repo_root: Path, baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    drift = {
        "canonical_turn_entrypoint": _drift_for_key(baseline, current, "canonical_turn_entrypoint"),
        "legacy_facade_entrypoints": _drift_for_key(baseline, current, "legacy_facade_entrypoints"),
        "legacy_service_entrypoints": _drift_for_key(baseline, current, "legacy_service_entrypoints"),
        "frozen_runtime_surface": {},
    }

    baseline_surface = dict(baseline.get("frozen_runtime_surface") or {})
    current_surface = dict(current.get("frozen_runtime_surface") or {})
    all_surface_keys = sorted(set(baseline_surface) | set(current_surface))
    for surface_key in all_surface_keys:
        drift["frozen_runtime_surface"][surface_key] = _list_diff(
            list(baseline_surface.get(surface_key) or []),
            list(current_surface.get(surface_key) or []),
        )

    has_drift = bool(drift["canonical_turn_entrypoint"].get("changed"))
    has_drift = has_drift or bool(drift["legacy_facade_entrypoints"]["added"])
    has_drift = has_drift or bool(drift["legacy_facade_entrypoints"]["removed"])
    has_drift = has_drift or bool(drift["legacy_service_entrypoints"]["added"])
    has_drift = has_drift or bool(drift["legacy_service_entrypoints"]["removed"])

    for item in drift["frozen_runtime_surface"].values():
        if item["added"] or item["removed"]:
            has_drift = True
            break

    return {
        "ok": not has_drift,
        "drift": drift,
        "current": current,
    }


def _print_human_report(report: dict[str, Any], baseline_path: Path) -> None:
    if report["ok"]:
        print(f"[turn_surface_freeze] OK baseline={baseline_path}")
        return

    print(f"[turn_surface_freeze] FAIL baseline={baseline_path}")
    drift = report["drift"]
    canonical = drift["canonical_turn_entrypoint"]
    if canonical.get("changed"):
        print("  canonical_turn_entrypoint changed")
        print(f"    expected={canonical['expected']!r}")
        print(f"    actual={canonical['actual']!r}")

    for key in ("legacy_facade_entrypoints", "legacy_service_entrypoints"):
        entry = drift[key]
        if entry["added"] or entry["removed"]:
            print(f"  {key} drift")
            if entry["added"]:
                print(f"    added={entry['added']}")
            if entry["removed"]:
                print(f"    removed={entry['removed']}")

    for surface_key, entry in sorted(drift["frozen_runtime_surface"].items()):
        if not entry["added"] and not entry["removed"]:
            continue
        print(f"  {surface_key} drift")
        if entry["added"]:
            print(f"    added={entry['added']}")
        if entry["removed"]:
            print(f"    removed={entry['removed']}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    baseline_path = args.baseline if args.baseline.is_absolute() else root / args.baseline
    current = collect_current_surface(root)

    if args.update_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[turn_surface_freeze] Baseline written -> {baseline_path}")
        return 0

    if not baseline_path.exists():
        print(
            f"[turn_surface_freeze] ERROR: baseline not found at {baseline_path}. "
            "Run with --update-baseline first.",
            file=sys.stderr,
        )
        return 2

    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    report = evaluate_freeze(root, baseline_payload, current)
    if args.json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human_report(report, baseline_path)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())