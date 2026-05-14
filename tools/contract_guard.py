"""Contract guard for execution context signatures.

Rules:
  - Functions must not accept `trace_id` directly.
  - Functions must not accept `kernel_step_id` directly.

This gate is baseline-aware so existing debt can be tracked without blocking
all progress. New violations fail the gate.

Usage:
  python tools/contract_guard.py
  python tools/contract_guard.py --update-baseline
  python tools/contract_guard.py --json

Exit codes:
  0 - pass
  1 - violations found
  2 - baseline missing
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

DEFAULT_BASELINE = Path("tools/contract_guard_baseline.json")
FORBIDDEN_PARAMS = {"trace_id", "kernel_step_id"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execution contract guard")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args(argv)


def _iter_python_files(root: Path) -> list[Path]:
    return [
        path
        for path in sorted((root / "dadbot").rglob("*.py"))
        if "__pycache__" not in path.parts and ".venv" not in path.parts
    ]


def _function_signature_violations(pyfile: Path, repo_root: Path) -> list[str]:
    try:
        source = pyfile.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(pyfile))
    except (OSError, SyntaxError):
        return []

    violations: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        arg_names = [arg.arg for arg in node.args.args]
        if node.args.vararg is not None:
            arg_names.append(node.args.vararg.arg)
        arg_names.extend(arg.arg for arg in node.args.kwonlyargs)
        if node.args.kwarg is not None:
            arg_names.append(node.args.kwarg.arg)

        forbidden = sorted(name for name in arg_names if name in FORBIDDEN_PARAMS)
        if not forbidden:
            continue

        rel_path = pyfile.relative_to(repo_root).as_posix()
        fn_name = getattr(node, "name", "<anonymous>")
        joined = ", ".join(forbidden)
        violations.append(f"{rel_path}:{node.lineno} {fn_name}({joined})")

    return violations


def _collect_violations(repo_root: Path) -> list[str]:
    violations: list[str] = []
    for pyfile in _iter_python_files(repo_root):
        violations.extend(_function_signature_violations(pyfile, repo_root))
    return sorted(violations)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    baseline_path = args.baseline if args.baseline.is_absolute() else root / args.baseline

    current = _collect_violations(root)

    if args.update_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "forbidden_params": sorted(FORBIDDEN_PARAMS),
            "violations": current,
        }
        baseline_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[contract_guard] Baseline written -> {baseline_path}")
        print(f"  tracked violations={len(current)}")
        return 0

    if not baseline_path.exists():
        print(
            f"[contract_guard] ERROR: baseline not found at {baseline_path}. "
            "Run with --update-baseline first.",
            file=sys.stderr,
        )
        return 2

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    approved = set(str(item) for item in baseline.get("violations", []))

    new_violations = [item for item in current if item not in approved]

    if args.json_output:
        print(
            json.dumps(
                {
                    "current_count": len(current),
                    "baseline_count": len(approved),
                    "new_violations": new_violations,
                },
                indent=2,
            ),
        )
    else:
        print(
            f"[contract_guard] current={len(current)} baseline={len(approved)} "
            f"new={len(new_violations)}",
        )
        if new_violations:
            print("[contract_guard] NEW violations:")
            for item in new_violations:
                print(f"  {item}")

    return 1 if new_violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
