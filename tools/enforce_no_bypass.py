"""Hard enforcement for contract-only execution pathways.

Blocks three classes of bypass:
  1) Direct ledger writes outside adapter/writer internals.
  2) Direct MutationKind construction outside contract module.
  3) Execution context primitive leakage from ctx into trace_id/kernel_step_id.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET_ROOT = ROOT / "dadbot"

ALLOWED_LEDGER_DIRECT = {
    "dadbot/core/ledger_writer.py",
    "dadbot/core/execution_ledger.py",
    "dadbot/core/execution_ledger_memory.py",
}

ALLOWED_MUTATION_KIND_CONSTRUCTION = {
    "dadbot/core/contracts/mutation.py",
    "dadbot/core/graph_types.py",
}

FORBIDDEN_PRIMITIVES = {"trace_id", "kernel_step_id"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enforce no bypass architecture boundaries")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args(argv)


def _iter_files() -> list[Path]:
    return [
        path
        for path in sorted(TARGET_ROOT.rglob("*.py"))
        if "__pycache__" not in path.parts and ".venv" not in path.parts
    ]


def _name_contains_ledger(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return "ledger" in node.id.lower()
    if isinstance(node, ast.Attribute):
        return _name_contains_ledger(node.value) or "ledger" in node.attr.lower()
    return False


def _analyze_file(pyfile: Path) -> dict[str, list[str]]:
    rel = pyfile.relative_to(ROOT).as_posix()
    text = pyfile.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError as exc:
        return {
            "ledger_bypass": [],
            "mutation_kind_bypass": [],
            "execution_context_unpacking": [f"{rel}:{exc.lineno} syntax_error: {exc.msg}"],
        }

    ledger_bypass: list[str] = []
    mutation_kind_bypass: list[str] = []
    execution_context_unpacking: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"write", "append"} and rel not in ALLOWED_LEDGER_DIRECT:
                if _name_contains_ledger(node.func.value):
                    ledger_bypass.append(
                        f"{rel}:{node.lineno} direct ledger {node.func.attr} call outside adapter/writer",
                    )

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "MutationKind" and rel not in ALLOWED_MUTATION_KIND_CONSTRUCTION:
                mutation_kind_bypass.append(
                    f"{rel}:{node.lineno} direct MutationKind construction outside contract module",
                )

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arg_names = {arg.arg for arg in node.args.args}
            if node.args.kwarg:
                arg_names.add(node.args.kwarg.arg)
            if node.args.vararg:
                arg_names.add(node.args.vararg.arg)
            arg_names.update(arg.arg for arg in node.args.kwonlyargs)

            if "ctx" in arg_names and FORBIDDEN_PRIMITIVES & arg_names:
                leaked = sorted(FORBIDDEN_PRIMITIVES & arg_names)
                execution_context_unpacking.append(
                    f"{rel}:{node.lineno} function {node.name} mixes ctx with primitive params {', '.join(leaked)}",
                )

        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in FORBIDDEN_PRIMITIVES:
                    if isinstance(kw.value, ast.Attribute) and isinstance(kw.value.value, ast.Name):
                        if kw.value.value.id == "ctx":
                            execution_context_unpacking.append(
                                f"{rel}:{node.lineno} forwards ctx.{kw.arg} into primitive kwarg {kw.arg}",
                            )

    return {
        "ledger_bypass": ledger_bypass,
        "mutation_kind_bypass": mutation_kind_bypass,
        "execution_context_unpacking": execution_context_unpacking,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    violations = {
        "ledger_bypass": [],
        "mutation_kind_bypass": [],
        "execution_context_unpacking": [],
    }

    for pyfile in _iter_files():
        res = _analyze_file(pyfile)
        for key in violations:
            violations[key].extend(res.get(key, []))

    total = sum(len(items) for items in violations.values())

    if args.json_output:
        print(json.dumps({"violations": violations, "total": total}, indent=2))
    else:
        print("[enforce_no_bypass] violations")
        for key, items in violations.items():
            print(f"  {key}={len(items)}")
            for item in items:
                print(f"    {item}")

    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
