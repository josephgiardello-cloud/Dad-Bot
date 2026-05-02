from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ci.import_graph_check import classify_layer

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIR_NAMES = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}

OBSERVABILITY_MUTATING_METHODS = {
    "append",
    "extend",
    "update",
    "clear",
    "pop",
    "setdefault",
}

KERNEL_INJECTION_PATTERNS = (
    "register_trace_hook(",
    "inject_debug(",
    "register_logging_hook(",
    "debug_interceptor",
)

DUAL_PURPOSE_OBS_KEYWORDS = (
    "health_report",
    "full_system_standings",
    "gap_analysis",
    "diagnostic_report",
    "trace_tab",
)

DUAL_PURPOSE_EXEC_KEYWORDS = (
    "handle_turn",
    "run_turn",
    "mutation_queue",
    "save_turn_event",
    "execute_tool",
)


@dataclass(frozen=True)
class Violation:
    rule: str
    path: str
    detail: str


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
            continue
        files.append(path)
    return files


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check_runtime_injection_ban() -> list[Violation]:
    violations: list[Violation] = []
    for file_path in _iter_python_files():
        rel = _rel(file_path)
        if classify_layer(rel) != "kernel":
            continue

        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for lineno, line in enumerate(lines, start=1):
            if any(pattern in line for pattern in KERNEL_INJECTION_PATTERNS):
                violations.append(
                    Violation(
                        rule="RULE9_RUNTIME_INJECTION_BAN",
                        path=f"{rel}:{lineno}",
                        detail="Kernel execution path includes runtime trace/debug hook injection",
                    )
                )
    return violations


def check_no_dual_purpose_files() -> list[Violation]:
    violations: list[Violation] = []
    for file_path in _iter_python_files():
        rel = _rel(file_path)
        if classify_layer(rel) != "kernel":
            continue

        source = file_path.read_text(encoding="utf-8", errors="replace").lower()
        has_obs = any(keyword in source for keyword in DUAL_PURPOSE_OBS_KEYWORDS)
        has_exec = any(keyword in source for keyword in DUAL_PURPOSE_EXEC_KEYWORDS)
        if has_obs and has_exec:
            violations.append(
                Violation(
                    rule="RULE10_NO_DUAL_PURPOSE",
                    path=rel,
                    detail="Kernel file appears to mix execution logic with reporting/diagnostic concerns",
                )
            )
    return violations


class SharedMutableVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[tuple[int, str]] = []
        self._arg_stack: list[set[str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        args = {arg.arg for arg in node.args.args}
        self._arg_stack.append(args)
        self.generic_visit(node)
        self._arg_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        args = {arg.arg for arg in node.args.args}
        self._arg_stack.append(args)
        self.generic_visit(node)
        self._arg_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        current_args = self._arg_stack[-1] if self._arg_stack else set()
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                if target.value.id in {"kernel", "runtime", "bot", "state", "graph", "memory"}:
                    self.violations.append((int(node.lineno), f"attribute assignment on '{target.value.id}'"))
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                if target.value.id in current_args:
                    self.violations.append((int(node.lineno), f"subscript mutation on function arg '{target.value.id}'"))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        current_args = self._arg_stack[-1] if self._arg_stack else set()
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            base = node.func.value.id
            method = node.func.attr
            if base in current_args and method in OBSERVABILITY_MUTATING_METHODS:
                self.violations.append((int(node.lineno), f"mutating call '{base}.{method}(...)'"))
        self.generic_visit(node)


def check_no_cross_layer_shared_mutation() -> list[Violation]:
    violations: list[Violation] = []
    for file_path in _iter_python_files():
        rel = _rel(file_path)
        if classify_layer(rel) != "observability":
            continue

        source = file_path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        visitor = SharedMutableVisitor()
        visitor.visit(tree)
        for lineno, detail in visitor.violations:
            violations.append(
                Violation(
                    rule="RULE4_NO_SHARED_MUTABLE",
                    path=f"{rel}:{lineno}",
                    detail=detail,
                )
            )
    return violations


def run_checks() -> list[Violation]:
    return [
        *check_runtime_injection_ban(),
        *check_no_dual_purpose_files(),
        *check_no_cross_layer_shared_mutation(),
    ]
