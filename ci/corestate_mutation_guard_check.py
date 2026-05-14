from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (ROOT / "dadbot", ROOT / "dadbot_system")
EXCLUDED_DIR_NAMES = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
ALLOWED_WRITE_FILES = {
    "dadbot/memory/storage.py",
}
STORAGE_FILE = "dadbot/memory/storage.py"
MUTATING_METHODS = {
    "append",
    "extend",
    "insert",
    "pop",
    "remove",
    "clear",
    "update",
    "setdefault",
}

ALLOWED_STORAGE_WRITE_CALLERS = {
    "commit",
    "rollback",
    "clear_memory_projection",
}

ALLOWED_STORAGE_TRANSITION_CALLERS = {
    "_emit_memory_event",
}

REQUIRED_STORAGE_EVENT_CALLS: dict[str, str] = {
    "_run_mutation_commit_path": "_emit_memory_events",
    "load_memory_store": "_emit_memory_event",
    "clear_memory_projection": "_emit_memory_event",
}


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    detail: str


class MemoryStoreWriteVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[tuple[int, str]] = []

    @staticmethod
    def _is_memory_store_expr(node: ast.AST) -> bool:
        if isinstance(node, ast.Subscript):
            return MemoryStoreWriteVisitor._is_memory_store_expr(node.value)
        if isinstance(node, ast.Attribute):
            if node.attr == "memory_store":
                return True
            return MemoryStoreWriteVisitor._is_memory_store_expr(node.value)
        return False

    def _check_assignment_target(self, target: ast.AST, lineno: int) -> None:
        if isinstance(target, ast.Subscript) and self._is_memory_store_expr(target.value):
            self.violations.append((lineno, "direct memory_store subscript assignment bypasses reducer"))
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._check_assignment_target(item, lineno)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._check_assignment_target(target, int(node.lineno))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_assignment_target(node.target, int(node.lineno))
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_assignment_target(node.target, int(node.lineno))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in MUTATING_METHODS
            and self._is_memory_store_expr(func.value)
        ):
            self.violations.append((int(node.lineno), f"direct memory_store mutating call '{func.attr}(...)' bypasses reducer"))
        self.generic_visit(node)


class StorageCanonicalEventVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[tuple[int, str]] = []
        self._function_stack: list[str] = []
        self._functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}

    def _current_function(self) -> str:
        if not self._function_stack:
            return "<module>"
        return self._function_stack[-1]

    def _push_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._function_stack.append(node.name)
        self._functions[node.name] = node
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._push_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._push_function(node)

    def _call_name(self, node: ast.Call) -> str | None:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def visit_Call(self, node: ast.Call) -> None:
        call_name = self._call_name(node)
        function_name = self._current_function()

        if call_name == "_write_memory_store_unlocked" and function_name not in ALLOWED_STORAGE_WRITE_CALLERS:
            self.violations.append(
                (
                    int(node.lineno),
                    "raw _write_memory_store_unlocked(...) is only allowed in commit/rollback/clear_memory_projection; "
                    "all other memory mutations must route through canonical events",
                )
            )

        if call_name in {"transition", "push_core_state_event"} and function_name not in ALLOWED_STORAGE_TRANSITION_CALLERS:
            self.violations.append(
                (
                    int(node.lineno),
                    f"direct {call_name}(...) is only allowed inside _emit_memory_event",
                )
            )

        self.generic_visit(node)

    def finalize_required_event_calls(self) -> None:
        for function_name, required_call in REQUIRED_STORAGE_EVENT_CALLS.items():
            function_node = self._functions.get(function_name)
            if function_node is None:
                self.violations.append(
                    (
                        1,
                        f"required canonical mutation function '{function_name}' not found",
                    )
                )
                continue

            found = False
            for node in ast.walk(function_node):
                if not isinstance(node, ast.Call):
                    continue
                call_name = self._call_name(node)
                if call_name == required_call:
                    found = True
                    break
            if not found:
                self.violations.append(
                    (
                        int(function_node.lineno),
                        f"function '{function_name}' must call {required_call}(...) to preserve canonical memory event routing",
                    )
                )


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            rel = path.relative_to(ROOT)
            if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
                continue
            if rel.as_posix().startswith("tests/"):
                continue
            files.append(path)
    return files


def run_check() -> list[Violation]:
    violations: list[Violation] = []
    for file_path in _iter_python_files():
        rel = file_path.relative_to(ROOT).as_posix()
        source = file_path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        if rel not in ALLOWED_WRITE_FILES:
            visitor = MemoryStoreWriteVisitor()
            visitor.visit(tree)
            for line, detail in visitor.violations:
                violations.append(Violation(path=rel, line=line, detail=detail))

        if rel == STORAGE_FILE:
            storage_visitor = StorageCanonicalEventVisitor()
            storage_visitor.visit(tree)
            storage_visitor.finalize_required_event_calls()
            for line, detail in storage_visitor.violations:
                violations.append(Violation(path=rel, line=line, detail=detail))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect memory mutation paths that bypass canonical CoreState event routing.",
    )
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Fail (exit 1) when violations are present. Without this flag, warn only.",
    )
    args = parser.parse_args()

    violations = run_check()
    mode = "ENFORCE" if args.enforce else "WARN"
    if not violations:
        print(f"PASS  corestate-mutation-guard ({mode}): no canonical memory-routing bypasses found")
        return 0

    print(f"WARN  corestate-mutation-guard ({mode}): found {len(violations)} potential bypass(es)")
    for item in violations:
        print(f"  - {item.path}:{item.line}  {item.detail}")

    if args.enforce:
        print("FAIL  corestate-mutation-guard: enforcement enabled and violations are present")
        return 1

    print("PASS  corestate-mutation-guard: warn mode only (non-blocking)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
