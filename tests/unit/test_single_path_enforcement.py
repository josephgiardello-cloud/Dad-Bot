from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "dadbot" / "core"

ALLOWED_LEDGER_WRITE_FILES = {
    "dadbot/core/ledger_writer.py",
    "dadbot/core/execution_ledger.py",
    "dadbot/core/execution_ledger_memory.py",
}


def _iter_core_files() -> list[Path]:
    return [
        path
        for path in sorted(CORE.rglob("*.py"))
        if "__pycache__" not in path.parts and path.name != "__init__.py"
    ]


def _name_contains_ledger(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return "ledger" in node.id.lower()
    if isinstance(node, ast.Attribute):
        return _name_contains_ledger(node.value) or "ledger" in node.attr.lower()
    return False


def test_single_execution_mutation_ledger_path_enforced() -> None:
    control_plane_source = (CORE / "control_plane.py").read_text(encoding="utf-8")
    control_plane_tree = ast.parse(control_plane_source)

    adapter_constructions = 0
    for node in ast.walk(control_plane_tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "LedgerWriterAdapter":
            adapter_constructions += 1

    assert adapter_constructions == 1, "Control plane must construct exactly one ledger adapter gateway"

    mutation_source = (CORE / "graph_mutation.py").read_text(encoding="utf-8")
    assert "coerce_mutation_kind" in mutation_source
    assert "MutationKind(self.type)" not in mutation_source

    bypasses: list[str] = []
    for pyfile in _iter_core_files():
        rel = pyfile.relative_to(ROOT).as_posix()
        if rel in ALLOWED_LEDGER_WRITE_FILES:
            continue
        tree = ast.parse(pyfile.read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"write", "append"} and _name_contains_ledger(node.func.value):
                    bypasses.append(f"{rel}:{node.lineno}")

    assert not bypasses, f"Detected direct ledger bypass path(s): {bypasses}"
