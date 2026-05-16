from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
CONTROL_PLANE_PATH = ROOT / "dadbot" / "core" / "control_plane.py"


def _find_class(tree: ast.AST, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"missing class: {name}")


def _find_method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing method: {class_node.name}.{name}")


def _calls_method(method_node: ast.FunctionDef, method_name: str) -> bool:
    for node in ast.walk(method_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == method_name:
                return True
    return False


def test_control_plane_declares_complete_run_contract_method() -> None:
    source = CONTROL_PLANE_PATH.read_text(encoding="utf-8-sig", errors="replace")
    tree = ast.parse(source)
    class_node = _find_class(tree, "ExecutionControlPlane")
    _find_method(class_node, "_assert_complete_run_contract")


def test_finalize_submit_success_enforces_complete_run_contract() -> None:
    source = CONTROL_PLANE_PATH.read_text(encoding="utf-8-sig", errors="replace")
    tree = ast.parse(source)
    class_node = _find_class(tree, "ExecutionControlPlane")
    finalize_method = _find_method(class_node, "_finalize_submit_success")
    assert _calls_method(finalize_method, "_assert_complete_run_contract"), (
        "ExecutionControlPlane._finalize_submit_success must call "
        "_assert_complete_run_contract before returning"
    )


def test_production_spine_docs_exist() -> None:
    assert (ROOT / "docs" / "PRODUCTION_SPINE.md").exists()
    assert (ROOT / "docs" / "ARCHIVE_POLICY.md").exists()
