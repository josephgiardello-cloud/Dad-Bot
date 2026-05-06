from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImplicitStateFinding:
    module: str
    symbol: str
    kind: str
    severity: str
    line: int


_MUTABLE_AST_NODES = (ast.Dict, ast.List, ast.Set)

_WINDOWS_CLOUD_RECALL_ATTRIBUTE_MASK = 0x1000 | 0x40000 | 0x400000

_DEFAULT_EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tmp",
        ".vscode",
        "__pycache__",
        "artifacts",
        "session_logs",
        "snapshots",
        "system_snapshot",
        "SYSTEM_SNAPSHOT",
        "runtime",
        "node_modules",
    },
)

_PACKAGE_SCAN_ROOTS: dict[str, tuple[str, ...]] = {
    "dadbot": ("core", "managers", "memory", "services"),
    "dadbot_system": (),
}


def _module_name(root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(root)
    return ".".join(rel.with_suffix("").parts)


def _is_windows_cloud_recall_file(file_path: Path) -> bool:
    if os.name != "nt":
        return False
    try:
        attrs = int(getattr(os.stat(file_path), "st_file_attributes", 0) or 0)
    except OSError:
        return False
    return bool(attrs & _WINDOWS_CLOUD_RECALL_ATTRIBUTE_MASK)


def _scan_file(root: Path, file_path: Path) -> list[ImplicitStateFinding]:
    if _is_windows_cloud_recall_file(file_path):
        return []
    try:
        source = file_path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    findings: list[ImplicitStateFinding] = []
    module = _module_name(root, file_path)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            value = node.value
            if isinstance(value, _MUTABLE_AST_NODES):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = str(target.id)
                        if name.isupper() or name.startswith("_"):
                            continue
                        findings.append(
                            ImplicitStateFinding(
                                module=module,
                                symbol=name,
                                kind="module_mutable_global",
                                severity="warning",
                                line=int(node.lineno),
                            ),
                        )
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = node.value
            if isinstance(value, _MUTABLE_AST_NODES):
                name = str(node.target.id)
                if name.isupper() or name.startswith("_"):
                    continue
                findings.append(
                    ImplicitStateFinding(
                        module=module,
                        symbol=name,
                        kind="module_mutable_global",
                        severity="warning",
                        line=int(node.lineno),
                    ),
                )
    return findings


def _iter_python_files(
    package_root: Path,
    excluded_dir_names: set[str],
) -> list[Path]:
    files: list[Path] = []
    for dir_path, dir_names, file_names in os.walk(package_root):
        dir_names[:] = [
            name
            for name in dir_names
            if name not in excluded_dir_names and name.lower() not in excluded_dir_names
        ]
        for file_name in file_names:
            if file_name.endswith(".py"):
                files.append(Path(dir_path) / file_name)
    return files


def scan_implicit_state(runtime_root: Path) -> list[ImplicitStateFinding]:
    root = Path(runtime_root)
    findings: list[ImplicitStateFinding] = []
    excluded_dir_names = {
        str(name).strip().lower()
        for name in _DEFAULT_EXCLUDED_DIR_NAMES
        if str(name).strip()
    }
    for package in ("dadbot", "dadbot_system"):
        package_root = root / package
        if not package_root.exists():
            continue
        scan_subdirs = _PACKAGE_SCAN_ROOTS.get(package, ())
        if scan_subdirs:
            candidate_roots = [package_root / subdir for subdir in scan_subdirs]
        else:
            candidate_roots = [package_root]
        for candidate_root in candidate_roots:
            if not candidate_root.exists():
                continue
            for file_path in _iter_python_files(candidate_root, excluded_dir_names):
                findings.extend(_scan_file(root, file_path))
    return findings


def enforce_implicit_state_guard(
    runtime_root: Path,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    findings = scan_implicit_state(runtime_root)
    report = {
        "count": len(findings),
        "strict": bool(strict),
        "findings": [
            {
                "module": item.module,
                "symbol": item.symbol,
                "kind": item.kind,
                "severity": item.severity,
                "line": int(item.line),
            }
            for item in findings
        ],
    }
    if strict and findings:
        first = findings[0]
        raise RuntimeError(
            "Implicit-state scanner violation: mutable module global detected "
            f"module={first.module!r} symbol={first.symbol!r} line={first.line}",
        )
    return report
