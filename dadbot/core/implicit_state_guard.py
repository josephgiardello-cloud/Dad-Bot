from __future__ import annotations

import ast
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


def _module_name(root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(root)
    return ".".join(rel.with_suffix("").parts)


def _scan_file(root: Path, file_path: Path) -> list[ImplicitStateFinding]:
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


def scan_implicit_state(runtime_root: Path) -> list[ImplicitStateFinding]:
    root = Path(runtime_root)
    findings: list[ImplicitStateFinding] = []
    for package in ("dadbot", "dadbot_system"):
        package_root = root / package
        if not package_root.exists():
            continue
        for file_path in package_root.rglob("*.py"):
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
