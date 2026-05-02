from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

KERNEL_PREFIXES = (
    "dadbot/core/",
    "dadbot/services/",
    "dadbot/memory/",
    "dadbot/managers/",
    "dadbot_system/",
)

OBSERVABILITY_KEYWORDS = (
    "trace",
    "health_report",
    "standings",
    "gap_analysis",
    "analysis",
    "observability",
    "diagnostic",
    "debug",
    "telemetry",
)

TOOL_RUNTIME_KEYWORDS = (
    "tool_runtime",
    "tool_sandbox",
    "tool_ir",
)

EXCLUDED_DIR_NAMES = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}


@dataclass(frozen=True)
class Violation:
    rule: str
    path: str
    detail: str


def _iter_python_files() -> Iterable[Path]:
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
            continue
        yield path


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _module_name(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_observability_path(rel: str) -> bool:
    lower = rel.lower()
    if lower.startswith("audit/") or lower.startswith("tools/"):
        return True
    if lower.startswith("dadbot/tools/") or lower.startswith("dadbot/telemetry/"):
        return True
    return any(keyword in lower for keyword in OBSERVABILITY_KEYWORDS)


def classify_layer(rel: str) -> str:
    lower = rel.lower()
    if _is_observability_path(lower):
        return "observability"
    if lower.startswith(KERNEL_PREFIXES):
        return "kernel"
    return "other"


def _collect_imports(path: Path) -> list[tuple[str, int]]:
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    imports: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((str(alias.name), int(node.lineno)))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append((str(node.module), int(node.lineno)))
    return imports


def _build_module_index(files: Iterable[Path]) -> dict[str, str]:
    index: dict[str, str] = {}
    for file_path in files:
        rel = _rel(file_path)
        index[_module_name(file_path)] = classify_layer(rel)
    return index


def check_import_isolation() -> list[Violation]:
    files = list(_iter_python_files())
    module_index = _build_module_index(files)

    violations: list[Violation] = []
    for file_path in files:
        rel = _rel(file_path)
        if classify_layer(rel) != "kernel":
            continue

        for module, lineno in _collect_imports(file_path):
            module_lower = module.lower()
            if any(keyword in module_lower for keyword in OBSERVABILITY_KEYWORDS):
                violations.append(
                    Violation(
                        rule="RULE1_IMPORT_ISOLATION",
                        path=f"{rel}:{lineno}",
                        detail=f"Kernel import references observability keyword via import '{module}'",
                    )
                )
                continue

            layer = module_index.get(module)
            if layer == "observability":
                violations.append(
                    Violation(
                        rule="RULE1_IMPORT_ISOLATION",
                        path=f"{rel}:{lineno}",
                        detail=f"Kernel import targets observability module '{module}'",
                    )
                )
    return violations


def check_dependency_graph() -> list[Violation]:
    files = list(_iter_python_files())
    module_index = _build_module_index(files)
    violations: list[Violation] = []

    for file_path in files:
        rel = _rel(file_path)
        src_layer = classify_layer(rel)
        src_module = _module_name(file_path)

        for module, lineno in _collect_imports(file_path):
            dst_layer = module_index.get(module, "other")

            if src_layer == "kernel" and dst_layer == "observability":
                violations.append(
                    Violation(
                        rule="RULE8_DEPENDENCY_DAG",
                        path=f"{rel}:{lineno}",
                        detail=f"Invalid edge kernel -> observability ({src_module} -> {module})",
                    )
                )

            if src_layer == "observability" and any(k in module.lower() for k in TOOL_RUNTIME_KEYWORDS):
                violations.append(
                    Violation(
                        rule="RULE8_DEPENDENCY_DAG",
                        path=f"{rel}:{lineno}",
                        detail=f"Invalid edge observability -> tool_runtime ({src_module} -> {module})",
                    )
                )

    return violations


def run_checks() -> list[Violation]:
    return [*check_import_isolation(), *check_dependency_graph()]
