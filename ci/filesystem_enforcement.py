from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ci.import_graph_check import classify_layer, iter_python_files

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_KERNEL_ARTIFACT_DIR_MARKERS = (
    "/reports",
    "\\reports",
    "/analysis",
    "\\analysis",
    "/debug",
    "\\debug",
    "/health",
    "\\health",
    "/standings",
    "\\standings",
)

KERNEL_WRITE_PATTERNS = (
    "write_text(",
    "write_bytes(",
    "open(",
    "json.dump(",
)

OBSERVABILITY_FORBIDDEN_MUTATIONS = (
    "append_event(",
    "write_event(",
    "mutate_memory_store(",
    "sync_graph_store(",
    "apply_mutation(",
    "mutation_queue.queue(",
    "KernelEventTotalityLock.note_event(",
)

EXCLUDED_DIR_NAMES = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
INCLUDED_TOP_LEVEL_DIRS = {"dadbot", "dadbot_system", "ci", "tools"}


@dataclass(frozen=True)
class Violation:
    rule: str
    path: str
    detail: str


def _iter_python_files() -> list[Path]:
    return list(iter_python_files(
        ROOT, excluded_dirs=EXCLUDED_DIR_NAMES, included_top_level_dirs=INCLUDED_TOP_LEVEL_DIRS,
    ))


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check_kernel_filesystem_boundaries() -> list[Violation]:
    violations: list[Violation] = []
    for file_path in _iter_python_files():
        rel = _rel(file_path)
        if classify_layer(rel) != "kernel":
            continue

        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for lineno, line in enumerate(lines, start=1):
            low = line.lower()
            if any(marker in low for marker in FORBIDDEN_KERNEL_ARTIFACT_DIR_MARKERS):
                if any(pattern in line for pattern in KERNEL_WRITE_PATTERNS):
                    violations.append(
                        Violation(
                            rule="RULE6_ARTIFACT_SEPARATION",
                            path=f"{rel}:{lineno}",
                            detail="Kernel write call targets forbidden artifact directory marker",
                        )
                    )
    return violations


def check_observability_write_directionality() -> list[Violation]:
    violations: list[Violation] = []
    for file_path in _iter_python_files():
        rel = _rel(file_path)
        if classify_layer(rel) != "observability":
            continue

        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for lineno, line in enumerate(lines, start=1):
            if any(pattern in line for pattern in OBSERVABILITY_FORBIDDEN_MUTATIONS):
                violations.append(
                    Violation(
                        rule="RULE2_WRITE_DIRECTIONALITY",
                        path=f"{rel}:{lineno}",
                        detail="Observability module appears to mutate kernel/event/tool/memory state",
                    )
                )
    return violations


def run_checks() -> list[Violation]:
    return [*check_kernel_filesystem_boundaries(), *check_observability_write_directionality()]
