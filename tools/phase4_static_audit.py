"""Phase 4 Static Hotspot Audit — repo-wide structural performance hygiene scan.

This is NOT a runtime test.  It scans Python source files for patterns that
are known performance or correctness risks and produces a human-readable report.

Usage:
    python tools/phase4_static_audit.py              # all findings
    python tools/phase4_static_audit.py --strict     # non-zero exit on any finding
    python tools/phase4_static_audit.py --json       # machine-readable JSON output
    python tools/phase4_static_audit.py --path dadbot/core  # restrict scan root

Exit codes:
    0 — no findings (or --strict not set)
    1 — findings present AND --strict flag was passed
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    file: str
    line: int
    rule: str
    severity: str  # "warn" | "error"
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "rule": self.rule,
            "severity": self.severity,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Rule implementations
# ---------------------------------------------------------------------------


class _DeepCopyRule:
    """copy.deepcopy() in hot execution paths is a performance risk."""

    id = "P001"
    severity = "warn"

    def check(self, path: Path, text: str, _tree: ast.Module) -> list[Finding]:
        findings: list[Finding] = []
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if "copy.deepcopy(" in stripped or "deepcopy(" in stripped:
                findings.append(
                    Finding(
                        file=str(path),
                        line=lineno,
                        rule=self.id,
                        severity=self.severity,
                        detail="copy.deepcopy() usage — consider shallow copy or structural sharing",
                    )
                )
        return findings


class _JsonRoundtripRule:
    """json.loads(json.dumps(...)) is a slow and redundant serialization roundtrip."""

    id = "P002"
    severity = "warn"

    _PATTERN = re.compile(r"json\.(loads|dumps)\s*\(\s*json\.(loads|dumps)\s*\(")

    def check(self, path: Path, text: str, _tree: ast.Module) -> list[Finding]:
        findings: list[Finding] = []
        for lineno, line in enumerate(text.splitlines(), 1):
            if self._PATTERN.search(line):
                findings.append(
                    Finding(
                        file=str(path),
                        line=lineno,
                        rule=self.id,
                        severity=self.severity,
                        detail="Redundant JSON serialization roundtrip (json.loads/json.dumps nesting)",
                    )
                )
        return findings


class _UnboundedWhileLoopRule:
    """while True: loops without a break statement risk hanging under fault injection."""

    id = "P003"
    severity = "warn"

    def check(self, path: Path, text: str, _tree: ast.Module) -> list[Finding]:
        findings: list[Finding] = []
        lines = text.splitlines()
        for lineno, line in enumerate(lines, 1):
            if re.search(r"\bwhile\s+True\s*:", line.strip()):
                # Scan the surrounding 30 lines for a break
                window_start = max(0, lineno - 1)
                window_end = min(len(lines), lineno + 30)
                window = "\n".join(lines[window_start:window_end])
                if "break" not in window and "return" not in window:
                    findings.append(
                        Finding(
                            file=str(path),
                            line=lineno,
                            rule=self.id,
                            severity=self.severity,
                            detail="while True: loop with no visible break/return in next 30 lines",
                        )
                    )
        return findings


class _DirectDatetimeNowInStageRule:
    """datetime.now() inside stage execute() methods bypasses VirtualClock."""

    id = "D001"
    severity = "error"

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.violations: list[int] = []

        def _is_in_execute(self, node: ast.FunctionDef) -> bool:
            return node.name == "execute"

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            if self._is_in_execute(node):
                for child in ast.walk(node):
                    if (
                        isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Attribute)
                        and child.func.attr == "now"
                        and isinstance(child.func.value, ast.Name)
                        and child.func.value.id == "datetime"
                    ):
                        self.violations.append(getattr(child, "lineno", 0))
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def check(self, path: Path, _text: str, tree: ast.Module) -> list[Finding]:
        # Only scan non-temporal-layer files
        _ALLOW = {
            "graph.py",
            "state.py",
            "compat_mixin.py",
            "turn_service.py",
            "background.py",
            "agentic.py",
            "storage.py",
            "advice_audit.py",
            "graph_manager.py",
            "control_plane.py",
            "conversation_persistence.py",
        }
        if path.name in _ALLOW:
            return []
        visitor = self._Visitor()
        visitor.visit(tree)
        return [
            Finding(
                file=str(path),
                line=line,
                rule=self.id,
                severity=self.severity,
                detail="datetime.now() inside execute() — use TurnTemporalAxis or VirtualClock",
            )
            for line in visitor.violations
        ]


class _RandomImportInCoreRule:
    """'import random' in core/services breaks deterministic replay."""

    id = "D002"
    severity = "error"

    def check(self, path: Path, _text: str, tree: ast.Module) -> list[Finding]:
        # Only report for core and services directories
        path_str = str(path).replace("\\", "/")
        if "/dadbot/core/" not in path_str and "/dadbot/services/" not in path_str:
            return []
        findings: list[Finding] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else ([str(node.module or "")])
                )
                if any(n == "random" or n.startswith("random.") for n in names):
                    findings.append(
                        Finding(
                            file=str(path),
                            line=getattr(node, "lineno", 0),
                            rule=self.id,
                            severity=self.severity,
                            detail="'import random' in core/services — use DeterminismBoundary.seal()",
                        )
                    )
        return findings


class _MutationOutsideSaveNodeRule:
    """mutation_queue.queue() calls outside approved files are a mutation-guard violation risk."""

    id = "M001"
    severity = "warn"

    _APPROVED = frozenset({"graph.py", "turn_service.py"})

    def check(self, path: Path, text: str, _tree: ast.Module) -> list[Finding]:
        if path.name in self._APPROVED:
            return []
        findings: list[Finding] = []
        for lineno, line in enumerate(text.splitlines(), 1):
            if "mutation_queue.queue(" in line:
                findings.append(
                    Finding(
                        file=str(path),
                        line=lineno,
                        rule=self.id,
                        severity=self.severity,
                        detail="mutation_queue.queue() call outside approved files (graph.py, turn_service.py)",
                    )
                )
        return findings


class _SilentExceptPassRule:
    """bare 'except: pass' silently swallows errors — use specific exception types."""

    id = "C001"
    severity = "warn"

    _PATTERN = re.compile(r"^\s*except\s*:\s*pass\s*$")

    def check(self, path: Path, text: str, _tree: ast.Module) -> list[Finding]:
        findings: list[Finding] = []
        for lineno, line in enumerate(text.splitlines(), 1):
            if self._PATTERN.match(line):
                findings.append(
                    Finding(
                        file=str(path),
                        line=lineno,
                        rule=self.id,
                        severity=self.severity,
                        detail="bare 'except: pass' silently swallows all exceptions",
                    )
                )
        return findings


class _LargeInlineListLiteralRule:
    """Inline list literals with >50 elements in module scope waste startup memory."""

    id = "C002"
    severity = "warn"
    _THRESHOLD = 50

    def check(self, path: Path, _text: str, tree: ast.Module) -> list[Finding]:
        findings: list[Finding] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.List) and len(node.elts) > self._THRESHOLD:
                findings.append(
                    Finding(
                        file=str(path),
                        line=getattr(node, "lineno", 0),
                        rule=self.id,
                        severity=self.severity,
                        detail=f"Inline list literal with {len(node.elts)} elements (>{self._THRESHOLD}) — consider a generator or external data file",
                    )
                )
        return findings


class _ForbiddenCouplingRule:
    """Detect forbidden cross-module coupling: graph ↔ persistence ↔ UX direct imports.

    Industry expectation: the execution graph layer, the persistence layer, and
    the UX/observability layer must NOT have direct circular or same-level
    imports.  All cross-layer communication must go through declared adapters.

    Forbidden patterns
    ------------------
    - ``dadbot/core/graph.py`` imports directly from ``dadbot/core/ux_projection``
      (should go through ``graph_side_effects`` orchestrator).
    - Any UI module (``dadbot/ui/``) imports from ``dadbot/core/graph``
      execution internals (must use the public API / registry only).
    - Any persistence module (``dadbot/managers/conversation_persistence``) imports
      from ``dadbot/core/graph`` at execution level (creates replay / coupling
      hazard).
    - ``dadbot/core/ux_projection.py`` imports from ``dadbot/core/graph``
      (UX is a projection layer — it must be downstream only).
    """

    id = "X001"
    severity = "error"

    # (subject_path_fragment, forbidden_import_fragment, hint)
    _RULES: list[tuple[str, str, str]] = [
        (
            "dadbot/core/graph.py",
            "dadbot.core.ux_projection",
            "graph.py must not import ux_projection directly — route through graph_side_effects.GraphSideEffectsOrchestrator",
        ),
        (
            "dadbot/ui/",
            "dadbot.core.graph",
            "UI layer must not import from dadbot.core.graph directly — use registry/public API",
        ),
        (
            "dadbot/core/ux_projection",
            "dadbot.core.graph",
            "ux_projection must not import from graph — UX is a downstream projection layer only",
        ),
        (
            "dadbot/core/ux_projection",
            "dadbot.managers",
            "ux_projection must not import from persistence managers",
        ),
    ]

    def check(self, path: Path, _text: str, tree: ast.Module) -> list[Finding]:
        path_str = str(path).replace("\\", "/")
        # Read the source text to detect # re-export suppression comments.
        try:
            source_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            source_lines = []

        findings: list[Finding] = []

        for subject_fragment, forbidden_fragment, hint in self._RULES:
            if subject_fragment not in path_str:
                continue
            # Scan all import nodes for the forbidden module.
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    module = str(node.module or "")
                    imported_names = [module]
                else:
                    continue
                lineno = getattr(node, "lineno", 0)
                # Allow lines explicitly annotated as re-exports.
                if lineno > 0 and lineno <= len(source_lines):
                    if "# re-export" in source_lines[lineno - 1]:
                        continue
                for name in imported_names:
                    if forbidden_fragment in name:
                        findings.append(
                            Finding(
                                file=str(path),
                                line=lineno,
                                rule=self.id,
                                severity=self.severity,
                                detail=f"Forbidden coupling: {hint} (import: {name!r})",
                            )
                        )
        return findings


class _ImportGraphCycleRule:
    """Detect cross-module dependency inversion violations using import-graph analysis.

    Enforces the architectural layer order:
        infrastructure ← core ← managers ← services ← ui

    Each layer may import from layers below it but NOT from layers above it.
    Violations indicate dependency inversion — the lower layer is coupled to
    a higher layer which makes independent testing and replay impossible.

    Layer definitions
    -----------------
    - ``dadbot/core/``         — base execution layer (lowest)
    - ``dadbot/managers/``     — persistence / state management layer
    - ``dadbot/services/``     — application service layer
    - ``dadbot/ui/``           — user interface layer (highest)

    Forbidden upward imports
    ------------------------
    - ``dadbot/core/`` importing from ``dadbot/managers/``
    - ``dadbot/core/`` importing from ``dadbot/services/``
    - ``dadbot/core/`` importing from ``dadbot/ui/``
    - ``dadbot/managers/`` importing from ``dadbot/services/``
    - ``dadbot/managers/`` importing from ``dadbot/ui/``
    - ``dadbot/services/`` importing from ``dadbot/ui/``

    Adapter allowlist
    -----------------
    Files whose names contain ``_adapter``, ``_gateway``, or ``_bridge`` are
    explicitly exempt because they are declared boundary-crossing adapters.
    """

    id = "X002"
    severity = "error"

    # (lower_layer_fragment, forbidden_upper_layer_fragment)
    _LAYER_VIOLATIONS: list[tuple[str, str]] = [
        ("dadbot/core/", "dadbot/managers/"),
        ("dadbot/core/", "dadbot/services/"),
        ("dadbot/core/", "dadbot/ui/"),
        ("dadbot/managers/", "dadbot/services/"),
        ("dadbot/managers/", "dadbot/ui/"),
        ("dadbot/services/", "dadbot/ui/"),
    ]

    # Import module prefixes corresponding to the above layer paths.
    _LAYER_IMPORT_MAP: list[tuple[str, str]] = [
        ("dadbot/core/", "dadbot.managers"),
        ("dadbot/core/", "dadbot.services"),
        ("dadbot/core/", "dadbot.ui"),
        ("dadbot/managers/", "dadbot.services"),
        ("dadbot/managers/", "dadbot.ui"),
        ("dadbot/services/", "dadbot.ui"),
    ]

    # Files that are explicitly allowed to cross layer boundaries.
    _ADAPTER_SUFFIXES = ("_adapter", "_gateway", "_bridge", "_side_effects")

    def _is_adapter(self, path: Path) -> bool:
        stem = path.stem.lower()
        return any(stem.endswith(s) for s in self._ADAPTER_SUFFIXES)

    def check(self, path: Path, _text: str, tree: ast.Module) -> list[Finding]:
        if self._is_adapter(path):
            return []

        path_str = str(path).replace("\\", "/")
        findings: list[Finding] = []

        for lower_layer, forbidden_import in self._LAYER_IMPORT_MAP:
            if lower_layer not in path_str:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    module = str(node.module or "")
                    imported_names = [module]
                else:
                    continue
                for name in imported_names:
                    if name.startswith(forbidden_import) or name == forbidden_import:
                        findings.append(
                            Finding(
                                file=str(path),
                                line=getattr(node, "lineno", 0),
                                rule=self.id,
                                severity=self.severity,
                                detail=(
                                    f"Layer inversion: {lower_layer!r} layer imports from "
                                    f"upper layer {forbidden_import!r} — violates "
                                    "infrastructure→core→managers→services→ui dependency order"
                                ),
                            )
                        )
        return findings


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

_ALL_RULES = [
    _DeepCopyRule(),
    _JsonRoundtripRule(),
    _UnboundedWhileLoopRule(),
    _DirectDatetimeNowInStageRule(),
    _RandomImportInCoreRule(),
    _MutationOutsideSaveNodeRule(),
    _SilentExceptPassRule(),
    _LargeInlineListLiteralRule(),
    _ForbiddenCouplingRule(),
    _ImportGraphCycleRule(),
]

_SKIP_DIRS = frozenset(
    {
        "__pycache__",
        ".venv",
        ".git",
        "node_modules",
        "dist",
        "build",
        ".mypy_cache",
        ".pytest_cache",
    }
)


def _should_skip(path: Path) -> bool:
    return any(part in _SKIP_DIRS for part in path.parts)


def scan_file(path: Path) -> list[Finding]:
    if _should_skip(path):
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(text)
    except (SyntaxError, OSError):
        return []
    findings: list[Finding] = []
    for rule in _ALL_RULES:
        findings.extend(rule.check(path, text, tree))
    return findings


def scan_root(root: Path) -> dict[str, list[Finding]]:
    report: dict[str, list[Finding]] = {}
    for py in sorted(root.rglob("*.py")):
        if _should_skip(py):
            continue
        findings = scan_file(py)
        if findings:
            report[str(py)] = findings
    return report


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def _print_text(report: dict[str, list[Finding]], *, use_color: bool = True) -> None:
    RESET = "\033[0m" if use_color else ""
    RED = "\033[31m" if use_color else ""
    YELLOW = "\033[33m" if use_color else ""
    BOLD = "\033[1m" if use_color else ""

    total = sum(len(v) for v in report.values())
    if not total:
        print("✓ No findings — Phase 4 static audit passed.")
        return

    print(f"\n{BOLD}Phase 4 Static Audit — {total} finding(s) across {len(report)} file(s){RESET}\n")
    for file_path, findings in sorted(report.items()):
        print(f"{BOLD}{file_path}{RESET}")
        for f in findings:
            color = RED if f.severity == "error" else YELLOW
            print(f"  {color}[{f.rule}] line {f.line}: {f.detail}{RESET}")
        print()


def _print_json(report: dict[str, list[Finding]]) -> None:
    out: list[dict] = []
    for file_path, findings in sorted(report.items()):
        for f in findings:
            d = f.to_dict()
            d["file"] = file_path
            out.append(d)
    print(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 4 static hotspot audit — structural performance and correctness scan."
    )
    parser.add_argument(
        "--path",
        default=".",
        help="Root directory or file to scan (default: current directory).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when any finding is present.",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="Output findings as JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Only report findings with severity='error'.",
    )
    args = parser.parse_args(argv)

    root = Path(args.path).resolve()
    report = scan_root(root)

    if args.errors_only:
        report = {
            k: [f for f in v if f.severity == "error"]
            for k, v in report.items()
            if any(f.severity == "error" for f in v)
        }

    use_color = sys.stdout.isatty()
    if args.output_json:
        _print_json(report)
    else:
        _print_text(report, use_color=use_color)

    total = sum(len(v) for v in report.values())
    if args.strict and total > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
