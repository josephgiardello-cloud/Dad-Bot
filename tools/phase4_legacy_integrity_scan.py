from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEGACY_TERMS = [
    "turn_service",
    "handle_graph_failure",
    "fallback",
    "degraded",
    "legacy",
    "graph_turns_enabled",
    "direct_path",
]

DUAL_PATH_HINTS = (
    "graph_enabled",
    "turn_graph",
    "strict_graph_mode",
    "fallback",
    "legacy",
    "direct_path",
)

GRAPH_CORE_SCAN_FILES = {
    "dadbot/core/graph.py",
    "dadbot/core/nodes.py",
    "dadbot/core/orchestrator.py",
    "dadbot/services/persistence.py",
    "dadbot/services/turn_service.py",
}

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "session_logs",
    "runtime",
}


@dataclass
class Finding:
    file: str
    line: int
    kind: str
    detail: str
    snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "kind": self.kind,
            "detail": self.detail,
            "snippet": self.snippet,
        }


class _ParentAwareVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.parents: dict[ast.AST, ast.AST] = {}

    def generic_visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            self.parents[child] = node
            self.visit(child)


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in path.parts)


def _iter_python_files(root: Path):
    for file_path in root.rglob("*.py"):
        rel = file_path.relative_to(root)
        if _is_excluded(rel):
            continue
        yield file_path, rel.as_posix()


def _line_at(text: str, line_no: int) -> str:
    lines = text.splitlines()
    if 1 <= line_no <= len(lines):
        return lines[line_no - 1].strip()
    return ""


def _ancestor_class_name(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, ast.ClassDef):
            return current.name
    return ""


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _call_target_repr(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts: list[str] = []
        cur: ast.AST | None = func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def _is_temporal_allowed(rel_path: str, class_name: str, call_repr: str) -> bool:
    if class_name == "TemporalNode":
        return True
    if rel_path == "dadbot/core/graph.py" and call_repr.startswith("datetime.now"):
        # Canonical temporal axis creation belongs here.
        return True
    return False


def run_scan(repo_root: Path) -> dict[str, Any]:
    legacy_paths: list[Finding] = []
    dual_execution_paths: list[Finding] = []
    temporal_violations: list[Finding] = []
    unsafe_mutations: list[Finding] = []
    dead_code: list[Finding] = []

    try_except_dual_path_re = re.compile(
        r"try\s*:\s*[\s\S]{0,800}?(run_graph|_run_graph_turn|graph\.execute)"
        r"[\s\S]{0,800}?except[\s\S]{0,800}?(fallback|legacy|direct_path|turn_service)",
        re.IGNORECASE,
    )

    for file_path, rel_path in _iter_python_files(repo_root):
        text = file_path.read_text(encoding="utf-8", errors="replace")

        # 1) Legacy path markers
        for line_no, line in enumerate(text.splitlines(), start=1):
            lower = line.lower()
            for term in LEGACY_TERMS:
                if term in lower:
                    legacy_paths.append(
                        Finding(
                            file=rel_path,
                            line=line_no,
                            kind="legacy_marker",
                            detail=f"Found legacy marker: {term}",
                            snippet=line.strip(),
                        )
                    )
                    if "backward-compatible alias" in lower or "compat" in lower:
                        dead_code.append(
                            Finding(
                                file=rel_path,
                                line=line_no,
                                kind="partial_migration",
                                detail="Compatibility/legacy alias indicates partial migration surface",
                                snippet=line.strip(),
                            )
                        )

        # 2) Try/except fallback dual path pattern
        if try_except_dual_path_re.search(text):
            dual_execution_paths.append(
                Finding(
                    file=rel_path,
                    line=1,
                    kind="try_except_fallback",
                    detail="Potential graph->legacy fallback pattern detected",
                    snippet="try/except run_graph ... fallback",
                )
            )

        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        parent_visitor = _ParentAwareVisitor()
        parent_visitor.visit(tree)
        parents = parent_visitor.parents

        for node in ast.walk(tree):
            # 2) If/else dual-path execution logic
            if isinstance(node, ast.If):
                if not node.orelse:
                    continue
                test_src = ast.get_source_segment(text, node.test) or ""
                test_lower = test_src.lower()
                if any(hint in test_lower for hint in DUAL_PATH_HINTS):
                    dual_execution_paths.append(
                        Finding(
                            file=rel_path,
                            line=getattr(node, "lineno", 1),
                            kind="if_else_dual_path",
                            detail="Conditional branch looks like dual execution path",
                            snippet=test_src.strip(),
                        )
                    )

            if not isinstance(node, ast.Call):
                continue

            call_name = _call_name(node)
            call_repr = _call_target_repr(node)
            class_name = _ancestor_class_name(node, parents)
            line_no = getattr(node, "lineno", 1)
            snippet = _line_at(text, line_no)

            # 3) Unsafe mutation candidates in graph/core modules outside SaveNode
            if rel_path in GRAPH_CORE_SCAN_FILES and class_name != "SaveNode":
                is_mutation = False
                detail = ""
                lower_repr = call_repr.lower()
                if "memory.save" in lower_repr or lower_repr.endswith("save_mood_state"):
                    is_mutation = True
                    detail = "Memory mutation outside SaveNode"
                elif "relationship.update" in lower_repr:
                    is_mutation = True
                    detail = "Relationship update outside SaveNode"
                elif call_name == "mutate_memory_store":
                    is_mutation = True
                    detail = "mutate_memory_store call outside SaveNode"
                elif call_name == "append":
                    is_mutation = True
                    detail = "append() in graph-core module outside SaveNode"
                elif call_name == "setattr":
                    is_mutation = True
                    detail = "setattr() in graph-core module outside SaveNode"

                if is_mutation:
                    unsafe_mutations.append(
                        Finding(
                            file=rel_path,
                            line=line_no,
                            kind="unsafe_mutation_candidate",
                            detail=detail,
                            snippet=snippet,
                        )
                    )

            # 4) Temporal violations outside strict temporal context
            if call_repr in {"datetime.now", "date.today", "time.time"}:
                if not _is_temporal_allowed(rel_path, class_name, call_repr):
                    temporal_violations.append(
                        Finding(
                            file=rel_path,
                            line=line_no,
                            kind="temporal_violation",
                            detail=f"{call_repr} used outside TemporalNode context",
                            snippet=snippet,
                        )
                    )

            # 5) Hidden async fallback behavior (reported via dead_code list)
            if call_repr in {"asyncio.create_task", "threading.Thread", "Thread", "create_task"}:
                dead_code.append(
                    Finding(
                        file=rel_path,
                        line=line_no,
                        kind="hidden_async_fallback",
                        detail="Potential fire-and-forget/background path",
                        snippet=snippet,
                    )
                )

            if call_name == "submit" and "mutate_memory_store" in snippet:
                dead_code.append(
                    Finding(
                        file=rel_path,
                        line=line_no,
                        kind="background_mutation_write",
                        detail="Background submit() appears to schedule memory mutation",
                        snippet=snippet,
                    )
                )

    legacy_payload = [f.to_dict() for f in legacy_paths]
    dual_payload = [f.to_dict() for f in dual_execution_paths]
    temporal_payload = [f.to_dict() for f in temporal_violations]
    unsafe_payload = [f.to_dict() for f in unsafe_mutations]
    dead_payload = [f.to_dict() for f in dead_code]

    overall_integrity = "PASS"
    if dual_payload or temporal_payload or unsafe_payload:
        overall_integrity = "FAIL"

    return {
        "legacy_paths": legacy_payload,
        "dead_code": dead_payload,
        "temporal_violations": temporal_payload,
        "dual_execution_paths": dual_payload,
        "unsafe_mutations": unsafe_payload,
        "overall_integrity": overall_integrity,
    }


def build_quarantine_registry(report: dict[str, Any]) -> dict[str, Any]:
    quarantined_symbols: set[str] = set(LEGACY_TERMS)

    def _ingest_findings(items: list[dict[str, Any]]) -> None:
        for item in items:
            detail = str(item.get("detail") or "").lower()
            snippet = str(item.get("snippet") or "").strip()
            for term in LEGACY_TERMS:
                if term in detail or term in snippet.lower():
                    quarantined_symbols.add(term)
            # If a finding references an explicit function def/call, capture it.
            m_def = re.search(r"\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", snippet)
            if m_def:
                quarantined_symbols.add(m_def.group(1).lower())
            m_call = re.search(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", snippet)
            if m_call:
                candidate = m_call.group(1).lower()
                if candidate in {"fallback", "legacy", "direct_path", "handle_graph_failure"}:
                    quarantined_symbols.add(candidate)

    _ingest_findings(list(report.get("legacy_paths", [])))
    _ingest_findings(list(report.get("dual_execution_paths", [])))

    return {
        "quarantined_symbols": sorted(quarantined_symbols),
        "source": "phase4_legacy_integrity_scan",
        "overall_integrity": str(report.get("overall_integrity") or "FAIL"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 legacy and integrity static scanner")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    parser.add_argument(
        "--quarantine-output",
        default="runtime/phase4_quarantine_registry.json",
        help="Path to write execution quarantine registry JSON",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    report = run_scan(root)

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)

    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")

    quarantine = build_quarantine_registry(report)
    quarantine_path = Path(args.quarantine_output)
    if not quarantine_path.is_absolute():
        quarantine_path = root / quarantine_path
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path.write_text(json.dumps(quarantine, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return 0 if report.get("overall_integrity") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
