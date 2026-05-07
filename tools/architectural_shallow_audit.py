from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CORE_FOCUS_DIRS = ("dadbot/core", "dadbot/services", "dadbot/managers", "runtime")
IGNORE_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
ARCH_CLAIM_KEYWORDS = ("compiler", "planner", "engine", "evaluator", "resolver")


@dataclass
class FileAudit:
    file: str
    score: int
    flags: list[str]
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "score": self.score,
            "flags": self.flags,
            "details": self.details,
        }


def _iter_py_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def _is_core_focus(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    return rel.startswith(CORE_FOCUS_DIRS)


def _function_metrics(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, int]:
    if_count = 0
    string_compare_count = 0
    return_in_loop_count = 0
    loop_stack = 0

    class Visitor(ast.NodeVisitor):
        def nonlocal_increment_if(self) -> None:
            nonlocal if_count
            if_count += 1

        def nonlocal_increment_strcmp(self) -> None:
            nonlocal string_compare_count
            string_compare_count += 1

        def nonlocal_increment_return_loop(self) -> None:
            nonlocal return_in_loop_count
            return_in_loop_count += 1

        def visit_If(self, node: ast.If) -> None:
            self.nonlocal_increment_if()
            self.generic_visit(node)

        def visit_Compare(self, node: ast.Compare) -> None:
            values: list[ast.AST] = [node.left, *node.comparators]
            if any(isinstance(v, ast.Constant) and isinstance(v.value, str) for v in values):
                self.nonlocal_increment_strcmp()
            self.generic_visit(node)

        def visit_For(self, node: ast.For) -> None:
            nonlocal loop_stack
            loop_stack += 1
            self.generic_visit(node)
            loop_stack -= 1

        def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
            nonlocal loop_stack
            loop_stack += 1
            self.generic_visit(node)
            loop_stack -= 1

        def visit_While(self, node: ast.While) -> None:
            nonlocal loop_stack
            loop_stack += 1
            self.generic_visit(node)
            loop_stack -= 1

        def visit_Return(self, node: ast.Return) -> None:
            if loop_stack > 0:
                self.nonlocal_increment_return_loop()
            self.generic_visit(node)

    Visitor().visit(fn)
    return {
        "if_count": if_count,
        "string_compare_count": string_compare_count,
        "return_in_loop_count": return_in_loop_count,
    }


def _module_metrics(tree: ast.Module) -> dict[str, Any]:
    getattr_count = 0
    hasattr_count = 0
    any_return_count = 0
    missing_return_annotation_count = 0
    any_param_count = 0
    dict_return_count = 0
    compile_like_name_count = 0
    compile_like_without_ir_count = 0
    high_if_function_count = 0
    string_dispatch_function_count = 0
    early_return_loop_function_count = 0
    class_names: list[str] = []
    function_names: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "getattr":
            getattr_count += 1
        if isinstance(node.func, ast.Name) and node.func.id == "hasattr":
            hasattr_count += 1

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_names.append(node.name)
            continue
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        function_names.append(node.name)

        metrics = _function_metrics(node)
        if metrics["if_count"] > 5:
            high_if_function_count += 1
        if metrics["string_compare_count"] > 0:
            string_dispatch_function_count += 1
        if metrics["return_in_loop_count"] > 0:
            early_return_loop_function_count += 1

        if node.returns is None:
            missing_return_annotation_count += 1
        elif isinstance(node.returns, ast.Name) and node.returns.id == "Any":
            any_return_count += 1
        elif isinstance(node.returns, ast.Subscript) and isinstance(node.returns.value, ast.Name):
            if node.returns.value.id == "dict":
                dict_return_count += 1

        for arg in [*node.args.args, *node.args.kwonlyargs, *node.args.posonlyargs]:
            if isinstance(arg.annotation, ast.Name) and arg.annotation.id == "Any":
                any_param_count += 1

        lowered = node.name.lower()
        if lowered.startswith(("compile", "build", "plan")):
            compile_like_name_count += 1
            fn_source_nodes = list(ast.walk(node))
            has_ir_hint = any(
                isinstance(n, ast.Name) and n.id.lower() in {"ir", "intent_graph", "plan", "policyplan"}
                for n in fn_source_nodes
            )
            has_loop_plus_return = metrics["return_in_loop_count"] > 0
            if has_loop_plus_return and not has_ir_hint:
                compile_like_without_ir_count += 1

    return {
        "getattr_count": getattr_count,
        "hasattr_count": hasattr_count,
        "any_return_count": any_return_count,
        "missing_return_annotation_count": missing_return_annotation_count,
        "any_param_count": any_param_count,
        "dict_return_count": dict_return_count,
        "compile_like_name_count": compile_like_name_count,
        "compile_like_without_ir_count": compile_like_without_ir_count,
        "high_if_function_count": high_if_function_count,
        "string_dispatch_function_count": string_dispatch_function_count,
        "early_return_loop_function_count": early_return_loop_function_count,
        "class_names": sorted(class_names),
        "function_names": sorted(function_names),
    }


def _name_reality_mismatch(
    *,
    path: Path,
    root: Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    claims: list[str] = []
    symbols = [
        path.stem,
        *[str(n) for n in list(metrics.get("class_names") or [])],
        *[str(n) for n in list(metrics.get("function_names") or [])],
    ]
    lowered_symbols = [s.lower() for s in symbols]
    for keyword in ARCH_CLAIM_KEYWORDS:
        if any(keyword in symbol for symbol in lowered_symbols):
            claims.append(keyword)

    claims = sorted(set(claims))
    if not claims:
        return {
            "claims": [],
            "mismatch": False,
            "mismatch_score": 0,
            "reasons": [],
        }

    reasons: list[str] = []
    if int(metrics.get("compile_like_without_ir_count") or 0) > 0:
        reasons.append("compile_or_plan_name_without_intermediate_representation")
    if int(metrics.get("string_dispatch_function_count") or 0) > 0:
        reasons.append("string_dispatch_in_claimed_arch_component")
    if int(metrics.get("high_if_function_count") or 0) > 0:
        reasons.append("imperative_control_flow_dominates_claimed_arch_component")
    if int(metrics.get("dict_return_count") or 0) >= 2:
        reasons.append("dict_heavy_contracts_at_arch_boundary")
    if int(metrics.get("any_return_count") or 0) > 0:
        reasons.append("any_return_weakens_claimed_arch_boundary")

    mismatch_score = min(6, len(reasons) * 2)
    return {
        "claims": claims,
        "mismatch": bool(reasons),
        "mismatch_score": mismatch_score,
        "reasons": reasons,
        "file": path.relative_to(root).as_posix(),
    }


def _score_metrics(metrics: dict[str, Any], *, core_focus: bool) -> tuple[int, list[str]]:
    score = 0
    flags: list[str] = []

    def flag_when(condition: bool, name: str, weight: int) -> None:
        nonlocal score
        if condition:
            score += weight
            flags.append(name)

    flag_when(metrics["any_return_count"] > 0, "uses_any_return", 3 if core_focus else 2)
    flag_when(metrics["missing_return_annotation_count"] >= 3, "many_missing_return_annotations", 1)
    flag_when(metrics["any_param_count"] > 0, "uses_any_param", 2 if core_focus else 1)
    flag_when(metrics["dict_return_count"] >= 2, "dict_return_contracts", 2)
    flag_when(metrics["getattr_count"] + metrics["hasattr_count"] >= 2, "introspection_heavy", 2)
    flag_when(metrics["string_dispatch_function_count"] >= 1, "string_dispatch", 2)
    flag_when(metrics["high_if_function_count"] >= 1, "high_if_density", 2)
    flag_when(metrics["early_return_loop_function_count"] >= 1, "early_return_in_loop", 2)
    flag_when(metrics["compile_like_without_ir_count"] >= 1, "fake_compile_pattern", 5)

    return score, sorted(set(flags))


def analyze_file(path: Path, root: Path) -> FileAudit | None:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        rel = path.relative_to(root).as_posix()
        return FileAudit(
            file=rel,
            score=2,
            flags=["syntax_error_unparsed"],
            details={"syntax_error": True},
        )

    metrics = _module_metrics(tree)
    core_focus = _is_core_focus(path, root)
    score, flags = _score_metrics(metrics, core_focus=core_focus)
    name_reality = _name_reality_mismatch(path=path, root=root, metrics=metrics)
    if bool(name_reality.get("mismatch")):
        score += int(name_reality.get("mismatch_score") or 0)
        flags.append("name_reality_mismatch")
    flags = sorted(set(flags))
    rel = path.relative_to(root).as_posix()
    return FileAudit(
        file=rel,
        score=score,
        flags=flags,
        details={
            **metrics,
            "core_focus": core_focus,
            "name_reality": name_reality,
        },
    )


def _build_manual_review_queue(items: list[FileAudit], *, max_items: int = 25) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for item in items:
        name_reality = dict(item.details.get("name_reality") or {})
        if not bool(name_reality.get("mismatch")):
            continue
        queue.append(
            {
                "file": item.file,
                "priority": "high" if item.score >= 8 else "medium",
                "score": item.score,
                "claims": list(name_reality.get("claims") or []),
                "reasons": list(name_reality.get("reasons") or []),
                "review_prompt": "Does the component name imply architecture depth that the implementation does not provide?",
            },
        )
    queue.sort(key=lambda q: (-int(q.get("score") or 0), str(q.get("file") or "")))
    return queue[: max(int(max_items), 0)]


def audit_repo(
    root: Path,
    *,
    min_score: int = 1,
    top_n: int = 50,
    include_tests: bool = False,
) -> dict[str, Any]:
    candidates: list[FileAudit] = []

    for path in _iter_py_files(root):
        rel = path.relative_to(root).as_posix()
        if not include_tests and rel.startswith("tests/"):
            continue
        audited = analyze_file(path, root)
        if audited is None:
            continue
        if audited.score >= min_score:
            candidates.append(audited)

    candidates.sort(key=lambda item: (-item.score, item.file))
    ranked = candidates[: max(int(top_n), 0)]
    manual_review_queue = _build_manual_review_queue(candidates)

    return {
        "root": root.as_posix(),
        "file_count_scored": len(candidates),
        "min_score": int(min_score),
        "top_n": int(top_n),
        "threshold_rewrite": 6,
        "rewrite_candidates": [item.as_dict() for item in ranked if item.score >= 6],
        "manual_review_queue": manual_review_queue,
        "ranked": [item.as_dict() for item in ranked],
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect architecturally shallow Python files using structural and semantic heuristics.",
    )
    parser.add_argument(
        "--root",
        default=str(ROOT),
        help="Repository root to scan (default: workspace root)",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=1,
        help="Only include files with score >= this value",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Maximum number of ranked files in output",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include tests/ in scoring",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    report = audit_repo(
        root,
        min_score=int(args.min_score),
        top_n=int(args.top),
        include_tests=bool(args.include_tests),
    )

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        out = Path(args.output)
        out.write_text(payload + "\n", encoding="utf-8")
    print(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
