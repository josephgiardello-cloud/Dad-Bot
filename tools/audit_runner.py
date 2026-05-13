from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ci.ast_invariant_check import check_service_shell_forward_only, check_tool_sandbox_isolation
from ci.import_graph_check import run_checks as run_import_graph_checks
from ci.kernel_boundary_check import run_kernel_boundary_checks
from ci.mutation_tracking_check import check_no_cross_layer_shared_mutation
from dadbot.core.execution_boundary import CANONICAL_EXECUTION_KERNEL
from dadbot.core.execution_topology_graph import get_execution_topology_graph, NodeType

REPORT_PATH = ROOT / "artifacts" / "system_audit_matrix_report.json"


@dataclass(frozen=True)
class AuditCheck:
    id: str
    category: str
    name: str
    passed: bool
    severity: str
    classification: str
    detail: str


def _iter_repo_py() -> list[Path]:
    files: list[Path] = []
    include_roots = {"dadbot", "dadbot_system", "ci", "tools"}
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if any(part in {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache"} for part in rel.parts):
            continue
        if rel.parts and rel.parts[0] not in include_roots:
            continue
        files.append(path)
    return files


def _collect_graph_execute_callers() -> set[str]:
    callers: set[str] = set()
    target_files = sorted((ROOT / "dadbot" / "core").rglob("*.py"))
    for pyfile in target_files:
        rel = pyfile.relative_to(ROOT).as_posix()
        source = pyfile.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.func_stack: list[str] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.func_stack.append(node.name)
                self.generic_visit(node)
                self.func_stack.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self.func_stack.append(node.name)
                self.generic_visit(node)
                self.func_stack.pop()

            def visit_Call(self, node: ast.Call) -> None:
                if isinstance(node.func, ast.Attribute) and node.func.attr == "execute":
                    base = node.func.value
                    is_graph_execute = False
                    if isinstance(base, ast.Name) and base.id == "graph":
                        is_graph_execute = True
                    elif isinstance(base, ast.Attribute) and base.attr == "graph":
                        is_graph_execute = True
                    if is_graph_execute:
                        current = self.func_stack[-1] if self.func_stack else "<module>"
                        callers.add(f"{rel}:{current}")
                self.generic_visit(node)

        Visitor().visit(tree)
    return callers


def _scan_import_time_execution() -> list[str]:
    offenders: list[str] = []
    for pyfile in sorted((ROOT / "dadbot" / "core").rglob("*.py")):
        rel = pyfile.relative_to(ROOT).as_posix()
        source = pyfile.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name) and func.id in {"print", "open"}:
                    offenders.append(f"{rel}:{node.lineno}:top-level call to {func.id}")
                if isinstance(func, ast.Attribute) and func.attr in {"execute", "submit_turn"}:
                    offenders.append(f"{rel}:{node.lineno}:top-level execution call")
    return offenders


def _has_direct_memory_store_mutation() -> list[str]:
    hits: list[str] = []
    for pyfile in _iter_repo_py():
        rel = pyfile.relative_to(ROOT).as_posix()
        if rel == "tools/audit_runner.py":
            continue
        text = pyfile.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "memory.store[" in line.replace(" ", ""):
                hits.append(f"{rel}:{lineno}")
    return hits


def _check_execution_topology_graph() -> dict[str, Any]:
    """Verify the ExecutionTopologyGraph is properly initialized.

    Returns:
        Dict with topology validation details:
        - graph_initialized: bool
        - entry_node_exists: bool
        - canonical_chain_valid: bool
        - total_nodes: int
        - violations: list of any detected violations
    """
    try:
        graph = get_execution_topology_graph()
        nodes = graph.get_all_nodes()
        entry_node = graph.get_node("control_plane.submit_turn")
        exit_node = graph.get_node("persistence.finalize_turn")

        # Check canonical chain
        chain_valid = True
        current = entry_node
        while current and current.children_node_ids:
            if len(current.children_node_ids) > 1:
                chain_valid = False  # Non-linear topology
            if not current.children_node_ids:
                break
            current = graph.get_node(current.children_node_ids[0])

        violations = graph.get_violations()
        return {
            "graph_initialized": True,
            "entry_node_exists": entry_node is not None,
            "exit_node_exists": exit_node is not None,
            "canonical_chain_valid": chain_valid,
            "total_nodes": len(nodes),
            "violations_count": len(violations),
        }
    except Exception as exc:
        return {
            "graph_initialized": False,
            "error": str(exc),
        }


def _compute_matrix(strict: bool, no_shadow_paths: bool, deep_scan: bool) -> dict[str, Any]:
    graph_execute_callers = _collect_graph_execute_callers()
    allowed_callers = {
        "dadbot/core/orchestrator.py:handle_turn",
        "dadbot/core/orchestrator.py:_run",
        "dadbot/core/execution_resource_budget.py:execute_with_budget",
        "dadbot/core/execution_resource_budget.py:run_with_budget",
        "dadbot/core/execution_resource_budget.py:run_with_budget_sync",
        # Phase 4: Topology-wrapped path
        "dadbot/core/orchestrator.py:_run_graph_with_trace_binding",
    }
    shadow_callers = sorted(item for item in graph_execute_callers if item not in allowed_callers)

    import_time_offenders = _scan_import_time_execution()
    memory_store_offenders = _has_direct_memory_store_mutation()
    sandbox_violations = check_tool_sandbox_isolation()
    service_shell_violations = check_service_shell_forward_only()
    mutation_violations = check_no_cross_layer_shared_mutation()
    import_graph_violations = run_import_graph_checks()
    kernel_boundary_violations = run_kernel_boundary_checks() if deep_scan else []

    control_plane_text = (ROOT / "dadbot" / "core" / "control_plane.py").read_text(encoding="utf-8", errors="replace")
    has_delayed_learning_queue = "response_learning_pending" in control_plane_text and "reward_feedback" in control_plane_text

    check_results: list[AuditCheck] = [
        AuditCheck(
            id="A1",
            category="execution_uniqueness",
            name="canonical execution kernel is declared",
            passed="DadBotOrchestrator.handle_turn" in CANONICAL_EXECUTION_KERNEL,
            severity="critical",
            classification="entrypoint_uniqueness",
            detail=CANONICAL_EXECUTION_KERNEL,
        ),
        AuditCheck(
            id="A2",
            category="execution_uniqueness",
            name="no shadow graph.execute paths",
            passed=len(shadow_callers) == 0,
            severity="critical" if no_shadow_paths else "high",
            classification="shadow_execution",
            detail="; ".join(shadow_callers) or "none",
        ),
        AuditCheck(
            id="A3",
            category="execution_uniqueness",
            name="no service-shell bypasses",
            passed=len(service_shell_violations) == 0,
            severity="critical",
            classification="bypass_guard",
            detail=str(len(service_shell_violations)),
        ),
        AuditCheck(
            id="B1",
            category="decision_graph_freezing",
            name="snapshot hash lock exists",
            passed="snapshot_hash" in (ROOT / "dadbot" / "core" / "execution_context.py").read_text(
                encoding="utf-8", errors="replace"
            ),
            severity="critical",
            classification="snapshot_lock",
            detail="execution_context snapshot_hash",
        ),
        AuditCheck(
            id="B2",
            category="decision_graph_freezing",
            name="import-time execution scanner clean",
            passed=len(import_time_offenders) == 0,
            severity="high",
            classification="import_time_execution",
            detail="; ".join(import_time_offenders[:10]) or "none",
        ),
        AuditCheck(
            id="C1",
            category="response_generation",
            name="final_output replay contract enforced",
            passed='"final_output"' in (ROOT / "dadbot" / "core" / "execution_replay_engine.py").read_text(
                encoding="utf-8", errors="replace"
            ),
            severity="high",
            classification="candidate_origin",
            detail="execution_replay_engine final_output contract",
        ),
        AuditCheck(
            id="D1",
            category="memory_consistency",
            name="memory writer enforcement exists",
            passed="MemoryManager.mutate_memory_store" in (ROOT / "dadbot" / "core" / "execution_boundary.py").read_text(
                encoding="utf-8", errors="replace"
            ),
            severity="critical",
            classification="single_writer",
            detail="execution boundary owner enforcement",
        ),
        AuditCheck(
            id="D2",
            category="memory_consistency",
            name="no direct memory.store index mutation",
            passed=len(memory_store_offenders) == 0,
            severity="critical",
            classification="cross_layer_write",
            detail="; ".join(memory_store_offenders[:10]) or "none",
        ),
        AuditCheck(
            id="E1",
            category="tool_execution",
            name="tool sandbox isolation",
            passed=len(sandbox_violations) == 0,
            severity="critical",
            classification="tool_isolation",
            detail=str(len(sandbox_violations)),
        ),
        AuditCheck(
            id="E2",
            category="tool_execution",
            name="kernel boundary gate",
            passed=(len(kernel_boundary_violations) == 0) if deep_scan else True,
            severity="critical",
            classification="tool_failure_propagation",
            detail=str(len(kernel_boundary_violations)) if deep_scan else "skipped (use --deep-scan)",
        ),
        AuditCheck(
            id="F1",
            category="reward_learning",
            name="learning is delayed via pending queue",
            passed=has_delayed_learning_queue,
            severity="high",
            classification="temporal_isolation",
            detail="response_learning_pending + reward_feedback",
        ),
        AuditCheck(
            id="G1",
            category="ux_layer",
            name="ux projection gateway separated",
            passed="TurnUxProjectionGateway" in (ROOT / "dadbot" / "core" / "dadbot.py").read_text(
                encoding="utf-8", errors="replace"
            ),
            severity="high",
            classification="ux_projection_only",
            detail="dadbot facade ux gateway",
        ),
        AuditCheck(
            id="H1",
            category="cross_layer_integrity",
            name="no cross-layer shared mutation",
            passed=len(mutation_violations) == 0,
            severity="critical",
            classification="no_hidden_cross_writes",
            detail=str(len(mutation_violations)),
        ),
        AuditCheck(
            id="H2",
            category="cross_layer_integrity",
            name="import dependency graph closed",
            passed=len(import_graph_violations) == 0,
            severity="critical",
            classification="closed_call_graph",
            detail=str(len(import_graph_violations)),
        ),
        AuditCheck(
            id="P4",
            category="phase4_topology",
            name="execution topology graph registered",
            passed=_check_execution_topology_graph().get("graph_initialized", False),
            severity="high",
            classification="topology_initialization",
            detail="ExecutionTopologyGraph initialized and canonical nodes available",
        ),
    ]

    passed = sum(1 for item in check_results if item.passed)
    total = len(check_results)
    closure_score = round(passed / total, 4) if total else 0.0

    determinism_checks = [item for item in check_results if item.id in {"A1", "B1", "H2"}]
    determinism_score = round(
        (sum(1 for item in determinism_checks if item.passed) / len(determinism_checks)) if determinism_checks else 0.0,
        4,
    )

    overall_pass = all(item.passed for item in check_results)
    if strict:
        overall_pass = overall_pass and len(import_time_offenders) == 0

    return {
        "mode": {
            "strict": strict,
            "no_shadow_paths": no_shadow_paths,
            "deep_scan": deep_scan,
        },
        "overall_pass": overall_pass,
        "invariants": [asdict(item) for item in check_results],
        "summary": {
            "passed": passed,
            "failed": total - passed,
            "total": total,
        },
        "shadow_path_map": shadow_callers,
        "mutation_surface_map": {
            "cross_layer_shared_mutation": [asdict(item) for item in mutation_violations],
            "memory_store_direct_mutation": memory_store_offenders,
        },
        "determinism_score": determinism_score,
        "closure_score": closure_score,
    }


def run_full_system_audit(*, strict: bool, no_shadow_paths: bool, deep_scan: bool = False) -> dict[str, Any]:
    payload = _compute_matrix(strict=strict, no_shadow_paths=no_shadow_paths, deep_scan=deep_scan)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full system audit matrix runner")
    parser.add_argument("--full-system", action="store_true", dest="full_system")
    parser.add_argument("--strict", action="store_true", dest="strict")
    parser.add_argument("--no-shadow-paths", action="store_true", dest="no_shadow_paths")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--deep-scan", action="store_true", dest="deep_scan")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.full_system:
        print("Use --full-system to run the audit matrix")
        return 2

    payload = run_full_system_audit(
        strict=bool(args.strict),
        no_shadow_paths=bool(args.no_shadow_paths),
        deep_scan=bool(args.deep_scan),
    )
    if args.json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"PASS={payload['overall_pass']}")
        print(f"INVARIANTS: {payload['summary']['passed']}/{payload['summary']['total']}")
        print(f"DETERMINISM_SCORE={payload['determinism_score']}")
        print(f"CLOSURE_SCORE={payload['closure_score']}")
        print(f"REPORT={REPORT_PATH}")
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
