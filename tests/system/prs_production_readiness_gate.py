from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import tools.arch_completeness_audit as arch_audit
import tools.contract_guard as contract_guard
import tools.enforce_no_bypass as no_bypass


ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = ROOT / "dadbot" / "core"
GATE_REL_PATH = "tests/system/prs_production_readiness_gate.py"
ARCH_MANIFEST = ROOT / "audit" / "spec" / "architecture_manifest.json"
CONTRACT_GUARD_BASELINE = ROOT / "tools" / "contract_guard_baseline.json"
SHADOW_MODULE_SUFFIXES = ("_old", "_v2", "_legacy", "_bak", "_copy")

EXPECTED_CORE_MODULES: dict[str, dict[str, Any]] = {
    "graph": {"required": True},
    "execution_policy": {"required": True},
    "tool_dag": {"required": True},
    "observability": {"required": True},
    "graph_types": {"required": True},
    "control_plane": {"required": True},
    "orchestrator": {"required": True},
    "nodes": {"required": True},
    # Standalone surfaces that currently exist by design even if they are not imported.
    "execution_commitment": {"required": False, "allow_standalone": True},
    "execution_resource_budget": {"required": False, "allow_standalone": True},
    "execution_terminal_state": {"required": False, "allow_standalone": True},
}

CRITICAL_EXECUTION_MODULES = {
    "dadbot.core.graph",
    "dadbot.core.graph_context",
    "dadbot.core.graph_mutation",
    "dadbot.core.nodes",
    "dadbot.core.orchestrator",
    "dadbot.core.control_plane",
}

TYPE_REGISTRY_OWNERS = {
    "MutationKind": "dadbot/core/graph_types.py",
    "MemoryMutationOp": "dadbot/core/graph_types.py",
    "RelationshipMutationOp": "dadbot/core/graph_types.py",
    "LedgerMutationOp": "dadbot/core/graph_types.py",
    "GoalMutationOp": "dadbot/core/graph_types.py",
    "NodeType": "dadbot/core/graph_types.py",
    "MutationTransactionStatus": "dadbot/core/graph_types.py",
}


@dataclass(slots=True)
class AnalyzerResult:
    name: str
    ok: bool
    report: str
    details: dict[str, Any] = field(default_factory=dict)


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _parse(path: Path) -> ast.Module:
    source = _read_text(path).replace("\ufeff", "")
    return ast.parse(source, filename=_rel(path))


def _iter_repo_python_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*.py"):
        if any(part in {".venv", "__pycache__", ".git"} for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def _iter_core_surface_files() -> list[Path]:
    return sorted(path for path in CORE_DIR.glob("*.py") if path.name != "__init__.py")


def _module_name_for_path(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _path_for_module_name(module_name: str) -> Path | None:
    base = ROOT / Path(*module_name.split("."))
    module_file = base.with_suffix(".py")
    if module_file.exists():
        return module_file
    init_file = base / "__init__.py"
    if init_file.exists():
        return init_file
    return None


def _is_type_checking_block(node: ast.If) -> bool:
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def _collect_type_checking_lines(tree: ast.Module) -> frozenset[int]:
    """Return line numbers of all imports inside TYPE_CHECKING guards."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking_block(node):
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    lines.add(child.lineno)
    return frozenset(lines)


def _resolve_imported_module(importer: Path, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    importer_module = _module_name_for_path(importer)
    # For __init__.py, the file IS its own package; for regular .py strip the module name.
    if importer.name == "__init__.py":
        package_parts = importer_module.split(".")
    else:
        package_parts = importer_module.split(".")[:-1]
    if node.level > len(package_parts):
        return None
    base_parts = package_parts[: len(package_parts) - node.level + 1]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(part for part in base_parts if part)


def _module_aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[-1]
                aliases[bound] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound = alias.asname or alias.name
                aliases[bound] = f"{module_name}.{alias.name}" if module_name else alias.name
    return aliases


def _format_block(title: str, items: list[str], *, empty: str = "none") -> str:
    if not items:
        return f"{title}: {empty}"
    lines = [f"{title}:"]
    lines.extend(f"  - {item}" for item in items)
    return "\n".join(lines)


def run_structural_completeness_check() -> AnalyzerResult:
    core_files = _iter_core_surface_files()
    stems = {path.stem: path for path in core_files}
    missing = [name for name, spec in EXPECTED_CORE_MODULES.items() if spec.get("required") and name not in stems]

    module_refs: dict[str, set[str]] = defaultdict(set)
    repo_files = _iter_repo_python_files()
    for path in repo_files:
        if path == ROOT / GATE_REL_PATH:
            continue
        text = _read_text(path)
        for stem, module_path in stems.items():
            rel_module = _module_name_for_path(module_path)
            if rel_module in text or re.search(rf"\b{re.escape(stem)}\b", text):
                module_refs[stem].add(_rel(path))

    orphaned = []
    for stem, refs in sorted(module_refs.items()):
        if refs:
            continue
        if EXPECTED_CORE_MODULES.get(stem, {}).get("allow_standalone"):
            continue
        orphaned.append(stem)

    duplicate_contracts: list[str] = []
    normalized_stems: dict[str, list[str]] = defaultdict(list)
    for path in core_files:
        stem = path.stem
        base = stem
        for suffix in SHADOW_MODULE_SUFFIXES:
            if stem.endswith(suffix):
                base = stem[: -len(suffix)]
                break
        normalized_stems[base].append(stem)

    for base, variants in sorted(normalized_stems.items()):
        if len(variants) > 1:
            duplicate_contracts.append(f"{base}: {', '.join(sorted(variants))}")

    shadow_modules = []
    for stem in sorted(stems):
        for suffix in SHADOW_MODULE_SUFFIXES:
            if stem.endswith(suffix) and stem[: -len(suffix)] in stems:
                shadow_modules.append(stem)
                break

    ok = not (missing or orphaned or duplicate_contracts or shadow_modules)
    report = "\n\n".join(
        [
            _format_block("Missing required modules", [f"dadbot/core/{name}.py" for name in missing]),
            _format_block("Unexpected orphan modules", [f"dadbot/core/{name}.py" for name in orphaned]),
            _format_block("Duplicate contract surfaces", duplicate_contracts),
            _format_block("Shadow modules", [f"dadbot/core/{name}.py" for name in shadow_modules]),
        ]
    )
    return AnalyzerResult(
        name="structural",
        ok=ok,
        report=report,
        details={
            "missing": missing,
            "orphaned": orphaned,
            "duplicate_contracts": duplicate_contracts,
            "shadow_modules": shadow_modules,
        },
    )


def _build_internal_import_graph() -> tuple[dict[str, set[str]], list[str], list[str], list[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    broken_imports: list[str] = []
    phantom_dependencies: list[str] = []
    hidden_coupling: list[str] = []

    required_surface = {
        _module_name_for_path(CORE_DIR / f"{name}.py")
        for name, spec in EXPECTED_CORE_MODULES.items()
        if spec.get("required") and (CORE_DIR / f"{name}.py").exists()
    }

    for path in _iter_repo_python_files():
        rel = _rel(path)
        is_test_file = any(part in {"tests", "test"} for part in path.parts)
        tree = _parse(path)
        importer = _module_name_for_path(path)
        aliases = _module_aliases(tree)
        tc_lines = _collect_type_checking_lines(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if node.lineno in tc_lines:
                    continue
                for alias in node.names:
                    if not alias.name.startswith("dadbot"):
                        continue
                    graph[importer].add(alias.name)
                    if _path_for_module_name(alias.name) is None:
                        broken_imports.append(f"{rel}:{node.lineno} -> {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.lineno in tc_lines:
                    continue
                module_name = _resolve_imported_module(path, node)
                if not module_name or not module_name.startswith("dadbot"):
                    continue
                graph[importer].add(module_name)
                if _path_for_module_name(module_name) is None:
                    broken_imports.append(f"{rel}:{node.lineno} -> {module_name}")

            elif isinstance(node, ast.Call) and not is_test_file:
                # Only flag hidden coupling in production code, not test files.
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in {"import_module", "__import__"}:
                    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        target = node.args[0].value
                        if target.startswith("dadbot"):
                            hidden_coupling.append(f"{rel}:{node.lineno} dynamic import -> {target}")

                if isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) >= 2:
                    target_obj = node.args[0]
                    attr_node = node.args[1]
                    if isinstance(attr_node, ast.Constant) and isinstance(attr_node.value, str):
                        attr_name = attr_node.value
                    else:
                        attr_name = "<dynamic>"
                    if isinstance(target_obj, ast.Name):
                        bound = aliases.get(target_obj.id, "")
                        if bound.startswith("dadbot") and importer in required_surface:
                            hidden_coupling.append(
                                f"{rel}:{node.lineno} getattr on imported module {bound} -> {attr_name}"
                            )

        for target in graph.get(importer, set()):
            if target.startswith("dadbot") and _path_for_module_name(target) is None:
                phantom_dependencies.append(f"{importer} -> {target}")

    return graph, sorted(set(broken_imports)), sorted(set(phantom_dependencies)), sorted(set(hidden_coupling))


def run_wiring_graph_check() -> AnalyzerResult:
    graph, broken_imports, phantom_dependencies, hidden_coupling = _build_internal_import_graph()
    edge_count = sum(len(targets) for targets in graph.values())
    # hidden_coupling is reported for visibility but does not block the integrity gate.
    ok = not (broken_imports or phantom_dependencies)
    report = "\n\n".join(
        [
            f"Internal dependency edges: {edge_count}",
            _format_block("Broken internal imports", broken_imports),
            _format_block("Phantom dependencies", phantom_dependencies),
            _format_block("Hidden coupling", hidden_coupling),
        ]
    )
    return AnalyzerResult(
        name="wiring",
        ok=ok,
        report=report,
        details={
            "edge_count": edge_count,
            "broken_imports": broken_imports,
            "phantom_dependencies": phantom_dependencies,
            "hidden_coupling": hidden_coupling,
        },
    )


def _probe_critical_module_imports() -> dict[str, bool]:
    """Verify each critical module is importable in an isolated subprocess."""
    results: dict[str, bool] = {}
    for module in sorted(CRITICAL_EXECUTION_MODULES):
        probe = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        results[module] = probe.returncode == 0
    return results


def _critical_module_test_references() -> dict[str, list[str]]:
    """Statically scan test files for references to each critical module."""
    references: dict[str, list[str]] = {m: [] for m in CRITICAL_EXECUTION_MODULES}
    for path in _iter_repo_python_files():
        if not any(part in {"tests", "test"} for part in path.parts):
            continue
        try:
            text = _read_text(path)
        except Exception:
            continue
        rel = _rel(path)
        for module in CRITICAL_EXECUTION_MODULES:
            short = module.split(".")[-1]
            if module in text or short in text:
                references[module].append(rel)
    return references


def run_execution_coverage_check() -> AnalyzerResult:
    """Verify critical execution modules are importable and referenced by tests."""
    import_results = _probe_critical_module_imports()
    test_refs = _critical_module_test_references()

    import_errors: list[str] = [m for m, ok in import_results.items() if not ok]
    unreferenced: list[str] = [m for m, refs in test_refs.items() if not refs]

    ok = not import_errors and not unreferenced
    report = "\n\n".join(
        [
            _format_block("Import errors in critical modules", import_errors),
            _format_block("Critical modules with no test references", unreferenced),
            _format_block(
                "Critical modules verified",
                [
                    f"{m} (importable, {len(test_refs[m])} test file(s))"
                    for m in sorted(CRITICAL_EXECUTION_MODULES)
                    if m not in import_errors and m not in unreferenced
                ],
                empty="none",
            ),
        ]
    )
    return AnalyzerResult(
        name="coverage",
        ok=ok,
        report=report,
        details={
            "import_errors": import_errors,
            "unreferenced_critical": unreferenced,
            "test_references": {m: refs for m, refs in test_refs.items()},
        },
    )


def _load_contract_guard_new_violations() -> list[str]:
    current = contract_guard._collect_violations(ROOT)
    baseline_payload = json.loads(CONTRACT_GUARD_BASELINE.read_text(encoding="utf-8"))
    approved = set(str(item) for item in baseline_payload.get("violations", []))
    return sorted(item for item in current if item not in approved)


def _aggregate_bypass_violations() -> list[str]:
    aggregated: list[str] = []
    for path in sorted(no_bypass._iter_files()):
        result = no_bypass._analyze_file(path)
        for key, items in result.items():
            aggregated.extend(f"{key}: {item}" for item in items)
    return sorted(aggregated)


def _class_contract_map(path: Path) -> dict[str, dict[str, Any]]:
    tree = _parse(path)
    contract_map: dict[str, dict[str, Any]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        attrs = set()
        methods = set()
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.add(item.name)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attrs.add(target.id)
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                attrs.add(item.target.id)
        contract_map[node.name] = {
            "attrs": attrs,
            "methods": methods,
            "bases": [base.id for base in node.bases if isinstance(base, ast.Name)],
        }
    return contract_map


def _inherits_contract(class_name: str, contract_map: dict[str, dict[str, Any]], seen: set[str] | None = None) -> dict[str, bool]:
    if seen is None:
        seen = set()
    if class_name in seen:
        return {"name": False, "dependencies": False, "run_or_execute": False}
    seen.add(class_name)
    info = contract_map.get(class_name, {"attrs": set(), "methods": set(), "bases": []})
    name_ok = "name" in info["attrs"]
    deps_ok = "dependencies" in info["methods"]
    run_ok = bool({"run", "execute"} & set(info["methods"]))
    for base in info.get("bases", []):
        inherited = _inherits_contract(base, contract_map, seen)
        name_ok = name_ok or inherited["name"]
        deps_ok = deps_ok or inherited["dependencies"]
        run_ok = run_ok or inherited["run_or_execute"]
    return {"name": name_ok, "dependencies": deps_ok, "run_or_execute": run_ok}


def _check_graph_node_contracts() -> list[str]:
    path = CORE_DIR / "graph_pipeline_nodes.py"
    contract_map = _class_contract_map(path)
    issues: list[str] = []
    protocol = contract_map.get("GraphNode", {})
    required_protocol_methods = {"dependencies", "run", "execute"}
    missing_protocol_methods = sorted(required_protocol_methods - set(protocol.get("methods", set())))
    if "name" not in set(protocol.get("methods", set())):
        missing_protocol_methods.insert(0, "name")
    if missing_protocol_methods:
        issues.append(f"GraphNode protocol missing: {', '.join(missing_protocol_methods)}")

    for class_name in sorted(contract_map):
        if class_name in {"GraphNode", "_NodeContractMixin"} or class_name.startswith("_"):
            continue
        if not class_name.endswith("Node"):
            continue
        state = _inherits_contract(class_name, contract_map)
        missing = [key for key, ok in state.items() if not ok]
        if missing:
            issues.append(f"{class_name} missing contract members: {', '.join(missing)}")
    return issues


def _check_node_run_annotations() -> list[str]:
    path = CORE_DIR / "nodes.py"
    tree = _parse(path)
    issues: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not node.name.endswith("Node"):
            continue
        run_methods = [item for item in node.body if isinstance(item, ast.AsyncFunctionDef) and item.name == "run"]
        if not run_methods:
            continue
        run_method = run_methods[0]
        if len(run_method.args.args) < 2:
            issues.append(f"{node.name}.run missing TurnContext parameter")
            continue
        context_arg = run_method.args.args[1]
        annotation = ast.unparse(context_arg.annotation) if context_arg.annotation is not None else ""
        if "TurnContext" not in annotation:
            issues.append(f"{node.name}.run parameter is not annotated as TurnContext")
    return issues


def _check_type_registry_integrity() -> list[str]:
    definitions: dict[str, list[str]] = defaultdict(list)
    for path in _iter_repo_python_files():
        if not _rel(path).startswith("dadbot/"):
            continue
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in TYPE_REGISTRY_OWNERS:
                definitions[node.name].append(_rel(path))

    issues: list[str] = []
    for class_name, expected_owner in sorted(TYPE_REGISTRY_OWNERS.items()):
        owners = sorted(set(definitions.get(class_name, [])))
        if owners != [expected_owner]:
            issues.append(f"{class_name} owners={owners or ['<missing>']} expected={[expected_owner]}")
    return issues


def run_contract_compliance_check() -> AnalyzerResult:
    manifest = json.loads(ARCH_MANIFEST.read_text(encoding="utf-8"))
    interface = arch_audit.check_interface_completeness(manifest)
    propagation = arch_audit.check_param_propagation(manifest)
    new_contract_violations = _load_contract_guard_new_violations()
    bypass_violations = _aggregate_bypass_violations()
    graph_node_contract_issues = _check_graph_node_contracts()
    node_annotation_issues = _check_node_run_annotations()
    type_registry_issues = _check_type_registry_integrity()

    ok = not any(
        [
            interface["drift_count"],
            propagation["violation_count"],
            new_contract_violations,
            bypass_violations,
            graph_node_contract_issues,
            node_annotation_issues,
            type_registry_issues,
        ]
    )
    report = "\n\n".join(
        [
            _format_block("Interface drift", [str(item) for item in interface["drift"]]),
            _format_block("Propagation violations", [str(item) for item in propagation["violations"]]),
            _format_block("New forbidden primitive signatures", new_contract_violations),
            _format_block("Bypass violations", bypass_violations),
            _format_block("GraphNode contract issues", graph_node_contract_issues),
            _format_block("Node run annotation issues", node_annotation_issues),
            _format_block("Type registry issues", type_registry_issues),
        ]
    )
    return AnalyzerResult(
        name="contracts",
        ok=ok,
        report=report,
        details={
            "interface": interface,
            "propagation": propagation,
            "new_contract_violations": new_contract_violations,
            "bypass_violations": bypass_violations,
            "graph_node_contract_issues": graph_node_contract_issues,
            "node_annotation_issues": node_annotation_issues,
            "type_registry_issues": type_registry_issues,
        },
    )


def _combined_report(results: list[AnalyzerResult]) -> str:
    sections = []
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        sections.append(f"[{status}] {result.name}\n{result.report}")
    return "\n\n".join(sections)


@pytest.mark.regression
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.timeout(300)
def test_system_readiness_gate() -> None:
    structural = run_structural_completeness_check()
    wiring = run_wiring_graph_check()
    coverage = run_execution_coverage_check()
    contracts = run_contract_compliance_check()

    results = [structural, wiring, coverage, contracts]
    assert all(result.ok for result in results), _combined_report(results)