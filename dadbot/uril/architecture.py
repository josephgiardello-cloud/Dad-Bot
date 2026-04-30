from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dadbot.uril.models import RepoSignalBus, SubsystemHealth

ROOT = Path(__file__).resolve().parents[2]
DADBOT_ROOT = ROOT / "dadbot"


@dataclass
class GraphMetrics:
    coupling: float
    centrality: float
    blast_radius: float


def _python_files(root: Path) -> list[Path]:
    return [
        p
        for p in root.rglob("*.py")
        if ".venv" not in p.parts and "__pycache__" not in p.parts and p.name != "__init__.py"
    ]


def _module_name(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    return ".".join(rel.parts)


def _imports_in(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return set()

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(str(alias.name or ""))
        elif isinstance(node, ast.ImportFrom):
            module = str(node.module or "")
            if module:
                imports.add(module)
    return imports


def _build_graph() -> dict[str, set[str]]:
    files = _python_files(DADBOT_ROOT)
    modules = {_module_name(p): p for p in files}

    graph: dict[str, set[str]] = {m: set() for m in modules}
    for module, path in modules.items():
        imports = _imports_in(path)
        for imported in imports:
            for candidate in modules:
                if imported == candidate or imported.startswith(candidate + "."):
                    graph[module].add(candidate)
    return graph


def _reverse_graph(graph: dict[str, set[str]]) -> dict[str, set[str]]:
    rev = {k: set() for k in graph}
    for src, targets in graph.items():
        for dst in targets:
            rev.setdefault(dst, set()).add(src)
    return rev


def _reachable_count(graph: dict[str, set[str]], start: str) -> int:
    seen: set[str] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        for nxt in graph.get(node, set()):
            if nxt in seen:
                continue
            seen.add(nxt)
            stack.append(nxt)
    return len(seen)


def _metrics_for_module(
    graph: dict[str, set[str]],
    rev: dict[str, set[str]],
    module: str,
) -> GraphMetrics:
    n = max(1, len(graph) - 1)
    out_d = len(graph.get(module, set())) / n
    in_d = len(rev.get(module, set())) / n
    coupling = min(1.0, (in_d + out_d) / 2.0)
    centrality = min(1.0, in_d + out_d)
    blast = min(1.0, _reachable_count(graph, module) / n)
    return GraphMetrics(coupling=coupling, centrality=centrality, blast_radius=blast)


def _coverage_ratio(module_hint: str, bus: RepoSignalBus) -> float:
    correctness = bus.by_category("correctness")
    if not correctness:
        return 0.0
    stem = module_hint.rsplit(".", maxsplit=1)[-1]
    matches = [s for s in correctness if stem in s.subsystem.replace("/", ".")]
    if not matches:
        repo = [s for s in correctness if s.subsystem == "repo"]
        return repo[0].score if repo else 0.0
    return sum(s.score for s in matches) / len(matches)


def build_subsystem_health(signal_bus: RepoSignalBus) -> list[SubsystemHealth]:
    graph = _build_graph()
    rev = _reverse_graph(graph)

    subsystem_map = {
        "dadbot_core": "dadbot.core.dadbot",
        "graph_engine": "dadbot.core.graph",
        "kernel": "dadbot.core.kernel",
        "validator": "dadbot.core.execution_policy",
        "persistence": "dadbot.services.persistence",
        "observability": "dadbot.core.observability",
        "mcp_layer": "dadbot.runtime.mcp.local_mcp_server_controller",
        "tool_registry": "dadbot.managers.tool_registry",
    }

    runtime_criticality_map = {
        "dadbot_core": 0.95,
        "graph_engine": 0.95,
        "kernel": 0.9,
        "validator": 0.8,
        "persistence": 0.9,
        "observability": 0.7,
        "mcp_layer": 0.6,
        "tool_registry": 0.7,
    }

    health_rows: list[SubsystemHealth] = []
    for subsystem, module in subsystem_map.items():
        if module not in graph:
            health_rows.append(
                SubsystemHealth(
                    subsystem=subsystem,
                    score=0.0,
                    coupling=1.0,
                    centrality=1.0,
                    blast_radius=1.0,
                    test_coverage_ratio=0.0,
                    runtime_criticality=runtime_criticality_map.get(subsystem, 0.5),
                ),
            )
            continue

        metrics = _metrics_for_module(graph, rev, module)
        coverage = _coverage_ratio(module, signal_bus)
        criticality = runtime_criticality_map.get(subsystem, 0.5)

        risk = (0.35 * metrics.coupling) + (0.25 * metrics.centrality) + (0.2 * metrics.blast_radius)
        confidence = (0.2 * coverage) + (0.2 * (1.0 - abs(criticality - coverage)))
        score = max(0.0, min(1.0, (1.0 - risk) * 0.8 + confidence))

        health_rows.append(
            SubsystemHealth(
                subsystem=subsystem,
                score=score,
                coupling=metrics.coupling,
                centrality=metrics.centrality,
                blast_radius=metrics.blast_radius,
                test_coverage_ratio=coverage,
                runtime_criticality=criticality,
            ),
        )

    return health_rows


def subsystem_health_map(signal_bus: RepoSignalBus) -> dict[str, float]:
    return {row.subsystem: round(row.score, 3) for row in build_subsystem_health(signal_bus)}


def subsystem_risk_heatmap(signal_bus: RepoSignalBus) -> list[dict[str, Any]]:
    rows = build_subsystem_health(signal_bus)
    heatmap: list[dict[str, Any]] = []
    for row in rows:
        risk_score = (row.coupling + row.centrality + row.blast_radius) / 3.0
        level = "LOW"
        if risk_score >= 0.75:
            level = "HIGH"
        elif risk_score >= 0.5:
            level = "MEDIUM"
        heatmap.append(
            {
                "subsystem": row.subsystem,
                "risk_level": level,
                "risk_score": round(risk_score, 3),
                "coupling": round(row.coupling, 3),
                "centrality": round(row.centrality, 3),
                "blast_radius": round(row.blast_radius, 3),
            },
        )
    heatmap.sort(key=lambda x: x["risk_score"], reverse=True)
    return heatmap


# ---------------------------------------------------------------------------
# Cycle detection — ROI #5
# ---------------------------------------------------------------------------


def _tarjan_scc(graph: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's strongly connected components — iterative to avoid recursion depth limits."""
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    stack: list[str] = []
    sccs: list[list[str]] = []
    counter = [0]

    for root in graph:
        if root in index:
            continue
        # Iterative Tarjan using an explicit call stack
        call_stack: list[tuple[str, list[str], int]] = []
        neighbours = list(graph.get(root, set()))
        call_stack.append((root, neighbours, 0))
        index[root] = lowlink[root] = counter[0]
        counter[0] += 1
        stack.append(root)
        on_stack[root] = True

        while call_stack:
            v, nbrs, ni = call_stack[-1]
            if ni < len(nbrs):
                call_stack[-1] = (v, nbrs, ni + 1)
                w = nbrs[ni]
                if w not in index:
                    index[w] = lowlink[w] = counter[0]
                    counter[0] += 1
                    stack.append(w)
                    on_stack[w] = True
                    call_stack.append((w, list(graph.get(w, set())), 0))
                elif on_stack.get(w, False):
                    lowlink[v] = min(lowlink[v], index[w])
            else:
                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[v])
                if lowlink[v] == index[v]:
                    scc: list[str] = []
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        scc.append(w)
                        if w == v:
                            break
                    sccs.append(scc)

    return sccs


def detect_cycles(graph: dict[str, set[str]] | None = None) -> list[list[str]]:
    """Return all cyclic SCCs (groups of ≥2 modules in a mutual-import cycle).

    If *graph* is not provided, the live dadbot import graph is built fresh.
    Each returned list is a group of module names that mutually import each
    other (directly or transitively).
    """
    if graph is None:
        graph = _build_graph()
    sccs = _tarjan_scc(graph)
    return [scc for scc in sccs if len(scc) > 1]


# Forbidden import edges: (from_module_fragment, to_module_fragment)
# These represent coupling directions that should never exist in this
# architecture.  The fragments are substring-matched against full module names.
FORBIDDEN_IMPORT_EDGES: list[tuple[str, str]] = [
    # Infrastructure must not import from the core application layer
    ("dadbot_system", "dadbot.core.dadbot"),
    # UI layer must not import from services directly (go through dadbot core)
    ("dadbot.ui", "dadbot_system.kernel"),
    # Tests are excluded — this applies to production modules only
]


def find_forbidden_cycles(
    forbidden_edges: list[tuple[str, str]] | None = None,
    graph: dict[str, set[str]] | None = None,
) -> list[tuple[str, str]]:
    """Return any forbidden import edges that actually exist in the codebase.

    Each entry in *forbidden_edges* is a tuple of ``(src_fragment, dst_fragment)``.
    Both are substring-matched against full module names.  If the live import
    graph contains an edge from any module matching *src_fragment* to any module
    matching *dst_fragment*, that edge is returned as a violation.

    Uses ``FORBIDDEN_IMPORT_EDGES`` by default.
    """
    if forbidden_edges is None:
        forbidden_edges = FORBIDDEN_IMPORT_EDGES
    if graph is None:
        graph = _build_graph()

    violations: list[tuple[str, str]] = []
    for src_frag, dst_frag in forbidden_edges:
        src_modules = [m for m in graph if src_frag in m]
        dst_modules = {m for m in graph if dst_frag in m}
        for src in src_modules:
            for dst in graph.get(src, set()):
                if dst in dst_modules:
                    violations.append((src, dst))
    return violations
