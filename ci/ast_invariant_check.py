"""AST-level invariant enforcement for the DadBot kernel.

RULE11_SETATTR_KERNEL_BAN
    Kernel files must not call ``setattr()`` outside a ``__setattr__`` method
    body.  Bare setattr() in kernel code is a hidden mutation affordance that
    bypasses the mutation gate.  ``__setattr__`` overrides are the one
    legitimate routing exception (e.g. config mirror in dadbot.py).

RULE12_GLOBALS_WRITE_BAN
    Kernel files must not write to ``globals()`` outside a ``__getattr__``
    function.  The only permitted use is lazy-import caching inside
    ``__getattr__`` (the standard module-level deferred-import pattern).

RULE13_DYNAMIC_PATCH_BAN
    Kernel files must not call ``importlib.reload()`` or assign to
    ``sys.modules[x]``.  These are dynamic patching surfaces that allow
    runtime module substitution and break the closed-world assumption.

RULE14_TOOL_SANDBOX
    Tool files (``tools/``) must not call ``setattr()`` on kernel-owned object
    names (bot, runtime, kernel, state, graph, memory, session, context).
    Tools are observers and executors — they must never reach back and mutate
    kernel state via reflection.

RULE15_POLICY_AUTHORITY
    The service layer (``dadbot/services/``) must not define tool authorization
    policy.  Specifically: no ``frozenset`` literal containing known tool names
    (``set_reminder``, ``web_search``) may appear in service-layer files.
    Policy belongs exclusively in the runtime contract
    (``DadBotActionMixin.authorize_tool_execution``).

RULE16_TOOL_SANDBOX_ISOLATION
    The private tool sandbox implementation must not be imported outside the
    allowed core execution-spine files. All callers must go through the single
    kernel-owned spine: ``dadbot.core.tool_executor.execute_tool`` or approved
    core-internal test adapters.

RULE17_LLM_COMMIT_SEPARATION
    The persistence module must remain post-commit pure. It may publish a
    post-commit-ready event, but it must not import or call memory-reasoning or
    LLM-adjacent functionality. Those operations belong exclusively in the
    post-commit worker.

RULE18_TURN_ENTRYPOINT_CLOSURE
    Production callers under ``dadbot/`` must use the canonical
    ``execute_turn(...)`` adapter. Direct calls to legacy turn entrypoints
    (``process_user_message*``, ``handle_turn*``) are compatibility-only and
    restricted to the owning wrapper modules.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ci.import_graph_check import classify_layer

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIR_NAMES = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}

# Objects that tools must not setattr() on — they belong to the kernel
KERNEL_OWNED_NAMES: frozenset[str] = frozenset(
    {"bot", "runtime", "kernel", "state", "graph", "memory", "session", "context"}
)


@dataclass(frozen=True)
class Violation:
    rule: str
    path: str
    detail: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
            continue
        files.append(path)
    return files


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _parse(file_path: Path) -> ast.Module | None:
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        return ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return None


def _enclosing_function_names(node: ast.AST, parents: list[ast.AST]) -> list[str]:
    """Return the names of all enclosing FunctionDef / AsyncFunctionDef nodes."""
    names: list[str] = []
    for parent in parents:
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(parent.name)
    return names


# ---------------------------------------------------------------------------
# Parent-map builder
# ---------------------------------------------------------------------------


def _build_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    """Return a map from node id() to its parent node."""
    parent_map: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent_map[id(child)] = node
    return parent_map


def _enclosing_func_names(node: ast.AST, parent_map: dict[int, ast.AST]) -> list[str]:
    """Walk up the parent chain and collect names of enclosing function defs."""
    names: list[str] = []
    current: ast.AST | None = parent_map.get(id(node))
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(current.name)
        current = parent_map.get(id(current))
    return names


# ---------------------------------------------------------------------------
# Scan functions (use ast.walk for clarity)
# ---------------------------------------------------------------------------


def _scan_kernel_setattr(tree: ast.AST) -> list[tuple[int, str, str]]:
    """RULE11: setattr(self/kernel-obj, ...) in kernel files outside __setattr__.

    Targets: setattr(self, ...) and setattr(<kernel-owned-name>, ...).
    Does NOT flag setattr on arbitrary external parameters (e.g. context_builder
    patching in adapters), which is a different concern handled at review time.
    """
    parent_map = _build_parent_map(tree)
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "setattr"):
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        # Only flag setattr on self or kernel-owned names
        is_self = isinstance(first_arg, ast.Name) and first_arg.id == "self"
        is_kernel = isinstance(first_arg, ast.Name) and first_arg.id in KERNEL_OWNED_NAMES
        if not (is_self or is_kernel):
            continue
        enclosing = _enclosing_func_names(node, parent_map)
        if "__setattr__" not in enclosing:
            target_name = first_arg.id  # type: ignore[union-attr]
            found.append((
                node.lineno,
                "RULE11_SETATTR_KERNEL_BAN",
                f"setattr({target_name}, ...) in kernel file outside __setattr__ "
                f"(line {node.lineno})",
            ))
    return found


def _scan_kernel_globals_write(tree: ast.AST) -> list[tuple[int, str, str]]:
    """RULE12: globals()[x]= writes outside __getattr__ in kernel files."""
    parent_map = _build_parent_map(tree)
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        lineno: int | None = None
        is_write = False

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if _is_globals_subscript(target):
                    is_write = True
                    lineno = node.lineno
                    break
        elif isinstance(node, ast.AugAssign):
            if _is_globals_subscript(node.target):
                is_write = True
                lineno = node.lineno

        if is_write and lineno is not None:
            enclosing = _enclosing_func_names(node, parent_map)
            if "__getattr__" not in enclosing:
                found.append((
                    lineno,
                    "RULE12_GLOBALS_WRITE_BAN",
                    f"globals()[x] write outside __getattr__ (line {lineno})",
                ))
    return found


def _scan_kernel_dynamic_patch(tree: ast.AST) -> list[tuple[int, str, str]]:
    """RULE13: importlib.reload() and sys.modules[x]= in kernel files."""
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # importlib.reload(...)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "reload"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
            ):
                found.append((
                    node.lineno,
                    "RULE13_DYNAMIC_PATCH_BAN",
                    f"importlib.reload() in kernel file (line {node.lineno})",
                ))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if _is_sys_modules_subscript(target):
                    found.append((
                        node.lineno,
                        "RULE13_DYNAMIC_PATCH_BAN",
                        f"sys.modules[x] = write in kernel file (line {node.lineno})",
                    ))
    return found


def _scan_tool_setattr(tree: ast.AST) -> list[tuple[int, str, str]]:
    """RULE14: setattr() on kernel-owned names in tools/ files."""
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "setattr"):
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Name) and first_arg.id in KERNEL_OWNED_NAMES:
            found.append((
                node.lineno,
                "RULE14_TOOL_SANDBOX",
                f"setattr({first_arg.id}, ...) — tools must not mutate "
                f"kernel-owned objects (line {node.lineno})",
            ))
    return found


# ---------------------------------------------------------------------------
# AST pattern helpers
# ---------------------------------------------------------------------------


def _is_globals_subscript(node: ast.expr) -> bool:
    """Return True if node is globals()[x] (i.e. a subscript of a globals() call)."""
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "globals"
    )


def _is_sys_modules_subscript(node: ast.expr) -> bool:
    """Return True if node is sys.modules[x]."""
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "modules"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "sys"
    )


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def check_kernel_setattr_ban() -> list[Violation]:
    """RULE11: setattr() in kernel files must only appear inside __setattr__."""
    violations: list[Violation] = []
    for file_path in _iter_python_files():
        rel = _rel(file_path)
        if classify_layer(rel) != "kernel":
            continue
        tree = _parse(file_path)
        if tree is None:
            continue
        for lineno, rule, detail in _scan_kernel_setattr(tree):
            violations.append(Violation(rule=rule, path=f"{rel}:{lineno}", detail=detail))
    return violations


def check_kernel_globals_write_ban() -> list[Violation]:
    """RULE12: globals()[x]= in kernel files only allowed inside __getattr__."""
    violations: list[Violation] = []
    for file_path in _iter_python_files():
        rel = _rel(file_path)
        if classify_layer(rel) != "kernel":
            continue
        tree = _parse(file_path)
        if tree is None:
            continue
        for lineno, rule, detail in _scan_kernel_globals_write(tree):
            violations.append(Violation(rule=rule, path=f"{rel}:{lineno}", detail=detail))
    return violations


def check_kernel_dynamic_patch_ban() -> list[Violation]:
    """RULE13: importlib.reload() and sys.modules[x]= banned in kernel files."""
    violations: list[Violation] = []
    for file_path in _iter_python_files():
        rel = _rel(file_path)
        if classify_layer(rel) != "kernel":
            continue
        tree = _parse(file_path)
        if tree is None:
            continue
        for lineno, rule, detail in _scan_kernel_dynamic_patch(tree):
            violations.append(Violation(rule=rule, path=f"{rel}:{lineno}", detail=detail))
    return violations


def check_tool_sandbox_setattr_ban() -> list[Violation]:
    """RULE14: Agentic tool execution files must not setattr() on kernel-owned objects.

    Scans ``dadbot/core/tool_*.py`` — the files implementing the tool execution
    sandbox.  Tool executor callables must not reach back into kernel-owned
    state via reflection.  CI infrastructure under ``tools/`` is excluded:
    those are dev/ops tools that legitimately set up test stubs.
    """
    violations: list[Violation] = []
    for file_path in _iter_python_files():
        rel = _rel(file_path)
        # Only scan the agentic tool kernel files, not the CI tools/ directory
        if not rel.startswith("dadbot/core/tool_"):
            continue
        tree = _parse(file_path)
        if tree is None:
            continue
        for lineno, rule, detail in _scan_tool_setattr(tree):
            violations.append(Violation(rule=rule, path=f"{rel}:{lineno}", detail=detail))
    return violations


# ---------------------------------------------------------------------------
# RULE15: Policy authority — no service-layer tool allowlists
# ---------------------------------------------------------------------------

_TOOL_NAME_LITERALS: frozenset[str] = frozenset({"set_reminder", "web_search"})


def _scan_service_policy_frozenset(tree: ast.AST) -> list[tuple[int, str]]:
    """Return (lineno, detail) for frozenset literals containing tool names."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "frozenset"):
            continue
        # frozenset({...}) or frozenset([...])
        if not node.args:
            continue
        arg = node.args[0]
        elements: list[ast.expr] = []
        if isinstance(arg, (ast.Set, ast.List, ast.Tuple)):
            elements = list(arg.elts)
        for elt in elements:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                if elt.value in _TOOL_NAME_LITERALS:
                    found.append((
                        node.lineno,
                        f"frozenset containing tool name '{elt.value}' — "
                        f"policy belongs in authorize_tool_execution() "
                        f"(line {node.lineno})",
                    ))
    return found


def check_service_policy_authority() -> list[Violation]:
    """RULE15: Service layer must not define tool allowlists."""
    violations: list[Violation] = []
    for file_path in _iter_python_files():
        rel = _rel(file_path)
        if not rel.startswith("dadbot/services/"):
            continue
        tree = _parse(file_path)
        if tree is None:
            continue
        for lineno, detail in _scan_service_policy_frozenset(tree):
            violations.append(Violation(
                rule="RULE15_POLICY_AUTHORITY",
                path=f"{rel}:{lineno}",
                detail=detail,
            ))
    return violations


# ---------------------------------------------------------------------------
# RULE16: Tool sandbox isolation — no private sandbox import outside allowlist
# ---------------------------------------------------------------------------

_TOOL_SANDBOX_ALLOWED_IMPORTERS: frozenset[str] = frozenset(
    {
        "dadbot/core/tool_executor.py",
        "dadbot/core/testing/tool_runtime_test_adapter.py",
    }
)


def _scan_tool_sandbox_import(tree: ast.AST) -> list[tuple[int, str]]:
    """Return (lineno, detail) for any private tool sandbox import."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.endswith("._tool_sandbox"):
                found.append((
                    node.lineno,
                    f"private _tool_sandbox import is forbidden outside the execution-spine allowlist "
                    f"(line {node.lineno})",
                ))
        elif isinstance(node, ast.Import):
            for alias in (node.names or []):
                if (alias.name or "").endswith("._tool_sandbox"):
                    found.append((
                        node.lineno,
                        f"private _tool_sandbox module import is forbidden outside the execution-spine allowlist "
                        f"(line {node.lineno})",
                    ))
    return found


def check_tool_sandbox_isolation() -> list[Violation]:
    """RULE16: private tool sandbox imports are repo-wide allowlist-only."""
    violations: list[Violation] = []
    for file_path in _iter_python_files():
        rel = _rel(file_path)
        if rel in _TOOL_SANDBOX_ALLOWED_IMPORTERS:
            continue
        tree = _parse(file_path)
        if tree is None:
            continue
        for lineno, detail in _scan_tool_sandbox_import(tree):
            violations.append(Violation(
                rule="RULE16_TOOL_SANDBOX_ISOLATION",
                path=f"{rel}:{lineno}",
                detail=detail,
            ))
    return violations


# ---------------------------------------------------------------------------
# RULE17: Persistence purity — no memory/LLM reasoning imports or calls
# ---------------------------------------------------------------------------

_PERSISTENCE_BANNED_CALLS: frozenset[str] = frozenset(
    {
        "consolidate_memories",
        "apply_controlled_forgetting",
        "call_ollama_chat",
        "build_memory_consolidation_prompt",
    }
)
_PERSISTENCE_BANNED_IMPORT_SUBSTRINGS: tuple[str, ...] = (
    "dadbot.managers.memory_coordination",
    "dadbot.memory.",
    "dadbot.services.llm_call_adapter",
    "ollama",
)


def _scan_persistence_reasoning_references(tree: ast.AST) -> list[tuple[int, str]]:
    """Return (lineno, detail) for banned persistence memory/LLM references."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_name = str(node.module or "")
            if any(part in module_name for part in _PERSISTENCE_BANNED_IMPORT_SUBSTRINGS):
                found.append((
                    node.lineno,
                    f"persistence must not import reasoning module {module_name!r} (line {node.lineno})",
                ))
        elif isinstance(node, ast.Import):
            for alias in (node.names or []):
                module_name = str(alias.name or "")
                if any(part in module_name for part in _PERSISTENCE_BANNED_IMPORT_SUBSTRINGS):
                    found.append((
                        node.lineno,
                        f"persistence must not import reasoning module {module_name!r} (line {node.lineno})",
                    ))
        elif isinstance(node, ast.Call):
            call_name: str | None = None
            if isinstance(node.func, ast.Name) and node.func.id in _PERSISTENCE_BANNED_CALLS:
                call_name = node.func.id
            elif isinstance(node.func, ast.Attribute) and node.func.attr in _PERSISTENCE_BANNED_CALLS:
                call_name = node.func.attr
            if call_name is not None:
                found.append((
                    node.lineno,
                    f"persistence must not call reasoning function {call_name}() (line {node.lineno})",
                ))
    return found


def check_llm_commit_separation() -> list[Violation]:
    """RULE17: persistence.py must stay free of memory/LLM reasoning references."""
    violations: list[Violation] = []
    target = ROOT / "dadbot" / "services" / "persistence.py"
    if not target.exists():
        return violations
    tree = _parse(target)
    if tree is None:
        return violations
    rel = _rel(target)
    for lineno, detail in _scan_persistence_reasoning_references(tree):
        violations.append(Violation(
            rule="RULE17_LLM_COMMIT_SEPARATION",
            path=f"{rel}:{lineno}",
            detail=detail,
        ))
    return violations


_LEGACY_TURN_CALL_NAMES: frozenset[str] = frozenset(
    {
        "process_user_message",
        "process_user_message_async",
        "process_user_message_stream",
        "process_user_message_stream_async",
        "handle_turn",
        "handle_turn_async",
        "handle_turn_sync",
    }
)
_TURN_ENTRYPOINT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "dadbot/core/turn_mixin.py",
        "dadbot/core/orchestrator.py",
        "dadbot/services/turn_service.py",
    }
)


def _scan_legacy_turn_calls(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in _LEGACY_TURN_CALL_NAMES:
            continue
        found.append((
            node.lineno,
            f"legacy turn entrypoint .{node.func.attr}() is forbidden in production callers; use execute_turn() instead (line {node.lineno})",
        ))
    return found


def check_turn_entrypoint_closure() -> list[Violation]:
    violations: list[Violation] = []
    for file_path in _iter_python_files():
        rel = _rel(file_path)
        if not rel.startswith("dadbot/"):
            continue
        if rel in _TURN_ENTRYPOINT_ALLOWLIST:
            continue
        tree = _parse(file_path)
        if tree is None:
            continue
        for lineno, detail in _scan_legacy_turn_calls(tree):
            violations.append(Violation(
                rule="RULE18_TURN_ENTRYPOINT_CLOSURE",
                path=f"{rel}:{lineno}",
                detail=detail,
            ))
    return violations


# ---------------------------------------------------------------------------
# RULE19 — No durable-synthesis calls in the execution-path hot path
# ---------------------------------------------------------------------------
# Post-commit memory operations (archive, consolidate, forget, evolve) must
# only run from the post-commit background worker or maintenance scheduler.
# They must never be called inline from nodes.py, turn_service.py, or
# agent_service.py (i.e. the live turn execution path).

_SYNTHESIS_CALL_NAMES: frozenset[str] = frozenset(
    {
        "consolidate_memories",
        "run_periodic_durable_synthesis",
        "apply_controlled_forgetting",
        "archive_session_context",
        "evolve_persona",
        "update_memory_store",
        "refresh_relationship_timeline",
    }
)

_EXECUTION_PATH_FILES: frozenset[str] = frozenset(
    {
        "dadbot/core/nodes.py",
        "dadbot/core/graph_pipeline_nodes.py",
        "dadbot/services/turn_service.py",
        "dadbot/services/agent_service.py",
        "dadbot/services/maintenance_service.py",
    }
)


def _scan_synthesis_calls(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        attr = None
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
        elif isinstance(node.func, ast.Name):
            attr = node.func.id
        if attr in _SYNTHESIS_CALL_NAMES:
            found.append((
                node.lineno,
                f"post-commit synthesis op .{attr}() must not run in execution path (line {node.lineno}); move to post-commit background worker",
            ))
    return found


def check_no_synthesis_in_execution_path() -> list[Violation]:
    violations: list[Violation] = []
    for file_path in _iter_python_files():
        rel = _rel(file_path)
        if rel not in _EXECUTION_PATH_FILES:
            continue
        tree = _parse(file_path)
        if tree is None:
            continue
        for lineno, detail in _scan_synthesis_calls(tree):
            violations.append(Violation(
                rule="RULE19_NO_SYNTHESIS_IN_EXECUTION_PATH",
                path=f"{rel}:{lineno}",
                detail=detail,
            ))
    return violations


def run_checks() -> list[Violation]:
    return [
        *check_kernel_setattr_ban(),
        *check_kernel_globals_write_ban(),
        *check_kernel_dynamic_patch_ban(),
        *check_tool_sandbox_setattr_ban(),
        *check_service_policy_authority(),
        *check_tool_sandbox_isolation(),
        *check_llm_commit_separation(),
        *check_turn_entrypoint_closure(),
        *check_no_synthesis_in_execution_path(),
    ]


if __name__ == "__main__":
    results = run_checks()
    if not results:
        print("PASS ast_invariant_check — no violations")
        raise SystemExit(0)
    print(f"FAIL ast_invariant_check — {len(results)} violation(s)")
    for v in results:
        print(f"  {v.rule}: {v.path}")
        print(f"    -> {v.detail}")
    raise SystemExit(1)
