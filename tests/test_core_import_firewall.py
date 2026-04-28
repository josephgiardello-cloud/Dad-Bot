from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DADBOT_ROOT = ROOT / "dadbot"


TOOLING_PREFIXES = (
    "dadbot.uril",
    "dadbot.ux_overlay",
    "tools",
    "tests",
)

EXPERIMENTAL_PREFIXES = (
    "dadbot.core.invariance_contract",
    "dadbot.core.transaction_coordinator",
    "dadbot.core.semantic_graph_validator",
    "dadbot.core.contract_propagation",
)

CORE_FIREWALL_MODULE_PREFIXES = (
    "dadbot.core",
    "dadbot.runtime_adapter",
    "dadbot.registry",
    "dadbot.memory",
)


def _module_name(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    return ".".join(rel.parts)


def _iter_runtime_python_files() -> list[Path]:
    return [
        p
        for p in DADBOT_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts
    ]


def _imported_modules(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return set()

    imports: set[str] = set()
    current_mod = _module_name(path)
    current_parts = current_mod.split(".")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = str(alias.name or "").strip()
                if name:
                    imports.add(name)
        elif isinstance(node, ast.ImportFrom):
            module = str(node.module or "").strip()
            level = int(getattr(node, "level", 0) or 0)

            if level > 0:
                base_parts = current_parts[:-level]
                if module:
                    resolved = ".".join(base_parts + module.split("."))
                else:
                    resolved = ".".join(base_parts)
                if resolved:
                    imports.add(resolved)
            elif module:
                imports.add(module)

    return imports


def _starts_with_any(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def test_core_runtime_import_firewall_blocks_tooling_and_tests() -> None:
    """Core runtime must not import tooling/CI/test layers."""
    violations: list[str] = []

    for path in _iter_runtime_python_files():
        source_mod = _module_name(path)
        if not _starts_with_any(source_mod, CORE_FIREWALL_MODULE_PREFIXES):
            continue

        for imported in _imported_modules(path):
            if _starts_with_any(imported, TOOLING_PREFIXES):
                rel = path.relative_to(ROOT).as_posix()
                violations.append(f"{source_mod} imports {imported} ({rel})")

    assert violations == [], "Core import firewall violations:\n" + "\n".join(sorted(violations))


def test_core_runtime_does_not_hard_depend_on_experimental_modules() -> None:
    """Experimental modules must stay opt-in and not be hard-required by core."""
    violations: list[str] = []

    for path in _iter_runtime_python_files():
        source_mod = _module_name(path)
        if not _starts_with_any(source_mod, CORE_FIREWALL_MODULE_PREFIXES):
            continue

        for imported in _imported_modules(path):
            if _starts_with_any(imported, EXPERIMENTAL_PREFIXES):
                rel = path.relative_to(ROOT).as_posix()
                violations.append(f"{source_mod} imports experimental {imported} ({rel})")

    assert violations == [], "Experimental hard-dependency violations:\n" + "\n".join(sorted(violations))


def test_dependency_direction_core_must_not_import_runtime_adapter() -> None:
    """Allowed direction: runtime_adapter -> core (not the reverse)."""
    violations: list[str] = []

    for path in _iter_runtime_python_files():
        source_mod = _module_name(path)
        if not (source_mod == "dadbot.core" or source_mod.startswith("dadbot.core.")):
            continue
        for imported in _imported_modules(path):
            if imported == "dadbot.runtime_adapter" or imported.startswith("dadbot.runtime_adapter."):
                rel = path.relative_to(ROOT).as_posix()
                violations.append(f"{source_mod} imports {imported} ({rel})")

    assert violations == [], "Direction violations (core -> runtime_adapter):\n" + "\n".join(sorted(violations))


def test_dependency_direction_registry_must_not_import_core_layer() -> None:
    """Registry is a leaf service locator for runtime wiring and must stay core-independent."""
    path = ROOT / "dadbot" / "registry.py"
    imported = sorted(_imported_modules(path))
    violations = [m for m in imported if m == "dadbot.core" or m.startswith("dadbot.core.")]
    assert violations == [], (
        "Direction violations (registry -> core). Found imports: "
        + ", ".join(violations)
    )


def test_entrypoints_route_startup_through_app_runtime_main() -> None:
    """Execution startup must route through dadbot.app_runtime.main()."""
    dad_py = (ROOT / "Dad.py").read_text(encoding="utf-8", errors="replace")
    launch_py = (ROOT / "launch.py").read_text(encoding="utf-8", errors="replace")
    install_py = (ROOT / "install.py").read_text(encoding="utf-8", errors="replace")

    assert "dadbot.app_runtime" in dad_py and "run_app_main" in dad_py
    assert "dadbot.app_runtime" in launch_py and "app_main" in launch_py
    # install.py may launch Dad.py, which is accepted as an app_runtime.main wrapper.
    assert "Dad.py" in install_py
