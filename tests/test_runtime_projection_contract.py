from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DADBOT_ROOT = REPO_ROOT / "dadbot"
CONSUMERS_ROOT = DADBOT_ROOT / "consumers"
STREAMLIT_CONSUMER_PATH = REPO_ROOT / "dad_streamlit.py"
CANONICAL_THREAD_VIEW_PATH = DADBOT_ROOT / "runtime_core" / "store.py"
CANONICAL_GET_VIEW_PATH = DADBOT_ROOT / "runtime_core" / "event_api.py"
EXPECTED_CONSUMER_PACKAGES = {
    CONSUMERS_ROOT,
    CONSUMERS_ROOT / "streamlit",
    CONSUMERS_ROOT / "api_clients",
    CONSUMERS_ROOT / "analytics",
}
ALLOWED_PROJECTION_CLIENT_MODULES = {
    (CONSUMERS_ROOT / "streamlit" / "projection_access.py").resolve(),
}
PROJECTION_IMPORT_MODULES = {
    "dadbot.runtime_core.event_api",
    "dadbot.runtime_core.store",
}
PROJECTION_IMPORT_NAMES = {
    "RuntimeEventAPI",
    "ConversationStore",
}
FORBIDDEN_CONSUMER_IMPORT_MODULES = {
    "dadbot.runtime_core.runtime",
    "dadbot.runtime_core.journal",
    "dadbot.runtime_core.models",
}
FORBIDDEN_CONSUMER_IMPORT_NAMES = {
    "AgentRuntime",
    "Event",
    "EventJournal",
    "FileEventJournal",
}
FORBIDDEN_CONSUMER_HELPERS = {
    "thread_view",
    "get_view",
    "runtime_thread_view",
    "runtime_thread_messages",
    "active_thread_messages",
}


def _parse_module(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def _defined_function_names(tree: ast.AST) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _iter_imports(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, None
        elif isinstance(node, ast.ImportFrom):
            module_name = str(node.module or "")
            if node.level:
                module_name = "." * int(node.level) + module_name
            yield module_name, {alias.name for alias in node.names}


def _repo_python_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*.py"):
        relative = path.relative_to(REPO_ROOT)
        if relative.parts and relative.parts[0] in {".venv", "tests", "session_logs", "runtime"}:
            continue
        files.append(path)
    return files


def _is_consumer_module(path: Path) -> bool:
    try:
        path.relative_to(CONSUMERS_ROOT)
        return True
    except ValueError:
        return False


def _is_runtime_core_module(path: Path) -> bool:
    try:
        path.relative_to(DADBOT_ROOT / "runtime_core")
        return True
    except ValueError:
        return False


def test_consumer_boundary_packages_exist() -> None:
    for package_path in EXPECTED_CONSUMER_PACKAGES:
        assert package_path.is_dir()
        assert (package_path / "__init__.py").is_file()


def test_streamlit_consumer_does_not_import_runtime_internals_or_raw_journal() -> None:
    tree = _parse_module(STREAMLIT_CONSUMER_PATH)

    imported_modules: set[str] = set()
    imported_names: set[str] = set()
    for module, names in _iter_imports(tree):
        imported_modules.add(module)
        if names is not None:
            imported_names.update(names)

    assert imported_modules.isdisjoint(FORBIDDEN_CONSUMER_IMPORT_MODULES)
    assert imported_names.isdisjoint(FORBIDDEN_CONSUMER_IMPORT_NAMES)


def test_only_consumer_modules_import_projection_boundary_clients() -> None:
    offending_paths: list[str] = []

    for path in _repo_python_files():
        resolved_path = path.resolve()
        if _is_runtime_core_module(path):
            continue
        tree = _parse_module(path)
        imported_modules: set[str] = set()
        imported_names: set[str] = set()
        for module, names in _iter_imports(tree):
            imported_modules.add(module)
            if names is not None:
                imported_names.update(names)
        imports_projection_boundary = bool(imported_modules.intersection(PROJECTION_IMPORT_MODULES)) or bool(
            imported_names.intersection(PROJECTION_IMPORT_NAMES)
        )
        if imports_projection_boundary and resolved_path not in ALLOWED_PROJECTION_CLIENT_MODULES:
            offending_paths.append(path.relative_to(REPO_ROOT).as_posix())

    assert offending_paths == []


def test_consumer_modules_do_not_define_projection_builders_or_local_slices() -> None:
    offending_paths: list[str] = []

    for path in _repo_python_files():
        if not _is_consumer_module(path):
            continue
        tree = _parse_module(path)
        if not _defined_function_names(tree).isdisjoint(FORBIDDEN_CONSUMER_HELPERS):
            offending_paths.append(path.relative_to(REPO_ROOT).as_posix())

    assert offending_paths == []


def test_non_consumer_modules_do_not_depend_on_consumer_packages() -> None:
    offending_paths: list[str] = []

    for path in DADBOT_ROOT.rglob("*.py"):
        if _is_consumer_module(path):
            continue
        tree = _parse_module(path)
        imported_modules = {module for module, _ in _iter_imports(tree)}
        if any(module == "dadbot.consumers" or module.startswith("dadbot.consumers.") for module in imported_modules):
            offending_paths.append(path.relative_to(REPO_ROOT).as_posix())

    assert offending_paths == []


def test_only_canonical_runtime_files_define_projection_builders() -> None:
    thread_view_owners: list[str] = []
    get_view_owners: list[str] = []

    for path in _repo_python_files():
        tree = _parse_module(path)
        names = _defined_function_names(tree)
        if "thread_view" in names:
            thread_view_owners.append(path.relative_to(REPO_ROOT).as_posix())
        if "get_view" in names:
            get_view_owners.append(path.relative_to(REPO_ROOT).as_posix())

    assert thread_view_owners == [CANONICAL_THREAD_VIEW_PATH.relative_to(REPO_ROOT).as_posix()]
    assert get_view_owners == [CANONICAL_GET_VIEW_PATH.relative_to(REPO_ROOT).as_posix()]