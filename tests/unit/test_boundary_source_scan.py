import glob
import os
from pathlib import Path

import pytest

from ci import ast_invariant_check, filesystem_enforcement, import_graph_check, mutation_tracking_check


@pytest.mark.parametrize(
    "scanner",
    [ast_invariant_check, filesystem_enforcement, import_graph_check, mutation_tracking_check],
)
def test_boundary_source_scan_prunes_excluded_trees(scanner, tmp_path, monkeypatch):
    included = {"entry.py", "dadbot/core/node.py", "dadbot_system/worker.py", "ci/check.py", "tools/check.py"}
    excluded = {
        ".venv/library/module.py",
        ".venv-other/library/module.py",
        "external/library/module.py",
        "tests/test_node.py",
        "dadbot/.venv-other/module.py",
        "dadbot/__pycache__/module.py",
        "dadbot/.git/module.py",
    }
    for relative in included | excluded:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    original_scandir = os.scandir
    forbidden_roots = {
        tmp_path / ".venv",
        tmp_path / ".venv-other",
        tmp_path / "external",
        tmp_path / "tests",
        tmp_path / "dadbot" / ".venv-other",
        tmp_path / "dadbot" / "__pycache__",
        tmp_path / "dadbot" / ".git",
    }

    def checked_scandir(path):
        assert Path(path) not in forbidden_roots, f"Excluded directory was traversed: {path}"
        return original_scandir(path)

    monkeypatch.setattr(scanner, "ROOT", tmp_path)
    monkeypatch.setattr(os, "scandir", checked_scandir)
    monkeypatch.setattr(glob._StringGlobber, "scandir", staticmethod(checked_scandir))
    actual = {path.relative_to(tmp_path).as_posix() for path in scanner._iter_python_files()}
    assert actual == included
