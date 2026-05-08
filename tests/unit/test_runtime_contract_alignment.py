from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
pytestmark = pytest.mark.unit


def test_python_runtime_contract_is_aligned() -> None:
    root = Path(__file__).resolve().parents[2]

    pyproject_data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    requires_python = str(pyproject_data.get("project", {}).get("requires-python") or "").strip()
    assert requires_python == ">=3.13"

    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "Python 3.13+" in readme

    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.13-slim" in dockerfile
