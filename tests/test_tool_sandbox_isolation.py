from __future__ import annotations

import pytest

from ci.ast_invariant_check import check_tool_sandbox_isolation


pytestmark = pytest.mark.unit


def test_private_tool_sandbox_imports_are_repo_isolated():
    violations = check_tool_sandbox_isolation()
    assert violations == []