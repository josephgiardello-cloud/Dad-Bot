from __future__ import annotations

import pytest

from ci.ast_invariant_check import check_service_shell_forward_only

pytestmark = pytest.mark.unit


def test_service_shell_is_forward_only():
    violations = check_service_shell_forward_only()
    assert violations == []