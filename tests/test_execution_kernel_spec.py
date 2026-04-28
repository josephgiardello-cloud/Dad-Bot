from __future__ import annotations

import pytest

from dadbot.core.execution_firewall import ExecutionFirewall
from dadbot.core.execution_kernel import ExecutionKernel
from dadbot.core.execution_kernel_spec import (
    ExecutionKernelContractViolation,
    validate_execution_kernel_spec,
)
from dadbot.core.invariant_registry import InvariantRegistry


class _BadKernel:
    strict = False

    def validate(self, *args, **kwargs):
        return None

    def run(self, *args, **kwargs):
        return None


def test_real_execution_kernel_conforms_to_spec():
    kernel = ExecutionKernel(
        firewall=ExecutionFirewall(),
        invariant_registry=InvariantRegistry(),
        quarantine=None,
        strict=False,
    )
    issues = validate_execution_kernel_spec(kernel)
    assert issues == []


def test_invalid_kernel_spec_raises_when_enforced():
    with pytest.raises(ExecutionKernelContractViolation):
        validate_execution_kernel_spec(_BadKernel(), raise_on_failure=True)
