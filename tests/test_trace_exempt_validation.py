from __future__ import annotations

import ast

import pytest

import tools.arch_completeness_audit as audit

pytestmark = pytest.mark.unit


def _first_func(source: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    node = tree.body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def test_trace_exempt_requires_explicit_justification() -> None:
    source = """
def run(context):
    # TRACE_EXEMPT:
    return context
"""
    func = _first_func(source)
    ok, errors = audit._validate_trace_exemption(source, func)

    assert ok is False
    assert any("reason too short" in err for err in errors)


def test_trace_exempt_rejects_side_effect_api_calls() -> None:
    source = """
def run(context):
    # TRACE_EXEMPT: Pure observer node with no persistence mutation path.
    save_turn_event({"bad": True})
    return context
"""
    func = _first_func(source)
    ok, errors = audit._validate_trace_exemption(source, func)

    assert ok is False
    assert any("side-effect API" in err for err in errors)


def test_trace_exempt_accepts_valid_marker() -> None:
    source = """
def run(context):
    # TRACE_EXEMPT: Pure structural observer; no ledger emission, no checkpoint writes, no downstream calls.
    context.state["x"] = 1
    return context
"""
    func = _first_func(source)
    ok, errors = audit._validate_trace_exemption(source, func)

    assert ok is True
    assert errors == []
