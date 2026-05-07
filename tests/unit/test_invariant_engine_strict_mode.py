from __future__ import annotations

import pytest

from dadbot.core.invariant_engine import ExecutionState, GlobalInvariantEngine


class _DagWithoutAcyclicMethod:
    pass


@pytest.mark.unit
def test_dag_acyclic_strict_mode_rejects_unknown_dag_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DADBOT_STRICT_INVARIANTS", "1")
    engine = GlobalInvariantEngine.default()
    state = ExecutionState(
        planner_output={"intent_type": "lookup", "strategy": "tool-first", "tool_plan": ["search"]},
        dag=_DagWithoutAcyclicMethod(),
        tool_events=[],
    )

    report = engine.validate_all(state)
    violations = {v.invariant_id: v for v in report.violations}
    assert "dag.acyclic" in violations
    assert report.ok is False


@pytest.mark.unit
def test_dag_acyclic_non_strict_mode_keeps_legacy_fallback() -> None:
    engine = GlobalInvariantEngine.default()
    state = ExecutionState(
        planner_output={"intent_type": "lookup", "strategy": "tool-first", "tool_plan": ["search"]},
        dag=_DagWithoutAcyclicMethod(),
        tool_events=[],
    )

    report = engine.validate_all(state)
    violations = {v.invariant_id for v in report.violations}
    assert "dag.acyclic" not in violations
