from __future__ import annotations

from dataclasses import dataclass

from tests.eval.datasets import EvalCase
from tests.eval.runner import RunnerConfig, run_eval_suite


@dataclass
class _FakeRuntime:
    def handle_turn(self, user_input: str):
        # Minimal runtime surface for eval-runner smoke tests.
        return (f"echo:{user_input}", True)


class _FakeToolRuntime:
    def execute_tool(self, tool_name: str, parameters: dict | None = None):
        return {"ok": True, "tool": tool_name, "parameters": parameters or {}}

    def handle_turn(self, user_input: str):
        self.execute_tool("echo_tool", {"text": user_input})
        return (f"echo:{user_input}", True)


def test_eval_suite_summary_shape_and_ranges():
    cases = [
        EvalCase(input="hello"),
        EvalCase(input="set a reminder", expected_tools=["set_reminder"]),
    ]

    summary = run_eval_suite(cases, _FakeRuntime())

    assert set(
        [
            "success_rate",
            "irrelevance_rate",
            "avg_tool_calls",
            "efficiency",
            "tool_execution_rate",
            "no_tool_rate",
            "robustness_suppression_rate",
        ],
    ).issubset(summary.keys())
    assert 0.0 <= summary["success_rate"] <= 1.0
    assert 0.0 <= summary["irrelevance_rate"] <= 1.0
    assert summary["avg_tool_calls"] >= 0.0
    assert 0.0 <= summary["efficiency"] <= 1.0
    assert 0.0 <= summary["tool_execution_rate"] <= 1.0
    assert 0.0 <= summary["no_tool_rate"] <= 1.0
    assert 0.0 <= summary["robustness_suppression_rate"] <= 1.0


def test_eval_suite_strict_mode_fails_when_no_tool_path_is_executed():
    cases = [EvalCase(input="hello", expected_tools=["set_reminder"])]

    summary = run_eval_suite(cases, _FakeRuntime(), config=RunnerConfig(mode="evaluation-strict"))
    strict_failures = summary.get("strict_failures", [])
    assert len(strict_failures) == 1
    assert "trace.tool_calls" in str(strict_failures[0].get("error", ""))


def test_eval_suite_strict_mode_passes_with_tool_execution_path():
    cases = [EvalCase(input="hello", expected_tools=["echo_tool"])]

    summary = run_eval_suite(cases, _FakeToolRuntime(), config=RunnerConfig(mode="evaluation-strict"))
    assert summary["avg_tool_calls"] > 0.0
    assert summary.get("strict_failures") == []
