from __future__ import annotations

from dataclasses import dataclass

from tests.benchmark_runner import BenchmarkRunner
from tests.scenario_suite import get_scenario


@dataclass
class _StubContext:
    state: dict


class _StubOrchestrator:
    def __init__(self, state: dict):
        self._state = state
        self._last_turn_context = None

    async def handle_turn(self, user_input: str, session_id: str, timeout_seconds: float):
        self._last_turn_context = _StubContext(state=dict(self._state))
        return "stub response", True


def _base_state() -> dict:
    return {
        "plan": [{"step": "mortgage"}, {"step": "inspection"}],
        "detected_goals": ["buy house"],
        "task_decomposition": {
            "tasks": ["mortgage", "inspection"],
            "dependencies": [("mortgage", "inspection")],
        },
        "tool_ir": {
            "executions": [
                {
                    "tool_name": "weather_api",
                    "sequence": 1,
                    "deterministic_id": "t1",
                    "duration_ms": 30.0,
                }
            ]
        },
        "tool_results": [
            {
                "tool_name": "weather_api",
                "sequence": 1,
                "deterministic_id": "t1",
                "status": "failed",
                "output": None,
            }
        ],
        "memory_structured": {"user_preference": "vietnamese cuisine"},
        "session_goals": [{"id": "g1"}],
    }


def test_benchmark_runner_orchestrator_path_uses_causal_trace_contracts():
    scenario = get_scenario("dependency_aware_task")

    coherent_state = {
        **_base_state(),
        "ux_trace": {
            "intent_shift_detected": True,
            "clarification_requested": True,
            "repair_event_emitted": True,
            "user_confusion_detected": False,
            "replan_triggered": True,
            "memory_correction_written": True,
        },
        "planner_causal_trace": {
            "planner_replan_reason": "user correction requested",
            "intent_delta_vector": ["topic shift"],
            "dependency_graph_diff": ["replace old edge with new edge"],
        },
        "memory_causal_trace": {
            "trigger": "user correction",
            "read_link_id": "mem-read-1",
            "write_link_id": "mem-write-1",
            "influenced_final_response": True,
            "overridden": False,
        },
        "tool_failure_semantics": [
            {
                "tool_name": "weather_api",
                "failure_class": "timeout",
                "reason": "request timed out",
            }
        ],
    }

    incoherent_state = {
        **_base_state(),
        "ux_trace": {
            "intent_shift_detected": False,
            "clarification_requested": False,
            "repair_event_emitted": False,
            "user_confusion_detected": True,
            "replan_triggered": False,
            "memory_correction_written": False,
        },
        "planner_causal_trace": {
            "planner_replan_reason": "",
            "intent_delta_vector": ["topic shift"],
            "dependency_graph_diff": [],
        },
        "memory_causal_trace": {
            "trigger": "user correction",
            "read_link_id": "",
            "write_link_id": "",
            "influenced_final_response": False,
            "overridden": True,
        },
        "tool_failure_semantics": [],
    }

    coherent_runner = BenchmarkRunner(mode="orchestrator", orchestrator=_StubOrchestrator(coherent_state))
    incoherent_runner = BenchmarkRunner(mode="orchestrator", orchestrator=_StubOrchestrator(incoherent_state))

    coherent_result = coherent_runner.run_scenario(scenario)
    incoherent_result = incoherent_runner.run_scenario(scenario)

    coherent_score = coherent_result["capability_score"]
    incoherent_score = incoherent_result["capability_score"]

    assert coherent_score is not None
    assert incoherent_score is not None
    assert coherent_result["execution"]["completed"] is True
    assert incoherent_result["execution"]["completed"] is True
    assert coherent_score["overall"] > incoherent_score["overall"]
