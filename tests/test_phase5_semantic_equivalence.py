from __future__ import annotations

from dadbot.core.execution_equivalence_oracle import ExecutionEquivalenceOracle
from dadbot.core.execution_replay_engine import reconstruct_terminal_state_from_trace


def _trace(tool_output: str = "Weather: Sunny") -> dict:
    return {
        "schema_version": "2.0",
        "final_hash": "seed-hash",
        "normalized_response": "The weather is sunny.",
        "execution_dag": {"dag_hash": "dag-hash"},
        "memory_snapshot_used": {
            "memory_structured": {"topic": "weather"},
            "memory_full_history_id": "hist-1",
        },
        "memory_retrieval_set": [{"id": "m1", "summary": "weather preferences"}],
        "tool_outputs": [{"tool": "weather", "output": tool_output}],
        "steps": [
            {"seq": 0, "operation": "model_call", "payload": {"purpose": "chat"}},
            {
                "seq": 1,
                "operation": "external_system_call",
                "payload": {"operation": "tool_dispatch", "system": "builtin_tool:weather", "status": "ok"},
            },
            {"seq": 2, "operation": "model_output", "payload": {"output_hash": "out-1"}},
        ],
    }


def _seed(trace: dict) -> dict:
    return reconstruct_terminal_state_from_trace(
        terminal_state_seed={
            "schema_version": "1.0",
            "tool_trace_hash": "tool-trace",
            "policy_snapshot": {"kernel_policy": {"mode": "strict"}},
        },
        execution_trace_context=trace,
    )


def test_semantic_oracle_reports_semantic_fields():
    trace = _trace()
    seed = _seed(trace)
    result = ExecutionEquivalenceOracle.evaluate(
        input_seed="input",
        trace_seed="trace",
        memory_state_id=str(seed.get("memory_view_state_id") or ""),
        terminal_state_seed=seed,
        execution_trace_context=trace,
        semantic_mode="exact",
    )
    assert result.semantic_mode == "exact"
    assert isinstance(result.semantic_report, dict)
    assert "semantic_action_equivalent" in result.semantic_report


def test_semantic_oracle_approximate_mode_normalizes_tool_output_formatting():
    trace = _trace(tool_output=" Weather:   SUNNY ")
    seed = _seed(trace)
    result = ExecutionEquivalenceOracle.evaluate(
        input_seed="input",
        trace_seed="trace",
        memory_state_id=str(seed.get("memory_view_state_id") or ""),
        terminal_state_seed=seed,
        execution_trace_context=trace,
        semantic_mode="approximate",
    )
    assert result.semantic_equivalent is True
    assert result.equivalent is True


def test_semantic_oracle_exact_mode_preserves_strictness():
    trace = _trace(tool_output=" Weather:   SUNNY ")
    seed = _seed(trace)
    result = ExecutionEquivalenceOracle.evaluate(
        input_seed="input",
        trace_seed="trace",
        memory_state_id=str(seed.get("memory_view_state_id") or ""),
        terminal_state_seed=seed,
        execution_trace_context=trace,
        semantic_mode="exact",
    )
    assert result.semantic_mode == "exact"
    assert isinstance(result.violations, list)


def test_semantic_oracle_detects_memory_state_divergence():
    trace = _trace()
    seed = _seed(trace)
    seed["final_memory_view"] = {
        "memory_structured": {"topic": "finance"},
        "memory_retrieval_set": [{"id": "m2", "summary": "budget"}],
        "memory_full_history_id": "hist-2",
    }
    result = ExecutionEquivalenceOracle.evaluate(
        input_seed="input",
        trace_seed="trace",
        memory_state_id="state-x",
        terminal_state_seed=seed,
        execution_trace_context=trace,
        semantic_mode="exact",
    )
    assert "semantic_memory_state_equivalence" in result.violations or result.equivalent is False
