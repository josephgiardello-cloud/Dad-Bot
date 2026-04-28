from __future__ import annotations

from types import SimpleNamespace

from dadbot.core.execution_trace_context import ExecutionTraceRecorder, build_execution_trace_context


def _fake_context() -> SimpleNamespace:
    return SimpleNamespace(
        user_input="How did my week go?",
        trace_id="trace-123",
        state={
            "memory_structured": {"claims": [{"summary": "user had a hard week"}]},
            "memory_full_history_id": "hist-abc",
            "memory_retrieval_set": [{"summary_key": "m1", "summary": "slept poorly"}],
            "tool_results": [{"tool": "calendar", "value": "2 meetings"}],
        },
        metadata={
            "determinism": {
                "memory_fingerprint": "fp-1",
                "llm_provider": "ollama",
                "llm_model": "llama3",
                "seed_policy": "fixed",
                "temperature_policy": "zero",
            },
            "control_plane": {"session_id": "s-1"},
        },
    )


def test_trace_context_contains_snapshot_and_dag():
    recorder = ExecutionTraceRecorder(trace_id="trace-123", prompt="How did my week go?")
    recorder.record("kernel_turn_start", {"job_id": "j1"})
    recorder.record("iteration_start", {"iteration": 0})
    recorder.record("model_call", {"iteration": 0, "input_hash": "in-1", "message_count": 2})
    recorder.record("model_output", {"iteration": 0, "output_hash": "out-1", "output_length": 42})
    recorder.record("iteration_output", {"iteration": 0, "reply_preview": "first draft"})
    recorder.record("critique_iteration", {"iteration": 0, "passed": True, "issue_count": 0})

    context = _fake_context()
    trace = build_execution_trace_context(context=context, result=("final answer", True), recorder=recorder)

    assert trace["schema_version"] == "2.0"
    assert trace["execution_snapshot"]["inputs"]["trace_id"] == "trace-123"
    assert trace["execution_snapshot"]["memory_snapshot"]["memory_full_history_id"] == "hist-abc"
    assert len(trace["execution_snapshot"]["outputs_per_step"]) == 6

    dag = trace["execution_dag"]
    assert isinstance(dag.get("nodes"), list)
    assert isinstance(dag.get("edges"), list)
    assert dag.get("entry", "").startswith("step:0")
    assert any(edge.get("type") == "iteration" for edge in dag.get("edges", []))


def test_trace_context_hash_is_stable_for_same_input():
    recorder = ExecutionTraceRecorder(trace_id="trace-123", prompt="How did my week go?")
    recorder.record("kernel_turn_start", {"job_id": "j1"})
    recorder.record("iteration_start", {"iteration": 0})
    recorder.record("iteration_output", {"iteration": 0, "reply_preview": "ok"})

    context = _fake_context()
    t1 = build_execution_trace_context(context=context, result=("final", True), recorder=recorder)
    t2 = build_execution_trace_context(context=context, result=("final", True), recorder=recorder)

    assert t1["execution_snapshot"]["snapshot_hash"] == t2["execution_snapshot"]["snapshot_hash"]
    assert t1["execution_dag"]["dag_hash"] == t2["execution_dag"]["dag_hash"]
    assert t1["final_hash"] == t2["final_hash"]
