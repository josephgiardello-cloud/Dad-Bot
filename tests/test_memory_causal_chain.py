from dataclasses import dataclass, field

from tests.scoring_engine import ScoringEngine
from tests.trace_schema import MemoryAccess, MemoryType, NormalizedTrace


@dataclass
class _Scenario:
    name: str = "contradictory_memory_handling"
    category: str = "memory"
    behavioral_spec: dict = field(default_factory=dict)


def _trace_with_memory(memory_causal_trace: dict) -> NormalizedTrace:
    accesses = [
        MemoryAccess(
            key="preference_cuisine",
            memory_type=MemoryType.SEMANTIC,
            retrieved=True,
            value_summary="user prefers vietnamese",
        )
    ]
    return NormalizedTrace(
        scenario_name="contradictory_memory_handling",
        category="memory",
        input_text="Use corrected preference.",
        final_response="",
        completed=True,
        total_duration_ms=14.0,
        memory_accesses=accesses,
        raw_state={
            "session_goals": [{"id": "g1"}],
            "memory_causal_trace": memory_causal_trace,
        },
        execution_mode="orchestrator",
    )


def test_memory_read_write_linkage_is_scored():
    engine = ScoringEngine()
    scenario = _Scenario()

    no_link = _trace_with_memory(
        {
            "trigger": "user correction",
            "read_link_id": "",
            "write_link_id": "",
            "influenced_final_response": False,
            "overridden": False,
        }
    )
    linked = _trace_with_memory(
        {
            "trigger": "user correction",
            "read_link_id": "mem-read-1",
            "write_link_id": "mem-write-1",
            "influenced_final_response": True,
            "overridden": False,
        }
    )

    no_link_score = engine.score(no_link, scenario)
    linked_score = engine.score(linked, scenario)

    assert no_link_score.memory is not None
    assert linked_score.memory is not None
    assert linked_score.memory.score > no_link_score.memory.score
