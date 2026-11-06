import pytest

from dadbot.core.dadbot import DadBot
from dadbot.core.execution_context import active_execution_trace, ensure_execution_trace_root
from dadbot.core.runtime_errors import ReplayInvariantViolation

pytestmark = pytest.mark.unit


def test_retired_alias_is_unavailable_in_replay_mode(bot) -> None:
    assert bot.__class__._DEPRECATED_FACADE_ALIAS_MAP == {}
    bot._replay_mode = True
    bot._alias_layer_enabled = False

    with pytest.raises(AttributeError):
        _ = bot.agentic_handler


def test_relationship_apply_feedback_records_boundary_trace() -> None:
    bot = DadBot()

    with ensure_execution_trace_root(operation="test.relationship.apply_feedback", required=True):
        payload = bot.relationship.apply_feedback("supportive")
        recorder = active_execution_trace()
        assert recorder is not None
        matching = [
            step
            for step in recorder.steps
            if str(step.get("operation") or "") == "relationship.apply_feedback"
        ]

    assert matching
    trace_payload = dict(matching[-1].get("payload") or {})
    assert str(trace_payload.get("before_state_hash") or "")
    assert str(trace_payload.get("after_state_hash") or "")
    assert dict(trace_payload.get("inputs") or {}).get("feedback_type") == "supportive"
    assert dict(trace_payload.get("outputs") or {}).get("feedback_applied") == "supportive"
    assert payload.get("feedback_applied") == "supportive"
