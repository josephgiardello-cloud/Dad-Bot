from __future__ import annotations

from copy import deepcopy

import pytest

from dadbot.core.kernel_locks import KernelReplaySequenceLock
from dadbot.core.kernel_mutation_gate import apply_event, emit_event


def test_no_direct_state_mutation() -> None:
    state = {"count": 1}
    with pytest.raises(RuntimeError, match="MutationEvent"):
        apply_event(  # type: ignore[arg-type]
            {"event_type": "MUTATION_EVENT"},
            state,
            lambda s, _evt: s,
        )


def test_all_writes_emit_events() -> None:
    state = {"count": 1}
    event = emit_event(
        "MUTATION_EVENT",
        {"key": "count", "value": 2},
        source="test",
    )
    updated = apply_event(
        event,
        state,
        lambda s, evt: {**s, str(evt.payload["key"]): evt.payload["value"]},
    )
    assert updated["count"] == 2


def test_mutation_gate_is_required() -> None:
    state = {"v": 0}
    with pytest.raises(RuntimeError, match="MutationEvent"):
        apply_event("not-an-event", state, lambda s, _evt: s)  # type: ignore[arg-type]


def test_no_shared_mutable_references() -> None:
    state = {"nested": {"x": 1}}
    event = emit_event("MUTATION_EVENT", {"key": "x", "value": 9}, source="test")

    updated = apply_event(
        event,
        state,
        lambda s, evt: {
            **s,
            "nested": {**dict(s.get("nested") or {}), str(evt.payload["key"]): evt.payload["value"]},
        },
    )

    assert updated["nested"]["x"] == 9
    assert state["nested"]["x"] == 1


def test_state_is_copy_on_write() -> None:
    state = {"items": [1, 2, 3]}
    event = emit_event("MUTATION_EVENT", {"append": 4}, source="test")

    updated = apply_event(
        event,
        state,
        lambda s, evt: {**s, "items": [*list(s.get("items") or []), int(evt.payload["append"])]},
    )

    assert updated["items"] == [1, 2, 3, 4]
    assert state["items"] == [1, 2, 3]


def test_replay_is_source_of_truth() -> None:
    events = [
        {"sequence": 1, "event_type": "TURN_START", "payload": {"v": 1}},
        {"sequence": 2, "event_type": "TURN_END", "payload": {"v": 2}},
    ]
    digest_a, canonical_a = KernelReplaySequenceLock.strict_hash(
        trace_id="tr-1",
        events=deepcopy(events),
    )
    digest_b, canonical_b = KernelReplaySequenceLock.strict_hash(
        trace_id="tr-1",
        events=deepcopy(events),
    )
    assert digest_a == digest_b
    assert canonical_a == canonical_b


def test_replay_matches_live_execution() -> None:
    live_events = [
        {"sequence": 1, "event_type": "NODE_ENTER", "payload": {"stage": "inference"}},
        {"sequence": 2, "event_type": "NODE_EXIT", "payload": {"stage": "inference"}},
    ]
    replay_events = deepcopy(live_events)
    live_digest, _ = KernelReplaySequenceLock.strict_hash(trace_id="tr-live", events=live_events)
    replay_digest, _ = KernelReplaySequenceLock.strict_hash(trace_id="tr-live", events=replay_events)
    assert live_digest == replay_digest


def test_observability_cannot_mutate_kernel() -> None:
    kernel_state = {"counter": 1}
    event = emit_event("MUTATION_EVENT", {"counter": 99}, source="observability")

    def observer_attempt(state: dict[str, int], _evt):
        state["counter"] = 99
        return state

    _updated = apply_event(event, kernel_state, observer_attempt)
    assert kernel_state["counter"] == 1
