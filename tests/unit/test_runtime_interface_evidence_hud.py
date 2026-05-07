from __future__ import annotations

from types import SimpleNamespace

import pytest

from dadbot.managers.runtime_interface import RuntimeInterfaceManager


pytestmark = pytest.mark.unit


def test_render_audit_hud_prints_evidence_chain_lines(capsys) -> None:
    manager = RuntimeInterfaceManager.__new__(RuntimeInterfaceManager)
    context = SimpleNamespace(
        state={
            "relational_state": {
                "topic_drift_detected": True,
                "topic_overlap_ratio": 0.2,
                "trust_credit": 0.3,
                "dominant_topic": "fishing",
            },
            "temporal_budget": {
                "turn_index": 7,
                "topic_drift_streak": 2,
                "budget_pressure": 0.6,
            },
            "reflection_summary": {
                "evidence_graph": {
                    "edges": [
                        {
                            "source": "event:topic:fishing",
                            "target": "episode:7-8",
                        },
                        {
                            "source": "episode:7-8",
                            "target": "outcome:unrecovered",
                        },
                    ]
                }
            },
            "ux_feedback": {
                "evidence_graph_digest": {
                    "chain_preview": [
                        "event:topic:fishing -> episode:7-8",
                        "episode:7-8 -> outcome:unrecovered",
                    ]
                }
            },
        },
        last_checkpoint_hash="abc123",
    )
    manager.bot = SimpleNamespace(NO_COLOR="1", _last_turn_context=context)

    manager._render_audit_hud()
    output = capsys.readouterr().out

    assert ("[EVIDENCE]" in output) or ("evidence:" in output.lower())
    assert "event:topic:fishing -> episode:7-8" in output
