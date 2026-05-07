from __future__ import annotations

from dadbot.core.graph_context import TurnContext
from dadbot.core.ux_projection import TurnUxProjector


def test_ux_feedback_contains_evidence_graph_digest() -> None:
    projector = TurnUxProjector()
    context = TurnContext(user_input="continue")
    context.state["reflection_summary"] = {
        "evidence_graph": {
            "node_count": 4,
            "edge_count": 3,
            "edges": [
                {
                    "source": "event:topic:fishing",
                    "target": "episode:10-11",
                    "weight": 1.2,
                    "observations": 2,
                },
                {
                    "source": "episode:10-11",
                    "target": "outcome:unrecovered",
                    "weight": 0.8,
                    "observations": 1,
                },
            ],
        }
    }

    projector.project(
        context,
        total_latency_ms=120.0,
        failed=False,
        stage_duration_lookup=lambda _ctx, _stage: 0.0,
    )

    ux_feedback = dict(context.state.get("ux_feedback") or {})
    digest = dict(ux_feedback.get("evidence_graph_digest") or {})
    assert digest.get("node_count") == 4
    assert digest.get("edge_count") == 3
    assert isinstance(digest.get("top_edges"), list)
    assert len(digest.get("top_edges") or []) >= 1
    assert isinstance(digest.get("chain_preview"), list)
    assert "->" in str((digest.get("chain_preview") or [""])[0])


def test_ux_feedback_omits_digest_without_evidence_graph() -> None:
    projector = TurnUxProjector()
    context = TurnContext(user_input="continue")

    projector.project(
        context,
        total_latency_ms=120.0,
        failed=False,
        stage_duration_lookup=lambda _ctx, _stage: 0.0,
    )

    ux_feedback = dict(context.state.get("ux_feedback") or {})
    assert "evidence_graph_digest" not in ux_feedback
