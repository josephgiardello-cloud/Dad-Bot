from __future__ import annotations

from typing import Any

from dadbot.contracts import FinalizedTurnResult
from dadbot.core.belief_state_engine import BeliefStateEngine
from dadbot.core.core_state import CoreState, InputEvent, project_views, transition
from dadbot.core.event_clock import now as _now
from dadbot.core.planning_utils import build_semantic_memory_candidates
from dadbot.core.execution_context import get_active_core_state


class PersistenceCoordinator:
    """Coordinates committed core state and derived semantic memory writes."""

    def __init__(self) -> None:
        self._belief_state_engine = BeliefStateEngine()

    def apply_turn_committed_core_state(
        self,
        *,
        session: dict[str, Any],
        job: Any,
        result: FinalizedTurnResult,
    ) -> None:
        session_state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(session_state, dict):
            return
        active_core_state = get_active_core_state()
        base_core_state = (
            active_core_state
            if active_core_state is not None
            else CoreState.from_dict(session_state.get("core_state"))
        )
        next_core_state = transition(
            base_core_state,
            InputEvent(
                event_type="turn_committed",
                payload={
                    "trace_id": str(job.trace_id or ""),
                    "response": str(result[0] if isinstance(result, tuple) and len(result) >= 1 else ""),
                    "should_end": bool(result[1] if isinstance(result, tuple) and len(result) >= 2 else False),
                    "memory_retrieval_set": base_core_state.memory.to_dict_list(),
                },
                metadata={
                    "job_id": str(job.job_id or ""),
                },
            ),
        )
        projections = project_views(next_core_state)
        session_state["core_state"] = next_core_state.to_dict()
        session_state["core_state_views"] = {
            "memory": {
                "entries": [dict(item.payload or {}) for item in projections.memory.entries],
            },
            "graph": {
                "adjacency": dict(projections.graph.adjacency),
            },
            "execution": {
                "trace_id": projections.execution.state.trace_id,
                "last_response": projections.execution.state.last_response,
                "should_end": bool(projections.execution.state.should_end),
            },
            "canonical": {
                "event_count": len(projections.canonical.events),
                "last_event_id": str(projections.canonical.events[-1].event_id if projections.canonical.events else ""),
            },
            "facade": projections.facade.as_payload(),
        }
        metadata = dict(job.metadata or {})
        metadata["core_state"] = dict(session_state["core_state"])
        job.metadata = metadata

    def promote_semantic_memory(
        self,
        *,
        session: dict[str, Any],
        job: Any,
        response_text: str,
    ) -> None:
        promoted = build_semantic_memory_candidates(
            user_input=str(job.user_input or ""),
            response=str(response_text or ""),
            trace_id=str(job.trace_id or ""),
            session_id=str(job.session_id or "default"),
        )
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if isinstance(state, dict) and promoted:
            projection = dict(state.get("semantic_memory_projection") or {})
            items = [
                dict(item)
                for item in list(projection.get("items") or [])
                if isinstance(item, dict)
            ]
            items.extend(promoted)
            projection["items"] = items[-128:]
            projection["last_updated"] = float(_now())
            projection["authority"] = "derived_read_only"
            projection["writer"] = "memory_manager"
            state["semantic_memory_projection"] = projection
        metadata = dict(job.metadata or {})
        metadata["semantic_memory_projection"] = {
            "authority": "derived_read_only",
            "writer": "memory_manager",
            "candidate_count": int(len(promoted)),
            "candidates": [dict(item) for item in list(promoted)[:8]],
        }
        job.metadata = metadata
