from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dadbot.runtime_core.runtime_types import PlanResult


logger = logging.getLogger(__name__)
_LEGACY_TUPLE_FLAG = "DADBOT_ENABLE_LEGACY_PLANNER_TUPLE"


def _allocate_agent_goals(
    *,
    runtime,
    event,
    user_text: str,
    attachments: list[dict],
    thread_state,
    intent_result,
):
    _ = (runtime, event, user_text, attachments, thread_state, intent_result)
    return []


def _multi_agent_plan_merge(
    *,
    runtime,
    event,
    user_text: str,
    attachments: list[dict],
    thread_state,
    intent_result,
):
    return runtime._multi_agent_orchestrator.plan_candidates(
        runtime=runtime,
        event=event,
        user_text=user_text,
        attachments=attachments,
        thread_state=thread_state,
        intent_result=intent_result,
    )


def _generate_candidates(
    *,
    runtime,
    enable_multi_agent_substrate: bool,
    event,
    user_text: str,
    attachments: list[dict],
    thread_state,
    intent_result,
):
    if enable_multi_agent_substrate:
        return _multi_agent_plan_merge(
            runtime=runtime,
            event=event,
            user_text=user_text,
            attachments=attachments,
            thread_state=thread_state,
            intent_result=intent_result,
        )
    return runtime.plan_candidates(
        event=event,
        user_text=user_text,
        attachments=attachments,
        thread_state=thread_state,
        intent_result=intent_result,
    ), {"enabled": False, "mode": "single_runtime"}


def _turn_phase_plan_result(
    *,
    turn_state: dict[str, Any],
    execution_result: dict[str, Any],
) -> PlanResult:
    # Defer import to avoid package-level runtime_core import cycles.
    from dadbot.runtime_core.runtime_types import PlanResult

    runtime = turn_state["runtime"]
    event = turn_state["event"]
    user_text = turn_state["user_text"]
    attachments = list(turn_state["attachments"])
    thread_state = turn_state["thread_state"]

    initial_result = execution_result["initial_result"]

    enable_multi_agent_substrate = bool(
        event.payload.get("enable_multi_agent_substrate", False),
    )
    candidates, multi_agent_substrate = _generate_candidates(
        runtime=runtime,
        enable_multi_agent_substrate=enable_multi_agent_substrate,
        event=event,
        user_text=user_text,
        attachments=attachments,
        thread_state=thread_state,
        intent_result=initial_result,
    )

    return PlanResult(
        candidates=list(candidates or []),
        metadata={
            "initial_result": initial_result,
            "enabled": bool(dict(multi_agent_substrate or {}).get("enabled", False)),
            "multi_agent_substrate": dict(multi_agent_substrate or {"enabled": False}),
        },
        evaluation_hint={
            "source": "legacy_planner_adapter",
            "requested": bool(enable_multi_agent_substrate),
        },
    )


def _tuple_rollback_mode_enabled() -> bool:
    flag = str(os.getenv(_LEGACY_TUPLE_FLAG, "0") or "0").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _legacy_tuple_mode_enabled() -> bool:
    """Backward-compatible alias for rollback mode checks."""
    return _tuple_rollback_mode_enabled()


def _turn_phase_plan(
    *,
    turn_state: dict[str, Any],
    execution_result: dict[str, Any],
) -> tuple:
    # LEGACY: tuple planner output maintained only for rollback compatibility.
    logger.warning(
        "LEGACY planner tuple output path invoked; this path is deprecated and gated by %s",
        _LEGACY_TUPLE_FLAG,
    )
    if not _tuple_rollback_mode_enabled():
        raise RuntimeError(
            "Legacy planner tuple output is disabled. Set DADBOT_ENABLE_LEGACY_PLANNER_TUPLE=1 to enable rollback mode.",
        )
    result = _turn_phase_plan_result(
        turn_state=turn_state,
        execution_result=execution_result,
    )
    return (
        dict(result.metadata or {}).get("initial_result"),
        list(result.candidates or []),
        dict(result.metadata or {}).get("multi_agent_substrate") or {"enabled": False},
    )


class Planner:
    def build(
        self,
        turn_state: dict[str, Any],
        execution_result: dict[str, Any],
    ) -> PlanResult:
        return _turn_phase_plan_result(
            turn_state=turn_state,
            execution_result=execution_result,
        )

    def build_legacy(
        self,
        turn_state: dict[str, Any],
        execution_result: dict[str, Any],
    ) -> tuple:
        """LEGACY tuple output entrypoint (deprecated, rollback only)."""
        return _turn_phase_plan(
            turn_state=turn_state,
            execution_result=execution_result,
        )
