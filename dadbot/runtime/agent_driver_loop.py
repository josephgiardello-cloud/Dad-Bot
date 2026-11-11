from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol, cast

from dadbot.core.execution_contract import SovereignContext, TurnDelivery, TurnResponse, live_turn_request
from dadbot.runtime.structured_output import AgentPlan, SchemaValidationError, verify_agent_plan

if TYPE_CHECKING:
    from dadbot.runtime_core.bus import EventBus


class KernelTurnExecutor(Protocol):
    def execute_turn(self, request: Any, **kwargs: Any) -> Any: ...


ReflectionHook = Callable[[dict[str, Any]], dict[str, Any]]
ObservationHook = Callable[[dict[str, Any]], str]


@dataclass(slots=True)
class DriverLoopPolicy:
    max_turns: int = 8
    max_failures: int = 2
    max_consecutive_noop: int = 2
    allowed_tools: list[str] | None = None
    trigger_type: str = "interactive"


@dataclass(slots=True)
class LoopTurnRecord:
    turn_index: int
    observation: str
    action_input: str
    reflection: dict[str, Any]
    reply: str
    should_end: bool
    duration_ms: float
    commit_status: str
    error: str = ""


@dataclass(slots=True)
class DriverLoopResult:
    records: list[LoopTurnRecord] = field(default_factory=list)
    stop_reason: str = "max_turns"
    completed_turns: int = 0
    failures: int = 0
    consecutive_noop: int = 0
    goal_blackboard: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class _LoopTurnContext:
    turn_index: int
    reflection_context: dict[str, Any]
    session_id: str
    initial_observation: str
    last_reply: str
    last_action_input: str
    system_observation: str
    observation: str = ""
    action_input: str = ""
    reflection: dict[str, Any] = field(default_factory=dict)


class AgentDriverLoop:
    """Run bounded autonomous turns through the canonical turn contract."""

    def __init__(
        self,
        kernel: KernelTurnExecutor,
        *,
        policy: DriverLoopPolicy | None = None,
        event_bus: EventBus | None = None,
        allowed_tools: list[str] | None = None,
    ) -> None:
        self.kernel = kernel
        self.policy = policy or DriverLoopPolicy()
        self._bus = event_bus
        policy_tools = self.policy.allowed_tools
        selected_tools = policy_tools if allowed_tools is None else allowed_tools
        self._allowed_tools = None if selected_tools is None else list(selected_tools)
        self._trigger_type = str(self.policy.trigger_type or "interactive")

    @staticmethod
    def _coerce_turn_result(response_obj: Any) -> tuple[str, bool]:
        if hasattr(response_obj, "as_result"):
            reply, should_end = cast(TurnResponse, response_obj).as_result()
            return str(reply or ""), bool(should_end)
        if isinstance(response_obj, tuple) and len(response_obj) >= 2:
            return str(response_obj[0] or ""), bool(response_obj[1])
        return str(getattr(response_obj, "reply", "") or ""), bool(getattr(response_obj, "should_end", False))

    def _emit(self, event_type: str, *, thread_id: str, payload: dict[str, Any]) -> None:
        if self._bus is None:
            return
        try:
            from dadbot.runtime_core.models import new_event  # type: ignore[import]
            self._bus.emit(new_event(event_type, thread_id=thread_id, payload=payload))  # type: ignore[arg-type]
        except Exception:
            pass  # telemetry must never break the loop

    def _record_failure(
        self,
        result: DriverLoopResult,
        record: LoopTurnRecord,
    ) -> bool:
        result.failures += 1
        result.records.append(record)
        if result.failures > int(self.policy.max_failures):
            result.stop_reason = "failure_budget"
            return True
        return False

    def _record_skipped(
        self,
        result: DriverLoopResult,
        turn: _LoopTurnContext,
        *,
        count_noop: bool,
    ) -> bool:
        result.records.append(
            LoopTurnRecord(
                turn_index=turn.turn_index,
                observation=turn.observation,
                action_input="",
                reflection=turn.reflection,
                reply="",
                should_end=False,
                duration_ms=0.0,
                commit_status="skipped",
            ),
        )
        if not count_noop:
            return False
        result.consecutive_noop += 1
        if result.consecutive_noop >= int(self.policy.max_consecutive_noop):
            result.stop_reason = "noop_threshold"
            return True
        return False

    def _prepare_turn(
        self,
        result: DriverLoopResult,
        reflection_hook: ReflectionHook | None,
        turn: _LoopTurnContext,
    ) -> tuple[AgentPlan | None, str, bool]:
        try:
            reflection = dict(reflection_hook(turn.reflection_context) or {}) if callable(reflection_hook) else {}
        except Exception as exc:  # noqa: BLE001
            error = f"Reflection failed: {type(exc).__name__}: {exc}"
            should_stop = self._record_failure(
                result,
                LoopTurnRecord(
                    turn_index=turn.turn_index,
                    observation=turn.observation,
                    action_input="",
                    reflection={},
                    reply="",
                    should_end=False,
                    duration_ms=0.0,
                    commit_status="failed",
                    error=error,
                ),
            )
            return None, f"System observation: {error}", should_stop

        turn.reflection = reflection
        if reflection.get("should_continue") is False:
            result.stop_reason = "reflection_stop"
            return None, "", True

        action_candidate = reflection.get("action_input")
        if action_candidate is None:
            action_candidate = turn.observation
        turn.action_input = str(action_candidate or "").strip()
        system_observation = str(reflection.get("system_observation") or "").strip()
        if isinstance(action_candidate, str) and not turn.action_input:
            if not system_observation:
                should_stop = self._record_skipped(
                    result,
                    turn,
                    count_noop=True,
                )
                return None, "", should_stop
            self._record_skipped(
                result,
                turn,
                count_noop=False,
            )
            return None, system_observation, False

        try:
            plan = verify_agent_plan(
                {**reflection, "action_input": action_candidate},
                allowed_tools=self._allowed_tools,
                current_goal_blackboard=result.goal_blackboard,
            )
        except SchemaValidationError as exc:
            error = f"Plan verifier rejected the reflection: {exc}"
            should_stop = self._record_failure(
                result,
                LoopTurnRecord(
                    turn_index=turn.turn_index,
                    observation=turn.observation,
                    action_input=turn.action_input,
                    reflection=reflection,
                    reply="",
                    should_end=False,
                    duration_ms=0.0,
                    commit_status="rejected",
                    error=str(exc),
                ),
            )
            return None, error, should_stop

        turn.action_input = plan.action_input
        turn.reflection = {
            **reflection,
            "should_continue": plan.should_continue,
            "action_input": plan.action_input,
            "goal_updates": plan.goal_updates,
        }
        return plan, "", False

    def _dispatch_plan(
        self,
        result: DriverLoopResult,
        turn: _LoopTurnContext,
        last_reply: str,
        last_action_input: str,
        tenant_id: str,
    ) -> tuple[str, str, bool]:
        started = perf_counter()
        try:
            response_obj = self.kernel.execute_turn(
                live_turn_request(
                    turn.action_input,
                    delivery=TurnDelivery.SYNC,
                    context=SovereignContext(
                        session_id=turn.session_id,
                        tenant_id=tenant_id,
                    ),
                    metadata={
                        "agent_loop": {
                            "turn_index": turn.turn_index,
                            "goal_blackboard": dict(sorted(result.goal_blackboard.items())),
                            "plan_verification": "passed",
                            "trigger_type": self._trigger_type,
                        },
                    },
                ),
            )
            reply, should_end = self._coerce_turn_result(response_obj)
            elapsed_ms = (perf_counter() - started) * 1000.0
            result.records.append(
                LoopTurnRecord(
                    turn_index=turn.turn_index,
                    observation=turn.observation,
                    action_input=turn.action_input,
                    reflection=turn.reflection,
                    reply=reply,
                    should_end=should_end,
                    duration_ms=elapsed_ms,
                    commit_status="committed",
                ),
            )
            result.completed_turns += 1
            if reply.strip():
                result.consecutive_noop = 0
            else:
                result.consecutive_noop += 1
            action_type = str(turn.reflection.get("action_type") or "kernel_turn").strip() or "kernel_turn"
            self._emit(
                "loop_turn_completed",
                thread_id=turn.session_id,
                payload={
                    "turn_index": turn.turn_index,
                    "action_type": action_type,
                    "action_input": turn.action_input,
                    "commit_status": "committed",
                    "reply": reply,
                    "should_end": should_end,
                    "duration_ms": elapsed_ms,
                    "consecutive_noop": result.consecutive_noop,
                },
            )
            if should_end:
                result.stop_reason = "kernel_should_end"
                return reply, turn.action_input, True
            if result.consecutive_noop >= int(self.policy.max_consecutive_noop):
                result.stop_reason = "noop_threshold"
                return reply, turn.action_input, True
            return reply, turn.action_input, False
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (perf_counter() - started) * 1000.0
            should_stop = self._record_failure(
                result,
                LoopTurnRecord(
                    turn_index=turn.turn_index,
                    observation=turn.observation,
                    action_input=turn.action_input,
                    reflection=turn.reflection,
                    reply="",
                    should_end=False,
                    duration_ms=elapsed_ms,
                    commit_status="failed",
                    error=str(exc),
                ),
            )
            return last_reply, last_action_input, should_stop

    @staticmethod
    def _build_observation(
        turn: _LoopTurnContext,
        observation_hook: ObservationHook | None,
    ) -> str:
        default_observation = (
            str(turn.initial_observation or "")
            if turn.turn_index == 1
            else str(turn.last_reply or turn.last_action_input or "")
        )
        if turn.system_observation:
            default_observation = f"{turn.system_observation}\n\n{default_observation}".strip()
        if callable(observation_hook):
            return str(observation_hook(turn.reflection_context) or "")
        return default_observation

    def run(
        self,
        initial_observation: str,
        *,
        session_id: str = "default",
        tenant_id: str = "default",
        reflection_hook: ReflectionHook | None = None,
        observation_hook: ObservationHook | None = None,
    ) -> DriverLoopResult:
        result = DriverLoopResult()
        last_reply = ""
        last_action_input = str(initial_observation or "")
        pending_system_observation = ""

        self._emit(
            "loop_started",
            thread_id=str(session_id or "default"),
            payload={"initial_observation": str(initial_observation or ""), "policy": {
                "max_turns": self.policy.max_turns,
                "max_failures": self.policy.max_failures,
                "max_consecutive_noop": self.policy.max_consecutive_noop,
            }, "trigger_type": self._trigger_type},
        )

        for turn_index in range(1, int(self.policy.max_turns) + 1):
            normalized_session_id = str(session_id or "default")
            ctx_payload = {
                "turn_index": turn_index,
                "last_reply": last_reply,
                "initial_observation": str(initial_observation or ""),
                "records": list(result.records),
                "system_observation": pending_system_observation,
                "goal_blackboard": dict(result.goal_blackboard),
            }
            turn = _LoopTurnContext(
                turn_index=turn_index,
                reflection_context=ctx_payload,
                session_id=normalized_session_id,
                initial_observation=str(initial_observation or ""),
                last_reply=last_reply,
                last_action_input=last_action_input,
                system_observation=pending_system_observation,
            )
            turn.observation = self._build_observation(turn, observation_hook)
            pending_system_observation = ""
            ctx_payload["observation"] = turn.observation

            plan, retry_observation, should_stop = self._prepare_turn(result, reflection_hook, turn)
            if should_stop:
                break
            if plan is None:
                pending_system_observation = retry_observation
                continue

            result.goal_blackboard.update(plan.goal_updates)

            last_reply, last_action_input, should_stop = self._dispatch_plan(
                result,
                turn,
                last_reply=last_reply,
                last_action_input=last_action_input,
                tenant_id=str(tenant_id or "default"),
            )
            if should_stop:
                break
        else:
            result.stop_reason = "max_turns"

        self._emit(
            "loop_stopped",
            thread_id=str(session_id or "default"),
            payload={
                "stop_reason": result.stop_reason,
                "completed_turns": result.completed_turns,
                "failures": result.failures,
            },
        )
        return result


__all__ = [
    "AgentDriverLoop",
    "DriverLoopPolicy",
    "DriverLoopResult",
    "LoopTurnRecord",
]
