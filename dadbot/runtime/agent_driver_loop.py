from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Protocol, cast

from dadbot.core.execution_contract import SovereignContext, TurnDelivery, TurnResponse, live_turn_request

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


class AgentDriverLoop:
    """Policy-bounded autonomous driver loop over the canonical turn contract."""

    def __init__(
        self,
        kernel: KernelTurnExecutor,
        *,
        policy: DriverLoopPolicy | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.kernel = kernel
        self.policy = policy or DriverLoopPolicy()
        self._bus = event_bus

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

    def run(
        self,
        initial_observation: str,
        *,
        session_id: str = "default",
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
            }},
        )

        for turn_index in range(1, int(self.policy.max_turns) + 1):
            ctx_payload = {
                "turn_index": turn_index,
                "last_reply": last_reply,
                "initial_observation": str(initial_observation or ""),
                "records": list(result.records),
                "system_observation": pending_system_observation,
            }
            default_observation = (
                str(initial_observation or "")
                if turn_index == 1
                else str(last_reply or last_action_input or "")
            )
            if pending_system_observation:
                default_observation = (
                    f"{pending_system_observation}\n\n{default_observation}".strip()
                )
                pending_system_observation = ""
            observation = (
                str(observation_hook(ctx_payload) or "")
                if callable(observation_hook)
                else default_observation
            )

            try:
                reflection = dict(reflection_hook(ctx_payload) or {}) if callable(reflection_hook) else {}
            except Exception as exc:
                reflection = {
                    "should_continue": True,
                    "action_input": "",
                    "system_observation": (
                        "System observation: Reflection planning failed "
                        f"({type(exc).__name__}: {exc}). Apologize briefly and retry with a valid tool choice."
                    ),
                }
            if reflection.get("should_continue") is False:
                result.stop_reason = "reflection_stop"
                break

            raw_action_input = reflection.get("action_input")
            action_input = (
                str(observation or "").strip()
                if raw_action_input is None
                else str(raw_action_input or "").strip()
            )
            system_observation = str(reflection.get("system_observation") or "").strip()
            if not action_input:
                if system_observation:
                    pending_system_observation = system_observation
                    result.records.append(
                        LoopTurnRecord(
                            turn_index=turn_index,
                            observation=observation,
                            action_input="",
                            reflection=reflection,
                            reply="",
                            should_end=False,
                            duration_ms=0.0,
                            commit_status="skipped",
                        ),
                    )
                    continue
                result.consecutive_noop += 1
                result.records.append(
                    LoopTurnRecord(
                        turn_index=turn_index,
                        observation=observation,
                        action_input="",
                        reflection=reflection,
                        reply="",
                        should_end=False,
                        duration_ms=0.0,
                        commit_status="skipped",
                    ),
                )
                if result.consecutive_noop >= int(self.policy.max_consecutive_noop):
                    result.stop_reason = "noop_threshold"
                    break
                continue

            started = perf_counter()
            try:
                response_obj = self.kernel.execute_turn(
                    live_turn_request(
                        action_input,
                        delivery=TurnDelivery.SYNC,
                        context=SovereignContext(session_id=str(session_id or "default")),
                    ),
                )
                reply, should_end = self._coerce_turn_result(response_obj)
                elapsed_ms = (perf_counter() - started) * 1000.0
                result.records.append(
                    LoopTurnRecord(
                        turn_index=turn_index,
                        observation=observation,
                        action_input=action_input,
                        reflection=reflection,
                        reply=reply,
                        should_end=should_end,
                        duration_ms=elapsed_ms,
                        commit_status="committed",
                    ),
                )
                result.completed_turns += 1
                last_reply = reply
                last_action_input = action_input
                if reply.strip():
                    result.consecutive_noop = 0
                else:
                    result.consecutive_noop += 1
                action_type = str(reflection.get("action_type") or "kernel_turn").strip() or "kernel_turn"
                self._emit(
                    "loop_turn_completed",
                    thread_id=str(session_id or "default"),
                    payload={
                        "turn_index": turn_index,
                        "action_type": action_type,
                        "action_input": action_input,
                        "commit_status": "committed",
                        "reply": reply,
                        "should_end": should_end,
                        "duration_ms": elapsed_ms,
                        "consecutive_noop": result.consecutive_noop,
                    },
                )
                if should_end:
                    result.stop_reason = "kernel_should_end"
                    break
                if result.consecutive_noop >= int(self.policy.max_consecutive_noop):
                    result.stop_reason = "noop_threshold"
                    break
            except Exception as exc:
                elapsed_ms = (perf_counter() - started) * 1000.0
                result.failures += 1
                result.records.append(
                    LoopTurnRecord(
                        turn_index=turn_index,
                        observation=observation,
                        action_input=action_input,
                        reflection=reflection,
                        reply="",
                        should_end=False,
                        duration_ms=elapsed_ms,
                        commit_status="failed",
                        error=str(exc),
                    ),
                )
                if result.failures > int(self.policy.max_failures):
                    result.stop_reason = "failure_budget"
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
