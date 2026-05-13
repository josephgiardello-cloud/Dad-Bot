from __future__ import annotations

import asyncio
import hashlib
import inspect
import time
from contextlib import ExitStack
from typing import Any

import dadbot.core.tool_executor as tool_executor_module
import dadbot.services.turn_service as turn_service_module

from tests.eval.tracer import Trace, TraceCollector


def _extract_planner_snapshot(runtime: Any) -> dict[str, str]:
    bot = getattr(runtime, "bot", None)
    if bot is None:
        return {"planner_status": "", "planner_tool": "", "planner_reason": ""}
    debug_snapshot = getattr(bot, "planner_debug_snapshot", None)
    if not callable(debug_snapshot):
        return {"planner_status": "", "planner_tool": "", "planner_reason": ""}
    try:
        snapshot = debug_snapshot() or {}
    except Exception:
        return {"planner_status": "", "planner_tool": "", "planner_reason": ""}
    if not isinstance(snapshot, dict):
        return {"planner_status": "", "planner_tool": "", "planner_reason": ""}
    return {
        "planner_status": str(snapshot.get("planner_status") or ""),
        "planner_tool": str(snapshot.get("planner_tool") or ""),
        "planner_reason": str(snapshot.get("planner_reason") or ""),
    }


def _derive_decision_outcome(*, runtime: Any, tool_call_count: int) -> tuple[bool, str, str, str, str | None]:
    planner_snapshot = _extract_planner_snapshot(runtime)
    planner_status = str(planner_snapshot.get("planner_status") or "").strip()
    planner_tool = str(planner_snapshot.get("planner_tool") or "").strip()
    planner_reason = str(planner_snapshot.get("planner_reason") or "").strip()

    if int(tool_call_count) > 0:
        return False, "executed_tool", planner_status, planner_tool, None

    normalized_status = planner_status.lower()
    if normalized_status == "low_confidence_rejected":
        reason = planner_reason or "low_confidence_rejected"
        return True, "robustness_suppressed", planner_status, planner_tool, reason

    return False, "no_tool_needed", planner_status, planner_tool, None


def _extract_final_output(result: Any) -> str:
    if isinstance(result, tuple):
        if len(result) >= 1:
            return str(result[0] or "")
    if isinstance(result, dict):
        for key in ("response_text", "output", "response", "final_output"):
            if key in result:
                return str(result.get(key) or "")
    return str(result or "")


def _extract_tool_name(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    if "tool_name" in kwargs:
        return str(kwargs.get("tool_name") or "")
    if args:
        return str(args[0] or "")
    return ""


def _extract_tool_input(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if "parameters" in kwargs:
        return kwargs.get("parameters")
    if "payload" in kwargs:
        return kwargs.get("payload")
    if len(args) >= 2:
        return args[1]
    return {}


def _discover_registry(runtime: Any) -> Any | None:
    for attr in ("registry", "service_registry"):
        candidate = getattr(runtime, attr, None)
        if candidate is not None:
            return candidate
    return None


def _safe_registry_get(registry: Any, name: str) -> Any | None:
    get_fn = getattr(registry, "get", None)
    if not callable(get_fn):
        return None
    try:
        return get_fn(name, optional=True)
    except Exception:
        try:
            return get_fn(name)
        except Exception:
            return None


def _patch_tool_method(
    stack: ExitStack,
    *,
    obj: Any,
    method_name: str,
    collector: TraceCollector,
) -> None:
    original = getattr(obj, method_name, None)
    if not callable(original):
        return

    def wrapped(*args: Any, **kwargs: Any):
        name = _extract_tool_name(args, kwargs)
        tool_input = _extract_tool_input(args, kwargs)
        try:
            out = original(*args, **kwargs)
        except Exception as exc:
            collector.record_tool_call(name=name, tool_input=tool_input, tool_output={"error": str(exc)})
            collector.bump_step()
            raise
        collector.record_tool_call(name=name, tool_input=tool_input, tool_output=out)
        collector.bump_step()
        return out

    setattr(obj, method_name, wrapped)
    stack.callback(lambda: setattr(obj, method_name, original))


def _patch_registry_tools(
    collector: TraceCollector,
    runtime: Any,
    tool_registry: Any | None,
) -> ExitStack:
    stack = ExitStack()

    # Explicit test-supplied registry first.
    if tool_registry is not None:
        _patch_tool_method(stack, obj=tool_registry, method_name="execute", collector=collector)

    # Runtime-level canonical execute_tool spine.
    _patch_tool_method(stack, obj=runtime, method_name="execute_tool", collector=collector)
    bot = getattr(runtime, "bot", None)
    if bot is not None:
        _patch_tool_method(stack, obj=bot, method_name="execute_tool", collector=collector)

    # Registry-level runtime service executes tools via dynamic runtime.
    registry = _discover_registry(runtime)
    if registry is not None:
        tool_runtime_service = _safe_registry_get(registry, "tool_runtime_service")
        if tool_runtime_service is not None:
            _patch_tool_method(
                stack,
                obj=tool_runtime_service,
                method_name="execute",
                collector=collector,
            )

    # TurnService imports execute_tool directly, so patch both module symbols.
    original_execute_tool = tool_executor_module.execute_tool

    def wrapped_execute_tool(*args: Any, **kwargs: Any):
        name = _extract_tool_name(args, kwargs)
        tool_input = _extract_tool_input(args, kwargs)
        try:
            out = original_execute_tool(*args, **kwargs)
        except Exception as exc:
            collector.record_tool_call(name=name, tool_input=tool_input, tool_output={"error": str(exc)})
            collector.bump_step()
            raise
        collector.record_tool_call(name=name, tool_input=tool_input, tool_output=out)
        collector.bump_step()
        return out

    tool_executor_module.execute_tool = wrapped_execute_tool
    turn_service_module.execute_tool = wrapped_execute_tool

    def _restore_execute_tool() -> None:
        tool_executor_module.execute_tool = original_execute_tool
        turn_service_module.execute_tool = original_execute_tool

    stack.callback(_restore_execute_tool)
    return stack


def _invoke_turn(handle_turn: Any, input_text: str, *, session_id: str | None = None):
    kwargs: dict[str, Any] = {}
    sig = None
    try:
        sig = inspect.signature(handle_turn)
    except Exception:
        sig = None

    if sig is not None:
        params = sig.parameters
        if "session_id" in params:
            kwargs["session_id"] = str(session_id or "eval-suite")
        if "confluence_key" in params:
            digest = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
            kwargs["confluence_key"] = f"eval:{digest}"
        if "timeout_seconds" in params:
            kwargs["timeout_seconds"] = 30.0

    try:
        return handle_turn(input_text, **kwargs)
    except TypeError:
        return handle_turn(user_input=input_text, **kwargs)


def _extract_execution_truth_contract(runtime: Any) -> dict | None:
    """Read execution_truth_contract written by TurnService into the last turn context state."""
    last_ctx = getattr(runtime, "_last_turn_context", None)
    if last_ctx is None:
        return None
    state = getattr(last_ctx, "state", None)
    if not isinstance(state, dict):
        return None
    contract = state.get("execution_truth_contract")
    if not isinstance(contract, dict):
        return None
    return dict(contract)


async def arun_with_trace(input: str, runtime: Any, tool_registry: Any | None = None) -> Trace:
    collector = TraceCollector(input_text=input)
    handle_turn = getattr(runtime, "handle_turn", None) or getattr(runtime, "run", None)
    if not callable(handle_turn):
        raise ValueError("runtime must expose handle_turn(input) or run(input)")

    start = time.perf_counter()
    error: str | None = None
    final_output = ""

    stack = _patch_registry_tools(collector, runtime, tool_registry)
    with stack:
        try:
            session_digest = hashlib.sha256(f"{input}:{time.perf_counter_ns()}".encode("utf-8")).hexdigest()[:12]
            result = _invoke_turn(handle_turn, input, session_id=f"eval-suite-{session_digest}")
            if inspect.isawaitable(result):
                result = await result
            final_output = _extract_final_output(result)
            collector.bump_step()
        except Exception as exc:
            error = str(exc)

    execution_truth_contract = _extract_execution_truth_contract(runtime)
    if not collector.tool_calls and isinstance(execution_truth_contract, dict):
        if str(execution_truth_contract.get("decision_outcome") or "").strip().lower() == "executed_tool":
            fallback_tool = str(execution_truth_contract.get("planner_tool") or "").strip()
            if fallback_tool:
                collector.record_tool_call(
                    name=fallback_tool,
                    tool_input={"source": "execution_truth_contract"},
                    tool_output={"status": "observed_via_contract"},
                )
                collector.bump_step()

    latency_ms = int((time.perf_counter() - start) * 1000)
    robustness_suppressed, decision_outcome, planner_status, planner_tool, robustness_reason = _derive_decision_outcome(
        runtime=runtime,
        tool_call_count=len(collector.tool_calls),
    )
    return collector.finalize(
        final_output=final_output,
        error=error,
        latency_ms=latency_ms,
        robustness_suppressed=robustness_suppressed,
        decision_outcome=decision_outcome,
        planner_status=planner_status,
        planner_tool=planner_tool,
        robustness_reason=robustness_reason,
        execution_truth_contract=execution_truth_contract,
    )


def run_with_trace(input: str, runtime: Any, tool_registry: Any | None = None) -> Trace:
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            raise RuntimeError("run_with_trace called inside an active event loop; use arun_with_trace")
    except RuntimeError as exc:
        if "active event loop" in str(exc):
            raise

    return asyncio.run(arun_with_trace(input=input, runtime=runtime, tool_registry=tool_registry))
