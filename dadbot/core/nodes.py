from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Any

from dadbot.core.graph import NodeType, TurnContext
from dadbot.core.invariant_gate import InvariantGate
from dadbot.core.policy_compiler import PolicyCompiler
from dadbot.core.tool_ir import ToolContractResult, ToolStatus, deterministic_tool_id

_MAX_DELEGATION_DEPTH: int = 2
_MAX_DELEGATION_SUBTASKS: int = 8
_ALLOWED_MEMORY_INTENTS = frozenset({"goal_lookup", "session_memory_fetch"})


@dataclass(frozen=True)
class ToolRegistration:
    handler: Any
    required_args: frozenset[str] = frozenset()
    allowed_intents: frozenset[str] | None = None
    require_expected_output: bool = True
    output_validator: Any | None = None
    retryable_exceptions: tuple[type[Exception], ...] = ()


_TOOL_REGISTRY: dict[str, ToolRegistration] = {}


def register_tool(
    name: str,
    *,
    handler: Any,
    required_args: set[str] | frozenset[str] | None = None,
    allowed_intents: set[str] | frozenset[str] | None = None,
    require_expected_output: bool = True,
    output_validator: Any | None = None,
    retryable_exceptions: tuple[type[Exception], ...] = (),
) -> None:
    tool_name = str(name or "").strip().lower()
    if not tool_name:
        raise ValueError("Tool name must be non-empty")
    if not callable(handler):
        raise ValueError(f"Tool handler for {tool_name!r} must be callable")
    _TOOL_REGISTRY[tool_name] = ToolRegistration(
        handler=handler,
        required_args=frozenset(required_args or ()),
        allowed_intents=frozenset(allowed_intents) if allowed_intents is not None else None,
        require_expected_output=bool(require_expected_output),
        output_validator=output_validator,
        retryable_exceptions=tuple(retryable_exceptions or ()),
    )


def get_registered_tool(name: str) -> ToolRegistration | None:
    return _TOOL_REGISTRY.get(str(name or "").strip().lower())


def get_registered_tool_names() -> frozenset[str]:
    return frozenset(_TOOL_REGISTRY.keys())


def get_tool_required_args() -> dict[str, frozenset[str]]:
    return {name: reg.required_args for name, reg in _TOOL_REGISTRY.items()}


class TemporalNode:
    name = "temporal"

    async def run(self, context: TurnContext) -> TurnContext:
        if getattr(context, "temporal", None) is None:
            raise RuntimeError("TemporalNode missing - deterministic execution violated")
        snap = context.temporal_snapshot() if callable(getattr(context, "temporal_snapshot", None)) else {}
        context.state.setdefault("temporal", snap)
        context.metadata.setdefault("temporal", snap)
        return context


class HealthNode:
    name = "health"

    def __init__(self, manager: Any = None) -> None:
        self.mgr = manager

    async def run(self, context: TurnContext) -> TurnContext:
        tick = getattr(self.mgr, "tick", None)
        if callable(tick):
            context.state["health"] = tick()
        return context


class ContextBuilderNode:
    name = "context_builder"

    def __init__(self, memory_manager: Any = None, *, goal_ranker: Any = None) -> None:
        self.mgr, self._goal_ranker = memory_manager, goal_ranker

    async def run(self, context: TurnContext) -> TurnContext:
        query = getattr(self.mgr, "query", None)
        memories = await query(context.user_input) if callable(query) else list(context.state.get("memories") or [])
        goals = list(context.state.get("session_goals") or [])
        if goals and self._goal_ranker is not None:
            rerank = getattr(self._goal_ranker, "rerank", None)
            if callable(rerank):
                try:
                    memories = rerank(memories, goals)
                except Exception:
                    context.state["goal_ranker_failure"] = {"failure_mode": "recoverable"}
        context.state["memories"] = list(memories or [])
        context.state["rich_context"] = {
            "memories": list(context.state.get("memories") or []),
            "session_goals": goals,
            "temporal": dict(context.state.get("temporal") or {}),
        }
        return context


MemoryNode = ContextBuilderNode


class ToolRouterNode:
    name = "tool_router"

    async def run(self, context: TurnContext) -> TurnContext:
        tool_ir = dict(context.state.get("tool_ir") or {})
        raw_requests = list(tool_ir.get("requests") or [])
        compiled: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_requests):
            item = dict(raw or {})
            name = str(item.get("tool_name") or "").strip().lower()
            args = item.get("args")
            intent = str(item.get("intent") or "").strip().lower()
            expected = str(item.get("expected_output") or "").strip()
            registration = get_registered_tool(name)
            try:
                priority = int(item.get("priority") or 100)
            except Exception:
                priority = -1
            if registration is None:
                rejected.append({"index": index, "reason": "unsupported_tool"})
                continue
            if not isinstance(args, dict):
                rejected.append({"index": index, "reason": "invalid_args"})
                continue
            if registration.allowed_intents is not None and intent not in registration.allowed_intents:
                rejected.append({"index": index, "reason": "invalid_intent"})
                continue
            if registration.require_expected_output and not expected:
                rejected.append({"index": index, "reason": "invalid_request"})
                continue
            if priority < 0:
                rejected.append({"index": index, "reason": "invalid_request"})
                continue
            req = {"tool_name": name, "args": dict(args), "intent": intent, "expected_output": expected, "priority": priority}
            req_id = deterministic_tool_id(req["tool_name"], req["args"])
            if req_id in seen:
                rejected.append({"index": index, "reason": "duplicate_request", "deterministic_id": req_id})
                continue
            seen.add(req_id)
            compiled.append(req)
        compiled.sort(key=lambda i: (int(i["priority"]), str(i["intent"]), deterministic_tool_id(i["tool_name"], i["args"])))
        tool_ir["execution_plan"] = [{"sequence": idx, "tool_name": r["tool_name"], "args": r["args"], "intent": r["intent"], "expected_output": r["expected_output"], "priority": r["priority"], "deterministic_id": deterministic_tool_id(r["tool_name"], r["args"])} for idx, r in enumerate(compiled)]
        tool_ir["compiler"] = {"strict": True, "compiled_count": len(compiled), "rejected_count": len(rejected), "rejected": rejected}
        context.state["tool_ir"] = tool_ir
        return context


class ToolExecutorNode:
    name = "tool_executor"
    _MAX_RETRY_ATTEMPTS: int = 1

    async def run(self, context: TurnContext) -> TurnContext:
        tool_ir = dict(context.state.get("tool_ir") or {})
        executions: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        for item in list(tool_ir.get("execution_plan") or []):
            seq = int(item.get("sequence") or 0)
            name = str(item.get("tool_name") or "").strip().lower()
            args = dict(item.get("args") or {})
            det_id = str(item.get("deterministic_id") or deterministic_tool_id(name, args))
            start = time.perf_counter()
            remediation_log: list[dict[str, Any]] = []
            raw: Any = None
            for attempt in range(self._MAX_RETRY_ATTEMPTS + 1):
                try:
                    raw = dispatch_registered_tool(name, args, context)
                except Exception as exc:
                    raw = ToolContractResult(
                        tool_name=name,
                        status=ToolStatus.FATAL,
                        data=None,
                        error_context={"exception": str(exc)},
                        repair_hint=f"Tool {name!r} raised an unexpected exception.",
                    )
                if not isinstance(raw, ToolContractResult):
                    break
                failure_class = {
                    ToolStatus.RETRY: "retryable_tool_failure",
                    ToolStatus.CONTRACT_VIOLATION: "execution_contract_violation",
                    ToolStatus.FATAL: "fatal_tool_failure",
                }.get(raw.status, "fatal_tool_failure")
                decision = InvariantGate.decide_remediation(
                    failure_class,
                    reason=str(raw.repair_hint or ""),
                    attempt=attempt,
                    max_attempts=self._MAX_RETRY_ATTEMPTS,
                    details={"tool_name": name, "status": raw.status.value},
                )
                remediation_log.append(
                    {
                        "action": decision.action.value,
                        "failure_class": decision.failure_class,
                        "attempt": decision.attempt,
                        "max_attempts": decision.max_attempts,
                        "reason": decision.reason,
                    },
                )
                if raw.status == ToolStatus.RETRY and decision.action.value == "retry":
                    continue
                break
            if isinstance(raw, ToolContractResult):
                status = raw.status.value
                output = raw
            else:
                status = "ok"
                output = raw
            rec = {
                "sequence": seq,
                "tool_name": name,
                "status": status,
                "output": output,
                "latency": time.perf_counter() - start,
                "deterministic_id": det_id,
                "remediation": remediation_log,
            }
            executions.append(rec)
            results.append({
                "sequence": seq,
                "tool_name": name,
                "status": status,
                "output": output,
                "deterministic_id": det_id,
                "remediation": remediation_log,
            })
        tool_ir["executions"] = executions
        context.state["tool_ir"] = tool_ir
        context.state["tool_results"] = results
        context.metadata["tool_execution_graph_hash"] = _stable_sha256({"executions": executions, "results": results})
        return context


class InferenceNode:
    name = "inference"

    def __init__(self, llm_manager: Any, *, critique_engine: Any = None, max_loop_iterations: int = 2) -> None:
        self.mgr, self._critique_engine, self._max_loop_iterations = llm_manager, critique_engine, max(1, int(max_loop_iterations))

    def _run_critique_check(self, context: TurnContext, candidate: Any, iteration: int) -> bool:
        if self._critique_engine is None or iteration >= self._max_loop_iterations - 1:
            return True
        plan = dict(context.state.get("turn_plan") or {})
        reply = candidate[0] if isinstance(candidate, tuple) else str(candidate or "")
        critique = self._critique_engine.critique(reply, context.user_input, plan, iteration)
        passed = bool(getattr(critique, "passed", False))
        hint = str(getattr(critique, "revision_hint", "") or "")
        context.state["critique_record"] = {"iteration": iteration, "score": getattr(critique, "score", 0.0), "passed": passed, "issues": list(getattr(critique, "issues", []) or []), "revision_hint": hint, "tool_necessity_score": getattr(critique, "tool_necessity_score", 0.0), "tool_correctness_score": getattr(critique, "tool_correctness_score", 0.0)}
        if passed:
            return True
        context.state["_critique_revision_context"] = hint
        return False

    async def run(self, context: TurnContext) -> TurnContext:
        run_agent = getattr(self.mgr, "run_agent", None)
        if not callable(run_agent):
            raise RuntimeError("InferenceNode requires manager.run_agent")
        rich_context = context.state.get("rich_context", {})
        candidate: Any = None
        for iteration in range(self._max_loop_iterations):
            candidate = await run_agent(context, rich_context)
            if self._run_critique_check(context, candidate, iteration):
                break
        context.state.pop("_critique_revision_context", None)
        context.state["candidate"] = candidate
        return context


class SafetyNode:
    name = "safety"

    def __init__(self, safety_manager: Any = None) -> None:
        self.mgr = safety_manager

    async def run(self, context: TurnContext) -> TurnContext:
        candidate = context.state.get("candidate")
        plan = PolicyCompiler.compile_safety(self.mgr)
        decision = PolicyCompiler.evaluate_safety(plan, context, candidate)
        context.state["safe_result"] = decision.output
        context.state["safety_policy_decision"] = {
            "action": decision.action,
            "step_name": decision.step_name,
            "details": dict(decision.details or {}),
            "trace": dict(decision.trace or {}),
        }
        policy_events = list(context.state.get("policy_trace_events") or [])
        policy_events.append(
            {
                "event_type": "policy_decision",
                "policy": "safety",
                "node": self.name,
                "sequence": len(policy_events) + 1,
                "trace": dict(decision.trace or {}),
            },
        )
        context.state["policy_trace_events"] = policy_events
        if decision.action == "passthrough":
            context.state["safety_passthrough"] = {
                "reason": "no_safety_manager",
                "failure_mode": "passthrough",
            }
        return context


class SaveNode:
    name = "save"
    node_type = NodeType.COMMIT

    def __init__(self, persistence_manager: Any = None) -> None:
        self.mgr = persistence_manager

    async def run(self, context: TurnContext) -> TurnContext:
        if getattr(context, "temporal", None) is None:
            raise RuntimeError("SaveNode requires temporal context")
        if self.mgr is None:
            context.fidelity.save = True
            return context
        begin = getattr(self.mgr, "begin_transaction", None)
        apply = getattr(self.mgr, "apply_mutations", None)
        finalize = getattr(self.mgr, "finalize_turn", None)
        commit = getattr(self.mgr, "commit_transaction", None)
        rollback = getattr(self.mgr, "rollback_transaction", None)
        try:
            if callable(begin):
                begin(context)
            if callable(apply):
                apply(context)
            if callable(finalize):
                context.state["safe_result"] = finalize(context, context.state.get("safe_result"))
            if callable(commit):
                commit(context)
            else:
                save_turn = getattr(self.mgr, "save_turn", None)
                if callable(save_turn):
                    save_turn(context, context.state.get("safe_result"))
            context.fidelity.save = True
            return context
        except Exception:
            if callable(rollback):
                rollback(context)
            raise


class ReflectionNode:
    name = "reflection"

    def __init__(self, reflection_manager: Any = None) -> None:
        self.mgr = reflection_manager

    async def run(self, context: TurnContext) -> TurnContext:
        reflect = getattr(self.mgr, "reflect", None)
        if callable(reflect):
            try:
                context.state["reflection"] = reflect(context)
            except Exception:
                context.state["reflection_error"] = True
        return context


def _stable_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")).hexdigest()


def _validate_tool_output(tool_name: str, registration: ToolRegistration, output: Any) -> ToolContractResult | Any:
    validator = registration.output_validator
    if not callable(validator):
        return output
    try:
        is_valid = bool(validator(output))
    except Exception as exc:
        return ToolContractResult(
            tool_name=tool_name,
            status=ToolStatus.FATAL,
            data=None,
            error_context={"validator_exception": str(exc)},
            repair_hint=f"Output validator for tool {tool_name!r} raised an exception.",
        )
    if is_valid:
        return output
    return ToolContractResult(
        tool_name=tool_name,
        status=ToolStatus.CONTRACT_VIOLATION,
        data=None,
        error_context={"output_validation": "failed"},
        repair_hint=f"Tool {tool_name!r} returned output that did not match expected shape.",
    )


def dispatch_registered_tool(name: str, args: dict[str, Any], context: TurnContext) -> Any:
    tool_name = str(name or "").strip().lower()
    payload = dict(args or {})
    registration = get_registered_tool(tool_name)
    if registration is None:
        raise ValueError(f"Unsupported built-in tool: {tool_name}")

    missing = sorted(registration.required_args - set(payload.keys()))
    if missing:
        return ToolContractResult(
            tool_name=tool_name,
            status=ToolStatus.CONTRACT_VIOLATION,
            data=None,
            error_context={"missing_args": missing, "received_args": sorted(payload.keys())},
            repair_hint=(
                f"Tool '{tool_name}' requires args {sorted(registration.required_args)}; "
                f"received {sorted(payload.keys())}. Missing: {missing}."
            ),
        )

    try:
        raw = registration.handler(payload, context)
    except registration.retryable_exceptions as exc:
        return ToolContractResult(
            tool_name=tool_name,
            status=ToolStatus.RETRY,
            data=None,
            error_context={"exception": str(exc)},
            repair_hint=f"Tool {tool_name!r} hit a retryable failure; retry with same args.",
        )
    except Exception as exc:
        return ToolContractResult(
            tool_name=tool_name,
            status=ToolStatus.FATAL,
            data=None,
            error_context={"exception": str(exc)},
            repair_hint=f"Tool {tool_name!r} raised an unexpected exception.",
        )

    if isinstance(raw, ToolContractResult):
        return raw
    return _validate_tool_output(tool_name, registration, raw)


def _builtin_echo(args: dict[str, Any], context: TurnContext) -> str:
    _ = context
    return str(args.get("message") or "")


def _builtin_current_time(args: dict[str, Any], context: TurnContext) -> str:
    _ = args, context
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _builtin_memory_lookup(args: dict[str, Any], context: TurnContext) -> ToolContractResult:
    query = str(args.get("query") or "").strip()
    if not query:
        return ToolContractResult(
            tool_name="memory_lookup",
            status=ToolStatus.CONTRACT_VIOLATION,
            data=None,
            error_context={"missing_args": ["query"], "received_args": list(args.keys())},
            repair_hint="memory_lookup requires a non-empty 'query' argument.",
        )
    scope = str(args.get("scope") or "session").strip().lower()
    if scope == "goals":
        goals = list(context.state.get("session_goals") or [])
        ids = set(str(g) for g in list(args.get("goal_ids") or []))
        if ids:
            goals = [g for g in goals if str(g.get("id") or "") in ids]
        data = {"query": query, "scope": "goals", "goals": goals}
    else:
        data = {"query": query, "scope": "session", "memories": list(context.state.get("memories") or [])}
    return ToolContractResult(
        tool_name="memory_lookup",
        status=ToolStatus.SUCCESS,
        data=data,
        error_context={},
    )


register_tool(
    "memory_lookup",
    handler=_builtin_memory_lookup,
    required_args={"query"},
    allowed_intents=set(_ALLOWED_MEMORY_INTENTS),
)
register_tool(
    "echo",
    handler=_builtin_echo,
    required_args={"message"},
    allowed_intents=None,
)
register_tool(
    "current_time",
    handler=_builtin_current_time,
    required_args=set(),
    allowed_intents=None,
)


__all__ = [
    "ContextBuilderNode",
    "HealthNode",
    "InferenceNode",
    "MemoryNode",
    "ReflectionNode",
    "SafetyNode",
    "SaveNode",
    "TemporalNode",
    "ToolExecutorNode",
    "ToolRouterNode",
    "ToolRegistration",
    "dispatch_registered_tool",
    "get_registered_tool",
    "get_registered_tool_names",
    "get_tool_required_args",
    "register_tool",
    "_MAX_DELEGATION_DEPTH",
    "_MAX_DELEGATION_SUBTASKS",
]
