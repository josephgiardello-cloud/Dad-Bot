from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from typing import Any

from dadbot.core.graph import NodeType, TurnContext
from dadbot.core.tool_ir import deterministic_tool_id

_MAX_DELEGATION_DEPTH: int = 2
_MAX_DELEGATION_SUBTASKS: int = 8
_ALLOWED_ROUTED_TOOLS = frozenset({"memory_lookup"})
_ALLOWED_MEMORY_INTENTS = frozenset({"goal_lookup", "session_memory_fetch"})


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
            try:
                priority = int(item.get("priority") or 100)
            except Exception:
                priority = -1
            if name not in _ALLOWED_ROUTED_TOOLS:
                rejected.append({"index": index, "reason": "unsupported_tool"})
                continue
            if not isinstance(args, dict):
                rejected.append({"index": index, "reason": "invalid_args"})
                continue
            if intent not in _ALLOWED_MEMORY_INTENTS:
                rejected.append({"index": index, "reason": "invalid_intent"})
                continue
            if not expected or priority < 0:
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
            try:
                output, status = _dispatch_builtin_tool(name, args, context), "ok"
            except Exception as exc:
                output, status = str(exc), "error"
            rec = {"sequence": seq, "tool_name": name, "status": status, "output": output, "latency": time.perf_counter() - start, "deterministic_id": det_id}
            executions.append(rec)
            results.append({"sequence": seq, "tool_name": name, "status": status, "output": output, "deterministic_id": det_id})
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
        enforce = getattr(self.mgr, "enforce_policies", None)
        validate = getattr(self.mgr, "validate", None)
        candidate = context.state.get("candidate")
        if callable(enforce):
            context.state["safe_result"] = enforce(context, candidate)
            return context
        if callable(validate):
            context.state["safe_result"] = validate(candidate)
            return context
        context.state["safe_result"] = candidate
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


def _dispatch_builtin_tool(name: str, args: dict[str, Any], context: TurnContext) -> Any:
    tool_name = str(name or "").strip().lower()
    payload = dict(args or {})
    if tool_name == "echo":
        return str(payload.get("message") or "")
    if tool_name == "current_time":
        return datetime.now().astimezone().isoformat(timespec="seconds")
    if tool_name == "memory_lookup":
        query = str(payload.get("query") or "").strip()
        scope = str(payload.get("scope") or "session").strip().lower()
        if scope == "goals":
            goals = list(context.state.get("session_goals") or [])
            ids = set(str(g) for g in list(payload.get("goal_ids") or []))
            if ids:
                goals = [g for g in goals if str(g.get("id") or "") in ids]
            return {"query": query, "scope": "goals", "goals": goals}
        return {"query": query, "scope": "session", "memories": list(context.state.get("memories") or [])}
    raise ValueError(f"Unsupported built-in tool: {tool_name}")


__all__ = ["ContextBuilderNode", "HealthNode", "InferenceNode", "MemoryNode", "ReflectionNode", "SafetyNode", "SaveNode", "TemporalNode", "ToolExecutorNode", "ToolRouterNode", "_MAX_DELEGATION_DEPTH", "_MAX_DELEGATION_SUBTASKS", "_dispatch_builtin_tool"]
