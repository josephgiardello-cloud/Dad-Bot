from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from dadbot.core.graph_context import TurnContext
from dadbot.core.graph_types import NodeType
from dadbot.core.invariant_gate import InvariantGate
from dadbot.core.policy_compiler import PolicyCompiler
from dadbot.core.runtime_errors import InvariantViolation
from dadbot.core.tool_ir import ToolContractResult, ToolStatus, deterministic_tool_id

_MAX_DELEGATION_DEPTH: int = 2
_MAX_DELEGATION_SUBTASKS: int = 8
_ALLOWED_MEMORY_INTENTS = frozenset({"goal_lookup", "session_memory_fetch"})
_ALLOWED_UTILITY_INTENTS = frozenset({"time_lookup", "utility_echo"})


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


def _deterministic_subtask_trace_id(
    *,
    root_trace_id: str,
    parent_trace_id: str,
    depth: int,
    branch_index: int,
    subtask: dict[str, Any],
) -> str:
    payload = {
        "root_trace_id": str(root_trace_id or ""),
        "parent_trace_id": str(parent_trace_id or ""),
        "depth": int(depth),
        "branch_index": int(branch_index),
        "agent": str(subtask.get("agent") or ""),
        "input": str(subtask.get("input") or ""),
    }
    return _stable_sha256(payload)[:32]


class TemporalNode:
    name = "temporal"

    async def run(self, context: TurnContext) -> TurnContext:
        # LEDGER_EXEMPT: TemporalNode predates ledger protocol; temporal snapshot written to context.state["temporal"]
        # TRACE_EXEMPT: No side-effects, pure data transform (temporal snapshot only); no downstream calls, no ledger emission.
        if getattr(context, "temporal", None) is None:
            raise InvariantViolation("TemporalNode missing - deterministic execution violated")
        vc = getattr(context, "virtual_clock", None)
        if vc is not None:
            from dadbot.core.graph_temporal import TurnTemporalAxis
            epoch = vc.tick()
            dt = vc.to_datetime()
            offset = dt.utcoffset()
            offset_minutes = int(offset.total_seconds() // 60) if offset is not None else 0
            new_temporal = TurnTemporalAxis(
                turn_started_at=dt.isoformat(timespec="seconds"),
                wall_time=dt.isoformat(timespec="seconds"),
                wall_date=dt.date().isoformat(),
                timezone=str(dt.tzname() or "local").strip() or "local",
                utc_offset_minutes=offset_minutes,
                epoch_seconds=epoch,
            )
            context.temporal = new_temporal
        snap = context.temporal_snapshot() if callable(getattr(context, "temporal_snapshot", None)) else {}
        context.state.setdefault("temporal", snap)
        context.metadata.setdefault("temporal", snap)
        return context


class HealthNode:
    name = "health"

    def __init__(self, manager: Any = None) -> None:
        self.mgr = manager

    async def run(self, context: TurnContext) -> TurnContext:
        # LEDGER_EXEMPT: HealthNode predates ledger protocol; health tick written to context.state["health"]
        # TRACE_EXEMPT: Structural observer; no ledger emission, no graph mutation, no replay influence.
        tick = getattr(self.mgr, "tick", None)
        if callable(tick):
            context.state["health"] = tick()
        return context


class ContextBuilderNode:
    name = "context_builder"

    def __init__(self, memory_manager: Any = None, *, goal_ranker: Any = None) -> None:
        self.mgr, self._goal_ranker = memory_manager, goal_ranker

    @staticmethod
    def _normalize_memories_payload(payload: Any) -> list[Any]:
        if payload is None:
            return []
        if isinstance(payload, list):
            return list(payload)
        if isinstance(payload, tuple | set):
            return list(payload)
        if isinstance(payload, dict):
            return [dict(payload)]
        return []

    async def run(self, context: TurnContext) -> TurnContext:
        # LEDGER_EXEMPT: ContextBuilderNode predates ledger protocol; memory context written to context.state["rich_context"]
        # TRACE_EXEMPT: Data enrichment phase (memory query, goal ranking); no ledger calls, no downstream node invocation.
        query = getattr(self.mgr, "query", None)
        if callable(query):
            queried = query(context.user_input)
            memories_raw = await queried if inspect.isawaitable(queried) else queried
            memories = self._normalize_memories_payload(memories_raw)
        else:
            memories = self._normalize_memories_payload(context.state.get("memories"))
        goals = list(context.state.get("session_goals") or [])
        if goals and self._goal_ranker is not None:
            rerank = getattr(self._goal_ranker, "rerank", None)
            if callable(rerank):
                try:
                    memories = rerank(memories, goals)
                except Exception:
                    context.state["goal_ranker_failure"] = {"failure_mode": "recoverable"}
        context.state["memories"] = self._normalize_memories_payload(memories)
        context.state["rich_context"] = {
            "memories": list(context.state.get("memories") or []),
            "session_goals": goals,
            "temporal": dict(context.state.get("temporal") or {}),
        }
        return context


MemoryNode = ContextBuilderNode


class ToolRouterNode:
    name = "tool_router"

    @staticmethod
    def _request_priority(item: dict[str, Any]) -> int:
        try:
            return int(item.get("priority") or 100)
        except Exception:
            return -1

    def _compile_request(
        self,
        *,
        index: int,
        raw: Any,
        seen: set[str],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        item = dict(raw or {})
        name = str(item.get("tool_name") or "").strip().lower()
        args = item.get("args")
        intent = str(item.get("intent") or "").strip().lower()
        expected = str(item.get("expected_output") or "").strip()
        registration = get_registered_tool(name)
        priority = self._request_priority(item)

        if registration is None:
            return None, {"index": index, "reason": "unsupported_tool"}
        if not isinstance(args, dict):
            return None, {"index": index, "reason": "invalid_args"}
        if registration.allowed_intents is not None and intent not in registration.allowed_intents:
            return None, {"index": index, "reason": "invalid_intent"}
        if registration.require_expected_output and not expected:
            return None, {"index": index, "reason": "invalid_request"}
        if priority < 0:
            return None, {"index": index, "reason": "invalid_request"}

        req = {
            "tool_name": name,
            "args": dict(args),
            "intent": intent,
            "expected_output": expected,
            "priority": priority,
        }
        req_id = deterministic_tool_id(req["tool_name"], req["args"])
        if req_id in seen:
            return None, {"index": index, "reason": "duplicate_request", "deterministic_id": req_id}
        seen.add(req_id)
        return req, None

    @staticmethod
    def _execution_plan(compiled: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ordered = sorted(
            compiled,
            key=lambda i: (
                int(i["priority"]),
                str(i["intent"]),
                deterministic_tool_id(i["tool_name"], i["args"]),
            ),
        )
        return [
            {
                "sequence": idx,
                "tool_name": r["tool_name"],
                "args": r["args"],
                "intent": r["intent"],
                "expected_output": r["expected_output"],
                "priority": r["priority"],
                "deterministic_id": deterministic_tool_id(r["tool_name"], r["args"]),
            }
            for idx, r in enumerate(ordered)
        ]

    async def run(self, context: TurnContext) -> TurnContext:
        # LEDGER_EXEMPT: ToolRouterNode predates ledger protocol; compiled plan written to context.state["tool_ir"]
        # TRACE_EXEMPT: Tool plan compilation; pure deterministic transformation, no side-effects, no downstream execution.
        tool_ir = dict(context.state.get("tool_ir") or {})
        raw_requests = list(tool_ir.get("requests") or [])
        compiled: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_requests):
            req, reject = self._compile_request(index=index, raw=raw, seen=seen)
            if reject is not None:
                rejected.append(reject)
                continue
            if req is not None:
                compiled.append(req)
        tool_ir["execution_plan"] = self._execution_plan(compiled)
        tool_ir["compiler"] = {"strict": True, "compiled_count": len(compiled), "rejected_count": len(rejected), "rejected": rejected}
        context.state["tool_ir"] = tool_ir
        return context


class ToolExecutorNode:
    name = "tool_executor"
    _MAX_RETRY_ATTEMPTS: int = 1
    _MAX_SAME_TOOL_CALLS: int = 2  # Penalize calls to same tool > 2x per turn

    def _execute_plan_item(
        self,
        *,
        item: dict[str, Any],
        context: TurnContext,
        tool_call_counts: dict[str, int],
        redundant_calls: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
        seq = int(item.get("sequence") or 0)
        name = str(item.get("tool_name") or "").strip().lower()
        args = dict(item.get("args") or {})
        det_id = str(item.get("deterministic_id") or deterministic_tool_id(name, args))

        tool_call_counts[name] = tool_call_counts.get(name, 0) + 1
        if tool_call_counts[name] > self._MAX_SAME_TOOL_CALLS:
            redundant_calls.append(f"{name}:call_{tool_call_counts[name]}")

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
            status = "ok" if raw.status == ToolStatus.SUCCESS else raw.status.value
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
        result = {
            "sequence": seq,
            "tool_name": name,
            "status": status,
            "output": output,
            "deterministic_id": det_id,
            "remediation": remediation_log,
        }
        failure_record = self._failure_semantics_for_result(
            tool_name=name,
            status=status,
            output=output,
            remediation=remediation_log,
        )
        recovery_path = self._canonical_recovery_path_for_result(
            tool_name=name,
            status=status,
            remediation=remediation_log,
            attempts=attempt + 1,
            failure_record=failure_record,
        )
        return rec, result, failure_record, recovery_path

    async def run(self, context: TurnContext) -> TurnContext:
        # LEDGER_EXEMPT: ToolExecutorNode predates ledger protocol; execution records in context.state["tool_results"]
        # TRACE_EXEMPT: Tool execution layer; results recorded in context.state, not ledger; no graph node calls.
        tool_ir = dict(context.state.get("tool_ir") or {})
        executions: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        failure_semantics: list[dict[str, Any]] = []
        recovery_paths: list[dict[str, Any]] = []
        
        # Track tool calls for redundancy detection.
        tool_call_counts: dict[str, int] = {}
        redundant_calls: list[str] = []

        for item in list(tool_ir.get("execution_plan") or []):
            rec, result, failure_record, recovery_path = self._execute_plan_item(
                item=dict(item),
                context=context,
                tool_call_counts=tool_call_counts,
                redundant_calls=redundant_calls,
            )
            executions.append(rec)
            results.append(result)
            if failure_record is not None:
                failure_semantics.append(failure_record)
            recovery_paths.append(recovery_path)
        tool_ir["executions"] = executions
        context.state["tool_ir"] = tool_ir
        context.state["tool_results"] = results
        context.state["tool_failure_semantics"] = failure_semantics
        context.state["tool_recovery_paths"] = recovery_paths
        context.state["tool_call_counts"] = tool_call_counts
        context.state["tool_redundant_calls"] = redundant_calls
        
        # If redundancy detected, set early convergence flag to trigger exit after scoring
        if len(redundant_calls) > 0:
            context.state["should_converge_early"] = True
            context.state["convergence_reason"] = f"tool_redundancy: {len(redundant_calls)} redundant calls"
        
        context.metadata["tool_execution_graph_hash"] = _stable_sha256({"executions": executions, "results": results})
        return context

    @staticmethod
    def _failure_semantics_for_result(
        *,
        tool_name: str,
        status: str,
        output: Any,
        remediation: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        normalized = str(status or "").strip().lower()
        if normalized in {"ok", "success", "cached", "skipped"}:
            return None

        failure_class = {
            "retry": "timeout",
            "contract_violation": "bad_input",
            "fatal": "runtime_exception",
            "error": "runtime_exception",
            "failed": "runtime_exception",
            "fail": "runtime_exception",
        }.get(normalized, "runtime_exception")

        reason = normalized
        if isinstance(output, ToolContractResult):
            reason = str(output.repair_hint or normalized)
        elif isinstance(output, dict):
            reason = str(output.get("error") or output.get("message") or normalized)

        return {
            "tool_name": str(tool_name or ""),
            "failure_class": failure_class,
            "status": normalized,
            "reason": reason,
            "remediation": list(remediation or []),
        }

    @staticmethod
    def _canonical_recovery_path_for_result(
        *,
        tool_name: str,
        status: str,
        remediation: list[dict[str, Any]],
        attempts: int,
        failure_record: dict[str, Any] | None,
    ) -> dict[str, Any]:
        normalized = str(status or "").strip().lower()
        recovered = normalized in {"ok", "success", "cached", "skipped"}
        remediation_actions = [
            str(step.get("action") or "").strip().lower()
            for step in list(remediation or [])
        ]
        had_recovery_attempt = any(
            action in {"retry", "fallback", "downgrade"} for action in remediation_actions
        )

        if not remediation:
            path = "direct_success"
        elif recovered and had_recovery_attempt:
            path = "clean_recovery"
        elif had_recovery_attempt:
            path = "partial_recovery"
        else:
            path = "failed_recovery"

        return {
            "tool_name": str(tool_name or ""),
            "attempts": max(int(attempts), 1),
            "recovered": recovered,
            "recovery_path": path,
            "failure_class": str((failure_record or {}).get("failure_class") or ""),
            "remediation_actions": remediation_actions,
        }


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

    async def _run_sub_agent(self, context: TurnContext, run_agent: Any) -> str:
        try:
            result = run_agent(context, context.state.get("rich_context", {}))
            if inspect.isawaitable(result):
                result = await result
            text = result[0] if isinstance(result, tuple) else str(result or "")
            return str(text)
        except Exception as exc:
            return f"[error: sub-task failed: {exc}]"

    async def _dispatch_delegation(
        self,
        context: TurnContext,
        block: dict[str, Any],
        run_agent: Any,
        depth: int = 0,
    ) -> list[str]:
        if depth >= _MAX_DELEGATION_DEPTH:
            context.metadata["delegation_depth_exceeded"] = True
            return ["[delegation depth exceeded]"]
        mode = str(block.get("mode") or "sequential").lower()
        subtasks = list(block.get("subtasks") or [])[:_MAX_DELEGATION_SUBTASKS]
        blackboard: dict[str, Any] = dict(context.state.get("agent_blackboard") or {})
        seed_fingerprint = _stable_sha256(blackboard)
        sub_ctxs, subtask_ids = self._build_delegation_subcontexts(context, subtasks, depth)
        if mode == "parallel":
            results = await self._run_parallel_subtasks(sub_ctxs, subtasks, blackboard, run_agent)
        else:
            results = await self._run_sequential_subtasks(sub_ctxs, subtasks, blackboard, run_agent)
        context.metadata["delegation_depth"] = depth
        context.metadata["subtasks_executed"] = len(results)
        context.state["delegation_results"] = results
        context.state["agent_blackboard"] = dict(blackboard)
        final_fingerprint = _stable_sha256(blackboard)
        failure_count = sum(1 for r in results if str(r).startswith("[error:"))
        arb_hash = _stable_sha256({"subtask_ids": subtask_ids, "results": results})
        context.state["arbitration_metadata"] = {
            "mode": mode,
            "agents_dispatched": len(subtasks),
            "failure_count": failure_count,
            "arbitration_hash": arb_hash,
            "subtask_ids": subtask_ids,
        }
        # Stamp blackboard fingerprints into determinism envelope
        det = dict(context.metadata.get("determinism") or {})
        det["agent_blackboard_seed_fingerprint"] = seed_fingerprint
        det["agent_blackboard_final_fingerprint"] = final_fingerprint
        context.metadata["determinism"] = det
        return results

    def _build_delegation_subcontexts(
        self,
        context: TurnContext,
        subtasks: list[dict[str, Any]],
        depth: int,
    ) -> tuple[list[TurnContext], list[str]]:
        sub_ctxs: list[TurnContext] = []
        subtask_ids: list[str] = []
        for branch_index, task in enumerate(subtasks):
            sub_ctx = TurnContext(user_input=str(task.get("input") or ""))
            sub_ctx.trace_id = _deterministic_subtask_trace_id(
                root_trace_id=str(context.trace_id or ""),
                parent_trace_id=str(context.trace_id or ""),
                depth=depth,
                branch_index=branch_index,
                subtask=task,
            )
            sub_ctx.state.update(dict(context.state))
            sub_ctx.metadata.update(dict(context.metadata))
            sub_ctx.metadata["parent_trace_id"] = context.trace_id
            sub_ctx.metadata["agent_name"] = str(task.get("agent") or "")
            sub_ctx.metadata["delegation_depth"] = depth + 1
            sub_ctx.state["rich_context"] = dict(context.state.get("rich_context") or {})
            subtask_ids.append(sub_ctx.trace_id)
            sub_ctxs.append(sub_ctx)
        return sub_ctxs, subtask_ids

    async def _run_parallel_subtasks(
        self,
        sub_ctxs: list[TurnContext],
        subtasks: list[dict[str, Any]],
        blackboard: dict[str, Any],
        run_agent: Any,
    ) -> list[str]:
        for sub_ctx in sub_ctxs:
            sub_ctx.state["agent_blackboard"] = dict(blackboard)
        raw = await asyncio.gather(*[self._run_sub_agent(sc, run_agent) for sc in sub_ctxs])
        results = list(raw)
        for i, task in enumerate(subtasks):
            agent_name = str(task.get("agent") or "")
            if agent_name:
                blackboard[agent_name] = results[i]
        return results

    async def _run_sequential_subtasks(
        self,
        sub_ctxs: list[TurnContext],
        subtasks: list[dict[str, Any]],
        blackboard: dict[str, Any],
        run_agent: Any,
    ) -> list[str]:
        results: list[str] = []
        for sub_ctx, task in zip(sub_ctxs, subtasks):
            sub_ctx.state["agent_blackboard"] = dict(blackboard)
            result = await self._run_sub_agent(sub_ctx, run_agent)
            agent_name = str(task.get("agent") or "")
            if agent_name:
                blackboard[agent_name] = result
            results.append(result)
        return results

    def _parse_candidate_block(self, candidate: Any) -> tuple[str, dict[str, Any] | None]:
        raw = candidate[0] if isinstance(candidate, tuple) else str(candidate or "")
        try:
            parsed = json.loads(raw.strip())
        except (json.JSONDecodeError, ValueError, TypeError):
            return raw, None
        if isinstance(parsed, dict) and "type" in parsed:
            return raw, parsed
        return raw, None

    async def _apply_tool_block(
        self,
        context: TurnContext,
        block: dict[str, Any],
        run_agent: Any,
        rich_context: dict[str, Any],
    ) -> Any:
        tool_name = str(block.get("name") or "")
        tool_args = dict(block.get("args") or {})
        context.metadata["tool_called"] = tool_name
        try:
            tool_result = dispatch_registered_tool(tool_name, tool_args, context)
            context.metadata["tool_call_executed"] = True
            context.state["tool_result"] = tool_result
        except Exception as exc:
            context.metadata["tool_call_executed"] = False
            context.state["tool_result"] = f"[tool error: {exc}]"
        context.metadata["parent_trace_id"] = context.trace_id
        follow_up = run_agent(context, rich_context)
        if inspect.isawaitable(follow_up):
            follow_up = await follow_up
        return follow_up

    async def _resolve_structured_block(
        self,
        context: TurnContext,
        block: dict[str, Any],
        raw: str,
        run_agent: Any,
        rich_context: dict[str, Any],
    ) -> Any | None:
        block_type = str(block.get("type") or "")
        if block_type == "reasoning":
            steps = list(block.get("steps") or [])
            conclusion = str(block.get("conclusion") or raw)
            context.metadata["reasoning_structured"] = True
            context.metadata["reasoning_steps_count"] = len(steps)
            return (conclusion, False)
        if block_type == "tool":
            return await self._apply_tool_block(context, block, run_agent, rich_context)
        if block_type == "delegate":
            del_results = await self._dispatch_delegation(context, block, run_agent, depth=0)
            failure_count = int((context.state.get("arbitration_metadata") or {}).get("failure_count") or 0)
            n = len(del_results)
            if failure_count > 0:
                summary = f"I delegated {n} sub-tasks. {failure_count} sub-task(s) failed. " + " ".join(
                    r for r in del_results if str(r).startswith("[error:")
                )
            else:
                summary = f"I delegated {n} sub-tasks. Results: " + "; ".join(str(r) for r in del_results)
            return (summary, False)
        return None

    async def run(self, context: TurnContext) -> TurnContext:
        # LEDGER_EXEMPT: InferenceNode predates ledger protocol; LLM candidate written to context.state["candidate"]
        control_plane = getattr(
            getattr(getattr(self.mgr, "bot", None), "turn_orchestrator", None),
            "control_plane",
            None,
        )
        execute_from_graph_context = getattr(control_plane, "execute_from_graph_context", None)
        if not callable(execute_from_graph_context):
            raise RuntimeError(
                "InferenceNode requires control_plane.execute_from_graph_context; "
                "legacy manager.run_agent authority is disabled",
            )

        async def run_agent(candidate_context: TurnContext, candidate_rich_context: dict[str, Any]) -> Any:
            result = execute_from_graph_context(candidate_context, candidate_rich_context)
            if inspect.isawaitable(result):
                return await result
            return result

        rich_context = context.state.get("rich_context", {})
        candidate: Any = None
        for iteration in range(self._max_loop_iterations):
            next_candidate = run_agent(context, rich_context)
            candidate = await next_candidate if inspect.isawaitable(next_candidate) else next_candidate
            if self._run_critique_check(context, candidate, iteration):
                break
        context.state.pop("_critique_revision_context", None)

        raw, block = self._parse_candidate_block(candidate)
        if block is not None:
            resolved_candidate = await self._resolve_structured_block(
                context,
                block,
                raw,
                run_agent,
                rich_context,
            )
            if resolved_candidate is not None:
                candidate = resolved_candidate

        context.state["candidate"] = candidate
        return context


class SafetyNode:
    name = "safety"

    def __init__(self, safety_manager: Any = None) -> None:
        self.mgr = safety_manager

    async def run(self, context: TurnContext) -> TurnContext:
        # LEDGER_EXEMPT: SafetyNode predates ledger protocol; policy decision in context.state["policy_trace_events"]
        # TRACE_EXEMPT: Policy evaluation (deterministic); decisions written to context.state, no ledger, no node calls.
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
        semantic_trace = dict((decision.trace or {}).get("semantic_eval") or {})
        eval_input_hash = str(semantic_trace.get("eval_input_hash") or "")
        if eval_input_hash:
            by_hash = dict(context.state.get("semantic_decision_by_eval_hash") or {})
            by_hash[eval_input_hash] = {
                "action": decision.action,
                "step_name": decision.step_name,
                "details": dict(decision.details or {}),
            }
            context.state["semantic_decision_by_eval_hash"] = by_hash
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

    @staticmethod
    def _build_clarity_markers(reply: str) -> list[str]:
        clarity_markers: list[str] = []
        if reply.count("\n") > reply.count("  "):
            clarity_markers.append("structured_newlines")
        if ":" in reply and reply.count(":") > 2:
            clarity_markers.append("key_value_format")
        if any(bullet in reply for bullet in ["• ", "- ", "* ", "1. ", "• "]):
            clarity_markers.append("bullet_list")
        return clarity_markers

    def _prepare_ux_trace(self, context: TurnContext) -> None:
        ux_trace = dict(context.state.get("ux_trace") or context.state.get("ux_feedback") or {})
        ux_trace["time_to_first_token_ms"] = float(context.compute_time_to_first_token_ms())
        ux_trace["time_to_resolution_ms"] = float(context.compute_time_to_resolution_ms())
        critique_record = dict(context.state.get("critique_record") or {})
        ux_trace["backtrack_count"] = int(critique_record.get("iteration", 0))
        candidate = context.state.get("candidate")
        reply = str(candidate[0] if isinstance(candidate, tuple) else candidate or "")
        ux_trace["clarity_markers"] = self._build_clarity_markers(reply)
        context.state["ux_trace"] = ux_trace

    def _apply_commit_flow(self, context: TurnContext) -> None:
        begin = getattr(self.mgr, "begin_transaction", None)
        apply = getattr(self.mgr, "apply_mutations", None)
        finalize = getattr(self.mgr, "finalize_turn", None)
        commit = getattr(self.mgr, "commit_transaction", None)
        if callable(begin):
            begin(context)
        if callable(apply):
            apply(context)
        if callable(finalize):
            context.state["safe_result"] = finalize(context, context.state.get("safe_result"))
        if callable(commit):
            commit(context)
            return
        save_turn = getattr(self.mgr, "save_turn", None)
        if callable(save_turn):
            save_turn(context, context.state.get("safe_result"))

    async def run(self, context: TurnContext) -> TurnContext:
        # LEDGER_EXEMPT: SaveNode predates ledger protocol; persistence outcome recorded in context.fidelity.save
        # Commit-boundary contract anchor: ledger_entry is emitted via persistence/ledger services.
        if getattr(context, "temporal", None) is None:
            raise RuntimeError("SaveNode requires temporal context")
        self._prepare_ux_trace(context)

        if self.mgr is None:
            context.fidelity.save = True
            context.state["last_commit_id"] = context.trace_id
            context.state["last_transaction_status"] = "committed"
            return context
        rollback = getattr(self.mgr, "rollback_transaction", None)
        try:
            self._apply_commit_flow(context)
            context.fidelity.save = True
            context.state["last_commit_id"] = context.trace_id
            context.state["last_transaction_status"] = "committed"
            return context
        except Exception as exc:
            rollback_failed = False
            if callable(rollback):
                try:
                    rollback(context)
                except Exception as rollback_exc:
                    rollback_failed = True
                    context.state["rollback_error"] = str(rollback_exc)
            context.state["last_transaction_status"] = (
                "rollback_failed"
                if rollback_failed
                else ("rolled_back" if callable(rollback) else "failed_no_rollback")
            )
            raise exc


class ReflectionNode:
    name = "reflection"

    def __init__(self, reflection_manager: Any = None) -> None:
        self.mgr = reflection_manager

    async def run(self, context: TurnContext) -> TurnContext:
        # LEDGER_EXEMPT: ReflectionNode predates ledger protocol; output in context.state["reflection"]
        # TRACE_EXEMPT: Post-processing reflection; no ledger emission, no downstream calls, no graph mutation.
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
    allowed_intents=set(_ALLOWED_UTILITY_INTENTS),
)
register_tool(
    "current_time",
    handler=_builtin_current_time,
    required_args=set(),
    allowed_intents=set(_ALLOWED_UTILITY_INTENTS),
)


__all__ = [
    "_MAX_DELEGATION_DEPTH",
    "_MAX_DELEGATION_SUBTASKS",
    "ContextBuilderNode",
    "HealthNode",
    "InferenceNode",
    "MemoryNode",
    "ReflectionNode",
    "SafetyNode",
    "SaveNode",
    "TemporalNode",
    "ToolExecutorNode",
    "ToolRegistration",
    "ToolRouterNode",
    "dispatch_registered_tool",
    "get_registered_tool",
    "get_registered_tool_names",
    "get_tool_required_args",
    "register_tool",
]
