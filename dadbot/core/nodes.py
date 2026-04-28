from __future__ import annotations

import hashlib
import json
import logging
import re
import asyncio
import time
from dataclasses import dataclass
from typing import Any


from dadbot.core.graph import NodeType, TurnContext
from dadbot.core.execution_trace_context import (
    record_execution_step,
    record_external_system_call,
)
from dadbot.core.tool_ir import (
    ToolEvent,
    ToolEventLog,
    ToolEventType,
    ToolExecutionPlan,
    ToolRequest,
    build_execution_event,
    deterministic_tool_id,
    normalize_tool_results,
    reduce_events_to_results,
)
from dadbot.core.tool_dag import build_dag_from_execution_plan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stable hash helper (delegation determinism envelope)
# ---------------------------------------------------------------------------


def _stable_sha256(payload: Any) -> str:
    """Compute a deterministic SHA-256 digest from any JSON-serialisable payload."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

# ---------------------------------------------------------------------------
# Structured output helpers (delegation / reasoning / tool calling)
# ---------------------------------------------------------------------------

#: Maximum recursion depth for internal sub-task delegation.
_MAX_DELEGATION_DEPTH: int = 2

#: Matches a JSON object inside a markdown fenced code block.
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

#: Recognised structured block types emitted by the LLM.
_KNOWN_STRUCTURED_TYPES = frozenset({"delegate", "tool", "reasoning", "plan"})

#: Recognised delegation execution modes.
_DELEGATION_MODES = frozenset({"sequential", "parallel"})

#: Hard cap on delegated sub-task count per delegation block.
_MAX_DELEGATION_SUBTASKS: int = 8

_ALLOWED_ROUTED_TOOLS = frozenset({"memory_lookup"})
_ALLOWED_MEMORY_INTENTS = frozenset({"goal_lookup", "session_memory_fetch"})


@dataclass
class DelegateResult:
    task_input: str
    agent_name: str
    success: bool
    payload: str | None = None
    error: str = ""


def _safe_snippet(value: str, *, limit: int = 120) -> str:
    """Return a compact single-line snippet suitable for user-facing traces."""
    compact = " ".join(str(value or "").split())
    return compact[:limit]


def _is_subtask_failure_text(value: str) -> bool:
    text = str(value or "").strip().lower()
    return text.startswith("[sub-task failed") or text.startswith("[sub-task error")


def _append_arbitration_log(context: TurnContext, payload: dict[str, Any]) -> None:
    """Append a delegation arbitration event to turn state for safe auditing."""
    events = list(context.state.get("delegation_arbitration_log") or [])
    events.append(dict(payload or {}))
    context.state["delegation_arbitration_log"] = events


def _parse_structured_block(text: str) -> "dict | None":
    """Extract a typed JSON structured block from LLM output text, or None.

    Checks fenced code blocks first, then bare JSON at the start of text.
    Returns the parsed dict only when ``type`` is in ``_KNOWN_STRUCTURED_TYPES``.
    """
    if not text or not isinstance(text, str):
        return None
    candidates: list[str] = []
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1))
    stripped = text.strip()
    if stripped.startswith("{"):
        candidates.append(stripped)
    for raw in candidates:
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        block_type = str(data.get("type") or "").strip().lower()
        if block_type in _KNOWN_STRUCTURED_TYPES:
            return {**data, "type": block_type}
    return None


def _extract_reasoning_reply(block: dict) -> str:
    """Pull the final human-readable reply out of a reasoning/plan block."""
    for key in ("conclusion", "answer", "reply", "response", "result"):
        v = block.get(key)
        if v and isinstance(v, str) and v.strip():
            return v.strip()
    steps = block.get("steps")
    if isinstance(steps, list) and steps:
        last = steps[-1]
        if isinstance(last, str):
            return last.strip()
        if isinstance(last, dict):
            for k in ("output", "result", "description", "step"):
                v = last.get(k)
                if v and isinstance(v, str):
                    return v.strip()
    return ""


def _dispatch_builtin_tool(name: str, args: dict, context: TurnContext) -> str:
    """Execute a built-in tool and return its string result.

    Built-in tools:
    - ``echo``: echo the ``message`` arg (passthrough / smoke test)
    - ``current_time``: return the frozen turn wall-time from the temporal axis
    - ``memory_lookup``: return a snippet from the turn's memory/context snapshot
    """
    if name == "echo":
        return str(args.get("message") or args)
    if name == "current_time":
        temporal = getattr(context, "temporal", None)
        return str(getattr(temporal, "wall_time", "") if temporal else "")
    if name == "memory_lookup":
        query = str(args.get("query") or args.get("q") or "").strip()
        scope = str(args.get("scope") or "").strip().lower()
        if scope == "goals":
            goals = list(context.state.get("session_goals") or [])
            goal_ids = {str(v) for v in list(args.get("goal_ids") or [])}
            if goal_ids:
                goals = [g for g in goals if str((g or {}).get("id") or "") in goal_ids]
            if goals:
                return str(goals[:3])
            return f"[No goal matches for: {query!r}]"
        if scope == "session":
            memories = list(context.state.get("memories") or [])
            if memories:
                return str(memories[:3])
            rich_ctx = dict(context.state.get("rich_context") or {})
            if rich_ctx:
                return str(list(rich_ctx.items())[:3])
            return f"[No session memory results for: {query!r}]"
        memories = list(context.state.get("memories") or [])
        rich_ctx = dict(context.state.get("rich_context") or {})
        if memories:
            return str(memories[:3])
        if rich_ctx:
            return str(list(rich_ctx.items())[:3])
        return f"[No memory results for: {query!r}]"
    return f"[Unknown tool: {name!r}]"


class TemporalNode:
    """Publishes the canonical frozen turn time into state and metadata."""

    name = "temporal"

    async def run(self, context: TurnContext) -> TurnContext:
        if getattr(context, "temporal", None) is None:
            raise RuntimeError("TemporalNode missing — deterministic execution violated")
        vc = getattr(context, "virtual_clock", None)
        if vc is not None:
            # Derive temporal axis from the virtual clock instead of the real wall clock,
            # giving tests and replay runs fully deterministic temporal fields.
            vc.tick()
            vdt = vc.to_datetime()
            offset = vdt.utcoffset()
            offset_minutes = int(offset.total_seconds() // 60) if offset is not None else 0
            wall_time = vdt.isoformat(timespec="seconds")
            from dadbot.core.graph import TurnTemporalAxis
            context.temporal = TurnTemporalAxis(
                turn_started_at=wall_time,
                wall_time=wall_time,
                wall_date=vdt.date().isoformat(),
                timezone=str(vdt.tzname() or "local").strip() or "local",
                utc_offset_minutes=offset_minutes,
                epoch_seconds=vc.now(),
            )
        temporal_payload = context.temporal_snapshot()
        context.state["temporal"] = temporal_payload
        context.metadata.setdefault("temporal", temporal_payload)
        return context


class HealthNode:
    """Runs periodic maintenance and proactive engagement checks before the turn."""

    def __init__(self, health_manager: Any):
        self.mgr = health_manager

    async def run(self, context: TurnContext) -> TurnContext:
        tick = getattr(self.mgr, "tick", None)
        if callable(tick):
            try:
                context.state["health"] = tick(context)
            except Exception as exc:
                logger.warning("HealthNode.tick failed (non-fatal): %s", exc)
        return context


class ContextBuilderNode:
    """Builds rich contextual payload (profile/relationship/memory/cross-session)."""

    name = "context_builder"

    def __init__(self, memory_manager: Any, *, goal_ranker: Any = None):
        self.mgr = memory_manager
        self._goal_ranker = goal_ranker

    async def run(self, context: TurnContext) -> TurnContext:
        if getattr(context, "temporal", None) is None:
            raise RuntimeError("TemporalNode missing — deterministic execution violated")
        if callable(getattr(self.mgr, "query", None)):
            try:
                context.state.setdefault("temporal", context.temporal_snapshot())
                context.state["memories"] = await self.mgr.query(context.user_input)
                # Goal-aware memory re-ranking: boost entries relevant to active goals.
                active_goals = list(context.state.get("session_goals") or [])
                if active_goals and self._goal_ranker is not None:
                    try:
                        context.state["memories"] = self._goal_ranker.rerank(
                            context.state["memories"], active_goals
                        )
                    except Exception as exc:
                        logger.warning("GoalAwareRanker.rerank failed (non-fatal): %s", exc)
            except Exception as exc:
                logger.warning("MemoryNode.query failed (non-fatal): %s", exc)
            return context

        build_context = getattr(self.mgr, "build_context", None)
        if callable(build_context):
            try:
                rich_context = build_context(context)
                if isinstance(rich_context, dict):
                    rich_context.setdefault("temporal", context.temporal_snapshot())
                context.state["rich_context"] = rich_context
            except Exception as exc:
                logger.warning("MemoryNode.build_context failed (non-fatal): %s", exc)
        return context


class MemoryNode(ContextBuilderNode):
    """Backward-compatible alias for legacy pipeline wiring."""

    name = "memory"


class ToolRouterNode:
    """Strict compiler for planner-emitted Tool IR requests.

    Enforces validation, deduplication, and deterministic execution ordering.
    """

    name = "tool_router"

    def _compile_request(self, raw: Any) -> tuple[ToolRequest | None, str | None]:
        item = dict(raw or {})
        tool_name = str(item.get("tool_name") or "").strip().lower()
        if tool_name not in _ALLOWED_ROUTED_TOOLS:
            return None, f"unsupported_tool:{tool_name or '<empty>'}"
        args = item.get("args")
        if not isinstance(args, dict):
            return None, "invalid_args"
        intent = str(item.get("intent") or "").strip().lower()
        if intent not in _ALLOWED_MEMORY_INTENTS:
            return None, f"invalid_intent:{intent or '<empty>'}"
        expected_output = str(item.get("expected_output") or "").strip()
        if not expected_output:
            return None, "missing_expected_output"
        try:
            priority = int(item.get("priority") or 100)
        except Exception:
            return None, "invalid_priority"
        if priority < 0:
            return None, "invalid_priority"
        return ToolRequest(
            tool_name=tool_name,
            args=dict(args),
            intent=intent,
            expected_output=expected_output,
            priority=priority,
        ), None

    async def run(self, context: TurnContext) -> TurnContext:
        tool_ir = dict(context.state.get("tool_ir") or {})
        raw_requests = list(tool_ir.get("requests") or [])

        compiled: list[ToolRequest] = []
        rejected: list[dict[str, Any]] = []
        seen: set[str] = set()

        for index, raw in enumerate(raw_requests):
            request, rejection = self._compile_request(raw)
            if request is None:
                rejected.append({"index": index, "reason": rejection or "invalid_request"})
                continue
            req_id = deterministic_tool_id(request.tool_name, request.args)
            if req_id in seen:
                rejected.append({"index": index, "reason": "duplicate_request", "deterministic_id": req_id})
                continue
            seen.add(req_id)
            compiled.append(request)

        compiled.sort(
            key=lambda item: (
                int(item.priority),
                str(item.intent),
                deterministic_tool_id(item.tool_name, item.args),
            )
        )
        plan = ToolExecutionPlan(requests=compiled)

        tool_ir["requests"] = [
            {
                "tool_name": request.tool_name,
                "args": request.args,
                "intent": request.intent,
                "expected_output": request.expected_output,
                "priority": request.priority,
            }
            for request in compiled
        ]
        tool_ir["execution_plan"] = [
            {
                "sequence": index,
                "tool_name": request.tool_name,
                "args": request.args,
                "intent": request.intent,
                "expected_output": request.expected_output,
                "priority": request.priority,
                "deterministic_id": deterministic_tool_id(request.tool_name, request.args),
            }
            for index, request in enumerate(plan.requests)
        ]
        tool_ir["compiler"] = {
            "strict": True,
            "compiled_count": len(compiled),
            "rejected_count": len(rejected),
            "rejected": rejected,
            "ordering": "priority:intent:deterministic_id",
        }
        # Phase 1+2: emit canonical ToolDAG for downstream consumers.
        execution_plan_for_dag = list(tool_ir.get("execution_plan") or [])
        if execution_plan_for_dag:
            dag = build_dag_from_execution_plan(execution_plan_for_dag)
            tool_ir["tool_dag"] = dag.to_dict()
        else:
            tool_ir["tool_dag"] = None
        context.metadata["tool_router_strict"] = True
        context.metadata["tool_router_rejected_count"] = len(rejected)
        context.state["tool_ir"] = tool_ir
        return context


class ToolExecutorNode:
    """Deterministic source-of-truth tool execution boundary.

    Guarantees:
    - Executes only items present in ToolRouter output.
    - No hidden or fallback tool execution paths.
    - Every planned step emits a logged execution and tool result.
    """

    name = "tool_executor"

    @staticmethod
    def _canonical_execution_trace(
        *, execution_plan: list[dict[str, Any]], executions: list[dict[str, Any]], results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "execution_plan": [
                {
                    "sequence": int(item.get("sequence") or 0),
                    "tool_name": str(item.get("tool_name") or ""),
                    "intent": str(item.get("intent") or ""),
                    "priority": int(item.get("priority") or 0),
                    "deterministic_id": str(item.get("deterministic_id") or ""),
                }
                for item in execution_plan
            ],
            "executions": [
                {
                    "sequence": int(item.get("sequence") or 0),
                    "tool_name": str(item.get("tool_name") or ""),
                    "input_hash": str(item.get("input_hash") or ""),
                    "status": str(item.get("status") or ""),
                    "output": item.get("output"),
                    "deterministic_id": str(item.get("deterministic_id") or ""),
                }
                for item in executions
            ],
            "results": normalize_tool_results(results),
        }

    async def run(self, context: TurnContext) -> TurnContext:
        tool_ir = dict(context.state.get("tool_ir") or {})
        execution_plan = list(tool_ir.get("execution_plan") or [])
        executions: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []

        for item in execution_plan:
            sequence = int(item.get("sequence") or 0)
            tool_name = str(item.get("tool_name") or "").strip().lower()
            args = item.get("args")
            expected_id = str(item.get("deterministic_id") or "")

            if tool_name not in _ALLOWED_ROUTED_TOOLS:
                raise RuntimeError(f"ToolExecutorNode refuses unsupported tool: {tool_name!r}")
            if not isinstance(args, dict):
                raise RuntimeError(f"ToolExecutorNode received invalid args for tool: {tool_name!r}")

            started = time.perf_counter()
            try:
                output = _dispatch_builtin_tool(tool_name, dict(args), context)
                event = build_execution_event(tool_name, dict(args), output, "ok", started)
                record_external_system_call(
                    operation="tool_dispatch",
                    system=f"builtin_tool:{tool_name}",
                    request_payload={
                        "sequence": sequence,
                        "args": dict(args),
                        "deterministic_id": expected_id,
                    },
                    response_payload={
                        "status": "ok",
                        "output": output,
                    },
                    status="ok",
                    source="tool_executor",
                    deterministic_id=expected_id,
                    required=False,
                )
            except Exception as exc:
                event = build_execution_event(tool_name, dict(args), str(exc), "error", started)
                record_external_system_call(
                    operation="tool_dispatch",
                    system=f"builtin_tool:{tool_name}",
                    request_payload={
                        "sequence": sequence,
                        "args": dict(args),
                        "deterministic_id": expected_id,
                    },
                    response_payload={
                        "status": "error",
                        "error": str(exc),
                    },
                    status="error",
                    source="tool_executor",
                    deterministic_id=expected_id,
                    required=False,
                )

            if expected_id and expected_id != event.deterministic_id:
                raise RuntimeError(
                    "ToolExecutorNode deterministic_id mismatch for "
                    f"{tool_name!r}: expected {expected_id!r}, got {event.deterministic_id!r}"
                )

            executions.append(
                {
                    "sequence": sequence,
                    "tool_name": event.tool_name,
                    "input_hash": event.input_hash,
                    "output": event.output,
                    "latency": event.latency,
                    "status": event.status,
                    "deterministic_id": event.deterministic_id,
                }
            )
            results.append(
                {
                    "sequence": sequence,
                    "tool_name": tool_name,
                    "status": event.status,
                    "output": event.output,
                    "deterministic_id": event.deterministic_id,
                }
            )

        if len(executions) != len(execution_plan):
            raise RuntimeError(
                "ToolExecutorNode execution log mismatch: "
                f"planned={len(execution_plan)} executed={len(executions)}"
            )

        # Phase 3: build event log from execution records.
        event_log = ToolEventLog()
        for seq, (item, exec_rec) in enumerate(zip(execution_plan, executions)):
            t_name = str(item.get("tool_name") or "")
            t_args = dict(item.get("args") or {})
            t_id = str(item.get("deterministic_id") or "")
            # REQUESTED event
            event_log.append(ToolEvent.requested(t_id, seq * 2, t_name, t_args))
            # EXECUTED or FAILED event
            if str(exec_rec.get("status") or "ok") == "error":
                event_log.append(ToolEvent.failed(t_id, seq * 2 + 1, t_name, t_args, str(exec_rec.get("output") or "")))
            else:
                event_log.append(ToolEvent.executed(t_id, seq * 2 + 1, t_name, t_args, exec_rec.get("output"), "ok"))
        tool_ir["event_stream"] = event_log.to_list()
        tool_ir["event_stream_replay_hash"] = event_log.replay_hash()

        tool_ir["executions"] = executions
        tool_ir["executor"] = {
            "strict": True,
            "planned_count": len(execution_plan),
            "executed_count": len(executions),
            "all_outputs_logged": len(executions) == len(results),
            "hidden_execution_paths": False,
        }
        context.state["tool_ir"] = tool_ir
        context.state["tool_results"] = normalize_tool_results(results)

        canonical_trace = self._canonical_execution_trace(
            execution_plan=execution_plan,
            executions=executions,
            results=results,
        )
        context.metadata["tool_execution_graph_hash"] = _stable_sha256(canonical_trace)
        context.metadata["tool_executor_strict"] = True
        return context



class InferenceNode:
    """Pure cognition loop executor - runs AgentService.run_agent as the sole inference path.

    Phase 4 extensions (all contained inside this node; no pipeline shape change):

    - **Sub-task delegation** (Priority 1): LLM emits
      ``{"type": "delegate", "subtasks": [...]}``.  Each sub-task is executed
      synchronously via ``run_agent`` at most ``_MAX_DELEGATION_DEPTH`` levels deep.
      Results are merged and committed through the existing SaveNode.
    - **Structured reasoning** (Priority 2): LLM emits
      ``{"type": "reasoning", "steps": [...], "conclusion": "..."}`` or
      ``{"type": "plan", ...}``.  The conclusion is extracted as the reply;
      step list is saved to ``context.state["reasoning_steps"]``.
    - **Tool calling** (Priority 3): LLM emits
      ``{"type": "tool", "name": "...", "args": {...}}``.  Dispatched to the
      built-in tool registry; result fed back for a follow-up inference call.
    - **Depth guard** (Priority 4): delegation depth is tracked and capped at
      ``_MAX_DELEGATION_DEPTH``; ``delegation_depth_exceeded`` is stamped in
      metadata when the guard fires.

    No fallback generation path exists.  If the bound manager does not expose
    ``run_agent``, the node raises ``RuntimeError`` at run-time rather than
    silently routing through a legacy path.
    """

    def __init__(
        self,
        llm_manager: Any,
        *,
        critique_engine: Any = None,
        max_loop_iterations: int = 2,
    ) -> None:
        self.mgr = llm_manager
        self._critique_engine = critique_engine
        self._max_loop_iterations = max(1, int(max_loop_iterations))

    @staticmethod
    def _fallback_candidate(message: str) -> tuple[str, bool]:
        return (str(message or "Unable to generate a reply right now."), False)

    # ------------------------------------------------------------------
    # Structured-output dispatch
    # ------------------------------------------------------------------

    async def _resolve_candidate(
        self,
        context: TurnContext,
        raw_result: Any,
        *,
        depth: int = 0,
    ) -> Any:
        """Resolve raw ``run_agent`` output, expanding any structured block inline.

        Returns the same ``(reply, should_exit)`` tuple shape; only the reply
        text is transformed when a recognised structured block is detected.
        """
        reply = raw_result[0] if isinstance(raw_result, tuple) else str(raw_result or "")
        should_exit = (
            raw_result[1]
            if isinstance(raw_result, tuple) and len(raw_result) > 1
            else False
        )
        block = _parse_structured_block(reply)
        if block is None:
            return raw_result
        resolved = await self._handle_block(context, block, depth=depth)
        return (resolved, should_exit)

    async def _handle_block(
        self, context: TurnContext, block: dict, *, depth: int
    ) -> str:
        """Dispatch a structured block to its typed handler."""
        block_type = str(block.get("type") or "").lower()
        if block_type == "delegate":
            return await self._handle_delegation(context, block, depth=depth)
        if block_type == "tool":
            return await self._handle_tool_call(context, block)
        if block_type in ("reasoning", "plan"):
            return self._handle_reasoning(context, block)
        return str(block)

    # ------------------------------------------------------------------
    # Priority 1: Sub-task delegation
    # ------------------------------------------------------------------

    async def _handle_delegation(
        self, context: TurnContext, block: dict, *, depth: int
    ) -> str:
        """Execute delegated sub-tasks and merge their results.

        Execution modes (``mode`` key in the delegate block):

        - ``"sequential"`` (default): tasks run one after another. Each
          agent's result is posted to ``context.state["agent_blackboard"]``
          so later agents can read earlier results (inter-agent messaging).
        - ``"parallel"``: tasks run concurrently via ``asyncio.gather``.

        A short delegation summary is included in the final reply so users can
        understand what happened without opening debug views.
        """
        if depth >= _MAX_DELEGATION_DEPTH:
            logger.warning(
                "Delegation depth guard triggered at depth=%d (max=%d).",
                depth,
                _MAX_DELEGATION_DEPTH,
            )
            context.metadata["delegation_depth_exceeded"] = True
            context.state["delegation_depth_exceeded"] = True
            context.state["arbitration_metadata"] = {
                "mode": "blocked",
                "agents_dispatched": 0,
                "depth": depth,
                "reason": "depth_limit",
            }
            _append_arbitration_log(
                context,
                {
                    "event": "depth_guard_block",
                    "depth": depth,
                    "max_depth": _MAX_DELEGATION_DEPTH,
                },
            )
            message = (
                f"I skipped delegation because depth limit {_MAX_DELEGATION_DEPTH} was reached."
            )
            if context.state.get("assistant_response"):
                context.state["assistant_response"] = (
                    f"{context.state['assistant_response']}\n\nDelegation depth limit reached. "
                    "Stopping further subtasks."
                )
            else:
                context.state["assistant_response"] = (
                    "Delegation depth limit reached. Stopping further subtasks."
                )
            return message

        raw_mode = str(block.get("mode") or "sequential").strip().lower()
        mode = raw_mode if raw_mode in _DELEGATION_MODES else "sequential"

        subtasks = list(block.get("subtasks") or [])
        requested_subtasks = len(subtasks)
        if requested_subtasks > _MAX_DELEGATION_SUBTASKS:
            subtasks = subtasks[:_MAX_DELEGATION_SUBTASKS]
            _append_arbitration_log(
                context,
                {
                    "event": "subtask_trimmed",
                    "requested": requested_subtasks,
                    "executed": len(subtasks),
                    "max_subtasks": _MAX_DELEGATION_SUBTASKS,
                },
            )

        # Resolve (input, agent_name) pairs.
        task_pairs: list[tuple[str, str]] = []
        for i, subtask in enumerate(subtasks):
            if isinstance(subtask, str):
                sub_input, agent_name = subtask.strip(), f"agent_{i}"
            else:
                sub_input = str(subtask.get("input") or "").strip()
                agent_name = str(subtask.get("agent") or f"agent_{i}").strip()
            if sub_input:
                task_pairs.append((sub_input, agent_name))

        # Assign deterministic subtask IDs and sort by ID to guarantee consistent
        # execution order regardless of LLM output ordering or async scheduling.
        _parent_trace = str(context.trace_id or "")
        _indexed: list[tuple[str, str, str]] = [
            (f"{_parent_trace}.del{depth}.{i}", inp, name)
            for i, (inp, name) in enumerate(task_pairs)
        ]
        _indexed.sort(key=lambda t: t[0])
        subtask_ids = [sid for sid, _, _ in _indexed]
        task_pairs = [(inp, name) for _, inp, name in _indexed]

        seed_board = dict(context.metadata.get("agent_blackboard_seed") or {})
        blackboard: dict[str, str]
        if depth <= 0:
            blackboard = dict(seed_board)
        else:
            blackboard = dict(context.state.get("agent_blackboard") or seed_board)
        context.state["agent_blackboard"] = blackboard

        if mode == "parallel":
            coros = [
                self._run_subtask(context, inp, depth=depth + 1, agent_name=name)
                for inp, name in task_pairs
            ]
            gathered = await asyncio.gather(*coros, return_exceptions=True)
            delegate_results: list[DelegateResult] = []
            for (_inp, name), outcome in zip(task_pairs, gathered):
                if isinstance(outcome, Exception):
                    text = f"[Sub-task failed: {name} -> {str(outcome)[:100]}]"
                    logger.warning("Parallel sub-task %r failed: %s", name, outcome)
                    delegate_results.append(
                        DelegateResult(
                            task_input=_inp,
                            agent_name=name,
                            success=False,
                            payload=None,
                            error=text,
                        )
                    )
                else:
                    delegate_results.append(outcome)
        else:
            delegate_results = []
            for inp, name in task_pairs:
                sub_result = await self._run_subtask(
                    context,
                    inp,
                    depth=depth + 1,
                    agent_name=name,
                    blackboard=blackboard,
                )
                delegate_results.append(sub_result)
                # Sequential mode supports inter-agent messaging: downstream subtasks
                # must see upstream outputs immediately via the shared blackboard.
                if sub_result.success:
                    blackboard[name] = str(sub_result.payload or "")
                else:
                    blackboard[name] = str(sub_result.error or "")

        result_texts: list[str] = []
        for item in delegate_results:
            if item.success:
                text = str(item.payload or "")
                blackboard[item.agent_name] = text
                result_texts.append(text)
            else:
                blackboard[item.agent_name] = str(item.error or "")
                result_texts.append(str(item.error or ""))

        successful = sum(1 for item in delegate_results if item.success)
        failures = sum(1 for item in delegate_results if not item.success)
        merged_details = "\n\n".join(r for r in result_texts if r) or "[No sub-task results]"
        summary = (
            f"I delegated {len(task_pairs)} task(s) in {mode} mode"
            f" ({successful} succeeded, {failures} failed)."
        )

        # Canonical arbitration record hash — locks in "same subtasks → same winner"
        # guarantee and detects any reordering or output substitution.
        _arb_record = {
            "subtask_ids": subtask_ids,
            "mode": mode,
            "depth": depth,
            "outputs": result_texts,
        }
        arbitration_hash = _stable_sha256(_arb_record)

        context.metadata["delegation_depth"] = depth
        context.metadata["subtasks_executed"] = len(task_pairs)
        context.state["delegation_results"] = result_texts
        context.state["arbitration_metadata"] = {
            "mode": mode,
            "agents_dispatched": len(task_pairs),
            "depth": depth,
            "requested_subtasks": requested_subtasks,
            "executed_subtasks": len(task_pairs),
            "success_count": successful,
            "failure_count": failures,
            "depth_exceeded": bool(context.metadata.get("delegation_depth_exceeded")),
            "subtask_ids": subtask_ids,
            "arbitration_hash": arbitration_hash,
        }
        _append_arbitration_log(
            context,
            {
                "event": "delegation_complete",
                "mode": mode,
                "depth": depth,
                "requested_subtasks": requested_subtasks,
                "executed_subtasks": len(task_pairs),
                "success_count": successful,
                "failure_count": failures,
            },
        )
        visible_summary = (
            f"\n\nI delegated {len(task_pairs)} subtasks (research, safety checks, etc.) "
            "and merged the results."
        )
        if failures > 0:
            visible_summary += (
                f"\n\nNote: {failures} subtask(s) encountered issues but the main goal was completed."
            )
        if context.state.get("assistant_response"):
            context.state["assistant_response"] = f"{context.state['assistant_response']}{visible_summary}"
        else:
            context.state["assistant_response"] = f"Task completed with delegation.{visible_summary}"
        return f"{summary}\n\n{merged_details}{visible_summary}"

    async def _run_subtask(
        self,
        parent_context: TurnContext,
        subtask_input: str,
        *,
        depth: int,
        agent_name: str = "",
        blackboard: "dict[str, str] | None" = None,
    ) -> DelegateResult:
        """Run a single delegated sub-task through inference only."""
        run_agent = getattr(self.mgr, "run_agent", None)
        if not callable(run_agent):
            return DelegateResult(
                task_input=subtask_input,
                agent_name=str(agent_name or "agent"),
                success=False,
                payload=None,
                error="[Sub-task failed: no inference provider available.]",
            )
        sub_ctx = TurnContext(
            user_input=subtask_input,
            metadata={
                "determinism": dict(parent_context.metadata.get("determinism") or {}),
                "temporal": dict(parent_context.metadata.get("temporal") or {}),
                "delegation_depth": depth,
                "parent_trace_id": parent_context.trace_id,
                "agent_name": str(agent_name or ""),
            },
            state={
                "rich_context": dict(parent_context.state.get("rich_context") or {}),
                "memories": list(parent_context.state.get("memories") or []),
                "agent_blackboard": dict(blackboard or {}),
            },
        )
        sub_ctx.temporal = parent_context.temporal
        rich_context = sub_ctx.state.get("rich_context", {})
        try:
            sub_raw = await run_agent(sub_ctx, rich_context)
            reply = sub_raw[0] if isinstance(sub_raw, tuple) else str(sub_raw or "")
            sub_block = _parse_structured_block(reply)
            if sub_block is not None:
                reply = await self._handle_block(sub_ctx, sub_block, depth=depth)
                if bool(sub_ctx.metadata.get("delegation_depth_exceeded")):
                    parent_context.metadata["delegation_depth_exceeded"] = True
                    parent_context.state["delegation_depth_exceeded"] = True
                    _append_arbitration_log(
                        parent_context,
                        {
                            "event": "depth_guard_block",
                            "depth": depth,
                            "max_depth": _MAX_DELEGATION_DEPTH,
                            "source": "subtask",
                            "agent": str(agent_name or ""),
                        },
                    )
            text = str(reply or "").strip()
            if not text:
                return DelegateResult(
                    task_input=subtask_input,
                    agent_name=str(agent_name or "agent"),
                    success=False,
                    payload=None,
                    error=(
                        f"[Sub-task failed: '{_safe_snippet(subtask_input)}' -> "
                        f"{agent_name or 'agent'} returned no output."
                        "]"
                    ),
                )
            return DelegateResult(
                task_input=subtask_input,
                agent_name=str(agent_name or "agent"),
                success=True,
                payload=text,
                error="",
            )
        except Exception as exc:
            logger.warning("Sub-task inference failed (depth=%d, agent=%s): %s", depth, agent_name, exc)
            error_msg = (
                f"[Sub-task failed: '{_safe_snippet(subtask_input)}' -> {str(exc)[:100]}]"
            )
            return DelegateResult(
                task_input=subtask_input,
                agent_name=str(agent_name or "agent"),
                success=False,
                payload=None,
                error=error_msg,
            )

    # ------------------------------------------------------------------
    # Priority 3: Tool calling
    # ------------------------------------------------------------------

    async def _handle_tool_call(self, context: TurnContext, block: dict) -> str:
        """Execute a built-in tool and optionally follow up with inference.

        Tool name and result are stamped into ``context.state`` and
        ``context.metadata``.  If the tool succeeds a follow-up ``run_agent``
        call is made so the LLM can produce a natural-language reply.
        """
        tool_name = str(block.get("name") or "").strip().lower()
        tool_args = dict(block.get("args") or {})
        try:
            tool_result = _dispatch_builtin_tool(tool_name, tool_args, context)
            record_external_system_call(
                operation="inference_tool_call",
                system=f"builtin_tool:{tool_name}",
                request_payload={"args": dict(tool_args)},
                response_payload={"output": tool_result, "status": "ok"},
                status="ok",
                source="inference_node",
                required=False,
            )
        except Exception as exc:
            record_external_system_call(
                operation="inference_tool_call",
                system=f"builtin_tool:{tool_name}",
                request_payload={"args": dict(tool_args)},
                response_payload={"status": "error", "error": str(exc)},
                status="error",
                source="inference_node",
                required=False,
            )
            raise
        context.state["tool_result"] = tool_result
        context.state["tool_called"] = tool_name
        context.metadata["tool_called"] = tool_name
        context.metadata["tool_call_executed"] = True

        run_agent = getattr(self.mgr, "run_agent", None)
        if callable(run_agent) and tool_result:
            follow_input = (
                f"[Tool result for '{tool_name}']: {tool_result}\n\n"
                f"Original question: {context.user_input}"
            )
            follow_ctx = TurnContext(
                user_input=follow_input,
                metadata={
                    "determinism": dict(context.metadata.get("determinism") or {}),
                    "temporal": dict(context.metadata.get("temporal") or {}),
                    "parent_trace_id": context.trace_id,
                    "tool_follow_up": True,
                },
                state={
                    "rich_context": dict(context.state.get("rich_context") or {}),
                    "tool_result": tool_result,
                },
            )
            follow_ctx.temporal = context.temporal
            try:
                follow_raw = await run_agent(
                    follow_ctx, follow_ctx.state.get("rich_context", {})
                )
                follow_reply = (
                    follow_raw[0]
                    if isinstance(follow_raw, tuple)
                    else str(follow_raw or "")
                )
                # Do not recurse on structured output from follow-up calls.
                if _parse_structured_block(follow_reply) is None:
                    return follow_reply
            except Exception as exc:
                logger.warning("Tool follow-up inference failed: %s", exc)
        return str(tool_result)

    # ------------------------------------------------------------------
    # Priority 2: Structured reasoning output
    # ------------------------------------------------------------------

    def _handle_reasoning(self, context: TurnContext, block: dict) -> str:
        """Extract the final reply from a structured reasoning/plan block.

        Full step list is saved to ``context.state["reasoning_steps"]`` for
        downstream inspection, auditing, or re-use.
        """
        steps = list(block.get("steps") or [])
        conclusion = _extract_reasoning_reply(block)
        context.state["reasoning_steps"] = steps
        context.metadata["reasoning_structured"] = True
        context.metadata["reasoning_steps_count"] = len(steps)
        return conclusion or (str(steps[-1]) if steps else "")

    # ------------------------------------------------------------------
    # Main run method
    # ------------------------------------------------------------------

    async def run(self, context: TurnContext) -> TurnContext:
        # Turn-scoped blackboard seed prevents cross-turn leakage.
        seed_board = dict(context.metadata.get("agent_blackboard_seed") or {})
        context.state["agent_blackboard"] = dict(seed_board)
        run_agent = getattr(self.mgr, "run_agent", None)
        if not callable(run_agent):
            raise RuntimeError(
                f"InferenceNode: bound manager {type(self.mgr).__name__!r} does not expose "
                "'run_agent'; only AgentService is a valid inference provider."
            )
        # Priority 4: determinism enforcement (strict-mode LLM knobs).
        determinism = dict(getattr(context, "metadata", {}).get("determinism") or {})
        if bool(determinism.get("enforced", False)):
            runtime_bot = getattr(self.mgr, "bot", None)
            if runtime_bot is not None:
                setattr(runtime_bot, "LLM_TEMPERATURE", 0.0)
                setattr(runtime_bot, "LLM_SEED", 42)
                set_deterministic = getattr(runtime_bot, "set_deterministic", None)
                if callable(set_deterministic):
                    set_deterministic(True)
        rich_context = context.state.get("rich_context", {})
        candidate: Any = None
        for iteration in range(self._max_loop_iterations):
            record_execution_step(
                "iteration_start",
                payload={"iteration": int(iteration)},
                required=False,
            )
            try:
                raw_result = await run_agent(context, rich_context)
                # Resolve structured blocks (delegation / reasoning / tool call).
                candidate = await self._resolve_candidate(context, raw_result, depth=0)
                reply_text = candidate[0] if isinstance(candidate, tuple) else str(candidate or "")
                record_execution_step(
                    "iteration_output",
                    payload={
                        "iteration": int(iteration),
                        "reply_preview": str(reply_text)[:160],
                    },
                    required=False,
                )
            except Exception as exc:
                logger.error("InferenceNode.run_agent failed (iteration %d): %s", iteration, exc)
                candidate = self._fallback_candidate("Something went sideways. Try again in a moment.")
                break

            # Critique loop: evaluate the candidate against the turn plan.
            if self._critique_engine is not None and iteration < self._max_loop_iterations - 1:
                turn_plan = dict(context.state.get("turn_plan") or {})
                reply_text = candidate[0] if isinstance(candidate, tuple) else str(candidate or "")
                try:
                    try:
                        critique = self._critique_engine.critique(
                            reply_text,
                            context.user_input,
                            turn_plan,
                            iteration,
                            tool_ir=dict(context.state.get("tool_ir") or {}),
                            tool_results=list(context.state.get("tool_results") or []),
                        )
                    except TypeError:
                        critique = self._critique_engine.critique(
                            reply_text,
                            context.user_input,
                            turn_plan,
                            iteration,
                        )
                    critique_passed = bool(getattr(critique, "passed", False))
                    critique_issues = list(getattr(critique, "issues", []) or [])
                    revision_hint = str(getattr(critique, "revision_hint", "") or "")
                    context.state["critique_record"] = {
                        "iteration": iteration,
                        "score": getattr(critique, "score", 0.0),
                        "passed": critique_passed,
                        "issues": critique_issues,
                        "revision_hint": revision_hint,
                        "tool_necessity_score": getattr(critique, "tool_necessity_score", 0.0),
                        "tool_correctness_score": getattr(critique, "tool_correctness_score", 0.0),
                    }
                    record_execution_step(
                        "critique_iteration",
                        payload={
                            "iteration": int(iteration),
                            "passed": critique_passed,
                            "issue_count": len(critique_issues),
                        },
                        required=False,
                    )
                    if critique_passed:
                        break
                    # Inject revision hint so the agent can read it on re-run.
                    context.state["_critique_revision_context"] = revision_hint
                except Exception as exc:
                    logger.warning("CritiqueEngine.critique failed (non-fatal): %s", exc)
                    break
            else:
                break

        context.state.pop("_critique_revision_context", None)
        context.state["candidate"] = candidate
        return context
class SafetyNode:
    """Applies TONY score tone constraints and reply policy enforcement."""

    def __init__(self, safety_manager: Any):
        self.mgr = safety_manager

    async def run(self, context: TurnContext) -> TurnContext:
        # Session exit was already fully handled by InferenceNode -- skip.
        if context.state.get("already_finalized"):
            return context

        candidate = context.state.get("candidate")
        enforce = getattr(self.mgr, "enforce_policies", None)
        if callable(enforce):
            try:
                context.state["safe_result"] = enforce(context, candidate)
            except Exception as exc:
                logger.error("SafetyNode.enforce_policies failed: %s", exc)
                context.state["safe_result"] = candidate
            return context

        validate = getattr(self.mgr, "validate", None)
        if callable(validate):
            try:
                context.state["safe_result"] = validate(candidate)
            except Exception as exc:
                logger.error("SafetyNode.validate failed: %s", exc)
                context.state["safe_result"] = candidate
            return context

        context.state["safe_result"] = candidate
        return context


class SaveNode:
    """Atomically commits history, maintenance, health snapshot, and persistence.

    Contract: All durable mutations MUST go through SaveNode. The graph
    guarantees speculative execution until that boundary.
    """

    node_type = NodeType.COMMIT

    def __init__(self, storage_manager: Any):
        self.mgr = storage_manager

    def _result_from_context(self, context: TurnContext) -> Any:
        return context.state.get("safe_result") or context.state.get("candidate")

    def _finalize_turn(self, context: TurnContext, result: Any) -> bool:
        finalize = getattr(self.mgr, "finalize_turn", None)
        if not callable(finalize):
            raise RuntimeError("SaveNode requires finalize_turn in Phase 4 strict mode")
        try:
            finalized = finalize(context, result)
            context.state["safe_result"] = finalized
            return True
        except Exception as exc:
            logger.error("SaveNode.finalize_turn failed in strict mode: %s", exc)
            raise

    async def run(self, context: TurnContext) -> TurnContext:
        if getattr(context, "temporal", None) is None:
            raise RuntimeError("TemporalNode required — execution invalid")
        context.state.setdefault("temporal", context.temporal_snapshot())
        context.metadata.setdefault("temporal", context.temporal_snapshot())
        result = self._result_from_context(context)
        begin_transaction = getattr(self.mgr, "begin_transaction", None)
        apply_mutations = getattr(self.mgr, "apply_mutations", None)
        commit_transaction = getattr(self.mgr, "commit_transaction", None)
        if not callable(begin_transaction) or not callable(apply_mutations) or not callable(commit_transaction):
            raise RuntimeError("SaveNode requires begin/apply/commit transaction hooks in Phase 4 strict mode")

        begin_transaction(context)
        try:
            apply_mutations(context)
            self._finalize_turn(context, result)
            if not bool(context.state.get("_atomic_checkpoint_saved", False)):
                checkpoint_snapshot = getattr(context, "checkpoint_snapshot", None)
                save_graph_checkpoint = getattr(self.mgr, "save_graph_checkpoint", None)
                if callable(checkpoint_snapshot) and callable(save_graph_checkpoint):
                    checkpoint = checkpoint_snapshot(stage="save", status="atomic_finalize", error=None)
                    save_graph_checkpoint(checkpoint, _skip_turn_event=True)
                    context.state["_atomic_checkpoint_saved"] = True
            commit_transaction(context)
        except Exception:
            rollback_transaction = getattr(self.mgr, "rollback_transaction", None)
            if callable(rollback_transaction):
                rollback_transaction(context)
            raise
        return context


class ReflectionNode:
    """Runs optional post-save reflection hooks without impacting reply continuity."""

    name = "reflection"

    def __init__(self, reflection_manager: Any):
        self.mgr = reflection_manager

    async def run(self, context: TurnContext) -> TurnContext:
        if self.mgr is None:
            return context

        result = context.state.get("safe_result") or context.state.get("candidate")
        turn_text = context.state.get("turn_text") or context.user_input
        current_mood = context.state.get("mood") or "neutral"
        reply_text = result[0] if isinstance(result, tuple) else str(result or "")

        reflect_after_turn = getattr(self.mgr, "reflect_after_turn", None)
        if callable(reflect_after_turn):
            try:
                context.state["reflection"] = reflect_after_turn(turn_text, current_mood, reply_text)
            except TypeError:
                context.state["reflection"] = reflect_after_turn(context, result)
            except Exception as exc:
                logger.warning("ReflectionNode.reflect_after_turn failed (non-fatal): %s", exc)
            return context

        reflect = getattr(self.mgr, "reflect", None)
        if callable(reflect):
            try:
                context.state["reflection"] = reflect(context, result)
            except TypeError:
                try:
                    context.state["reflection"] = reflect(force=True)
                except TypeError:
                    context.state["reflection"] = reflect(context)
                except Exception as exc:
                    logger.warning("ReflectionNode.reflect failed (non-fatal): %s", exc)
            except Exception as exc:
                logger.warning("ReflectionNode.reflect failed (non-fatal): %s", exc)
        return context


