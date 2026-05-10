from __future__ import annotations

import asyncio
import json
import logging
import warnings
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from dadbot.contracts import (
    AttachmentList,
    ChunkCallback,
    DadBotContext,
    FinalizedTurnResult,
    PreparedTurnResult,
    SupportsTurnProcessingRuntime,
)
from dadbot.core.execution_context import ensure_execution_trace_root
from dadbot.core.execution_result_unified import get_unified_execution_result
from dadbot.core.memory_set_invariants import (
    assert_memory_set_invariants,
    record_causal_step_locked,
    MemorySetInvariantViolation,
)
from dadbot.core.system_state_algebra import (
    evaluate_system_state_algebra,
    persist_system_state_algebra,
)
from dadbot.core.graph import (
    LedgerMutationOp,
    MutationIntent,
    MutationKind,
    TurnContext,
)
from dadbot.core.tool_executor import execute_tool
from dadbot.core.turn_coherence import mark_turn_coherence, reset_turn_coherence
from dadbot.managers.reply_generation import ReplyGenerationManager
from dadbot.models import AgenticToolPlan
from dadbot.services.memory_service import MemoryService
from dadbot.services.llm_call_adapter import LLMCallAdapter
from dadbot.services.turn_state_mutator import TurnStateMutator

# Keep the historic logger name so existing tests and log pipelines remain stable
# while the implementation moves from managers/ to services/.
logger = logging.getLogger("dadbot.managers.turn_processing")

_EVENT_LOOP_CLOSED_ERROR = "Event loop is closed"
_SET_REMINDER_TOOL = "set_reminder"
_WEB_SEARCH_TOOL = "web_search"
_TOOL_VISIBILITY_SETTINGS = {
    _SET_REMINDER_TOOL: "auto_reminders",
    _WEB_SEARCH_TOOL: "auto_web_lookup",
}
_TOOL_NAME_ALIASES = {
    "web_lookup": _WEB_SEARCH_TOOL,
    "websearch": _WEB_SEARCH_TOOL,
    "web-search": _WEB_SEARCH_TOOL,
    "web search": _WEB_SEARCH_TOOL,
    "set reminder": _SET_REMINDER_TOOL,
    "set-reminder": _SET_REMINDER_TOOL,
}
_REMINDER_TOOL_NAMES = frozenset({_SET_REMINDER_TOOL})
_DEFER_TOOL_BIASES = frozenset({"defer_tools_unless_explicit"})


class _ReflectionDecision(BaseModel):
    sufficient: bool
    refined_query: str | None = None
    reason: str = ""


class _PlannerDecision(BaseModel):
    needs_tool: bool
    tool: str | None = None
    parameters: dict[str, object] | None = None
    reason: str = ""


def _is_event_loop_closed_error(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and str(exc).find(_EVENT_LOOP_CLOSED_ERROR) != -1


class TurnService:
    """Own per-turn preparation, reply generation, and finalization flows.

    This replaces the old TurnProcessingManager. The concrete implementation
    now lives under dadbot.services and is exposed through ``turn_service``.
    """

    def __init__(self, bot: DadBotContext | SupportsTurnProcessingRuntime):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot
        self.reply_generation = ReplyGenerationManager(self.context)
        self._llm_adapter = LLMCallAdapter(self.bot)
        self._memory_service = MemoryService(self.bot)
        self._state_mutator = TurnStateMutator(self.bot)

    def _pipeline_timestamp(self) -> str:
        return self._state_mutator.pipeline_timestamp()

    def _store_turn_pipeline(self, payload: dict[str, object]) -> dict[str, object]:
        return self._state_mutator.store_turn_pipeline(payload)

    def _start_turn_pipeline(self, mode: str, user_input: str) -> dict[str, object]:
        return self._state_mutator.start_turn_pipeline(mode, user_input)

    def _update_turn_pipeline(self, **fields) -> dict[str, object] | None:
        return self._state_mutator.update_turn_pipeline(**fields)

    def _append_turn_pipeline_step(
        self,
        name: str,
        status: str = "completed",
        detail: str = "",
        **metadata,
    ) -> dict[str, object] | None:
        return self._state_mutator.append_turn_pipeline_step(
            name,
            status,
            detail,
            **metadata,
        )

    def _complete_turn_pipeline(
        self,
        *,
        final_path: str = "",
        reply_source: str = "",
        should_end: bool = False,
        error: str = "",
    ) -> dict[str, object] | None:
        return self._state_mutator.complete_turn_pipeline(
            final_path=final_path,
            reply_source=reply_source,
            should_end=should_end,
            error=error,
        )

    def turn_pipeline_snapshot(self):
        return self._state_mutator.turn_pipeline_snapshot()

    def should_offer_daily_checkin_for_turn(self) -> bool:
        return self.bot.memory.should_do_daily_checkin() and self.bot.session_turn_count() == 0

    def record_user_turn_state(
        self,
        stripped_input: str,
        current_mood: str,
        turn_context: Any | None = None,
    ) -> None:
        should_offer_daily_checkin = self.should_offer_daily_checkin_for_turn()

        # If turn_context is provided, use the strict-mode path through mutation queue
        if turn_context is not None:
            mutation_queue = getattr(turn_context, "mutation_queue", None)
            if mutation_queue is None:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Strict mode turn_context lacks mutation_queue; using fallback",
                )
                # Fall through to direct write below
            else:
                try:
                    mutation_queue.queue(
                        MutationIntent(
                            type=MutationKind.LEDGER,
                            payload={
                                "op": LedgerMutationOp.RECORD_TURN_STATE.value,
                                "mood": str(current_mood or "neutral"),
                                "should_offer_daily_checkin": bool(
                                    should_offer_daily_checkin,
                                ),
                            },
                            requires_temporal=False,
                            source="turn_service.record_user_turn_state",
                        ),
                    )

                    # Queue mood mutation for SaveNode drain at the turn commit boundary.
                    # All persistent mood writes flow through state["_pending_mood_updates"] →
                    # PersistenceService._apply_pending_save_boundary_mutations → SaveNode.
                    queued_mood = str(current_mood or "neutral")
                    state = getattr(turn_context, "state", None)
                    if isinstance(state, dict):
                        pending_moods = list(state.get("_pending_mood_updates") or [])
                        pending_moods.append({"mood": queued_mood})
                        state["_pending_mood_updates"] = pending_moods
                    # Preserve same-turn UX behavior: tone blending depends on this
                    # transient flag before SaveNode drains queued mutations.
                    self.bot._last_should_offer_daily_checkin = bool(
                        should_offer_daily_checkin,
                    )
                    self.bot._pending_daily_checkin_context = bool(
                        should_offer_daily_checkin,
                    )
                    return
                except RuntimeError as guard_exc:
                    # MutationGuard violation: mutations locked outside SaveNode.
                    # Gracefully fall through to compatibility path for legacy callers.
                    if "MutationGuard" in str(guard_exc):
                        logger = logging.getLogger(__name__)
                        logger.debug(
                            f"MutationGuard prevented recording outside SaveNode; using fallback: {guard_exc}",
                        )
                    else:
                        raise

        # Fallback: direct write when turn_context is not available (e.g., in tests)
        # This maintains compatibility with legacy code paths.
        # All direct attribute writes are owned by TurnStateMutator.
        self._state_mutator.write_mood_fallback(
            current_mood,
            should_offer_daily_checkin,
        )

    def direct_reply_for_input(
        self,
        stripped_input: str,
        current_mood: str,
    ) -> str | None:
        parse_tool_command = getattr(self.bot, "parse_tool_command", None)
        parsed_tool_command = parse_tool_command(stripped_input) if callable(parse_tool_command) else None
        parsed_action = str((parsed_tool_command or {}).get("action") or "").strip()
        # Executable tool commands must flow through planner->executor coupling.
        if parsed_action in {_SET_REMINDER_TOOL, "web_lookup"}:
            return None

        crisis_reply = self.bot.safety_support.direct_reply_for_input(stripped_input)
        if crisis_reply is not None:
            return crisis_reply

        for direct_reply in (
            self.bot.handle_memory_command(stripped_input),
            self.bot.handle_tool_command(stripped_input),
            self.bot.get_memory_reply(stripped_input),
            self.bot.get_fact_reply(stripped_input),
        ):
            if direct_reply is not None:
                return self.bot.reply_finalization.finalize(
                    direct_reply,
                    current_mood,
                    stripped_input,
                )
        return None

    def _deterministic_tool_route(self, stripped_input: str) -> tuple[str, dict[str, object], str] | None:
        parse_tool_command = getattr(self.bot, "parse_tool_command", None)
        if not callable(parse_tool_command):
            return None

        command = parse_tool_command(stripped_input)
        if not isinstance(command, dict):
            return None

        action = str(command.get("action") or "").strip()
        if action == _SET_REMINDER_TOOL:
            params = {
                "title": str(command.get("title") or stripped_input[:100]).strip(),
                "due_text": str(command.get("due_text") or "").strip(),
            }
            return _SET_REMINDER_TOOL, params, "Deterministic reminder command routed to executor."

        if action == "web_lookup":
            query = str(command.get("query") or stripped_input).strip()
            if not query:
                return None
            return _WEB_SEARCH_TOOL, {"query": query}, "Deterministic web lookup command routed to executor."

        return None

    def _available_agentic_tools(
        self,
        settings: dict[str, object],
    ) -> list[dict[str, object]]:
        tools = []
        for tool in self.bot.get_available_tools():
            name = str(tool.get("function", {}).get("name") or "").strip()
            setting_key = _TOOL_VISIBILITY_SETTINGS.get(name)
            if setting_key and not bool(settings.get(setting_key)):
                continue
            tools.append(tool)
        return tools

    def _record_tool_decision_outcome(
        self,
        *,
        turn_context: Any | None,
        decision_outcome: str,
    ) -> None:
        if turn_context is None:
            return
        state = getattr(turn_context, "state", None)
        if not isinstance(state, dict):
            return
        state["decision_outcome"] = str(decision_outcome)
        state["robustness_suppressed"] = str(decision_outcome) == "robustness_suppressed"

    @staticmethod
    def _extract_executed_tools_from_state(state: dict[str, Any]) -> list[str]:
        events = list(state.get("sovereign_events") or [])
        tool_names: list[str] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            name = payload.get("tool_name")
            if not name:
                nested = payload.get("tool_execution")
                if isinstance(nested, dict):
                    name = nested.get("tool_name") or nested.get("tool")
            if not name:
                name = payload.get("tool")
            normalized = str(name or "").strip()
            if not normalized:
                continue
            if normalized not in tool_names:
                tool_names.append(normalized)
        return tool_names

    def build_execution_truth_contract(self, turn_context: Any | None) -> dict[str, Any]:
        base_contract = {
            "version": "execution-truth-v1",
            "consistent": True,
            "failure_code": "ok",
            "reason": "",
            "decision_outcome": "no_tool_needed",
            "planner_status": "",
            "planner_tool": "",
            "executed_tools": [],
            "executed_tool_count": 0,
        }
        if turn_context is None:
            return base_contract

        state = getattr(turn_context, "state", None)
        if not isinstance(state, dict):
            return {
                **base_contract,
                "consistent": False,
                "failure_code": "missing_turn_state",
                "reason": "Turn context state is missing or invalid.",
            }

        planner_snapshot = self.bot.planner_debug_snapshot()
        decision_outcome = str(state.get("decision_outcome") or "no_tool_needed").strip().lower()
        planner_status = str(planner_snapshot.get("planner_status") or "").strip().lower()
        planner_tool = str(planner_snapshot.get("planner_tool") or "").strip()
        executed_tools = self._extract_executed_tools_from_state(state)

        consistent = True
        failure_code = "ok"
        reason = ""

        if planner_status == "execution_contract_violation":
            consistent = False
            failure_code = "contract_violation"
            reason = "Planner reported execution_contract_violation before commit."
        elif decision_outcome == "executed_tool":
            if not planner_tool:
                consistent = False
                failure_code = "selection_missing"
                reason = "decision_outcome=executed_tool but planner_tool is empty."
            elif not executed_tools:
                consistent = False
                failure_code = "execution_missing"
                reason = "decision_outcome=executed_tool but no execution event was recorded."
            elif planner_tool not in executed_tools:
                consistent = False
                failure_code = "selection_execution_mismatch"
                reason = (
                    "planner_tool does not match executed tools "
                    f"(planner_tool={planner_tool}, executed={executed_tools})."
                )
        elif decision_outcome == "no_tool_needed":
            if executed_tools:
                consistent = False
                failure_code = "unexpected_execution"
                reason = (
                    "decision_outcome=no_tool_needed but tool execution events were recorded "
                    f"({executed_tools})."
                )
        elif decision_outcome == "robustness_suppressed":
            if executed_tools:
                consistent = False
                failure_code = "suppression_bypass"
                reason = (
                    "decision_outcome=robustness_suppressed but tool execution events were recorded "
                    f"({executed_tools})."
                )
        else:
            consistent = False
            failure_code = "unknown_decision_outcome"
            reason = f"Unrecognized decision_outcome '{decision_outcome}'."

        contract = {
            "version": "execution-truth-v1",
            "consistent": bool(consistent),
            "failure_code": str(failure_code),
            "reason": str(reason),
            "decision_outcome": str(decision_outcome),
            "planner_status": str(planner_status),
            "planner_tool": str(planner_tool),
            "executed_tools": list(executed_tools),
            "executed_tool_count": len(executed_tools),
        }
        state["execution_truth_contract"] = dict(contract)
        return contract

    def validate_execution_truth_contract(
        self,
        turn_context: Any | None,
        *,
        enforce: bool = True,
    ) -> dict[str, Any]:
        contract = self.build_execution_truth_contract(turn_context)
        if bool(enforce) and not bool(contract.get("consistent", False)):
            failure_code = str(contract.get("failure_code") or "unknown")
            reason = str(contract.get("reason") or "")
            raise RuntimeError(
                f"Execution truth contract violation ({failure_code}): {reason}",
            )
        return contract

    def resolve_turn_truth(self, turn_context: Any | None) -> dict[str, Any]:
        """Resolve turn truth via canonical system-state algebra projections."""
        state = getattr(turn_context, "state", None) if turn_context else None
        if not isinstance(state, dict):
            state = {}

        execution_contract = self.build_execution_truth_contract(turn_context)
        execution_result = get_unified_execution_result(turn_context)
        trace_id = str(getattr(turn_context, "trace_id", "") or "") if turn_context else ""
        algebra = evaluate_system_state_algebra(
            state=state,
            execution_result_payload=execution_result,
            trace_id=trace_id,
            context="resolve_turn_truth",
        )

        violations = list(algebra.get("violations") or [])
        memory_axis = dict(algebra.get("axes", {}).get("memory") or {})
        causal_axis = dict(algebra.get("axes", {}).get("causal") or {})
        memory_detail = str(memory_axis.get("inline_violation") or "")
        memory_ok = not any("memory_invariant" in str(item) for item in violations)
        causal_ok = not any("Causal order violation" in str(item) for item in violations)

        truth = {
            "overall_consistent": bool(algebra.get("overall_consistent", False)),
            "execution": execution_contract,
            "memory_invariants": {"ok": memory_ok, "detail": memory_detail},
            "causal_order": {
                "ok": causal_ok,
                "detail": "" if causal_ok else "; ".join(
                    str(item) for item in violations if "Causal order violation" in str(item)
                ),
                "steps": list(causal_axis.get("steps") or []),
            },
            "violations": violations,
            "cognitive_authority_boundary": dict(
                algebra.get("axes", {}).get("cognitive_authority_boundary") or {}
            ),
        }
        if isinstance(state, dict):
            persist_system_state_algebra(
                state=state,
                algebra=algebra,
                trace_context="resolve_turn_truth",
                persist_legacy_projections=True,
                terminal_snapshot=False,
            )
        return truth

    def _parse_structured_json(
        self,
        *,
        content: str,
        schema: type[BaseModel],
        failure_status: str,
        failure_reason: str,
    ) -> BaseModel | None:
        try:
            payload = self.bot.parse_model_json_content(content)
        except (TypeError, json.JSONDecodeError, KeyError):
            self.bot.update_planner_debug(
                planner_status=failure_status,
                planner_reason=failure_reason,
            )
            return None
        if not isinstance(payload, dict):
            self.bot.update_planner_debug(
                planner_status=failure_status,
                planner_reason=failure_reason,
            )
            return None
        try:
            return schema.model_validate(payload)
        except ValidationError:
            self.bot.update_planner_debug(
                planner_status=failure_status,
                planner_reason=failure_reason,
            )
        return None

    async def _call_json_with_validation_async(
        self,
        *,
        messages: list[dict[str, str]],
        purpose: str,
        schema: type[BaseModel],
        max_parse_attempts: int = 2,
    ) -> BaseModel | None:
        for parse_attempt in range(max(1, int(max_parse_attempts))):
            try:
                response = await self._llm_adapter.call_async(
                    messages=messages,
                    options={"temperature": 0.1},
                    response_format="json",
                    purpose=purpose,
                )
            except Exception as exc:
                logger.warning(
                    "Structured LLM call failed (%s, attempt %d): %s",
                    purpose,
                    parse_attempt + 1,
                    exc,
                )
                continue
            parsed = self._parse_structured_json(
                content=str(response.get("message", {}).get("content") or ""),
                schema=schema,
                failure_status="fallback",
                failure_reason="Model returned invalid structured JSON.",
            )
            if parsed is not None:
                return parsed
        return None

    def _call_json_with_validation_sync(
        self,
        *,
        messages: list[dict[str, str]],
        purpose: str,
        schema: type[BaseModel],
        max_parse_attempts: int = 2,
    ) -> BaseModel | None:
        for parse_attempt in range(max(1, int(max_parse_attempts))):
            try:
                response = self._llm_adapter.call(
                    messages=messages,
                    options={"temperature": 0.1},
                    response_format="json",
                    purpose=purpose,
                )
            except Exception as exc:
                logger.warning(
                    "Structured LLM call failed (%s, attempt %d): %s",
                    purpose,
                    parse_attempt + 1,
                    exc,
                )
                continue
            parsed = self._parse_structured_json(
                content=str(response.get("message", {}).get("content") or ""),
                schema=schema,
                failure_status="fallback",
                failure_reason="Model returned invalid structured JSON.",
            )
            if parsed is not None:
                return parsed
        return None

    def _reflection_prompt(
        self,
        original_input: str,
        query: str,
        observation: str,
    ) -> str:
        prompt_budget_fn = getattr(getattr(self.bot, "prompt_assembly", None), "_compute_prompt_budget", None)
        if callable(prompt_budget_fn):
            prompt_budget = max(128, int(prompt_budget_fn() or 0))
        else:
            context_budget = max(
                256,
                int(
                    self.bot.effective_context_token_budget(getattr(self.bot, "ACTIVE_MODEL", None))
                    or getattr(self.bot, "CONTEXT_TOKEN_BUDGET", 0)
                    or 0
                ),
            )
            reserved_tokens = max(64, int(getattr(self.bot, "RESERVED_RESPONSE_TOKENS", 0) or 0))
            prompt_budget = max(128, context_budget - reserved_tokens)

        input_text = str(original_input or "").strip()
        query_text = str(query or "").strip()
        observation_text = str(observation or "").strip()

        def _render_prompt(current_observation: str) -> str:
            return (
                "You are checking whether a web lookup is good enough before Dad replies to Tony.\n\n"
                f"User request: {input_text}\n"
                f"Current search query: {query_text}\n"
                "Observed result:\n"
                f"{current_observation}\n\n"
                "Return ONLY valid JSON with this shape:\n"
                '{"sufficient": true or false, "refined_query": "better query" or null, "reason": "short explanation"}\n\n'
                "Rules:\n"
                "- Set sufficient=true when the observed result is good enough to answer naturally.\n"
                "- Set sufficient=false only when a more specific query is needed.\n"
                "- If sufficient=false, provide a refined_query that is different from the current search query.\n"
                "- Keep reason short and concrete."
            )

        candidate_observation = observation_text
        message = {"role": "user", "content": _render_prompt(candidate_observation)}
        while self.bot.message_token_cost(message) > prompt_budget and len(candidate_observation) > 96:
            candidate_observation = candidate_observation[: max(96, int(len(candidate_observation) * 0.75))].rstrip()
            if not candidate_observation.endswith("..."):
                candidate_observation += "..."
            message = {"role": "user", "content": _render_prompt(candidate_observation)}
        return str(message["content"])

    @staticmethod
    def _reminder_confirmation_reply(reminder: Any) -> str:
        reminder_dict = dict(reminder or {}) if isinstance(reminder, dict) else {}
        title = str(reminder_dict.get("title") or "that").strip() or "that"
        due_text = str(reminder_dict.get("due_text") or "").strip()
        if due_text:
            return f"I set that reminder for you: {title} ({due_text})."
        return f"I set that reminder for you: {title}."

    def _reflect_on_web_observation(
        self,
        original_input: str,
        query: str,
        observation: str,
        settings: dict[str, object],
        max_retries: int = 2,
    ) -> str:
        current_query = query
        current_observation = observation
        attempts_used = 0
        retry_count = 0

        for attempt in range(max_retries):
            attempts_used = attempt + 1
            reflection = self._call_json_with_validation_sync(
                messages=[
                    {
                        "role": "user",
                        "content": self._reflection_prompt(
                            original_input,
                            current_query,
                            current_observation,
                        ),
                    },
                ],
                purpose="agentic reflection",
                schema=_ReflectionDecision,
            )
            if reflection is None:
                break

            if reflection.sufficient:
                logger.info(
                    "Agentic web reflection accepted observation on attempt %d/%d",
                    attempt + 1,
                    max_retries,
                )
                self.bot.update_planner_debug(
                    planner_status="reflected",
                    planner_reason=f"Reflection pass {attempt + 1}: observation accepted. {reflection.reason}",
                )
                break

            refined_query = str(reflection.refined_query or "").strip()
            if not refined_query or refined_query == current_query:
                break

            self.bot.update_planner_debug(
                planner_status="refining",
                planner_reason=f"Reflection pass {attempt + 1}: refining query. {reflection.reason}",
                planner_parameters={"query": refined_query},
            )
            retry_count += 1
            logger.info(
                "Agentic web reflection retry %d/%d with refined query: %s",
                retry_count,
                max_retries,
                refined_query,
            )

            result = self.bot.lookup_web(refined_query)
            if result:
                source = f" Source: {result['source_label']}." if result.get("source_label") else ""
                current_observation = f"{result['heading']}: {result['summary']}{source}"
                current_query = refined_query
            else:
                logger.info(
                    "Agentic web reflection retry %d/%d returned no result for query: %s",
                    retry_count,
                    max_retries,
                    refined_query,
                )
                break

        logger.info(
            "Agentic web reflection completed after %d/%d attempts with %d retries",
            attempts_used,
            max_retries,
            retry_count,
        )

        return current_observation

    async def _reflect_on_web_observation_async(
        self,
        original_input: str,
        query: str,
        observation: str,
        settings: dict[str, object],
        max_retries: int = 2,
    ) -> str:
        current_query = query
        current_observation = observation
        attempts_used = 0
        retry_count = 0

        for attempt in range(max_retries):
            attempts_used = attempt + 1
            reflection = await self._call_json_with_validation_async(
                messages=[
                    {
                        "role": "user",
                        "content": self._reflection_prompt(
                            original_input,
                            current_query,
                            current_observation,
                        ),
                    },
                ],
                purpose="agentic reflection",
                schema=_ReflectionDecision,
            )
            if reflection is None:
                break

            if reflection.sufficient:
                logger.info(
                    "Agentic web reflection accepted observation on attempt %d/%d",
                    attempt + 1,
                    max_retries,
                )
                self.bot.update_planner_debug(
                    planner_status="reflected",
                    planner_reason=f"Reflection pass {attempt + 1}: observation accepted. {reflection.reason}",
                )
                break

            refined_query = str(reflection.refined_query or "").strip()
            if not refined_query or refined_query == current_query:
                break

            self.bot.update_planner_debug(
                planner_status="refining",
                planner_reason=f"Reflection pass {attempt + 1}: refining query. {reflection.reason}",
                planner_parameters={"query": refined_query},
            )
            retry_count += 1
            logger.info(
                "Agentic web reflection retry %d/%d with refined query: %s",
                retry_count,
                max_retries,
                refined_query,
            )

            result = self.bot.lookup_web(refined_query)
            if result:
                source = f" Source: {result['source_label']}." if result.get("source_label") else ""
                current_observation = f"{result['heading']}: {result['summary']}{source}"
                current_query = refined_query
            else:
                logger.info(
                    "Agentic web reflection retry %d/%d returned no result for query: %s",
                    retry_count,
                    max_retries,
                    refined_query,
                )
                break

        logger.info(
            "Agentic web reflection completed after %d/%d attempts with %d retries",
            attempts_used,
            max_retries,
            retry_count,
        )

        return current_observation

    @staticmethod
    def _format_retrieval_hints(retrieval_set: list[dict]) -> str:
        fragments = []
        for item in list(retrieval_set or [])[:5]:
            summary = str(item.get("summary") or item.get("content") or "").strip()
            if summary:
                fragments.append(f"  - {summary[:120]}")
        if not fragments:
            return ""
        return "\nRetrieved memory signals for this turn:\n" + "\n".join(fragments)

    @staticmethod
    def _format_memory_brief(memory_influence_brief: str | None) -> str:
        text = str(memory_influence_brief or "").strip()
        if not text:
            return ""
        return f"\n{text}"

    @staticmethod
    def _normalize_tool_name(tool_name: str | None) -> str:
        normalized = str(tool_name or "").strip().lower()
        return _TOOL_NAME_ALIASES.get(normalized, normalized)

    def _normalize_tool_list(self, value: Any) -> set[str]:
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, (list, tuple, set, frozenset)):
            candidates = list(value)
        else:
            candidates = []
        normalized = {
            self._normalize_tool_name(str(item or ""))
            for item in candidates
        }
        return {item for item in normalized if item}

    def _extract_memory_authority(self, turn_context: Any | None) -> dict[str, Any]:
        state = getattr(turn_context, "state", None)
        retrieval_set = list((state or {}).get("memory_retrieval_set") or [])
        blocked_tools: set[str] = set()
        allowed_tools: set[str] = set()
        require_clarification = False
        evidence: list[str] = []

        for item in retrieval_set:
            if not isinstance(item, dict):
                continue
            policy_blocks = [
                item.get("tool_policy"),
                item.get("tool_constraints"),
                item.get("constraints"),
                item.get("policy"),
            ]
            for policy in policy_blocks:
                if not isinstance(policy, dict):
                    continue
                blocked_tools.update(
                    self._normalize_tool_list(
                        policy.get("forbid_tools")
                        or policy.get("disallow_tools")
                        or policy.get("blocked_tools"),
                    ),
                )
                allowed_tools.update(
                    self._normalize_tool_list(
                        policy.get("allow_tools")
                        or policy.get("allowed_tools")
                        or policy.get("permitted_tools"),
                    ),
                )
                if bool(
                    policy.get("require_clarification")
                    or policy.get("clarification_required")
                    or policy.get("ask_clarifying_question"),
                ):
                    require_clarification = True
                policy_reason = str(policy.get("reason") or "").strip()
                if policy_reason:
                    evidence.append(policy_reason)

            text_blob = " ".join(
                str(item.get(key) or "")
                for key in ("summary", "content", "title", "memory_influence")
            ).strip().lower()
            if not text_blob:
                continue
            if any(token in text_blob for token in ("do not use web", "don't use web", "no web search")):
                blocked_tools.add(_WEB_SEARCH_TOOL)
            if any(token in text_blob for token in ("do not set reminder", "don't set reminder", "no reminder")):
                blocked_tools.add(_SET_REMINDER_TOOL)

        return {
            "blocked_tools": sorted(blocked_tools),
            "allowed_tools": sorted(allowed_tools),
            "require_clarification": bool(require_clarification),
            "evidence": evidence[:3],
            "retrieval_count": len([item for item in retrieval_set if isinstance(item, dict)]),
        }

    def _memory_authority_decision(
        self,
        *,
        turn_context: Any | None,
        tool_name: str,
        plan_reason: str,
    ) -> tuple[bool, str]:
        authority = self._extract_memory_authority(turn_context)
        normalized_tool = self._normalize_tool_name(tool_name)
        blocked = set(authority.get("blocked_tools") or [])
        allowed = set(authority.get("allowed_tools") or [])

        state = getattr(turn_context, "state", None)
        if isinstance(state, dict):
            state["memory_authority"] = {
                **authority,
                "selected_tool": normalized_tool,
                "vetoed": False,
                "veto_reason": "",
            }

        veto_reason = ""
        if normalized_tool and normalized_tool in blocked:
            veto_reason = (
                f"Memory authority blocked tool '{normalized_tool}' for this turn. "
                "Ask a clarifying question before using tools."
            )
        elif normalized_tool and allowed and normalized_tool not in allowed:
            veto_reason = (
                f"Memory authority requires tools {sorted(allowed)}, so '{normalized_tool}' is blocked. "
                "Ask a clarifying question before using tools."
            )

        if veto_reason and isinstance(state, dict):
            state["memory_authority"] = {
                **dict(state.get("memory_authority") or {}),
                "vetoed": True,
                "veto_reason": veto_reason,
            }

        if veto_reason:
            reason_prefix = str(plan_reason or "").strip()
            if reason_prefix:
                return False, f"{reason_prefix} | {veto_reason}"
            return False, veto_reason

        return True, str(plan_reason or "").strip() or "Memory authority permits tool execution."

    @staticmethod
    def _planning_prompt(
        stripped_input: str,
        current_mood: str,
        tools: list[dict[str, object]],
        shared_context: str,
        memory_influence_brief: str | None = None,
        retrieval_set: list[dict] | None = None,
    ) -> str:
        memory_hints = TurnService._format_retrieval_hints(retrieval_set or [])
        memory_brief = TurnService._format_memory_brief(memory_influence_brief)
        return f"""
You are Dad helping Tony. Think carefully about his latest message.

Shared turn context used for response generation:
{shared_context}{memory_brief}{memory_hints}

Memory decision constraints:
- Retrieved memories are active constraints, not optional decoration.
- If a retrieved memory directly bears on the user's request, let it shape tool choice and parameters.
- If memory and the message conflict, preserve the memory fact and ask for clarification rather than guessing.

Message: "{stripped_input}"

Current mood: {current_mood}

Available tools:
{json.dumps(tools, indent=2)}

Decide if you should use a tool before replying:
- set_reminder: if he mentions something he wants to remember or do later
- web_search: if he asks for facts, weather, news, how-to, or current info

Return ONLY valid JSON (no extra text):

{{
  "needs_tool": true or false,
  "tool": "set_reminder" or "web_search" or null,
  "parameters": {{ ... }} or null,
  "reason": "short one-sentence explanation"
}}
""".strip()

    def _parse_agentic_tool_plan(self, content: str) -> AgenticToolPlan | None:
        parsed = self._parse_structured_json(
            content=content,
            schema=_PlannerDecision,
            failure_status="fallback",
            failure_reason="Planner returned invalid JSON, so heuristic fallback stayed available.",
        )
        if parsed is None:
            return None

        tool_name = str(parsed.tool or "").strip()
        payload = {
            "needs_tool": bool(parsed.needs_tool),
            "tool": tool_name if self.bot.authorize_tool_execution(tool_name) else None,
            "parameters": dict(parsed.parameters or {}),
            "reason": str(parsed.reason or "").strip(),
        }
        try:
            return AgenticToolPlan.model_validate(payload)
        except ValidationError:
            self.bot.update_planner_debug(
                planner_status="fallback",
                planner_reason="Planner returned an invalid tool decision, so heuristic fallback stayed available.",
            )
            return None

    def _refresh_memory_retrieval_after_tool(
        self,
        *,
        turn_context: Any | None,
        user_input: str,
        tool_feedback: str,
    ) -> None:
        if turn_context is None:
            return
        state = getattr(turn_context, "state", None)
        if not isinstance(state, dict):
            return

        query = " ".join(
            part
            for part in [
                str(user_input or "").strip(),
                str(tool_feedback or "").strip(),
                str(state.get("memory_influence_brief") or "").strip(),
            ]
            if part
        ).strip()

        refreshed = self._memory_service.retrieve_for_query(
            query=query,
            graph_limit=3,
            memory_limit=5,
        )

        if not refreshed:
            return

        existing = [dict(item) for item in list(state.get("memory_retrieval_set") or []) if isinstance(item, dict)]
        merged, reconciliation = self._memory_service.merge_retrieval_sets(
            existing,
            refreshed,
            source_labels=["pre_turn", "post_tool"],
        )

        # Gap A: enforce shrink/salience invariants at every merge boundary,
        # not only at tool-gate time.  Soft violation: log and annotate state
        # rather than aborting the turn, since post-tool refresh is best-effort.
        try:
            assert_memory_set_invariants(
                existing,
                merged,
                context="post_tool_refresh",
            )
        except MemorySetInvariantViolation as inv_exc:
            logger.warning("Memory set invariant violation after tool refresh: %s", inv_exc)
            state["memory_invariant_violation"] = str(inv_exc)

        if not list(state.get("_causal_step_log") or []):
            # Some test/direct-entry paths call post-tool refresh without routing
            # through planner/tool-execution wrappers. Seed the minimal causal spine.
            record_causal_step_locked(state, "retrieval", context="turn_service.refresh.bootstrap")
            record_causal_step_locked(state, "planning", context="turn_service.refresh.bootstrap")
            record_causal_step_locked(state, "tool_execution", context="turn_service.refresh.bootstrap")
        record_causal_step_locked(state, "post_tool_refresh", context="turn_service.post_tool_refresh")

        state["memory_retrieval_set"] = merged
        state["memory_reconciliation"] = reconciliation
        state["memory_retrieval_refined"] = True
        state["memory_influence_brief"] = state.get("memory_influence_brief") or ""

    def _execute_set_reminder_tool_sync(
        self,
        *,
        params: dict[str, object],
        stripped_input: str,
        current_mood: str,
        plan_reason: str,
        turn_context: Any | None = None,
    ) -> tuple[str | None, str | None]:
        title = str(params.get("title") or stripped_input[:100]).strip()
        due_text = str(params.get("due_text") or "").strip()

        # _result_holder lets the compensating action reference the execution
        # result without capturing a ToolSandbox instance directly.  It is
        # populated after execute_tool() returns; the compensating action is
        # only ever called on an explicit rollback, which never precedes the
        # execute_tool() call returning.
        _result_holder: list = []

        def _executor():
            return self.bot.add_reminder(title, due_text)

        def _compensate():
            created = _result_holder[0] if _result_holder else None
            if isinstance(created, dict) and created.get("id"):
                delete_fn = getattr(self.bot, "delete_reminder", None)
                if callable(delete_fn):
                    delete_fn(str(created["id"]))

        record = execute_tool(
            tool_name="set_reminder",
            parameters=dict(params),
            executor=_executor,
            compensating_action=_compensate,
            turn_context=turn_context,
        )
        _result_holder.append(record.result)
        if record.status == "failed":
            self.bot.update_planner_debug(
                planner_status="execution_failed",
                planner_reason=f"set_reminder raised during execution: {record.error}",
                planner_tool="set_reminder",
                planner_parameters=params,
            )
            failure_reply = "I tried to set that reminder, but something failed during execution."
            return self.bot.reply_finalization.finalize(
                failure_reply,
                current_mood,
                stripped_input,
            ), None
        reminder = record.result
        if reminder:
            self.bot.update_planner_debug(
                planner_status="used_tool",
                planner_reason=plan_reason or "Planner selected a reminder tool.",
                planner_tool="set_reminder",
                planner_parameters=params,
                final_path="planner_tool",
            )
            reply = self._reminder_confirmation_reply(reminder)
            return self.bot.reply_finalization.finalize(
                reply,
                current_mood,
                stripped_input,
            ), None
        self.bot.update_planner_debug(
            planner_status="execution_failed",
            planner_reason="Planner selected set_reminder, but Dad couldn't create the reminder cleanly.",
            planner_tool="set_reminder",
            planner_parameters=params,
        )
        failure_reply = "I couldn't confirm that reminder was created, so I need you to try once more."
        return self.bot.reply_finalization.finalize(
            failure_reply,
            current_mood,
            stripped_input,
        ), None

    async def _execute_set_reminder_tool_async(
        self,
        *,
        params: dict[str, object],
        stripped_input: str,
        current_mood: str,
        plan_reason: str,
        turn_context: Any | None = None,
    ) -> tuple[str | None, str | None]:
        title = str(params.get("title") or stripped_input[:100]).strip()
        due_text = str(params.get("due_text") or "").strip()

        def _executor():
            return self.bot.add_reminder(title, due_text)

        record = execute_tool(
            tool_name="set_reminder",
            parameters=dict(params),
            executor=_executor,
            turn_context=turn_context,
        )
        if record.status == "failed":
            self.bot.update_planner_debug(
                planner_status="execution_failed",
                planner_reason=f"set_reminder raised during execution: {record.error}",
                planner_tool="set_reminder",
                planner_parameters=params,
            )
            failure_reply = "I tried to set that reminder, but something failed during execution."
            return await self.bot.reply_finalization.finalize_async(
                failure_reply,
                current_mood,
                stripped_input,
            ), None
        reminder = record.result
        if reminder:
            self.bot.update_planner_debug(
                planner_status="used_tool",
                planner_reason=plan_reason or "Planner selected a reminder tool.",
                planner_tool="set_reminder",
                planner_parameters=params,
                final_path="planner_tool",
            )
            reply = self._reminder_confirmation_reply(reminder)
            return await self.bot.reply_finalization.finalize_async(
                reply,
                current_mood,
                stripped_input,
            ), None
        self.bot.update_planner_debug(
            planner_status="execution_failed",
            planner_reason="Planner selected set_reminder, but Dad couldn't create the reminder cleanly.",
            planner_tool="set_reminder",
            planner_parameters=params,
        )
        failure_reply = "I couldn't confirm that reminder was created, so I need you to try once more."
        return await self.bot.reply_finalization.finalize_async(
            failure_reply,
            current_mood,
            stripped_input,
        ), None

    def _execute_web_search_tool_sync(
        self,
        *,
        params: dict[str, object],
        stripped_input: str,
        settings: dict[str, object],
        plan_reason: str,
        turn_context: Any | None = None,
    ) -> tuple[str | None, str | None]:
        query = str(params.get("query") or stripped_input).strip()
        normalized_query = self.bot.normalize_lookup_query(query)

        record = execute_tool(
            tool_name="web_search",
            parameters={"query": normalized_query},
            executor=lambda: self.bot.lookup_web(normalized_query),
            # web_search is read-only; no compensating action needed
            turn_context=turn_context,
        )
        if record.status == "failed":
            self.bot.update_planner_debug(
                planner_status="execution_failed",
                planner_reason=f"web_search raised during execution: {record.error}",
                planner_tool="web_search",
                planner_parameters=params,
            )
            return None, "Web lookup execution failed before a result could be produced."
        result = record.result
        if result:
            source = f" Source: {result['source_label']}." if result.get("source_label") else ""
            observation = f"{result['heading']}: {result['summary']}{source}"
            observation = self._reflect_on_web_observation(
                stripped_input,
                normalized_query,
                observation,
                settings,
            )
            self.bot.update_planner_debug(
                planner_status="used_tool",
                planner_reason=plan_reason or "Planner selected a web lookup.",
                planner_tool="web_search",
                planner_parameters=params,
                planner_observation=observation,
                final_path="planner_tool",
            )
            return None, observation
        self.bot.update_planner_debug(
            planner_status="execution_failed",
            planner_reason="Planner selected web_search, but no clean lookup result was available.",
            planner_tool="web_search",
            planner_parameters=params,
        )
        return None, "Web lookup executed but returned no usable result."

    async def _execute_web_search_tool_async(
        self,
        *,
        params: dict[str, object],
        stripped_input: str,
        settings: dict[str, object],
        plan_reason: str,
        turn_context: Any | None = None,
    ) -> tuple[str | None, str | None]:
        query = str(params.get("query") or stripped_input).strip()
        normalized_query = self.bot.normalize_lookup_query(query)

        record = execute_tool(
            tool_name="web_search",
            parameters={"query": normalized_query},
            executor=lambda: self.bot.lookup_web(normalized_query),
            turn_context=turn_context,
        )
        if record.status == "failed":
            self.bot.update_planner_debug(
                planner_status="execution_failed",
                planner_reason=f"web_search raised during execution: {record.error}",
                planner_tool="web_search",
                planner_parameters=params,
            )
            return None, "Web lookup execution failed before a result could be produced."
        result = record.result
        if result:
            source = f" Source: {result['source_label']}." if result.get("source_label") else ""
            observation = f"{result['heading']}: {result['summary']}{source}"
            observation = await self._reflect_on_web_observation_async(
                stripped_input,
                normalized_query,
                observation,
                settings,
            )
            self.bot.update_planner_debug(
                planner_status="used_tool",
                planner_reason=plan_reason or "Planner selected a web lookup.",
                planner_tool="web_search",
                planner_parameters=params,
                planner_observation=observation,
                final_path="planner_tool",
            )
            return None, observation
        self.bot.update_planner_debug(
            planner_status="execution_failed",
            planner_reason="Planner selected web_search, but no clean lookup result was available.",
            planner_tool="web_search",
            planner_parameters=params,
        )
        return None, "Web lookup executed but returned no usable result."

    def _execute_planned_tool_sync(
        self,
        *,
        tool_name: str,
        params: dict[str, object],
        stripped_input: str,
        current_mood: str,
        settings: dict[str, object],
        plan_reason: str,
        turn_context: Any | None = None,
    ) -> tuple[str | None, str | None]:
        # Gap C: record planning + tool_execution causal steps.
        _tc_state = getattr(turn_context, "state", None)
        if isinstance(_tc_state, dict):
            record_causal_step_locked(_tc_state, "planning", context="turn_service.sync")
            record_causal_step_locked(_tc_state, "tool_execution", context="turn_service.sync")
        executors = {
            _SET_REMINDER_TOOL: self._execute_set_reminder_tool_sync,
            _WEB_SEARCH_TOOL: self._execute_web_search_tool_sync,
        }
        executor = executors.get(tool_name)
        if executor is None:
            self.bot.update_planner_debug(
                planner_status="execution_contract_violation",
                planner_reason=f"Planner selected unsupported tool: {tool_name or 'unknown'}.",
            )
            return None, f"Tool routing mismatch: unsupported tool '{tool_name or 'unknown'}'."
        if tool_name in _REMINDER_TOOL_NAMES:
            return executor(
                params=params,
                stripped_input=stripped_input,
                current_mood=current_mood,
                plan_reason=plan_reason,
                turn_context=turn_context,
            )
        return executor(
            params=params,
            stripped_input=stripped_input,
            settings=settings,
            plan_reason=plan_reason,
            turn_context=turn_context,
        )

    async def _execute_planned_tool_async(
        self,
        *,
        tool_name: str,
        params: dict[str, object],
        stripped_input: str,
        current_mood: str,
        settings: dict[str, object],
        plan_reason: str,
        turn_context: Any | None = None,
    ) -> tuple[str | None, str | None]:
        # Gap C hardening: async path must follow same causal transition graph.
        _tc_state = getattr(turn_context, "state", None)
        if isinstance(_tc_state, dict):
            record_causal_step_locked(_tc_state, "planning", context="turn_service.async")
            record_causal_step_locked(_tc_state, "tool_execution", context="turn_service.async")
        executors = {
            _SET_REMINDER_TOOL: self._execute_set_reminder_tool_async,
            _WEB_SEARCH_TOOL: self._execute_web_search_tool_async,
        }
        executor = executors.get(tool_name)
        if executor is None:
            self.bot.update_planner_debug(
                planner_status="execution_contract_violation",
                planner_reason=f"Planner selected unsupported tool: {tool_name or 'unknown'}.",
            )
            return None, f"Tool routing mismatch: unsupported tool '{tool_name or 'unknown'}'."
        if tool_name in _REMINDER_TOOL_NAMES:
            return await executor(
                params=params,
                stripped_input=stripped_input,
                current_mood=current_mood,
                plan_reason=plan_reason,
                turn_context=turn_context,
            )
        return await executor(
            params=params,
            stripped_input=stripped_input,
            settings=settings,
            plan_reason=plan_reason,
            turn_context=turn_context,
        )

    def _bayesian_tool_gate(
        self,
        *,
        tool_name: str,
        tool_bias: str,
        plan_reason: str,
    ) -> tuple[bool, str]:
        """Return (allowed, reason) using runtime-owned Bayesian policy."""
        normalized_tool_name = str(tool_name or "").strip()
        normalized_tool_bias = str(tool_bias or "planner_default").strip() or "planner_default"
        if normalized_tool_bias in _DEFER_TOOL_BIASES:
            return False, (
                f"Bayesian policy '{normalized_tool_bias}' blocks all tools; "
                "tool selection overridden by governing Bayesian authority."
            )
        if not self.bot.authorize_tool_execution_for_bias(normalized_tool_name, normalized_tool_bias):
            return False, (
                f"Bayesian policy '{normalized_tool_bias}' does not permit tool '{normalized_tool_name}'."
            )
        return (
            True,
            plan_reason or f"Bayesian policy '{normalized_tool_bias}' permits tool '{normalized_tool_name}'.",
        )

    def _validate_tool_intent_robustness(
        self,
        *,
        tool_name: str,
        params: dict[str, object],
        user_input: str,
        plan_reason: str,
    ) -> tuple[bool, str]:
        """
        Pre-execution validation of tool intent quality.
        
        Checks semantic relevance, parameter validity, and confidence signals
        before attempting tool execution.
        
        Returns:
            (is_robust: bool, reason: str)
            If is_robust=False, tool will NOT execute (soft recovery to direct response).
        """
        # CHECK 1: Semantic relevance (intent matches user input)
        user_keywords = self._extract_intent_keywords(user_input)
        tool_keywords = self._get_tool_intent_keywords(tool_name, params)
        
        if user_keywords and tool_keywords:  # Both non-empty
            relevance = self._compute_semantic_overlap(user_keywords, tool_keywords)
            if relevance < 0.35:  # Tunable threshold
                return False, (
                    f"Intent mismatch: user asks '{user_input[:40].strip()}...' but "
                    f"tool is '{tool_name}' (semantic overlap={relevance:.2f})"
                )
        
        # CHECK 2: Confidence from LLM reason field
        uncertainty_patterns = ["uncertain", "not sure", "guess", "maybe", "probably", "might"]
        if any(pat in plan_reason.lower() for pat in uncertainty_patterns):
            return False, f"Low confidence indicated in reason: '{plan_reason}'"
        
        # CHECK 3: Tool-specific pre-conditions
        if tool_name == _SET_REMINDER_TOOL:
            is_valid, reason = self._validate_reminder_params(params, user_input)
            if not is_valid:
                return False, reason
        
        elif tool_name == _WEB_SEARCH_TOOL:
            is_valid, reason = self._validate_search_params(params, user_input)
            if not is_valid:
                return False, reason
        
        # CHECK 4: Parameter sanity
        if params is None or (isinstance(params, dict) and not params):
            return False, "Tool parameters are empty"
        
        if isinstance(params, dict):
            for v in params.values():
                if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    return False, f"Tool parameters contain invalid type: {type(v).__name__}"
        
        # ✅ ALL CHECKS PASSED
        return True, "Intent validated"

    def _validate_reminder_params(self, params: dict, user_input: str) -> tuple[bool, str]:
        """Validate set_reminder parameters for semantic sanity."""
        title = str(params.get("title", "")).strip()
        due_text = str(params.get("due_text", "")).strip()
        minutes_from_now = params.get("minutes_from_now", None)

        if not title and not str(user_input or "").strip():
            return False, "Reminder missing content: neither title nor user input is available"

        if title and len(title) > 300:
            return False, "Reminder title too long"

        if due_text and len(due_text) > 200:
            return False, "Reminder due_text too long"

        if minutes_from_now is not None:
            try:
                minutes_value = float(minutes_from_now)
            except (TypeError, ValueError):
                return False, "Reminder minutes_from_now must be numeric"
            if minutes_value < 0:
                return False, "Reminder time is invalid: minutes_from_now cannot be in the past"

        return True, "Reminder params valid"

    def _validate_search_params(self, params: dict, user_input: str) -> tuple[bool, str]:
        """Validate web_search parameters for semantic sanity."""
        if "query" not in params:
            return False, "Search missing required parameter: query"
        
        query = str(params.get("query", "")).strip()
        if len(query) < 3:
            return False, f"Search query too short ({len(query)} chars): '{query}'"
        
        if len(query) > 200:
            return False, f"Search query too long ({len(query)} chars)"
        
        # Reject suspicious patterns (basic safety check)
        suspicious = ["hack", "crack", "exploit", "payload", "injection"]
        if any(pat in query.lower() for pat in suspicious):
            return False, f"Search query contains suspicious patterns: '{query[:50]}...'"
        
        return True, "Search params valid"

    def _extract_intent_keywords(self, text: str) -> set[str]:
        """Extract significant keywords from text for semantic matching."""
        import string
        
        words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
        # Common stopwords to ignore
        stopwords = {
            "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "do", "does", "did", "will", "would", "should", "could", "may",
            "can", "have", "has", "had", "will", "would", "should", "could"
        }
        
        return {w for w in words if w not in stopwords and len(w) > 2}

    def _get_tool_intent_keywords(self, tool_name: str, params: dict) -> set[str]:
        """Extract keywords from tool intent (tool name + parameters)."""
        keywords = {tool_name.lower()}
        
        if tool_name == _SET_REMINDER_TOOL:
            reminder_text = str(params.get("reminder_text", "")).strip()
            if reminder_text:
                keywords.update(self._extract_intent_keywords(reminder_text))
        
        elif tool_name == _WEB_SEARCH_TOOL:
            query = str(params.get("query", "")).strip()
            if query:
                keywords.update(self._extract_intent_keywords(query))
        
        return keywords

    def _compute_semantic_overlap(self, set_a: set[str], set_b: set[str]) -> float:
        """Compute Jaccard similarity (intersection / union) between two keyword sets."""
        if not set_a or not set_b:
            return 0.0
        
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        return intersection / union if union > 0 else 0.0

    def plan_agentic_tools(
        self,
        stripped_input: str,
        current_mood: str,
        attachments: AttachmentList | None = None,
        turn_context: Any | None = None,
    ) -> tuple[str | None, str | None]:
        settings = self.bot.agentic_tool_settings()
        if not settings["enabled"]:
            self.bot.update_planner_debug(
                planner_status="disabled",
                planner_reason="Agentic tools are disabled in profile settings.",
                final_path="disabled",
            )
            return None, None

        tools = self._available_agentic_tools(settings)
        if not tools:
            self.bot.update_planner_debug(
                planner_status="disabled",
                planner_reason="No agentic tools are currently enabled.",
                final_path="disabled",
            )
            return None, None

        try:
            deterministic_route = self._deterministic_tool_route(stripped_input)
            if deterministic_route is not None:
                tool_name, params, route_reason = deterministic_route
                memory_allowed, memory_reason = self._memory_authority_decision(
                    turn_context=turn_context,
                    tool_name=tool_name,
                    plan_reason=route_reason,
                )
                if not memory_allowed:
                    self._record_tool_decision_outcome(
                        turn_context=turn_context,
                        decision_outcome="no_tool_needed",
                    )
                    self.bot.update_planner_debug(
                        planner_status="memory_authority_veto",
                        planner_reason=memory_reason,
                        planner_tool=tool_name,
                        planner_parameters=params,
                        final_path="memory_authority",
                    )
                    return None, memory_reason
                self.bot.update_planner_debug(
                    planner_status="tool_selected",
                    planner_reason=memory_reason,
                    planner_tool=tool_name,
                    planner_parameters=params,
                )
                self._record_tool_decision_outcome(
                    turn_context=turn_context,
                    decision_outcome="executed_tool",
                )
                return self._execute_planned_tool_sync(
                    tool_name=tool_name,
                    params=params,
                    stripped_input=stripped_input,
                    current_mood=current_mood,
                    settings=settings,
                    plan_reason=memory_reason,
                    turn_context=turn_context,
                )

            shared_context = self.bot.prompt_assembly.build_request_system_prompt(
                stripped_input,
                current_mood,
                attachments,
            )
            _retrieval_set = list(
                (getattr(turn_context, "state", None) or {}).get("memory_retrieval_set") or []
            )
            _memory_brief = str((getattr(turn_context, "state", None) or {}).get("memory_influence_brief") or "")
            response = self._llm_adapter.call(
                messages=[
                    {
                        "role": "user",
                        "content": self._planning_prompt(
                            stripped_input,
                            current_mood,
                            tools,
                            shared_context,
                            memory_influence_brief=_memory_brief,
                            retrieval_set=_retrieval_set,
                        ),
                    },
                ],
                options={"temperature": 0.1},
                response_format="json",
                purpose="agentic tool planning",
            )
            content = response["message"]["content"]
            plan = self._parse_agentic_tool_plan(content)
            if plan is None:
                return None, None

            plan_reason = plan.reason
            tool_name = str(plan.tool or "")
            params = dict(plan.parameters)
            self.bot.update_planner_debug(
                planner_status="considered",
                planner_reason=plan_reason or "Planner evaluated the turn.",
                planner_tool=tool_name,
                planner_parameters=params,
            )

            if not plan.needs_tool or plan.tool is None:
                self._record_tool_decision_outcome(
                    turn_context=turn_context,
                    decision_outcome="no_tool_needed",
                )
                self.bot.update_planner_debug(
                    planner_status="no_tool",
                    planner_reason=plan_reason or "Planner decided no tool was needed.",
                )
                return None, None

            self.bot.update_planner_debug(
                planner_status="tool_selected",
                planner_reason=plan_reason or "Planner selected tool execution.",
                planner_tool=str(plan.tool or ""),
                planner_parameters=params,
            )

            memory_allowed, memory_reason = self._memory_authority_decision(
                turn_context=turn_context,
                tool_name=str(plan.tool or "").strip(),
                plan_reason=plan_reason,
            )
            if not memory_allowed:
                self._record_tool_decision_outcome(
                    turn_context=turn_context,
                    decision_outcome="no_tool_needed",
                )
                self.bot.update_planner_debug(
                    planner_status="memory_authority_veto",
                    planner_reason=memory_reason,
                    planner_tool=str(plan.tool or ""),
                    planner_parameters=params,
                    final_path="memory_authority",
                )
                return None, memory_reason

            self._record_tool_decision_outcome(
                turn_context=turn_context,
                decision_outcome="executed_tool",
            )
            tool_reply, tool_obs = self._execute_planned_tool_sync(
                tool_name=str(plan.tool or "").strip(),
                params=params,
                stripped_input=stripped_input,
                current_mood=current_mood,
                settings=settings,
                plan_reason=memory_reason,
                turn_context=turn_context,
            )
            _result_text = str(tool_obs or tool_reply or "").strip()
            if _result_text and turn_context is not None:
                _tc_state = getattr(turn_context, "state", None)
                if isinstance(_tc_state, dict):
                    _tc_state["last_tool_result"] = {
                        "tool": str(plan.tool or ""),
                        "text": _result_text[:500],
                    }
                self._refresh_memory_retrieval_after_tool(
                    turn_context=turn_context,
                    user_input=stripped_input,
                    tool_feedback=_result_text,
                )
            return tool_reply, tool_obs
        except Exception as exc:
            if _is_event_loop_closed_error(exc):
                self.bot.update_planner_debug(
                    planner_status="fallback",
                    planner_reason="Planner skipped: event loop closed",
                )
                logger.info("Skipping planner call because event loop is closed")
                return None, None
            self.bot.update_planner_debug(
                planner_status="fallback",
                planner_reason=f"Planner error: {exc}",
            )
            logger.warning("Agentic tool planning failed: %s", exc)

        return None, None

    async def plan_agentic_tools_async(
        self,
        stripped_input: str,
        current_mood: str,
        attachments: AttachmentList | None = None,
        turn_context: Any | None = None,
    ) -> tuple[str | None, str | None]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_closed():
            self.bot.update_planner_debug(
                planner_status="fallback",
                planner_reason="Planner skipped: event loop closed",
            )
            logger.info("Skipping async planner call because event loop is closed")
            return None, None

        settings = self.bot.agentic_tool_settings()
        if not settings["enabled"]:
            self.bot.update_planner_debug(
                planner_status="disabled",
                planner_reason="Agentic tools are disabled in profile settings.",
                final_path="disabled",
            )
            return None, None

        tools = self._available_agentic_tools(settings)
        if not tools:
            self.bot.update_planner_debug(
                planner_status="disabled",
                planner_reason="No agentic tools are currently enabled.",
                final_path="disabled",
            )
            return None, None

        try:
            deterministic_route = self._deterministic_tool_route(stripped_input)
            if deterministic_route is not None:
                tool_name, params, route_reason = deterministic_route
                memory_allowed, memory_reason = self._memory_authority_decision(
                    turn_context=turn_context,
                    tool_name=tool_name,
                    plan_reason=route_reason,
                )
                if not memory_allowed:
                    self._record_tool_decision_outcome(
                        turn_context=turn_context,
                        decision_outcome="no_tool_needed",
                    )
                    self.bot.update_planner_debug(
                        planner_status="memory_authority_veto",
                        planner_reason=memory_reason,
                        planner_tool=tool_name,
                        planner_parameters=params,
                        final_path="memory_authority",
                    )
                    return None, memory_reason
                self.bot.update_planner_debug(
                    planner_status="tool_selected",
                    planner_reason=memory_reason,
                    planner_tool=tool_name,
                    planner_parameters=params,
                )
                self._record_tool_decision_outcome(
                    turn_context=turn_context,
                    decision_outcome="executed_tool",
                )
                return await self._execute_planned_tool_async(
                    tool_name=tool_name,
                    params=params,
                    stripped_input=stripped_input,
                    current_mood=current_mood,
                    settings=settings,
                    plan_reason=memory_reason,
                    turn_context=turn_context,
                )

            shared_context = self.bot.prompt_assembly.build_request_system_prompt(
                stripped_input,
                current_mood,
                attachments,
            )
            _retrieval_set = list(
                (getattr(turn_context, "state", None) or {}).get("memory_retrieval_set") or []
            )
            _memory_brief = str((getattr(turn_context, "state", None) or {}).get("memory_influence_brief") or "")
            response = await self._llm_adapter.call_async(
                messages=[
                    {
                        "role": "user",
                        "content": self._planning_prompt(
                            stripped_input,
                            current_mood,
                            tools,
                            shared_context,
                            memory_influence_brief=_memory_brief,
                            retrieval_set=_retrieval_set,
                        ),
                    },
                ],
                options={"temperature": 0.1},
                response_format="json",
                purpose="agentic tool planning",
            )
            content = response["message"]["content"]
            plan = self._parse_agentic_tool_plan(content)
            if plan is None:
                return None, None

            plan_reason = plan.reason
            tool_name = str(plan.tool or "")
            params = dict(plan.parameters)
            self.bot.update_planner_debug(
                planner_status="considered",
                planner_reason=plan_reason or "Planner evaluated the turn.",
                planner_tool=tool_name,
                planner_parameters=params,
            )

            if not plan.needs_tool or plan.tool is None:
                self._record_tool_decision_outcome(
                    turn_context=turn_context,
                    decision_outcome="no_tool_needed",
                )
                self.bot.update_planner_debug(
                    planner_status="no_tool",
                    planner_reason=plan_reason or "Planner decided no tool was needed.",
                )
                return None, None

            self.bot.update_planner_debug(
                planner_status="tool_selected",
                planner_reason=plan_reason or "Planner selected tool execution.",
                planner_tool=str(plan.tool or ""),
                planner_parameters=params,
            )

            memory_allowed, memory_reason = self._memory_authority_decision(
                turn_context=turn_context,
                tool_name=str(plan.tool or "").strip(),
                plan_reason=plan_reason,
            )
            if not memory_allowed:
                self._record_tool_decision_outcome(
                    turn_context=turn_context,
                    decision_outcome="no_tool_needed",
                )
                self.bot.update_planner_debug(
                    planner_status="memory_authority_veto",
                    planner_reason=memory_reason,
                    planner_tool=str(plan.tool or ""),
                    planner_parameters=params,
                    final_path="memory_authority",
                )
                return None, memory_reason

            self._record_tool_decision_outcome(
                turn_context=turn_context,
                decision_outcome="executed_tool",
            )
            tool_reply, tool_obs = await self._execute_planned_tool_async(
                tool_name=str(plan.tool or "").strip(),
                params=params,
                stripped_input=stripped_input,
                current_mood=current_mood,
                settings=settings,
                plan_reason=memory_reason,
                turn_context=turn_context,
            )
            _result_text = str(tool_obs or tool_reply or "").strip()
            if _result_text and turn_context is not None:
                _tc_state = getattr(turn_context, "state", None)
                if isinstance(_tc_state, dict):
                    _tc_state["last_tool_result"] = {
                        "tool": str(plan.tool or ""),
                        "text": _result_text[:500],
                    }
                self._refresh_memory_retrieval_after_tool(
                    turn_context=turn_context,
                    user_input=stripped_input,
                    tool_feedback=_result_text,
                )
            return tool_reply, tool_obs
        except Exception as exc:
            if _is_event_loop_closed_error(exc):
                self.bot.update_planner_debug(
                    planner_status="fallback",
                    planner_reason="Planner skipped: event loop closed",
                )
                logger.info("Skipping async planner call because event loop is closed")
                return None, None
            self.bot.update_planner_debug(
                planner_status="fallback",
                planner_reason=f"Planner error: {exc}",
            )
            logger.warning("Agentic tool planning failed: %s", exc)

        return None, None

    def prepare_user_turn(
        self,
        stripped_input: str,
        attachments: AttachmentList | None = None,
        turn_context: Any | None = None,
    ) -> PreparedTurnResult:
        reset_turn_coherence(self.bot)
        self._start_turn_pipeline("sync", stripped_input)
        normalized_attachments = self.bot.normalize_chat_attachments(attachments)
        normalized_attachments = self.bot.enrich_multimodal_attachments(
            normalized_attachments,
            user_input=stripped_input,
        )
        self._append_turn_pipeline_step(
            "normalize_attachments",
            detail=f"attachments={len(normalized_attachments)}",
        )
        turn_text = self.bot.compose_user_turn_text(
            stripped_input,
            normalized_attachments,
        )
        self._append_turn_pipeline_step(
            "compose_turn_text",
            detail="composed user turn text" if turn_text else "empty composed turn",
        )
        if not turn_text:
            self._complete_turn_pipeline(final_path="empty_input", reply_source="none")
            return None, None, False, "", normalized_attachments

        if self.bot.is_session_exit_command(stripped_input):
            self.bot.persist_conversation()
            self.bot.mark_chat_thread_closed(closed=True)
            self._append_turn_pipeline_step(
                "session_exit",
                detail="handled exit command",
            )
            self._complete_turn_pipeline(
                final_path="session_exit",
                reply_source="exit_command",
                should_end=True,
            )
            return (
                None,
                self.bot.reply_finalization.append_signoff(
                    "Catch ya later, Tony! Always here if you need me.",
                ),
                True,
                turn_text,
                normalized_attachments,
            )

        mood_history_window = self.bot.runtime_config.window(
            "mood_detection_context",
            6,
        )
        if self.bot.LIGHT_MODE:
            current_mood = "neutral"
        else:
            current_mood = self.bot.mood_manager.detect(
                turn_text,
                self.bot.prompt_history()[-mood_history_window:],
            )
        self._update_turn_pipeline(current_mood=current_mood)
        self._append_turn_pipeline_step(
            "detect_mood",
            detail=f"current_mood={current_mood}",
        )
        self.record_user_turn_state(turn_text, current_mood, turn_context=turn_context)
        self._append_turn_pipeline_step(
            "record_turn_state",
            detail="saved mood and relationship state",
        )
        self.bot.begin_planner_debug(stripped_input, current_mood)
        self._append_turn_pipeline_step(
            "begin_planner_debug",
            detail="initialized planner debug state",
        )
        self.bot.prompt_assembly.begin_turn_memory_context(
            stripped_input,
            user_id=str(getattr(self.bot, "TENANT_ID", "default") or "default"),
            session_id=str(getattr(self.bot, "active_thread_id", "default") or "default"),
        )

        if direct_reply := self.direct_reply_for_input(stripped_input, current_mood):
            self.bot.update_planner_debug(
                planner_status="skipped",
                planner_reason="A direct reply path handled this turn before tool planning.",
                final_path="direct_reply",
            )
            self._append_turn_pipeline_step(
                "direct_reply",
                detail="reply produced before tool planning",
            )
            self._update_turn_pipeline(
                final_path="direct_reply",
                reply_source="direct_reply",
            )
            return current_mood, direct_reply, False, turn_text, normalized_attachments

        mark_turn_coherence(self.bot, "tool_decision_origin")
        auto_reply, tool_observation = self.plan_agentic_tools(
            stripped_input,
            current_mood,
            normalized_attachments,
            turn_context=turn_context,
        )
        planner_snapshot = self.bot.planner_debug_snapshot()
        self._append_turn_pipeline_step(
            "plan_tools",
            detail=f"planner_status={planner_snapshot.get('planner_status', 'idle')}",
        )
        if auto_reply is not None:
            self._update_turn_pipeline(
                final_path=str(
                    planner_snapshot.get("final_path") or "planner_tool",
                ).strip()
                or "planner_tool",
                reply_source="planner_tool",
            )
            return current_mood, auto_reply, False, turn_text, normalized_attachments

        self.bot.set_active_tool_observation(tool_observation)
        if tool_observation:
            self._append_turn_pipeline_step(
                "tool_observation",
                detail="captured tool observation for reply generation",
            )
        if auto_reply is not None:
            planner_snapshot = self.bot.planner_debug_snapshot()
            self._update_turn_pipeline(
                final_path=str(
                    planner_snapshot.get("final_path") or "heuristic_tool",
                ).strip()
                or "heuristic_tool",
                reply_source="heuristic_tool",
            )
            return current_mood, auto_reply, False, turn_text, normalized_attachments

        self._update_turn_pipeline(
            final_path="model_reply",
            reply_source="model_generation",
        )
        return current_mood, None, False, turn_text, normalized_attachments

    async def prepare_user_turn_async(
        self,
        stripped_input: str,
        attachments: AttachmentList | None = None,
        turn_context: Any | None = None,
    ) -> PreparedTurnResult:
        reset_turn_coherence(self.bot)
        self._start_turn_pipeline("async", stripped_input)
        normalized_attachments = self.bot.normalize_chat_attachments(attachments)
        normalized_attachments = self.bot.enrich_multimodal_attachments(
            normalized_attachments,
            user_input=stripped_input,
        )
        self._append_turn_pipeline_step(
            "normalize_attachments",
            detail=f"attachments={len(normalized_attachments)}",
        )
        turn_text = self.bot.compose_user_turn_text(
            stripped_input,
            normalized_attachments,
        )
        self._append_turn_pipeline_step(
            "compose_turn_text",
            detail="composed user turn text" if turn_text else "empty composed turn",
        )
        if not turn_text:
            self._complete_turn_pipeline(final_path="empty_input", reply_source="none")
            return None, None, False, "", normalized_attachments

        if self.bot.is_session_exit_command(stripped_input):
            self.bot.persist_conversation()
            self.bot.mark_chat_thread_closed(closed=True)
            self._append_turn_pipeline_step(
                "session_exit",
                detail="handled exit command",
            )
            self._complete_turn_pipeline(
                final_path="session_exit",
                reply_source="exit_command",
                should_end=True,
            )
            return (
                None,
                self.bot.reply_finalization.append_signoff(
                    "Catch ya later, Tony! Always here if you need me.",
                ),
                True,
                turn_text,
                normalized_attachments,
            )

        mood_history_window = self.bot.runtime_config.window(
            "mood_detection_context",
            6,
        )
        if self.bot.LIGHT_MODE:
            current_mood = "neutral"
        else:
            current_mood = await self.bot.mood_manager.detect_async(
                turn_text,
                self.bot.prompt_history()[-mood_history_window:],
            )
        self._update_turn_pipeline(current_mood=current_mood)
        self._append_turn_pipeline_step(
            "detect_mood",
            detail=f"current_mood={current_mood}",
        )
        self.record_user_turn_state(turn_text, current_mood, turn_context=turn_context)
        self._append_turn_pipeline_step(
            "record_turn_state",
            detail="saved mood and relationship state",
        )
        self.bot.begin_planner_debug(stripped_input, current_mood)
        self._append_turn_pipeline_step(
            "begin_planner_debug",
            detail="initialized planner debug state",
        )
        self.bot.prompt_assembly.begin_turn_memory_context(
            stripped_input,
            user_id=str(getattr(self.bot, "TENANT_ID", "default") or "default"),
            session_id=str(getattr(self.bot, "active_thread_id", "default") or "default"),
        )

        if direct_reply := self.direct_reply_for_input(stripped_input, current_mood):
            self.bot.update_planner_debug(
                planner_status="skipped",
                planner_reason="A direct reply path handled this turn before tool planning.",
                final_path="direct_reply",
            )
            self._append_turn_pipeline_step(
                "direct_reply",
                detail="reply produced before tool planning",
            )
            self._update_turn_pipeline(
                final_path="direct_reply",
                reply_source="direct_reply",
            )
            return current_mood, direct_reply, False, turn_text, normalized_attachments

        mark_turn_coherence(self.bot, "tool_decision_origin")
        auto_reply, tool_observation = await self.plan_agentic_tools_async(
            stripped_input,
            current_mood,
            normalized_attachments,
            turn_context=turn_context,
        )
        planner_snapshot = self.bot.planner_debug_snapshot()
        self._append_turn_pipeline_step(
            "plan_tools",
            detail=f"planner_status={planner_snapshot.get('planner_status', 'idle')}",
        )
        if auto_reply is not None:
            self._update_turn_pipeline(
                final_path=str(
                    planner_snapshot.get("final_path") or "planner_tool",
                ).strip()
                or "planner_tool",
                reply_source="planner_tool",
            )
            return current_mood, auto_reply, False, turn_text, normalized_attachments

        self.bot.set_active_tool_observation(tool_observation)
        if tool_observation:
            self._append_turn_pipeline_step(
                "tool_observation",
                detail="captured tool observation for reply generation",
            )
        if auto_reply is not None:
            planner_snapshot = self.bot.planner_debug_snapshot()
            self._update_turn_pipeline(
                final_path=str(
                    planner_snapshot.get("final_path") or "heuristic_tool",
                ).strip()
                or "heuristic_tool",
                reply_source="heuristic_tool",
            )
            return current_mood, auto_reply, False, turn_text, normalized_attachments

        self._update_turn_pipeline(
            final_path="model_reply",
            reply_source="model_generation",
        )
        return current_mood, None, False, turn_text, normalized_attachments

    def finalize_user_turn(
        self,
        stripped_input: str,
        current_mood: str | None,
        dad_reply: str | None,
        attachments: AttachmentList | None = None,
        turn_context: Any | None = None,
    ) -> FinalizedTurnResult:
        if turn_context is None:
            raise RuntimeError(
                "Strict mode requires turn_context for finalize_user_turn",
            )
        mutation_queue = getattr(turn_context, "mutation_queue", None)
        if mutation_queue is None:
            raise RuntimeError("SaveNode context missing mutation_queue in strict mode")

        self.validate_execution_truth_contract(turn_context, enforce=True)

        self._append_turn_pipeline_step(
            "finalize_turn",
            detail="persisted conversation turn",
        )

        user_turn = {"role": "user", "content": stripped_input, "mood": current_mood}
        if attachments:
            user_turn["attachments"] = [self.bot.history_attachment_metadata(attachment) for attachment in attachments]

        mutation_queue.queue(
            MutationIntent(
                type=MutationKind.LEDGER,
                payload={
                    "op": LedgerMutationOp.APPEND_HISTORY.value,
                    "entry": user_turn,
                },
                requires_temporal=False,
                source="turn_service.finalize_user_turn.user_entry",
            ),
        )
        mutation_queue.queue(
            MutationIntent(
                type=MutationKind.LEDGER,
                payload={
                    "op": LedgerMutationOp.APPEND_HISTORY.value,
                    "entry": {"role": "assistant", "content": dad_reply},
                },
                requires_temporal=False,
                source="turn_service.finalize_user_turn.assistant_entry",
            ),
        )
        mutation_queue.queue(
            MutationIntent(
                type=MutationKind.LEDGER,
                payload={"op": LedgerMutationOp.CLEAR_TURN_CONTEXT.value},
                requires_temporal=False,
                source="turn_service.finalize_user_turn.clear_turn_context",
            ),
        )
        mutation_queue.queue(
            MutationIntent(
                type=MutationKind.LEDGER,
                payload={"op": LedgerMutationOp.SYNC_THREAD_SNAPSHOT.value},
                requires_temporal=False,
                source="turn_service.finalize_user_turn.sync_thread_snapshot",
            ),
        )
        mutation_queue.queue(
            MutationIntent(
                type=MutationKind.LEDGER,
                payload={
                    "op": LedgerMutationOp.SCHEDULE_MAINTENANCE.value,
                    "turn_text": stripped_input,
                    "mood": current_mood,
                },
                requires_temporal=False,
                source="turn_service.finalize_user_turn.schedule_maintenance",
            ),
        )
        mutation_queue.queue(
            MutationIntent(
                type=MutationKind.LEDGER,
                payload={"op": LedgerMutationOp.HEALTH_SNAPSHOT.value},
                requires_temporal=False,
                source="turn_service.finalize_user_turn.health_snapshot",
            ),
        )
        policy_trace_events = list(getattr(turn_context, "state", {}).get("policy_trace_events") or [])
        if policy_trace_events:
            mutation_queue.queue(
                MutationIntent(
                    type=MutationKind.LEDGER,
                    payload={
                        "op": LedgerMutationOp.POLICY_TRACE_EVENT.value,
                        "events": policy_trace_events,
                    },
                    requires_temporal=False,
                    source="turn_service.finalize_user_turn.policy_trace_event",
                ),
            )
        if bool(getattr(turn_context, "metadata", {}).get("audit_mode", False)):
            mutation_queue.queue(
                MutationIntent(
                    type=MutationKind.LEDGER,
                    payload={
                        "op": LedgerMutationOp.CAPABILITY_AUDIT_EVENT.value,
                        "scenario": "runtime_turn",
                    },
                    requires_temporal=False,
                    source="turn_service.finalize_user_turn.capability_audit_event",
                ),
            )
        return dad_reply, False

    # ---------------------------------------------------------------------------
    # Public API: thin shells — accept input, invoke kernel, return output.
    # TurnService is an input adapter only.  No state mutations happen here.
    # ---------------------------------------------------------------------------

    def _resolve_persistence_service(self) -> Any:
        services = getattr(self.bot, "services", None)
        persistence_service = getattr(services, "persistence_service", None)
        if persistence_service is not None:
            return persistence_service
        registry = getattr(self.bot, "service_registry", None)
        get = getattr(registry, "get", None)
        if callable(get):
            try:
                persistence_service = get("persistence_service")
            except Exception:
                persistence_service = None
        if persistence_service is None:
            cached = getattr(self.bot, "_compat_persistence_service", None)
            if cached is not None:
                return cached
            from dadbot.managers.conversation_persistence import (
                ConversationPersistenceManager,
            )
            from dadbot.services.persistence import PersistenceService
            from dadbot.services.post_commit_worker import PostCommitWorker

            persistence_service = PersistenceService(
                ConversationPersistenceManager(self.bot),
                turn_service=self,
            )
            if getattr(self.bot, "_post_commit_worker", None) is None:
                self.bot._post_commit_worker = PostCommitWorker(self.bot)
            self.bot._compat_persistence_service = persistence_service
        return persistence_service

    @staticmethod
    def _compat_turn_context(
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> TurnContext:
        return TurnContext(user_input=user_input, attachments=attachments)

    # DEPRECATED — NO NEW CALLERS. This compat shim bridges legacy direct-call
    # surfaces that bypass the TurnGraph execution context. Target removal once
    # all callers are migrated to the graph pipeline. Expiry: 2026-Q3.
    # All callers must be migrated to process_user_message() or graph.execute().
    def _finalize_direct_compat_turn(
        self,
        stripped_input: str,
        current_mood: str | None,
        dad_reply: str | None,
        attachments: AttachmentList | None = None,
        *,
        turn_context: TurnContext | None = None,
    ) -> FinalizedTurnResult:
        return self._finalize_turn_compat_context(
            stripped_input,
            current_mood,
            dad_reply,
            attachments,
            turn_context=turn_context,
            mark_legacy_direct_compat=True,
            emit_deprecation_warning=True,
        )

    def _finalize_turn_compat_context(
        self,
        stripped_input: str,
        current_mood: str | None,
        dad_reply: str | None,
        attachments: AttachmentList | None = None,
        *,
        turn_context: TurnContext | None = None,
        mark_legacy_direct_compat: bool = False,
        emit_deprecation_warning: bool = False,
    ) -> FinalizedTurnResult:
        if emit_deprecation_warning:
            warnings.warn(
                "_finalize_direct_compat_turn is a legacy compat shim scheduled for removal in 2026-Q3. "
                "Migrate callers to process_user_message() or the TurnGraph pipeline.",
                DeprecationWarning,
                stacklevel=3,
            )
        context = turn_context or self._compat_turn_context(stripped_input, attachments)
        metadata = getattr(context, "metadata", None)
        if mark_legacy_direct_compat and isinstance(metadata, dict):
            metadata.setdefault("legacy_direct_compat", True)
        context.state["turn_text"] = stripped_input
        context.state["mood"] = current_mood or "neutral"
        context.state["norm_attachments"] = attachments or []
        persistence_service = self._resolve_persistence_service()
        with ensure_execution_trace_root(
            operation="turn_service_finalize_direct_compat_turn",
            prompt="[turn-service-direct-compat-finalize]",
            metadata={"source": "TurnService._finalize_direct_compat_turn"},
            required=True,
        ):
            return persistence_service.finalize_turn(context, (dad_reply, False))

    def _run_orchestrator_sync(
        self,
        user_input: str,
        attachments: AttachmentList | None,
    ) -> FinalizedTurnResult:
        response = self.bot.execute_turn(
            live_turn_request(
                user_input,
                attachments=list(attachments or []),
                delivery=TurnDelivery.SYNC,
                session_id=str(getattr(self.bot, "active_thread_id", "") or "default"),
            ),
        )
        return cast(TurnResponse, response).as_result()

    def process_user_message(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        stripped_input = user_input.strip()
        if not stripped_input and not attachments:
            return None, False
        turn_context = self._compat_turn_context(stripped_input, attachments)
        current_mood, dad_reply, should_end, turn_text, normalized_attachments = self.prepare_user_turn(
            stripped_input,
            attachments=attachments,
            turn_context=turn_context,
        )
        if should_end:
            return dad_reply, True
        if dad_reply is None:
            self._append_turn_pipeline_step(
                "generate_reply",
                status="running",
                detail="generating sync reply",
            )
            dad_reply = self.reply_generation.generate_validated_reply(
                stripped_input,
                turn_text,
                current_mood or "neutral",
                normalized_attachments,
                stream=False,
            )
            self._append_turn_pipeline_step(
                "generate_reply",
                detail="generated sync reply",
            )
        return self._finalize_turn_compat_context(
            turn_text,
            current_mood,
            dad_reply,
            normalized_attachments,
            turn_context=turn_context,
        )

    async def process_user_message_async(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        stripped_input = user_input.strip()
        if not stripped_input and not attachments:
            return None, False
        turn_context = self._compat_turn_context(stripped_input, attachments)
        (
            current_mood,
            dad_reply,
            should_end,
            turn_text,
            normalized_attachments,
        ) = await self.prepare_user_turn_async(
            stripped_input,
            attachments=attachments,
            turn_context=turn_context,
        )
        if should_end:
            return dad_reply, True
        if dad_reply is None:
            self._append_turn_pipeline_step(
                "generate_reply",
                status="running",
                detail="generating async reply",
            )
            dad_reply = await self.reply_generation.generate_validated_reply_async(
                stripped_input,
                turn_text,
                current_mood or "neutral",
                normalized_attachments,
                stream=False,
            )
            self._append_turn_pipeline_step(
                "generate_reply",
                detail="generated async reply",
            )
        return self._finalize_turn_compat_context(
            turn_text,
            current_mood,
            dad_reply,
            normalized_attachments,
            turn_context=turn_context,
        )

    def process_user_message_stream(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        chunk_callback: ChunkCallback | None = None,
    ) -> FinalizedTurnResult:
        stripped_input = user_input.strip()
        if not stripped_input and not attachments:
            return None, False
        turn_context = self._compat_turn_context(stripped_input, attachments)
        current_mood, dad_reply, should_end, turn_text, normalized_attachments = self.prepare_user_turn(
            stripped_input,
            attachments=attachments,
            turn_context=turn_context,
        )
        if should_end:
            return dad_reply, True
        if dad_reply is None:
            self._update_turn_pipeline(mode="stream_sync")
            self._append_turn_pipeline_step(
                "generate_reply",
                status="running",
                detail="generating streamed sync reply",
            )
            dad_reply = self.reply_generation.generate_validated_reply(
                stripped_input,
                turn_text,
                current_mood or "neutral",
                normalized_attachments,
                stream=True,
                chunk_callback=chunk_callback,
            )
            self._append_turn_pipeline_step(
                "generate_reply",
                detail="generated streamed sync reply",
            )
        elif callable(chunk_callback) and dad_reply:
            chunk_callback(dad_reply)
        return self._finalize_turn_compat_context(
            turn_text,
            current_mood,
            dad_reply,
            normalized_attachments,
            turn_context=turn_context,
        )

    async def process_user_message_stream_async(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        chunk_callback: ChunkCallback | None = None,
    ) -> FinalizedTurnResult:
        stripped_input = user_input.strip()
        if not stripped_input and not attachments:
            return None, False
        turn_context = self._compat_turn_context(stripped_input, attachments)
        (
            current_mood,
            dad_reply,
            should_end,
            turn_text,
            normalized_attachments,
        ) = await self.prepare_user_turn_async(
            stripped_input,
            attachments=attachments,
            turn_context=turn_context,
        )
        if should_end:
            return dad_reply, True
        if dad_reply is None:
            self._update_turn_pipeline(mode="stream_async")
            self._append_turn_pipeline_step(
                "generate_reply",
                status="running",
                detail="generating streamed async reply",
            )
            dad_reply = await self.reply_generation.generate_validated_reply_async(
                stripped_input,
                turn_text,
                current_mood or "neutral",
                normalized_attachments,
                stream=True,
                chunk_callback=chunk_callback,
            )
            self._append_turn_pipeline_step(
                "generate_reply",
                detail="generated streamed async reply",
            )
        elif callable(chunk_callback) and dad_reply:
            maybe_coro = chunk_callback(dad_reply)
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro
        return self._finalize_turn_compat_context(
            turn_text,
            current_mood,
            dad_reply,
            normalized_attachments,
            turn_context=turn_context,
        )


__all__ = ["TurnService"]
