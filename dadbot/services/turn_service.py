from __future__ import annotations

import asyncio
import json
import logging
import warnings
from typing import Any, cast

from pydantic import BaseModel, Field, ValidationError

from dadbot.contracts import (
    AttachmentList,
    ChunkCallback,
    DadBotContext,
    FinalizedTurnResult,
    PreparedTurnResult,
    SupportsTurnProcessingRuntime,
)
from dadbot.core.graph import (
    LedgerMutationOp,
    MutationIntent,
    MutationKind,
    TurnContext,
)
from dadbot.core.tool_executor import execute_tool
from dadbot.core.execution_context import ensure_execution_trace_root
from dadbot.managers.reply_generation import ReplyGenerationManager
from dadbot.models import AgenticToolPlan
from dadbot.services.llm_call_adapter import LLMCallAdapter
from dadbot.services.turn_state_mutator import TurnStateMutator
from dadbot.core.turn_coherence import mark_turn_coherence, reset_turn_coherence

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

    @staticmethod
    def _reflection_prompt(original_input: str, query: str, observation: str) -> str:
        return f"""
You are Dad's internal quality checker. Tony asked: "{original_input}"

You searched for: "{query}"
You got back: "{observation}"

Is this observation sufficient to give Tony a genuinely useful answer?

Return ONLY valid JSON (no extra text):

{{
  "sufficient": true or false,
  "refined_query": "a better search query if not sufficient, otherwise null",
  "reason": "one sentence"
}}
""".strip()

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
    def _planning_prompt(
        stripped_input: str,
        current_mood: str,
        tools: list[dict[str, object]],
        shared_context: str,
    ) -> str:
        return f"""
You are Dad helping Tony. Think carefully about his latest message.

Shared turn context used for response generation:
{shared_context}

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

    @staticmethod
    def _reminder_confirmation_reply(reminder: dict[str, object]) -> str:
        reply = f"I went ahead and set that reminder for you, Tony: {reminder['title']}."
        if reminder.get("due_text"):
            reply += f" ({reminder['due_text']})"
        return reply

    def _execute_set_reminder_tool_sync(
        self,
        *,
        params: dict[str, object],
        stripped_input: str,
        current_mood: str,
        plan_reason: str,
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
        )
        _result_holder.append(record.result)
        if record.status == "failed":
            self.bot.update_planner_debug(
                planner_status="fallback",
                planner_reason=f"set_reminder raised during execution: {record.error}",
                planner_tool="set_reminder",
                planner_parameters=params,
            )
            return None, None
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
            planner_status="fallback",
            planner_reason="Planner selected set_reminder, but Dad couldn't create the reminder cleanly.",
            planner_tool="set_reminder",
            planner_parameters=params,
        )
        return None, None

    async def _execute_set_reminder_tool_async(
        self,
        *,
        params: dict[str, object],
        stripped_input: str,
        current_mood: str,
        plan_reason: str,
    ) -> tuple[str | None, str | None]:
        title = str(params.get("title") or stripped_input[:100]).strip()
        due_text = str(params.get("due_text") or "").strip()

        def _executor():
            return self.bot.add_reminder(title, due_text)

        record = execute_tool(
            tool_name="set_reminder",
            parameters=dict(params),
            executor=_executor,
        )
        if record.status == "failed":
            self.bot.update_planner_debug(
                planner_status="fallback",
                planner_reason=f"set_reminder raised during execution: {record.error}",
                planner_tool="set_reminder",
                planner_parameters=params,
            )
            return None, None
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
            planner_status="fallback",
            planner_reason="Planner selected set_reminder, but Dad couldn't create the reminder cleanly.",
            planner_tool="set_reminder",
            planner_parameters=params,
        )
        return None, None

    def _execute_web_search_tool_sync(
        self,
        *,
        params: dict[str, object],
        stripped_input: str,
        settings: dict[str, object],
        plan_reason: str,
    ) -> tuple[str | None, str | None]:
        query = str(params.get("query") or stripped_input).strip()
        normalized_query = self.bot.normalize_lookup_query(query)

        record = execute_tool(
            tool_name="web_search",
            parameters={"query": normalized_query},
            executor=lambda: self.bot.lookup_web(normalized_query),
            # web_search is read-only; no compensating action needed
        )
        if record.status == "failed":
            self.bot.update_planner_debug(
                planner_status="fallback",
                planner_reason=f"web_search raised during execution: {record.error}",
                planner_tool="web_search",
                planner_parameters=params,
            )
            return None, None
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
            planner_status="fallback",
            planner_reason="Planner selected web_search, but no clean lookup result was available.",
            planner_tool="web_search",
            planner_parameters=params,
        )
        return None, None

    async def _execute_web_search_tool_async(
        self,
        *,
        params: dict[str, object],
        stripped_input: str,
        settings: dict[str, object],
        plan_reason: str,
    ) -> tuple[str | None, str | None]:
        query = str(params.get("query") or stripped_input).strip()
        normalized_query = self.bot.normalize_lookup_query(query)

        record = execute_tool(
            tool_name="web_search",
            parameters={"query": normalized_query},
            executor=lambda: self.bot.lookup_web(normalized_query),
        )
        if record.status == "failed":
            self.bot.update_planner_debug(
                planner_status="fallback",
                planner_reason=f"web_search raised during execution: {record.error}",
                planner_tool="web_search",
                planner_parameters=params,
            )
            return None, None
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
            planner_status="fallback",
            planner_reason="Planner selected web_search, but no clean lookup result was available.",
            planner_tool="web_search",
            planner_parameters=params,
        )
        return None, None

    def _execute_planned_tool_sync(
        self,
        *,
        tool_name: str,
        params: dict[str, object],
        stripped_input: str,
        current_mood: str,
        settings: dict[str, object],
        plan_reason: str,
    ) -> tuple[str | None, str | None]:
        executors = {
            _SET_REMINDER_TOOL: self._execute_set_reminder_tool_sync,
            _WEB_SEARCH_TOOL: self._execute_web_search_tool_sync,
        }
        enabled = {
            _SET_REMINDER_TOOL: bool(settings.get("auto_reminders")),
            _WEB_SEARCH_TOOL: bool(settings.get("auto_web_lookup")),
        }
        executor = executors.get(tool_name)
        if executor is None:
            self.bot.update_planner_debug(
                planner_status="fallback",
                planner_reason=f"Planner selected unsupported tool: {tool_name or 'unknown'}.",
            )
            return None, None
        if not enabled.get(tool_name, False):
            self.bot.update_planner_debug(
                planner_status="fallback",
                planner_reason=f"Planner selected disabled tool: {tool_name}.",
                planner_tool=tool_name,
                planner_parameters=params,
            )
            return None, None
        if tool_name in _REMINDER_TOOL_NAMES:
            return executor(
                params=params,
                stripped_input=stripped_input,
                current_mood=current_mood,
                plan_reason=plan_reason,
            )
        return executor(
            params=params,
            stripped_input=stripped_input,
            settings=settings,
            plan_reason=plan_reason,
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
    ) -> tuple[str | None, str | None]:
        executors = {
            _SET_REMINDER_TOOL: self._execute_set_reminder_tool_async,
            _WEB_SEARCH_TOOL: self._execute_web_search_tool_async,
        }
        enabled = {
            _SET_REMINDER_TOOL: bool(settings.get("auto_reminders")),
            _WEB_SEARCH_TOOL: bool(settings.get("auto_web_lookup")),
        }
        executor = executors.get(tool_name)
        if executor is None:
            self.bot.update_planner_debug(
                planner_status="fallback",
                planner_reason=f"Planner selected unsupported tool: {tool_name or 'unknown'}.",
            )
            return None, None
        if not enabled.get(tool_name, False):
            self.bot.update_planner_debug(
                planner_status="fallback",
                planner_reason=f"Planner selected disabled tool: {tool_name}.",
                planner_tool=tool_name,
                planner_parameters=params,
            )
            return None, None
        if tool_name in _REMINDER_TOOL_NAMES:
            return await executor(
                params=params,
                stripped_input=stripped_input,
                current_mood=current_mood,
                plan_reason=plan_reason,
            )
        return await executor(
            params=params,
            stripped_input=stripped_input,
            settings=settings,
            plan_reason=plan_reason,
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

    def plan_agentic_tools(
        self,
        stripped_input: str,
        current_mood: str,
        attachments: AttachmentList | None = None,
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
            shared_context = self.bot.prompt_assembly.build_request_system_prompt(
                stripped_input,
                current_mood,
                attachments,
            )
            response = self._llm_adapter.call(
                messages=[
                    {
                        "role": "user",
                        "content": self._planning_prompt(
                            stripped_input,
                            current_mood,
                            tools,
                            shared_context,
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
                self.bot.update_planner_debug(
                    planner_status="no_tool",
                    planner_reason=plan_reason or "Planner decided no tool was needed.",
                )
                return None, None

            # Bayesian gate: the Bayesian policy is the FINAL authority.
            # The planner is advisory; if the policy blocks the tool, skip it.
            tool_bias = str(
                self.bot.planner_debug_snapshot().get("bayesian_tool_bias") or "planner_default",
            )
            allowed, gate_reason = self._bayesian_tool_gate(
                tool_name=str(plan.tool or ""),
                tool_bias=tool_bias,
                plan_reason=plan_reason,
            )
            if not allowed:
                self.bot.update_planner_debug(
                    planner_status="bayesian_blocked",
                    planner_reason=gate_reason,
                    planner_tool=str(plan.tool or ""),
                    planner_parameters=params,
                )
                logger.info("Bayesian gate blocked tool %r: %s", plan.tool, gate_reason)
                return None, None

            return self._execute_planned_tool_sync(
                tool_name=str(plan.tool or "").strip(),
                params=params,
                stripped_input=stripped_input,
                current_mood=current_mood,
                settings=settings,
                plan_reason=gate_reason,
            )
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
            shared_context = self.bot.prompt_assembly.build_request_system_prompt(
                stripped_input,
                current_mood,
                attachments,
            )
            response = await self._llm_adapter.call_async(
                messages=[
                    {
                        "role": "user",
                        "content": self._planning_prompt(
                            stripped_input,
                            current_mood,
                            tools,
                            shared_context,
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
                self.bot.update_planner_debug(
                    planner_status="no_tool",
                    planner_reason=plan_reason or "Planner decided no tool was needed.",
                )
                return None, None

            # Bayesian gate: governing authority over tool selection.
            tool_bias = str(
                self.bot.planner_debug_snapshot().get("bayesian_tool_bias") or "planner_default",
            )
            allowed, gate_reason = self._bayesian_tool_gate(
                tool_name=str(plan.tool or ""),
                tool_bias=tool_bias,
                plan_reason=plan_reason,
            )
            if not allowed:
                self.bot.update_planner_debug(
                    planner_status="bayesian_blocked",
                    planner_reason=gate_reason,
                    planner_tool=str(plan.tool or ""),
                    planner_parameters=params,
                )
                logger.info("Bayesian gate blocked tool %r: %s", plan.tool, gate_reason)
                return None, None

            return await self._execute_planned_tool_async(
                tool_name=str(plan.tool or "").strip(),
                params=params,
                stripped_input=stripped_input,
                current_mood=current_mood,
                settings=settings,
                plan_reason=gate_reason,
            )
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
            from dadbot.services.post_commit_worker import PostCommitWorker
            from dadbot.services.persistence import PersistenceService

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
