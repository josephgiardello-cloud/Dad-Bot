from __future__ import annotations

import json
import logging
from datetime import datetime

from dadbot.contracts import (
    AttachmentList,
    ChunkCallback,
    DadBotContext,
    FinalizedTurnResult,
    PreparedTurnResult,
    SupportsTurnProcessingRuntime,
)
from dadbot.core.tool_sandbox import ToolSandbox
from dadbot.managers.reply_generation import ReplyGenerationManager
from dadbot.models import AgenticToolPlan, TurnPipelineSnapshot, TurnPipelineStep
from pydantic import ValidationError

# Keep the historic logger name so existing tests and log pipelines remain stable
# while the implementation moves from managers/ to services/.
logger = logging.getLogger("dadbot.managers.turn_processing")


class TurnService:
    """Own per-turn preparation, reply generation, and finalization flows.

    This replaces the old TurnProcessingManager. The concrete implementation
    now lives under dadbot.services and is exposed through ``turn_service``.
    """

    TOOL_EXECUTOR_NAMES = frozenset({"set_reminder", "web_search"})

    def __init__(self, bot: DadBotContext | SupportsTurnProcessingRuntime):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot
        self.reply_generation = ReplyGenerationManager(self.context)

    @staticmethod
    def _pipeline_timestamp() -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _store_turn_pipeline(self, payload: dict[str, object]) -> dict[str, object]:
        validated = TurnPipelineSnapshot.model_validate(payload)
        self.bot._last_turn_pipeline = validated.model_dump(mode="python")
        return self.bot._last_turn_pipeline

    def _start_turn_pipeline(self, mode: str, user_input: str) -> dict[str, object]:
        return self._store_turn_pipeline(
            {
                "mode": str(mode or "sync").strip() or "sync",
                "user_input": str(user_input or "").strip(),
                "started_at": self._pipeline_timestamp(),
                "steps": [],
            }
        )

    def _update_turn_pipeline(self, **fields) -> dict[str, object] | None:
        current = dict(getattr(self.bot, "_last_turn_pipeline", {}) or {})
        if not current:
            return None
        current.update(fields)
        return self._store_turn_pipeline(current)

    def _append_turn_pipeline_step(
        self,
        name: str,
        status: str = "completed",
        detail: str = "",
        **metadata,
    ) -> dict[str, object] | None:
        current = dict(getattr(self.bot, "_last_turn_pipeline", {}) or {})
        if not current:
            return None
        steps = list(current.get("steps", []))
        steps.append(
            TurnPipelineStep.model_validate(
                {
                    "name": str(name or "step").strip() or "step",
                    "status": str(status or "completed").strip().lower() or "completed",
                    "detail": str(detail or "").strip(),
                    "timestamp": self._pipeline_timestamp(),
                    "metadata": dict(metadata or {}),
                }
            ).model_dump(mode="python")
        )
        current["steps"] = steps
        return self._store_turn_pipeline(current)

    def _complete_turn_pipeline(
        self,
        *,
        final_path: str = "",
        reply_source: str = "",
        should_end: bool = False,
        error: str = "",
    ) -> dict[str, object] | None:
        current = dict(getattr(self.bot, "_last_turn_pipeline", {}) or {})
        if not current:
            return None
        current["completed_at"] = self._pipeline_timestamp()
        if final_path:
            current["final_path"] = str(final_path).strip()
        if reply_source:
            current["reply_source"] = str(reply_source).strip()
        current["should_end"] = bool(should_end)
        current["error"] = str(error or "").strip()
        return self._store_turn_pipeline(current)

    def turn_pipeline_snapshot(self):
        payload = getattr(self.bot, "_last_turn_pipeline", None)
        if not isinstance(payload, dict):
            return None
        return TurnPipelineSnapshot.model_validate(payload).model_dump(mode="python")

    def should_offer_daily_checkin_for_turn(self) -> bool:
        return self.bot.memory.should_do_daily_checkin() and self.bot.session_turn_count() == 0

    def record_user_turn_state(self, stripped_input: str, current_mood: str) -> None:
        should_offer_daily_checkin = self.should_offer_daily_checkin_for_turn()
        with self.bot._session_lock:
            self.bot.session_moods.append(current_mood)
            self.bot._pending_daily_checkin_context = should_offer_daily_checkin
        self.bot.memory.save_mood_state(current_mood)
        self.bot.relationship.update(stripped_input, current_mood)

    def direct_reply_for_input(self, stripped_input: str, current_mood: str) -> str | None:
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
                return self.bot.reply_finalization.finalize(direct_reply, current_mood, stripped_input)
        return None

    def _available_agentic_tools(self, settings: dict[str, object]) -> list[dict[str, object]]:
        tools = []
        for tool in self.bot.get_available_tools():
            name = str(tool.get("function", {}).get("name") or "").strip()
            if name == "set_reminder" and not settings["auto_reminders"]:
                continue
            if name == "web_search" and not settings["auto_web_lookup"]:
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
            try:
                response = self.bot.call_ollama_chat(
                    messages=[
                        {
                            "role": "user",
                            "content": self._reflection_prompt(original_input, current_query, current_observation),
                        }
                    ],
                    options={"temperature": 0.1},
                    response_format="json",
                    purpose="agentic reflection",
                )
                reflection = self.bot.parse_model_json_content(response["message"]["content"])
            except Exception as exc:
                logger.warning("Agentic reflection call failed (attempt %d): %s", attempt + 1, exc)
                break

            if not isinstance(reflection, dict):
                break

            if reflection.get("sufficient"):
                logger.info(
                    "Agentic web reflection accepted observation on attempt %d/%d",
                    attempt + 1,
                    max_retries,
                )
                self.bot.update_planner_debug(
                    planner_status="reflected",
                    planner_reason=f"Reflection pass {attempt + 1}: observation accepted. {reflection.get('reason', '')}",
                )
                break

            refined_query = str(reflection.get("refined_query") or "").strip()
            if not refined_query or refined_query == current_query:
                break

            self.bot.update_planner_debug(
                planner_status="refining",
                planner_reason=f"Reflection pass {attempt + 1}: refining query. {reflection.get('reason', '')}",
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
            try:
                response = await self.bot.call_ollama_chat_async(
                    messages=[
                        {
                            "role": "user",
                            "content": self._reflection_prompt(original_input, current_query, current_observation),
                        }
                    ],
                    options={"temperature": 0.1},
                    response_format="json",
                    purpose="agentic reflection",
                )
                reflection = self.bot.parse_model_json_content(response["message"]["content"])
            except Exception as exc:
                logger.warning("Agentic reflection call failed (attempt %d): %s", attempt + 1, exc)
                break

            if not isinstance(reflection, dict):
                break

            if reflection.get("sufficient"):
                logger.info(
                    "Agentic web reflection accepted observation on attempt %d/%d",
                    attempt + 1,
                    max_retries,
                )
                self.bot.update_planner_debug(
                    planner_status="reflected",
                    planner_reason=f"Reflection pass {attempt + 1}: observation accepted. {reflection.get('reason', '')}",
                )
                break

            refined_query = str(reflection.get("refined_query") or "").strip()
            if not refined_query or refined_query == current_query:
                break

            self.bot.update_planner_debug(
                planner_status="refining",
                planner_reason=f"Reflection pass {attempt + 1}: refining query. {reflection.get('reason', '')}",
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
    def _planning_prompt(stripped_input: str, current_mood: str, tools: list[dict[str, object]]) -> str:
        return f"""
You are Dad helping Tony. Think carefully about his latest message.

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
        try:
            plan = self.bot.parse_model_json_content(content)
        except (TypeError, json.JSONDecodeError, KeyError):
            self.bot.update_planner_debug(
                planner_status="fallback",
                planner_reason="Planner returned invalid JSON, so heuristic fallback stayed available.",
            )
            return None
        if not isinstance(plan, dict):
            self.bot.update_planner_debug(
                planner_status="fallback",
                planner_reason="Planner returned an unexpected payload shape, so heuristic fallback stayed available.",
            )
            return None

        tool_name = str(plan.get("tool") or "").strip()
        payload = {
            "needs_tool": bool(plan.get("needs_tool")),
            "tool": tool_name if tool_name in self.TOOL_EXECUTOR_NAMES else None,
            "parameters": dict(plan.get("parameters") or {}) if isinstance(plan.get("parameters"), dict) else {},
            "reason": str(plan.get("reason") or "").strip(),
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
        sandbox = ToolSandbox()
        title = str(params.get("title") or stripped_input[:100]).strip()
        due_text = str(params.get("due_text") or "").strip()

        def _executor():
            return self.bot.add_reminder(title, due_text)

        def _compensate():
            created = sandbox.snapshot()["records"][-1]["result"] if sandbox.snapshot()["records"] else None
            if isinstance(created, dict) and created.get("id"):
                delete_fn = getattr(self.bot, "delete_reminder", None)
                if callable(delete_fn):
                    delete_fn(str(created["id"]))

        record = sandbox.execute(
            tool_name="set_reminder",
            parameters=dict(params),
            executor=_executor,
            compensating_action=_compensate,
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
            return self.bot.reply_finalization.finalize(reply, current_mood, stripped_input), None
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
        sandbox = ToolSandbox()
        title = str(params.get("title") or stripped_input[:100]).strip()
        due_text = str(params.get("due_text") or "").strip()

        def _executor():
            return self.bot.add_reminder(title, due_text)

        record = sandbox.execute(
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
            return await self.bot.reply_finalization.finalize_async(reply, current_mood, stripped_input), None
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
        sandbox = ToolSandbox()
        query = str(params.get("query") or stripped_input).strip()
        normalized_query = self.bot.normalize_lookup_query(query)

        record = sandbox.execute(
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
            observation = self._reflect_on_web_observation(stripped_input, normalized_query, observation, settings)
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
        sandbox = ToolSandbox()
        query = str(params.get("query") or stripped_input).strip()
        normalized_query = self.bot.normalize_lookup_query(query)

        record = sandbox.execute(
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
            "set_reminder": self._execute_set_reminder_tool_sync,
            "web_search": self._execute_web_search_tool_sync,
        }
        enabled = {
            "set_reminder": bool(settings.get("auto_reminders")),
            "web_search": bool(settings.get("auto_web_lookup")),
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
        if tool_name == "set_reminder":
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
            "set_reminder": self._execute_set_reminder_tool_async,
            "web_search": self._execute_web_search_tool_async,
        }
        enabled = {
            "set_reminder": bool(settings.get("auto_reminders")),
            "web_search": bool(settings.get("auto_web_lookup")),
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
        if tool_name == "set_reminder":
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

    # Tool bias → permitted tools mapping.  The Bayesian layer is the SOLE authority
    # over which tools may be executed; the planner is advisory within that gate.
    _TOOL_BIAS_PERMISSIONS: dict[str, frozenset[str]] = {
        "planner_default": frozenset({"set_reminder", "web_search"}),
        "optional_tools": frozenset({"set_reminder", "web_search"}),
        "minimal_tools": frozenset({"set_reminder"}),           # acute_stress: no web noise
        "defer_tools_unless_explicit": frozenset(),             # guarded: block all tools
    }
    _DEFAULT_TOOL_BIAS_PERMISSIONS: frozenset[str] = frozenset({"set_reminder", "web_search"})

    @classmethod
    def _permitted_tools_for_bias(cls, tool_bias: str) -> frozenset[str]:
        return cls._TOOL_BIAS_PERMISSIONS.get(
            str(tool_bias or "planner_default"),
            cls._DEFAULT_TOOL_BIAS_PERMISSIONS,
        )

    def _bayesian_tool_gate(
        self,
        *,
        tool_name: str,
        tool_bias: str,
        plan_reason: str,
    ) -> tuple[bool, str]:
        """Return (allowed, reason).  The Bayesian policy is the FINAL authority."""
        permitted = self._permitted_tools_for_bias(tool_bias)
        if not permitted:
            return False, (
                f"Bayesian policy '{tool_bias}' blocks all tools; "
                "tool selection overridden by governing Bayesian authority."
            )
        if tool_name not in permitted:
            return False, (
                f"Bayesian policy '{tool_bias}' does not permit tool '{tool_name}'; "
                f"permitted: {sorted(permitted)}."
            )
        return True, plan_reason or f"Bayesian policy '{tool_bias}' permits tool '{tool_name}'."

    def plan_agentic_tools(self, stripped_input: str, current_mood: str) -> tuple[str | None, str | None]:
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
            response = self.bot.call_ollama_chat(
                messages=[{"role": "user", "content": self._planning_prompt(stripped_input, current_mood, tools)}],
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
            tool_bias = str(self.bot.planner_debug_snapshot().get("bayesian_tool_bias") or "planner_default")
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
            self.bot.update_planner_debug(
                planner_status="fallback",
                planner_reason=f"Planner error: {exc}",
            )
            logger.warning("Agentic tool planning failed: %s", exc)

        return None, None

    async def plan_agentic_tools_async(self, stripped_input: str, current_mood: str) -> tuple[str | None, str | None]:
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
            response = await self.bot.call_ollama_chat_async(
                messages=[{"role": "user", "content": self._planning_prompt(stripped_input, current_mood, tools)}],
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
            tool_bias = str(self.bot.planner_debug_snapshot().get("bayesian_tool_bias") or "planner_default")
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
            self.bot.update_planner_debug(
                planner_status="fallback",
                planner_reason=f"Planner error: {exc}",
            )
            logger.warning("Agentic tool planning failed: %s", exc)

        return None, None

    def prepare_user_turn(self, stripped_input: str, attachments: AttachmentList | None = None) -> PreparedTurnResult:
        self._start_turn_pipeline("sync", stripped_input)
        normalized_attachments = self.bot.normalize_chat_attachments(attachments)
        normalized_attachments = self.bot.enrich_multimodal_attachments(normalized_attachments, user_input=stripped_input)
        self._append_turn_pipeline_step("normalize_attachments", detail=f"attachments={len(normalized_attachments)}")
        turn_text = self.bot.compose_user_turn_text(stripped_input, normalized_attachments)
        self._append_turn_pipeline_step("compose_turn_text", detail="composed user turn text" if turn_text else "empty composed turn")
        if not turn_text:
            self._complete_turn_pipeline(final_path="empty_input", reply_source="none")
            return None, None, False, "", normalized_attachments

        if self.bot.is_session_exit_command(stripped_input):
            self.bot.persist_conversation()
            self.bot.mark_chat_thread_closed(closed=True)
            self._append_turn_pipeline_step("session_exit", detail="handled exit command")
            self._complete_turn_pipeline(final_path="session_exit", reply_source="exit_command", should_end=True)
            return None, self.bot.reply_finalization.append_signoff("Catch ya later, Tony! Always here if you need me."), True, turn_text, normalized_attachments

        mood_history_window = self.bot.runtime_config.window("mood_detection_context", 6)
        if self.bot.LIGHT_MODE:
            current_mood = "neutral"
        else:
            current_mood = self.bot.mood_manager.detect(turn_text, self.bot.prompt_history()[-mood_history_window:])
        self._update_turn_pipeline(current_mood=current_mood)
        self._append_turn_pipeline_step("detect_mood", detail=f"current_mood={current_mood}")
        self.record_user_turn_state(turn_text, current_mood)
        self._append_turn_pipeline_step("record_turn_state", detail="saved mood and relationship state")
        self.bot.begin_planner_debug(stripped_input, current_mood)
        self._append_turn_pipeline_step("begin_planner_debug", detail="initialized planner debug state")

        if direct_reply := self.direct_reply_for_input(stripped_input, current_mood):
            self.bot.update_planner_debug(
                planner_status="skipped",
                planner_reason="A direct reply path handled this turn before tool planning.",
                final_path="direct_reply",
            )
            self._append_turn_pipeline_step("direct_reply", detail="reply produced before tool planning")
            self._update_turn_pipeline(final_path="direct_reply", reply_source="direct_reply")
            return current_mood, direct_reply, False, turn_text, normalized_attachments

        auto_reply, tool_observation = self.plan_agentic_tools(stripped_input, current_mood)
        planner_snapshot = self.bot.planner_debug_snapshot()
        self._append_turn_pipeline_step("plan_tools", detail=f"planner_status={planner_snapshot.get('planner_status', 'idle')}")
        if auto_reply is not None:
            self._update_turn_pipeline(
                final_path=str(planner_snapshot.get("final_path") or "planner_tool").strip() or "planner_tool",
                reply_source="planner_tool",
            )
            return current_mood, auto_reply, False, turn_text, normalized_attachments

        if tool_observation is None:
            auto_reply, tool_observation = self.bot.autonomous_tool_result_for_input(
                stripped_input,
                current_mood,
                normalized_attachments,
            )
            self._append_turn_pipeline_step("heuristic_tools", detail="evaluated heuristic tool routing")
        self.bot.set_active_tool_observation(tool_observation)
        if tool_observation:
            self._append_turn_pipeline_step("tool_observation", detail="captured tool observation for reply generation")
        if auto_reply is not None:
            planner_snapshot = self.bot.planner_debug_snapshot()
            self._update_turn_pipeline(
                final_path=str(planner_snapshot.get("final_path") or "heuristic_tool").strip() or "heuristic_tool",
                reply_source="heuristic_tool",
            )
            return current_mood, auto_reply, False, turn_text, normalized_attachments

        self._update_turn_pipeline(final_path="model_reply", reply_source="model_generation")
        return current_mood, None, False, turn_text, normalized_attachments

    async def prepare_user_turn_async(self, stripped_input: str, attachments: AttachmentList | None = None) -> PreparedTurnResult:
        self._start_turn_pipeline("async", stripped_input)
        normalized_attachments = self.bot.normalize_chat_attachments(attachments)
        normalized_attachments = self.bot.enrich_multimodal_attachments(normalized_attachments, user_input=stripped_input)
        self._append_turn_pipeline_step("normalize_attachments", detail=f"attachments={len(normalized_attachments)}")
        turn_text = self.bot.compose_user_turn_text(stripped_input, normalized_attachments)
        self._append_turn_pipeline_step("compose_turn_text", detail="composed user turn text" if turn_text else "empty composed turn")
        if not turn_text:
            self._complete_turn_pipeline(final_path="empty_input", reply_source="none")
            return None, None, False, "", normalized_attachments

        if self.bot.is_session_exit_command(stripped_input):
            self.bot.persist_conversation()
            self.bot.mark_chat_thread_closed(closed=True)
            self._append_turn_pipeline_step("session_exit", detail="handled exit command")
            self._complete_turn_pipeline(final_path="session_exit", reply_source="exit_command", should_end=True)
            return None, self.bot.reply_finalization.append_signoff("Catch ya later, Tony! Always here if you need me."), True, turn_text, normalized_attachments

        mood_history_window = self.bot.runtime_config.window("mood_detection_context", 6)
        if self.bot.LIGHT_MODE:
            current_mood = "neutral"
        else:
            current_mood = await self.bot.mood_manager.detect_async(turn_text, self.bot.prompt_history()[-mood_history_window:])
        self._update_turn_pipeline(current_mood=current_mood)
        self._append_turn_pipeline_step("detect_mood", detail=f"current_mood={current_mood}")
        self.record_user_turn_state(turn_text, current_mood)
        self._append_turn_pipeline_step("record_turn_state", detail="saved mood and relationship state")
        self.bot.begin_planner_debug(stripped_input, current_mood)
        self._append_turn_pipeline_step("begin_planner_debug", detail="initialized planner debug state")

        if direct_reply := self.direct_reply_for_input(stripped_input, current_mood):
            self.bot.update_planner_debug(
                planner_status="skipped",
                planner_reason="A direct reply path handled this turn before tool planning.",
                final_path="direct_reply",
            )
            self._append_turn_pipeline_step("direct_reply", detail="reply produced before tool planning")
            self._update_turn_pipeline(final_path="direct_reply", reply_source="direct_reply")
            return current_mood, direct_reply, False, turn_text, normalized_attachments

        auto_reply, tool_observation = await self.plan_agentic_tools_async(stripped_input, current_mood)
        planner_snapshot = self.bot.planner_debug_snapshot()
        self._append_turn_pipeline_step("plan_tools", detail=f"planner_status={planner_snapshot.get('planner_status', 'idle')}")
        if auto_reply is not None:
            self._update_turn_pipeline(
                final_path=str(planner_snapshot.get("final_path") or "planner_tool").strip() or "planner_tool",
                reply_source="planner_tool",
            )
            return current_mood, auto_reply, False, turn_text, normalized_attachments

        if tool_observation is None:
            auto_reply, tool_observation = self.bot.autonomous_tool_result_for_input(
                stripped_input,
                current_mood,
                normalized_attachments,
            )
            self._append_turn_pipeline_step("heuristic_tools", detail="evaluated heuristic tool routing")
        self.bot.set_active_tool_observation(tool_observation)
        if tool_observation:
            self._append_turn_pipeline_step("tool_observation", detail="captured tool observation for reply generation")
        if auto_reply is not None:
            planner_snapshot = self.bot.planner_debug_snapshot()
            self._update_turn_pipeline(
                final_path=str(planner_snapshot.get("final_path") or "heuristic_tool").strip() or "heuristic_tool",
                reply_source="heuristic_tool",
            )
            return current_mood, auto_reply, False, turn_text, normalized_attachments

        self._update_turn_pipeline(final_path="model_reply", reply_source="model_generation")
        return current_mood, None, False, turn_text, normalized_attachments

    def finalize_user_turn(
        self,
        stripped_input: str,
        current_mood: str | None,
        dad_reply: str | None,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        self._append_turn_pipeline_step("finalize_turn", detail="persisted conversation turn")
        with self.bot._session_lock:
            user_turn = {"role": "user", "content": stripped_input, "mood": current_mood}
            if attachments:
                user_turn["attachments"] = [self.bot.history_attachment_metadata(attachment) for attachment in attachments]
            self.bot.history.append(user_turn)
            self.bot.history.append({"role": "assistant", "content": dad_reply})
            self.bot._pending_daily_checkin_context = False
            self.bot._active_tool_observation_context = None
        self.bot.sync_active_thread_snapshot()
        if not self.bot.LIGHT_MODE:
            self.bot.schedule_post_turn_maintenance(stripped_input, current_mood)
            self._append_turn_pipeline_step("schedule_maintenance", detail="queued post-turn maintenance")
        else:
            self._append_turn_pipeline_step("schedule_maintenance", status="skipped", detail="light mode skips maintenance")
        try:
            self.bot.internal_state_manager.reflect_after_turn(stripped_input, current_mood or "neutral", dad_reply or "")
            self._append_turn_pipeline_step("internal_reflection", detail="updated persistent internal state")
        except Exception as exc:
            logger.warning("Internal state reflection failed: %s", exc)
            self._append_turn_pipeline_step("internal_reflection", status="error", detail=str(exc))
        self.bot.current_runtime_health_snapshot(force=True, log_warnings=True, persist=True)
        self._append_turn_pipeline_step("health_snapshot", detail="refreshed runtime health snapshot")
        self._complete_turn_pipeline(should_end=False)
        return dad_reply, False

    def process_user_message(self, user_input: str, attachments: AttachmentList | None = None) -> FinalizedTurnResult:
        stripped_input = user_input.strip()
        if not stripped_input and not attachments:
            return None, False

        current_mood, dad_reply, should_end, turn_text, normalized_attachments = self.prepare_user_turn(
            stripped_input,
            attachments=attachments,
        )
        if should_end:
            return dad_reply, True

        if dad_reply is None:
            self._append_turn_pipeline_step("generate_reply", status="running", detail="generating sync reply")
            dad_reply = self.reply_generation.generate_validated_reply(
                stripped_input,
                turn_text,
                current_mood,
                normalized_attachments,
                stream=False,
            )
            self._append_turn_pipeline_step("generate_reply", detail="generated sync reply")

        return self.finalize_user_turn(turn_text, current_mood, dad_reply, normalized_attachments)

    async def process_user_message_async(self, user_input: str, attachments: AttachmentList | None = None) -> FinalizedTurnResult:
        stripped_input = user_input.strip()
        if not stripped_input and not attachments:
            return None, False

        current_mood, dad_reply, should_end, turn_text, normalized_attachments = await self.prepare_user_turn_async(
            stripped_input,
            attachments=attachments,
        )
        if should_end:
            return dad_reply, True

        if dad_reply is None:
            self._append_turn_pipeline_step("generate_reply", status="running", detail="generating async reply")
            dad_reply = await self.reply_generation.generate_validated_reply_async(
                stripped_input,
                turn_text,
                current_mood,
                normalized_attachments,
            )
            self._append_turn_pipeline_step("generate_reply", detail="generated async reply")

        return self.finalize_user_turn(turn_text, current_mood, dad_reply, normalized_attachments)

    def process_user_message_stream(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        chunk_callback: ChunkCallback | None = None,
    ) -> FinalizedTurnResult:
        stripped_input = user_input.strip()
        if not stripped_input and not attachments:
            return None, False

        current_mood, dad_reply, should_end, turn_text, normalized_attachments = self.prepare_user_turn(
            stripped_input,
            attachments=attachments,
        )
        if should_end:
            return dad_reply, True

        if dad_reply is None:
            self._update_turn_pipeline(mode="stream_sync")
            self._append_turn_pipeline_step("generate_reply", status="running", detail="generating streamed sync reply")
            dad_reply = self.reply_generation.generate_validated_reply(
                stripped_input,
                turn_text,
                current_mood,
                normalized_attachments,
                stream=True,
                chunk_callback=chunk_callback,
            )
            self._append_turn_pipeline_step("generate_reply", detail="generated streamed sync reply")

        return self.finalize_user_turn(turn_text, current_mood, dad_reply, normalized_attachments)

    async def process_user_message_stream_async(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        chunk_callback: ChunkCallback | None = None,
    ) -> FinalizedTurnResult:
        stripped_input = user_input.strip()
        if not stripped_input and not attachments:
            return None, False

        current_mood, dad_reply, should_end, turn_text, normalized_attachments = await self.prepare_user_turn_async(
            stripped_input,
            attachments=attachments,
        )
        if should_end:
            return dad_reply, True

        if dad_reply is None:
            self._update_turn_pipeline(mode="stream_async")
            self._append_turn_pipeline_step("generate_reply", status="running", detail="generating streamed async reply")
            dad_reply = await self.reply_generation.generate_validated_reply_async(
                stripped_input,
                turn_text,
                current_mood,
                normalized_attachments,
                stream=True,
                chunk_callback=chunk_callback,
            )
            self._append_turn_pipeline_step("generate_reply", detail="generated streamed async reply")

        return self.finalize_user_turn(turn_text, current_mood, dad_reply, normalized_attachments)


__all__ = ["TurnService"]