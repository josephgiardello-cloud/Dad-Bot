"""Tool-routing, planning, reflection, and execution methods for TurnService.

Extracted as a mixin to reduce the size of turn_service.py while keeping
the public API identical. TurnService inherits from this class.
"""
from __future__ import annotations

import asyncio
import json
import logging
import string
from typing import Any

from pydantic import BaseModel, ValidationError

from dadbot.contracts import AttachmentList
from dadbot.core.execution_result_unified import get_unified_execution_result
from dadbot.core.memory_set_invariants import (
    MemorySetInvariantViolation,
    assert_memory_set_invariants,
    record_causal_step_locked,
)
from dadbot.core.system_state_algebra import (
    evaluate_system_state_algebra,
    persist_system_state_algebra,
)
from dadbot.core.tool_executor import execute_tool
from dadbot.models import AgenticToolPlan

# Keep the historic logger name so existing tests and log pipelines remain stable.
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


class PlannerExecutionError(RuntimeError):
    """Planner runtime failure that should be distinguished from 'no tool needed'."""


def _is_event_loop_closed_error(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and str(exc).find(_EVENT_LOOP_CLOSED_ERROR) != -1


class ToolPipelineMixin:
    """Tool routing, planning, reflection, validation, and execution methods.

    Mixed into TurnService. All methods access instance attributes set by
    TurnService.__init__ (self.bot, self._llm_adapter, self._memory_service, etc.).
    """

    @staticmethod
    def _classify_planner_failure(exc: BaseException) -> tuple[str, str]:
        if _is_event_loop_closed_error(exc):
            return "fallback", "Planner skipped: event loop closed"
        if isinstance(exc, ValidationError):
            return "planner_validation_error", f"Planner response validation failed: {exc}"
        if isinstance(exc, (json.JSONDecodeError, ValueError, TypeError)):
            return "planner_parse_error", f"Planner response parse failed: {exc}"
        return "planner_error", f"Planner error: {exc}"

    def _handle_planner_failure(
        self,
        *,
        exc: BaseException,
        turn_context: Any | None,
    ) -> tuple[str | None, str | None]:
        status, reason = self._classify_planner_failure(exc)
        self._record_tool_decision_outcome(
            turn_context=turn_context,
            decision_outcome="planner_failed",
        )
        self.bot.update_planner_debug(
            planner_status=status,
            planner_reason=reason,
        )
        logger.warning("Agentic tool planning failed: %s", exc)
        if status == "fallback":
            return None, None
        return None, reason

    def _deterministic_tool_route(self, stripped_input: str) -> tuple[str, dict[str, object], str] | None:
        parse_tool_command = getattr(self.bot, "parse_tool_command", None)
        if not callable(parse_tool_command):
            command = None
        else:
            command = parse_tool_command(stripped_input)

        if not isinstance(command, dict):
            command = {}

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

        normalized = str(stripped_input or "").strip()
        lowered = normalized.lower()
        reminder_prefixes = (
            "remind me to ",
            "remind me ",
            "set a reminder to ",
            "set reminder to ",
        )
        if any(lowered.startswith(prefix) for prefix in reminder_prefixes):
            detail = normalized
            for prefix in reminder_prefixes:
                if lowered.startswith(prefix):
                    detail = normalized[len(prefix):].strip()
                    break
            detail = detail or normalized
            split_details = getattr(self.bot, "split_reminder_details", None)
            if callable(split_details):
                title, due_text = split_details(detail)
            else:
                title, due_text = detail, ""
            params = {
                "title": str(title or detail or normalized[:100]).strip(),
                "due_text": str(due_text or "").strip(),
            }
            return _SET_REMINDER_TOOL, params, "Natural-language reminder request routed to executor."

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
        memory_hints = ToolPipelineMixin._format_retrieval_hints(retrieval_set or [])
        memory_brief = ToolPipelineMixin._format_memory_brief(memory_influence_brief)
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
            failure_observation = (
                "Reminder tool execution failed before a confirmation could be generated."
            )
            recorder = getattr(self.bot, "record_shadow_decision", None)
            if callable(recorder):
                recorder(
                    source="tool",
                    type="suggestion",
                    content_preview=failure_observation,
                    reason="Reminder tool failed; observation emitted for ResponseEngine consideration.",
                    would_replace=False,
                    priority=0.35,
                    metadata={"tool": "set_reminder", "status": "failed", "path": "sync"},
                    turn_context=turn_context,
                )
            return None, failure_observation
        reminder = record.result
        if reminder:
            self.bot.update_planner_debug(
                planner_status="used_tool",
                planner_reason=plan_reason or "Planner selected a reminder tool.",
                planner_tool="set_reminder",
                planner_parameters=params,
                final_path="planner_tool",
            )
            observation = self._reminder_confirmation_reply(reminder)
            recorder = getattr(self.bot, "record_shadow_decision", None)
            if callable(recorder):
                recorder(
                    source="tool",
                    type="suggestion",
                    content_preview=observation,
                    reason="Reminder tool succeeded; observation emitted for ResponseEngine consideration.",
                    would_replace=False,
                    priority=0.55,
                    metadata={"tool": "set_reminder", "status": "success", "path": "sync"},
                    turn_context=turn_context,
                )
            # Legacy planning contract: reminder confirmations are returned as
            # tool replies, not planner observations.
            return observation, None
        self.bot.update_planner_debug(
            planner_status="execution_failed",
            planner_reason="Planner selected set_reminder, but Dad couldn't create the reminder cleanly.",
            planner_tool="set_reminder",
            planner_parameters=params,
        )
        failure_observation = (
            "Reminder tool returned no usable confirmation payload for this turn."
        )
        recorder = getattr(self.bot, "record_shadow_decision", None)
        if callable(recorder):
            recorder(
                source="tool",
                type="suggestion",
                content_preview=failure_observation,
                reason="Reminder tool returned no usable payload; observation emitted for ResponseEngine consideration.",
                would_replace=False,
                priority=0.30,
                metadata={"tool": "set_reminder", "status": "empty", "path": "sync"},
                turn_context=turn_context,
            )
        return None, failure_observation

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
            failure_observation = (
                "Reminder tool execution failed before a confirmation could be generated."
            )
            recorder = getattr(self.bot, "record_shadow_decision", None)
            if callable(recorder):
                recorder(
                    source="tool",
                    type="suggestion",
                    content_preview=failure_observation,
                    reason="Reminder tool failed; async observation emitted for ResponseEngine consideration.",
                    would_replace=False,
                    priority=0.35,
                    metadata={"tool": "set_reminder", "status": "failed", "path": "async"},
                    turn_context=turn_context,
                )
            return None, failure_observation
        reminder = record.result
        if reminder:
            self.bot.update_planner_debug(
                planner_status="used_tool",
                planner_reason=plan_reason or "Planner selected a reminder tool.",
                planner_tool="set_reminder",
                planner_parameters=params,
                final_path="planner_tool",
            )
            observation = self._reminder_confirmation_reply(reminder)
            recorder = getattr(self.bot, "record_shadow_decision", None)
            if callable(recorder):
                recorder(
                    source="tool",
                    type="suggestion",
                    content_preview=observation,
                    reason="Reminder tool succeeded; async observation emitted for ResponseEngine consideration.",
                    would_replace=False,
                    priority=0.55,
                    metadata={"tool": "set_reminder", "status": "success", "path": "async"},
                    turn_context=turn_context,
                )
            # Keep async contract aligned with sync reminder planning behavior.
            return observation, None
        self.bot.update_planner_debug(
            planner_status="execution_failed",
            planner_reason="Planner selected set_reminder, but Dad couldn't create the reminder cleanly.",
            planner_tool="set_reminder",
            planner_parameters=params,
        )
        failure_observation = (
            "Reminder tool returned no usable confirmation payload for this turn."
        )
        recorder = getattr(self.bot, "record_shadow_decision", None)
        if callable(recorder):
            recorder(
                source="tool",
                type="suggestion",
                content_preview=failure_observation,
                reason="Reminder tool returned no usable payload; async observation emitted for ResponseEngine consideration.",
                would_replace=False,
                priority=0.30,
                metadata={"tool": "set_reminder", "status": "empty", "path": "async"},
                turn_context=turn_context,
            )
        return None, failure_observation

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

        # ALL CHECKS PASSED
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
        words = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
        # Common stopwords to ignore
        stopwords = {
            "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "do", "does", "did", "will", "would", "should", "could", "may",
            "can", "have", "has", "had", "will", "would", "should", "could",
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
            return self._handle_planner_failure(exc=exc, turn_context=turn_context)

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
            return self._handle_planner_failure(exc=exc, turn_context=turn_context)

        return None, None
