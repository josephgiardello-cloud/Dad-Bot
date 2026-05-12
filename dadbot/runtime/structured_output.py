"""Structured output layer: LLM reflection → validated Pydantic plan → typed reflection dict.

Usage in AgentDriverLoop:
    from dadbot.runtime.structured_output import build_llm_reflection_hook
    hook = build_llm_reflection_hook(llm_client, tool_names=["search", "memory_write"])
    loop = AgentDriverLoop(kernel, policy=policy)
    result = loop.run(initial_input, reflection_hook=hook)
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator


# ---------------------------------------------------------------------------
# Plan schema — what the LLM must return for each reflection step
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """A single tool invocation the agent plans to make."""

    name: str = Field(..., description="Exact tool name from the allowed list")
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentPlan(BaseModel):
    """Structured reflection output from the LLM. Every field must be present."""

    should_continue: bool = Field(
        True,
        description="False instructs the loop to stop gracefully",
    )
    action_input: str = Field(
        ...,
        description="The exact text to send to the kernel on the next turn",
    )
    reasoning: str = Field(
        "",
        description="Brief internal reasoning (not sent to kernel)",
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Ordered list of tool calls to embed in the action_input",
    )
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Agent self-assessed confidence in this plan (0–1)",
    )

    @field_validator("action_input")
    @classmethod
    def action_input_not_empty(cls, v: str) -> str:
        if not str(v or "").strip():
            raise ValueError("action_input must not be blank")
        return v.strip()


# ---------------------------------------------------------------------------
# Schema validator — parses raw LLM text into AgentPlan
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class SchemaValidationError(Exception):
    """Raised when LLM output cannot be coerced into a valid AgentPlan."""


def _extract_json_block(text: str) -> str:
    """Pull JSON from fenced block or bare object in text."""
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1)
    m = _BARE_JSON_RE.search(text)
    if m:
        return m.group(0)
    raise SchemaValidationError(f"No JSON object found in LLM output: {text!r}")


def parse_agent_plan(raw_text: str, *, allowed_tools: list[str] | None = None) -> AgentPlan:
    """Parse + validate raw LLM text into an AgentPlan.

    Args:
        raw_text: The raw string returned by the LLM.
        allowed_tools: If provided, tool names in tool_calls are validated against this list.

    Returns:
        A fully-validated AgentPlan instance.

    Raises:
        SchemaValidationError: on any parse/validation failure.
    """
    json_str = _extract_json_block(raw_text)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"JSON decode error: {exc} — raw: {json_str!r}") from exc

    try:
        plan = AgentPlan.model_validate(data)
    except ValidationError as exc:
        raise SchemaValidationError(f"Schema validation failed: {exc}") from exc

    if allowed_tools is not None:
        for tc in plan.tool_calls:
            if tc.name not in allowed_tools:
                raise SchemaValidationError(
                    f"Tool {tc.name!r} not in allowed list: {allowed_tools}"
                )

    return plan


# ---------------------------------------------------------------------------
# Reflection prompt builder
# ---------------------------------------------------------------------------

_REFLECTION_SYSTEM_PROMPT = """\
You are an autonomous agent reflection engine.
Given the current turn context, decide:
  1. Whether to continue the loop (`should_continue`)
  2. What text to pass to the kernel next (`action_input`)
  3. Any tool calls needed (`tool_calls`)

Architect pillar behavior (tool chaining):
- Prefer multi-tool sequences when a task needs discovery, impact analysis, and a precise edit.
- For code-change tasks, a canonical sequence is:
    1) `computer_search_directory` to find candidate files/symbols.
    2) `computer_summarize_directory` to estimate scope and blast radius.
    3) `computer_refactor_python_function` (or other edit tool) to apply the fix.
- Keep tool_calls ordered to reflect this chain; do not skip discovery for non-trivial edits.
- Use `reasoning` to explain why a chain is required before acting.

You MUST respond with ONLY valid JSON matching this schema exactly:
{{
  "should_continue": true,
  "action_input": "<next instruction or query>",
  "reasoning": "<brief internal thought>",
  "tool_calls": [{{"name": "<tool>", "arguments": {{}}}}],
  "confidence": 0.9
}}

Allowed tools: {allowed_tools}
Do NOT include any text outside the JSON object.
"""


def build_reflection_prompt(
    ctx: dict[str, Any],
    *,
    allowed_tools: list[str] | None = None,
) -> list[dict[str, str]]:
    """Build the messages list for the LLM reflection call."""
    tools_str = json.dumps(allowed_tools or [])
    system = _REFLECTION_SYSTEM_PROMPT.format(allowed_tools=tools_str)
    user_content = json.dumps(
        {
            "turn_index": ctx.get("turn_index", 1),
            "initial_observation": ctx.get("initial_observation", ""),
            "last_reply": ctx.get("last_reply", ""),
            "completed_turns": len(ctx.get("records", [])),
        },
        ensure_ascii=False,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# LLM reflection hook factory
# ---------------------------------------------------------------------------


def build_llm_reflection_hook(
    llm_client: Any,
    *,
    tool_names: list[str] | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    fallback_on_error: bool = True,
) -> Any:
    """Return a ReflectionHook callable backed by the given LLM client.

    The hook calls the LLM, parses structured output, validates tool names,
    and returns the dict contract expected by AgentDriverLoop.

    Args:
        llm_client: Any object with a `call_llm(messages, ...)` method that
                    returns a string (e.g. dadbot RuntimeClient).
        tool_names: Allowed tool names for schema validation.
        model: Override model name (None = use client default).
        temperature: Sampling temperature for reflection calls.
        fallback_on_error: If True, return a safe no-op plan on parse failure
                           instead of propagating the exception.
    """
    allowed = list(tool_names or [])

    def reflection_hook(ctx: dict[str, Any]) -> dict[str, Any]:
        messages = build_reflection_prompt(ctx, allowed_tools=allowed)
        try:
            call_fn = getattr(llm_client, "call_llm", None)
            if not callable(call_fn):
                raise SchemaValidationError("llm_client has no call_llm method")
            raw: str = call_fn(
                messages,
                model=model,
                temperature=temperature,
                response_format="json",
                purpose="agent_reflection",
            )
            plan = parse_agent_plan(str(raw or ""), allowed_tools=allowed if allowed else None)
            return {
                "should_continue": plan.should_continue,
                "action_input": plan.action_input,
                "reasoning": plan.reasoning,
                "tool_calls": [tc.model_dump() for tc in plan.tool_calls],
                "confidence": plan.confidence,
            }
        except SchemaValidationError as exc:
            error_text = str(exc)
            if fallback_on_error:
                if "not in allowed list" in error_text:
                    return {
                        "should_continue": True,
                        "action_input": "",
                        "reasoning": "tool_hallucination_detected",
                        "tool_calls": [],
                        "confidence": 0.0,
                        "system_observation": (
                            "System observation: attempted a tool that does not exist or is not allowed. "
                            "Apologize and retry with only allowed tools. "
                            f"Validator details: {error_text}"
                        ),
                    }
                # Safe degraded plan: re-use the last observation as action_input
                return {
                    "should_continue": True,
                    "action_input": str(ctx.get("last_reply") or ctx.get("initial_observation") or ""),
                    "reasoning": "fallback: schema validation failed",
                    "tool_calls": [],
                    "confidence": 0.0,
                    "system_observation": "System observation: structured output was invalid. Retry with valid JSON schema.",
                }
            raise

    return reflection_hook


__all__ = [
    "AgentPlan",
    "ToolCall",
    "SchemaValidationError",
    "parse_agent_plan",
    "build_reflection_prompt",
    "build_llm_reflection_hook",
]
