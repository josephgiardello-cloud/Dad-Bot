"""LLM response extraction, JSON parsing, and transcript utilities.

Pure functions with no DadBot dependency::

    from dadbot.utils.llm import extract_ollama_message_payload, parse_model_json_content
"""

from __future__ import annotations

import json
import re


def extract_ollama_message_payload(response) -> dict:
    """Extract the message payload dict from an Ollama (or OpenAI-compat) response."""
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            message = first_choice.get("message") or first_choice.get("delta") or {}
            return dict(message) if isinstance(message, dict) else {}
        message = response.get("message") or {}
        return dict(message) if isinstance(message, dict) else {}

    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        message = getattr(first_choice, "message", None) or getattr(
            first_choice,
            "delta",
            None,
        )
        if message is not None and hasattr(message, "model_dump"):
            dumped = message.model_dump(exclude_none=True)
            return dumped if isinstance(dumped, dict) else {}
        if isinstance(message, dict):
            return dict(message)

    message = getattr(response, "message", None)
    if message is None:
        return {}
    if hasattr(message, "model_dump"):
        dumped = message.model_dump(exclude_none=True)
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(message, dict):
        return dict(message)

    payload: dict = {}
    for field_name in ("role", "content", "thinking", "tool_calls"):
        value = getattr(message, field_name, None)
        if value is not None:
            payload[field_name] = value
    return payload


def extract_ollama_message_content(response) -> str:
    """Return the text content of an Ollama response, with a thinking fallback."""
    payload = extract_ollama_message_payload(response)
    content = payload.get("content")
    if not content and "thinking" in payload:
        return f"(Thought: {payload['thinking']})"
    return str(content or "").strip()


def transcript_from_messages(messages) -> str:
    """Format a list of chat history dicts as a readable Tony/Dad transcript."""
    lines: list[str] = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        speaker = "Tony" if role == "user" else "Dad"
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def parse_model_json_content(content):
    """Parse JSON from a model reply, stripping optional markdown fences.

    Tries each ``{`` / ``[`` start position so partial prefix text (like
    a brief explanation before the JSON block) does not cause a parse failure.
    """
    from dadbot.utils import json_loads  # local import to avoid circular dependency

    text = str(content or "").strip()
    if not text:
        raise json.JSONDecodeError("Empty JSON content", text, 0)

    fenced_match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced_match:
        text = fenced_match.group(1).strip()

    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            return json_loads(text[index:])
        except json.JSONDecodeError:
            continue

    return json_loads(text)


__all__ = [
    "extract_ollama_message_content",
    "extract_ollama_message_payload",
    "parse_model_json_content",
    "transcript_from_messages",
]
