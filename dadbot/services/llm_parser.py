from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, TypeVar

from pydantic import BaseModel, ValidationError

TModel = TypeVar("TModel", bound=BaseModel)


def parse_json_object(
    content: str,
    parse_json: Callable[[str], Any],
) -> dict[str, Any] | None:
    try:
        payload = parse_json(content)
    except (json.JSONDecodeError, TypeError, KeyError):
        return None
    if not isinstance(payload, dict):
        return None
    return dict(payload)


def parse_json_model(
    payload: dict[str, Any],
    schema: type[TModel],
) -> TModel | None:
    try:
        return schema.model_validate(payload)
    except ValidationError:
        return None


def call_json_object_sync(
    *,
    call_llm: Callable[[], Any],
    extract_content: Callable[[Any], str],
    parse_json: Callable[[str], Any],
    max_attempts: int = 2,
) -> dict[str, Any] | None:
    for _ in range(max(1, int(max_attempts))):
        try:
            response = call_llm()
            content = extract_content(response)
        except (RuntimeError, KeyError, TypeError):
            return None
        parsed = parse_json_object(str(content or ""), parse_json)
        if parsed is not None:
            return parsed
    return None


async def call_json_object_async(
    *,
    call_llm: Callable[[], Awaitable[Any]],
    extract_content: Callable[[Any], str],
    parse_json: Callable[[str], Any],
    max_attempts: int = 2,
) -> dict[str, Any] | None:
    for _ in range(max(1, int(max_attempts))):
        try:
            response = await call_llm()
            content = extract_content(response)
        except (RuntimeError, KeyError, TypeError):
            return None
        parsed = parse_json_object(str(content or ""), parse_json)
        if parsed is not None:
            return parsed
    return None


__all__ = [
    "call_json_object_async",
    "call_json_object_sync",
    "parse_json_model",
    "parse_json_object",
]
