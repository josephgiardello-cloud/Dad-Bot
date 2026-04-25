"""Small shared utilities for the dadbot package split."""

from __future__ import annotations

import json
from functools import lru_cache

try:
    import orjson
except ImportError:
    orjson = None


SIGNIFICANT_TOKEN_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "buddy", "but", "by", "did",
    "do", "for", "from", "got", "had", "have", "i", "in", "is", "it", "kid", "my",
    "of", "on", "our", "really", "since", "so", "some", "that", "the", "their", "then",
    "to", "too", "up", "was", "we", "were", "where", "your",
})


def env_truthy(name, default=False):
    import os

    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def json_dumps(value, *, indent=None, sort_keys=False):
    if orjson is not None:
        option = 0
        if indent:
            option |= orjson.OPT_INDENT_2
        if sort_keys:
            option |= orjson.OPT_SORT_KEYS
        return orjson.dumps(value, option=option).decode("utf-8")
    return json.dumps(value, indent=indent, sort_keys=sort_keys)


def json_dump(value, file_handle, *, indent=None, sort_keys=False):
    file_handle.write(json_dumps(value, indent=indent, sort_keys=sort_keys))


def json_loads(value):
    if orjson is not None:
        return orjson.loads(value)
    return json.loads(value)


def json_load(file_handle):
    return json_loads(file_handle.read())


@lru_cache(maxsize=8192)
def normalize_memory_text(text):
    return " ".join(str(text or "").strip().lower().split())


@lru_cache(maxsize=8192)
def tokenize_text(text):
    import re

    return frozenset(re.findall(r"[a-z0-9']+", str(text or "").lower()))


@lru_cache(maxsize=8192)
def significant_tokens(text):
    return frozenset(
        token
        for token in tokenize_text(text)
        if token not in SIGNIFICANT_TOKEN_STOPWORDS and (len(token) > 3 or token.isdigit())
    )


__all__ = [
    "env_truthy",
    "json_dump",
    "json_dumps",
    "json_load",
    "json_loads",
    "normalize_memory_text",
    "significant_tokens",
    "tokenize_text",
]
