"""Shared UI utility helpers â€” no dependency on dad_streamlit."""

from __future__ import annotations

import streamlit as st

__all__ = [
    "ambient_fragment",
    "enabled_label",
    "filter_memory_entries",
    "maybe_fragment",
    "option_index",
    "titleize_token",
]


def maybe_fragment(func):
    """Use st.fragment when available; otherwise keep a no-op wrapper."""
    fragment = getattr(st, "fragment", None)
    if callable(fragment):
        return fragment(func)
    return func


def ambient_fragment(run_every: float = 2):
    """Decorator factory that creates a self-refreshing fragment.

    ``@ambient_fragment(run_every=2)`` reruns the decorated function every
    *run_every* seconds via ``@st.fragment(run_every=N)``.  Falls back to a
    plain ``@maybe_fragment`` if ``st.fragment`` does not accept that kwarg
    (Streamlit < 1.37).
    """

    def decorator(func):
        fragment_fn = getattr(st, "fragment", None)
        if callable(fragment_fn):
            try:
                return fragment_fn(run_every=run_every)(func)
            except TypeError:
                # Streamlit version doesn't support run_every
                return fragment_fn(func)
        return func

    return decorator


@st.cache_data(show_spinner=False, ttl=90)
def filter_memory_entries(
    all_memories: list[dict],
    search: str,
    category: str,
) -> list[dict]:
    filtered = list(all_memories or [])
    search_text = str(search or "").strip().lower()
    category_text = str(category or "all").strip().lower()

    if search_text:
        filtered = [entry for entry in filtered if search_text in str(entry.get("summary") or "").lower()]
    if category_text and category_text != "all":
        filtered = [
            entry for entry in filtered if str(entry.get("category") or "general").strip().lower() == category_text
        ]

    filtered.sort(
        key=lambda entry: (
            not bool(entry.get("pinned")),
            -float(
                entry.get("importance_score") or entry.get("importance", 0.5) or 0.5,
            ),
        ),
    )
    return filtered


def option_index(options, value, fallback=0):
    try:
        return list(options).index(value)
    except ValueError:
        return int(fallback)


def titleize_token(value, fallback="Unknown"):
    cleaned = str(value or "").strip()
    if not cleaned:
        return fallback
    return cleaned.replace("_", " ").title()


def enabled_label(value):
    return "On" if value else "Off"
