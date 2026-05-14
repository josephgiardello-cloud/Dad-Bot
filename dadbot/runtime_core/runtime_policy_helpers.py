"""Policy-level helpers for runtime intent classification."""
from __future__ import annotations

import re


def intent_label(user_input: str) -> str:
    """Return a normalized intent label derived from user input text."""
    text = (user_input or "").strip().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = text.split()
    return "_".join(tokens[:4]) if tokens else "default"


__all__ = ["intent_label"]
