"""Memory formatting, sorting, and naturalization utilities.

Pure functions with no DadBot dependency::

    from dadbot.utils.memory import naturalize_memory_summary, memory_sort_key
"""

from __future__ import annotations

import re


def naturalize_memory_summary(summary: str) -> str:
    """Rewrite a first-person memory summary into third-person Tony-centric prose."""
    summary = re.sub(r"\s+", " ", summary.strip())
    if not summary:
        return ""

    pattern_replacements = (
        (r"^i(?:'ve| have) been\s+", "Tony has been "),
        (r"^i(?:'m| am) trying to\s+", "Tony is trying to "),
        (r"^i(?:'m| am)\s+", "Tony is "),
        (r"^i was\s+", "Tony was "),
        (r"^i feel\s+", "Tony feels "),
        (r"^i(?:'ve| have)\s+", "Tony has "),
        (r"^i want to\s+", "Tony wants to "),
        (r"^i want\s+", "Tony wants "),
        (r"^i need to\s+", "Tony needs to "),
        (r"^i need\s+", "Tony needs "),
        (r"^trying to\s+", "Tony is trying to "),
        (r"^stressed about\s+", "Tony has been stressed about "),
    )

    lowered = summary.lower()
    for pattern, replacement in pattern_replacements:
        rewritten, replacement_count = re.subn(
            pattern,
            replacement,
            summary,
            count=1,
            flags=re.IGNORECASE,
        )
        if replacement_count:
            summary = rewritten
            break
    else:
        if not lowered.startswith("tony "):
            if len(summary) > 1:
                summary = f"Tony shared that {summary[0].lower()}{summary[1:]}"
            else:
                summary = f"Tony shared that {summary.lower()}"

    if summary[-1] not in ".!?":
        summary += "."

    return summary[0].upper() + summary[1:]


def memory_sort_key(memory: dict) -> tuple:
    """Return a sort key that orders memories by recency (newest first)."""
    created_at = memory.get("created_at", "")
    updated_at = memory.get("updated_at", "")
    return (updated_at, created_at, memory.get("summary", ""))


__all__ = [
    "memory_sort_key",
    "naturalize_memory_summary",
]
