"""PII scrubber for memory entries before persistence.

Detects and redacts personally-identifiable information that should never be
stored verbatim in Dad Bot's memory store.  This is a defence-in-depth layer,
not a substitute for a full DLP pipeline, but it satisfies the basic GDPR/CCPA
requirement that raw sensitive data (cards, SSNs, phone numbers, emails) must
not persist unredacted.

Patterns are compiled once at import time so the scrubbing hot-path is fast.
"""
from __future__ import annotations

import re

# ─── compiled PII patterns ───────────────────────────────────────────────────
_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Visa / MC / Amex / Discover card numbers (with optional separators)
    ("credit_card", re.compile(
        r"\b(?:4[0-9]{3}|5[1-5][0-9]{2}|3[47][0-9]{2}|6(?:011|5[0-9]{2}))"
        r"[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{1,4}\b"
    )),
    # US Social Security Numbers
    ("ssn", re.compile(
        r"\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0{4})\d{4}\b"
    )),
    # US phone numbers in common formats
    ("phone", re.compile(
        r"\b(?:\+?1[-.\s]?)?"
        r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    )),
    # Email addresses
    ("email", re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    )),
    # Passwords / API keys heuristic: long hex or base64-like tokens (≥32 chars)
    ("token", re.compile(
        r"\b[A-Za-z0-9+/]{32,}={0,2}\b"
    )),
    # Routing + account number pairs (bank)
    ("bank_routing", re.compile(
        r"\b(?:routing(?:\s+(?:number|#))?[\s:]+)(\d{9})\b",
        re.IGNORECASE,
    )),
]

_PLACEHOLDER = {
    "credit_card": "[CARD REDACTED]",
    "ssn":         "[SSN REDACTED]",
    "phone":       "[PHONE REDACTED]",
    "email":       "[EMAIL REDACTED]",
    "token":       "[TOKEN REDACTED]",
    "bank_routing": "[ROUTING REDACTED]",
}


def scrub_text(text: str) -> tuple[str, list[str]]:
    """Return (scrubbed_text, list_of_pii_types_found).

    ``list_of_pii_types_found`` is empty when nothing was detected.
    """
    if not text:
        return text, []

    result = text
    found: list[str] = []
    for pii_type, pattern in _PATTERNS:
        replaced, n = pattern.subn(_PLACEHOLDER[pii_type], result)
        if n:
            result = replaced
            found.append(pii_type)

    return result, found


def scrub_memory_entry(entry: dict) -> dict:
    """Return a *new* dict with PII scrubbed from 'summary' and 'detail' fields."""
    if not isinstance(entry, dict):
        return entry

    scrubbed = dict(entry)
    for field in ("summary", "detail", "text", "content"):
        raw = scrubbed.get(field)
        if isinstance(raw, str) and raw:
            cleaned, found = scrub_text(raw)
            scrubbed[field] = cleaned
            if found:
                tags = scrubbed.get("_pii_scrubbed", [])
                scrubbed["_pii_scrubbed"] = list(set(list(tags) + found))

    return scrubbed


def scrub_memory_list(memories: list[dict]) -> list[dict]:
    """Scrub an entire list of memory entries in-place (returns the same list)."""
    return [scrub_memory_entry(m) for m in memories]


def contains_pii(text: str) -> bool:
    """Quick check: True if any PII pattern matches *text*."""
    return any(pattern.search(text) for _, pattern in _PATTERNS)


__all__ = [
    "contains_pii",
    "scrub_memory_entry",
    "scrub_memory_list",
    "scrub_text",
]
