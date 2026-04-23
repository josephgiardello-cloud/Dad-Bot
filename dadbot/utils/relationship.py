"""Relationship scoring, decay, recency, and confidence utilities.

Pure functions with no DadBot dependency.  Use directly::

    from dadbot.utils.relationship import clamp_score, decay_relationship_level
"""

from __future__ import annotations

from datetime import date


def clamp_score(value, minimum: int = 0, maximum: int = 100) -> int:
    """Clamp *value* to [minimum, maximum] as an integer."""
    return max(minimum, min(maximum, int(value)))


def parse_iso_date(value):
    """Parse the first 10 characters of *value* as an ISO-8601 date.

    Returns a :class:`datetime.date` on success, ``None`` on any failure.
    """
    normalized = str(value or "").strip()
    if not normalized:
        return None
    try:
        return date.fromisoformat(normalized[:10])
    except ValueError:
        return None


def days_since_iso_date(value) -> int | None:
    """Return the number of days since *value* (an ISO date string), or ``None``."""
    parsed = parse_iso_date(value)
    if parsed is None:
        return None
    return max(0, (date.today() - parsed).days)


def decay_relationship_level(
    score,
    last_updated,
    midpoint: float = 50,
    daily_decay: float = 0.985,
) -> int:
    """Decay a relationship score back toward *midpoint* based on elapsed days.

    Prevents stale high (or low) scores from freezing in place when there has
    been no recent interaction.
    """
    clamped = clamp_score(score)
    elapsed_days = days_since_iso_date(last_updated)
    if elapsed_days is None or elapsed_days <= 0:
        return clamped

    decay_factor = daily_decay ** min(elapsed_days, 120)
    decayed = midpoint + (clamped - midpoint) * decay_factor
    return clamp_score(round(decayed))


def recency_weight(value, freshest: int = 4) -> int:
    """Return a small integer weight that decreases as *value* (an ISO date) ages."""
    elapsed_days = days_since_iso_date(value)
    if elapsed_days is None:
        return 1
    return max(1, freshest - min(elapsed_days, freshest - 1))


def normalize_confidence(
    value,
    source_count: int = 1,
    contradiction_count: int = 0,
    updated_at=None,
) -> float:
    """Infer a [0.05, 0.98] confidence score from repetition, contradiction, and recency."""
    try:
        if value is None:
            numeric = 0.34 + min(max(int(source_count), 1), 6) * 0.1
            if contradiction_count:
                numeric -= min(0.4, contradiction_count * 0.16)
            parsed = parse_iso_date(updated_at)
            if parsed is not None:
                elapsed_days = max(0, (date.today() - parsed).days)
                numeric += max(0.0, 0.12 - min(elapsed_days, 30) * 0.004)
        else:
            numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.5

    return round(max(0.05, min(0.98, numeric)), 2)


def confidence_label(confidence: float) -> str:
    """Map a numeric confidence value to a human-readable label."""
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "tentative"


__all__ = [
    "clamp_score",
    "confidence_label",
    "days_since_iso_date",
    "decay_relationship_level",
    "normalize_confidence",
    "parse_iso_date",
    "recency_weight",
]
