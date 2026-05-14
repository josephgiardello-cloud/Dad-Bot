"""Memory → Model Context Injection.

Gap 1 of the causal loop: memory is written correctly (tool → memory ✔) but the
read side — deciding *which* memories get injected into model context and *how* —
was missing.

This module provides the read-side control layer:

  MemoryContextBudget   — hard caps on slots and character count
  InjectionRankStrategy — how candidates are ranked before budget trimming
  MemoryContextCandidate — a scored, annotated entry ready for injection
  MemoryContextInjector — full pipeline: query → score → filter → format

Usage
-----
    injector = MemoryContextInjector(
        budget=MemoryContextBudget(max_slots=5, max_chars=2000),
        strategy=InjectionRankStrategy.SEVERITY_THEN_RECENCY,
    )
    snippets = injector.select(entries, query_context="user asked about payment")
    # snippets is a list of formatted strings ready for prompt construction
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from dadbot.core.tool_memory_causal_contract import CausalMemoryEntry


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryContextBudget:
    """Hard limits on how much memory can be injected into a single model context.

    Attributes
    ----------
    max_slots:
        Maximum number of individual memory entries to inject.
    max_chars:
        Maximum total character count across all injected snippets.
        0 means unlimited.
    min_relevance_score:
        Entries with a computed relevance score below this threshold are dropped.
        Score is in [0, 1].
    """

    max_slots: int = 8
    max_chars: int = 4000
    min_relevance_score: float = 0.0

    def __post_init__(self) -> None:
        if self.max_slots < 1:
            raise ValueError("max_slots must be >= 1")
        if self.max_chars < 0:
            raise ValueError("max_chars must be >= 0")
        if not (0.0 <= self.min_relevance_score <= 1.0):
            raise ValueError("min_relevance_score must be in [0, 1]")


# ---------------------------------------------------------------------------
# Ranking strategy
# ---------------------------------------------------------------------------


class InjectionRankStrategy(Enum):
    """Controls how memory candidates are ordered before budget trimming."""

    RECENCY = "recency"
    """Most-recently written entries first."""

    RELEVANCE = "relevance"
    """Highest lexical/keyword relevance to the current query first."""

    SEVERITY = "severity"
    """Most severe failures first (CRITICAL > HIGH > MEDIUM > LOW > UNKNOWN)."""

    SEVERITY_THEN_RECENCY = "severity_then_recency"
    """Primary: severity; secondary: recency.  Best default for alert-style injection."""

    POLICY_ACTION_WEIGHT = "policy_action_weight"
    """Entries whose policy_action was ESCALATE or ABORT are prioritized as they signal
    the highest-consequence decisions for the model to be aware of."""


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------

# Severity rank for ordering (higher = more critical)
_SEVERITY_RANK: dict[str | None, int] = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
    None: 0,
    "unknown": 0,
}

# Policy action urgency rank
_POLICY_ACTION_RANK: dict[str | None, int] = {
    "escalate": 4,
    "abort": 4,
    "reconcile": 3,
    "retry": 2,
    "fallback": 1,
    "degrade": 1,
    None: 0,
}


@dataclass
class MemoryContextCandidate:
    """An entry that has been scored and annotated for potential injection.

    Attributes
    ----------
    entry:
        The original CausalMemoryEntry.
    relevance_score:
        Float in [0, 1] representing match to the current query context.
        1.0 = exact match; 0.0 = no match.
    formatted_snippet:
        Pre-rendered string for insertion into a prompt.
    """

    entry: CausalMemoryEntry
    relevance_score: float = 0.0
    formatted_snippet: str = ""

    def __post_init__(self) -> None:
        self.relevance_score = max(0.0, min(1.0, self.relevance_score))
        if not self.formatted_snippet:
            self.formatted_snippet = _default_format(self.entry)


def _default_format(entry: CausalMemoryEntry) -> str:
    """Produce a compact, readable string for prompt injection."""
    parts = [f"[TOOL:{entry.tool_name}]", f"status={entry.status}"]
    if entry.failure_class:
        parts.append(f"failure={entry.failure_class}")
    if entry.failure_severity:
        parts.append(f"severity={entry.failure_severity}")
    if entry.policy_action:
        parts.append(f"action={entry.policy_action}")
    if entry.output_preview:
        preview = entry.output_preview[:200]
        parts.append(f"output={preview!r}")
    if entry.error:
        err = entry.error[:200]
        parts.append(f"error={err!r}")
    ts = entry.timestamp_ms // 1000
    parts.append(f"at={ts}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------


def _lexical_relevance(entry: CausalMemoryEntry, query: str) -> float:
    """Simple token-overlap relevance score between entry fields and query."""
    if not query:
        return 0.5  # Neutral when no query
    query_tokens = set(query.lower().split())
    if not query_tokens:
        return 0.5

    # Collect searchable text from entry
    haystack_parts: list[str] = [
        entry.tool_name or "",
        entry.status or "",
        entry.failure_class or "",
        entry.output_preview or "",
        entry.error or "",
        entry.policy_action or "",
    ]
    # Include any extra_metadata values
    for v in entry.extra_metadata.values():
        haystack_parts.append(str(v))

    haystack_tokens = set(" ".join(haystack_parts).lower().split())
    if not haystack_tokens:
        return 0.0

    overlap = len(query_tokens & haystack_tokens)
    return min(1.0, overlap / max(len(query_tokens), 1))


# ---------------------------------------------------------------------------
# Sorting keys
# ---------------------------------------------------------------------------


def _sort_key_recency(candidate: MemoryContextCandidate) -> tuple:
    return (-candidate.entry.timestamp_ms,)


def _sort_key_relevance(candidate: MemoryContextCandidate) -> tuple:
    return (-candidate.relevance_score, -candidate.entry.timestamp_ms)


def _sort_key_severity(candidate: MemoryContextCandidate) -> tuple:
    sev = _SEVERITY_RANK.get(
        (candidate.entry.failure_severity or "").lower(), 0
    )
    return (-sev, -candidate.entry.timestamp_ms)


def _sort_key_severity_then_recency(candidate: MemoryContextCandidate) -> tuple:
    sev = _SEVERITY_RANK.get(
        (candidate.entry.failure_severity or "").lower(), 0
    )
    return (-sev, -candidate.entry.timestamp_ms)


def _sort_key_policy_action_weight(candidate: MemoryContextCandidate) -> tuple:
    rank = _POLICY_ACTION_RANK.get(
        (candidate.entry.policy_action or "").lower(), 0
    )
    return (-rank, -candidate.entry.timestamp_ms)


_SORT_KEY_MAP = {
    InjectionRankStrategy.RECENCY: _sort_key_recency,
    InjectionRankStrategy.RELEVANCE: _sort_key_relevance,
    InjectionRankStrategy.SEVERITY: _sort_key_severity,
    InjectionRankStrategy.SEVERITY_THEN_RECENCY: _sort_key_severity_then_recency,
    InjectionRankStrategy.POLICY_ACTION_WEIGHT: _sort_key_policy_action_weight,
}


# ---------------------------------------------------------------------------
# Injector
# ---------------------------------------------------------------------------


class MemoryContextInjector:
    """Full pipeline: score → rank → budget-trim → format entries for prompt injection.

    Parameters
    ----------
    budget:
        Hard limits on injection volume.
    strategy:
        How candidates are ranked before budget trimming.

    Methods
    -------
    select(entries, query_context="") → list[str]
        Returns formatted, budget-trimmed snippets ready for prompt insertion.
    score(entries, query_context="") → list[MemoryContextCandidate]
        Returns fully scored and sorted candidates (without budget trimming).
    """

    def __init__(
        self,
        *,
        budget: MemoryContextBudget | None = None,
        strategy: InjectionRankStrategy = InjectionRankStrategy.SEVERITY_THEN_RECENCY,
    ) -> None:
        self._budget = budget or MemoryContextBudget()
        self._strategy = strategy

    @property
    def budget(self) -> MemoryContextBudget:
        return self._budget

    @property
    def strategy(self) -> InjectionRankStrategy:
        return self._strategy

    def score(
        self,
        entries: Sequence[CausalMemoryEntry],
        query_context: str = "",
    ) -> list[MemoryContextCandidate]:
        """Score and rank all entries without applying budget limits."""
        candidates = [
            MemoryContextCandidate(
                entry=e,
                relevance_score=_lexical_relevance(e, query_context),
            )
            for e in entries
        ]
        sort_key = _SORT_KEY_MAP.get(self._strategy, _sort_key_severity_then_recency)
        candidates.sort(key=sort_key)
        return candidates

    def select(
        self,
        entries: Sequence[CausalMemoryEntry],
        query_context: str = "",
    ) -> list[str]:
        """Score, rank, filter by budget, and return formatted snippets.

        Returns
        -------
        list[str]
            Formatted memory snippets, ordered by rank, within budget.
        """
        candidates = self.score(entries, query_context)

        # Apply minimum relevance filter
        if self._budget.min_relevance_score > 0.0:
            candidates = [
                c for c in candidates
                if c.relevance_score >= self._budget.min_relevance_score
            ]

        # Apply slot and char budgets
        result: list[str] = []
        total_chars = 0
        for candidate in candidates:
            if len(result) >= self._budget.max_slots:
                break
            snippet = candidate.formatted_snippet
            snippet_chars = len(snippet)
            if self._budget.max_chars > 0 and total_chars + snippet_chars > self._budget.max_chars:
                break
            result.append(snippet)
            total_chars += snippet_chars

        return result


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_default_injector(
    *,
    max_slots: int = 6,
    max_chars: int = 3000,
    strategy: InjectionRankStrategy = InjectionRankStrategy.SEVERITY_THEN_RECENCY,
) -> MemoryContextInjector:
    """Build a sensible default injector for most turn-time injection scenarios."""
    return MemoryContextInjector(
        budget=MemoryContextBudget(max_slots=max_slots, max_chars=max_chars),
        strategy=strategy,
    )


__all__ = [
    "MemoryContextBudget",
    "InjectionRankStrategy",
    "MemoryContextCandidate",
    "MemoryContextInjector",
    "build_default_injector",
]
