"""Context window pruner for the AgentDriverLoop.

Implements the "Core Identity + Last N Turns + Relevant Snippets" strategy.
Prevents context bloat while keeping the full turn history in the durable ledger.

Usage:
    pruner = ContextWindowPruner(max_turns=10, max_chars=12_000)
    pruned_records = pruner.prune(result.records)
    ctx_text = pruner.build_context_text(core_identity, pruned_records, relevant_snippets)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PrunedContext:
    """The assembled context window ready to pass as action_input or system text."""

    core_identity: str
    turns: list[dict[str, Any]]  # each has keys: turn_index, observation, reply, commit_status
    relevant_snippets: list[str]
    total_chars: int
    dropped_turns: int
    strategy: str


class ContextWindowPruner:
    """Builds a bounded context window from loop turn records.

    Strategy:
      1. Always include `core_identity` (pinned prefix).
      2. Keep the last `max_turns` committed turns.
      3. Append up to `max_snippets` relevant snippets (caller-provided).
      4. Hard-cap total character count at `max_chars`.
      5. If still over budget, trim oldest kept turns further.
    """

    def __init__(
        self,
        *,
        max_turns: int = 10,
        max_chars: int = 12_000,
        max_snippets: int = 5,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if max_chars < 256:
            raise ValueError("max_chars must be >= 256")
        self.max_turns = int(max_turns)
        self.max_chars = int(max_chars)
        self.max_snippets = int(max_snippets)

    # ------------------------------------------------------------------
    # Core pruning logic
    # ------------------------------------------------------------------

    def prune(self, records: list[Any]) -> list[dict[str, Any]]:
        """Return the last `max_turns` committed turns as plain dicts.

        Args:
            records: List of LoopTurnRecord (or any object with turn_index,
                     observation, reply, commit_status attributes).

        Returns:
            Ordered list of plain dicts, newest-last, len <= max_turns.
        """
        committed = [
            r for r in records if str(getattr(r, "commit_status", "")) == "committed"
        ]
        window = committed[-self.max_turns :]
        return [
            {
                "turn_index": getattr(r, "turn_index", i + 1),
                "observation": str(getattr(r, "observation", "") or ""),
                "reply": str(getattr(r, "reply", "") or ""),
                "commit_status": str(getattr(r, "commit_status", "") or ""),
            }
            for i, r in enumerate(window)
        ]

    def build_context_text(
        self,
        core_identity: str,
        turn_dicts: list[dict[str, Any]],
        relevant_snippets: list[str] | None = None,
    ) -> PrunedContext:
        """Assemble the final context string within the character budget.

        Args:
            core_identity: Fixed preamble (persona, rules, goal). Always included.
            turn_dicts: Output of prune(). Newest-last ordered.
            relevant_snippets: Optional short memory snippets (semantic recall).

        Returns:
            PrunedContext with assembled context and metadata.
        """
        snippets = list((relevant_snippets or [])[: self.max_snippets])
        identity_chars = len(str(core_identity or ""))
        snippet_chars = sum(len(s) for s in snippets)
        budget_for_turns = max(0, self.max_chars - identity_chars - snippet_chars - 200)

        # Greedily include turns newest-first until we hit the budget
        kept: list[dict[str, Any]] = []
        chars_used = 0
        for t in reversed(turn_dicts):
            t_text = f"[T{t['turn_index']}] obs={t['observation']!r} reply={t['reply']!r}"
            if chars_used + len(t_text) > budget_for_turns and kept:
                break
            kept.insert(0, t)
            chars_used += len(t_text)

        dropped = len(turn_dicts) - len(kept)
        strategy = f"last_{self.max_turns}_turns_char_budget_{self.max_chars}"

        return PrunedContext(
            core_identity=str(core_identity or ""),
            turns=kept,
            relevant_snippets=snippets,
            total_chars=identity_chars + chars_used + snippet_chars,
            dropped_turns=dropped,
            strategy=strategy,
        )

    def format_for_llm(
        self,
        ctx: PrunedContext,
        *,
        current_task: str = "",
    ) -> str:
        """Render a PrunedContext into the text that goes into the LLM prompt."""
        lines: list[str] = []

        # Core identity block
        lines.append("=== CORE IDENTITY ===")
        lines.append(ctx.core_identity.strip())
        lines.append("")

        # Relevant memory snippets
        if ctx.relevant_snippets:
            lines.append("=== RELEVANT MEMORY ===")
            for i, s in enumerate(ctx.relevant_snippets, 1):
                lines.append(f"[{i}] {s.strip()}")
            lines.append("")

        # History (pruned)
        if ctx.turns:
            lines.append(
                f"=== RECENT HISTORY (last {len(ctx.turns)} turns"
                + (f", {ctx.dropped_turns} older dropped" if ctx.dropped_turns else "")
                + ") ==="
            )
            for t in ctx.turns:
                lines.append(f"Turn {t['turn_index']}")
                lines.append(f"  > {t['observation']}")
                lines.append(f"  < {t['reply']}")
            lines.append("")

        # Current task
        if current_task:
            lines.append("=== CURRENT TASK ===")
            lines.append(current_task.strip())

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: build an ObservationHook using the pruner
# ---------------------------------------------------------------------------


def build_pruned_observation_hook(
    pruner: ContextWindowPruner,
    *,
    core_identity: str = "",
    snippet_provider: Any = None,
) -> Any:
    """Return an ObservationHook that feeds a pruned context window into the loop.

    The hook is called before each turn and returns the assembled context text
    as the observation string passed to the kernel.

    Args:
        pruner: A ContextWindowPruner instance.
        core_identity: Fixed persona/rules text.
        snippet_provider: Optional callable(query: str) -> list[str] for memory recall.
    """

    def observation_hook(ctx: dict[str, Any]) -> str:
        records = ctx.get("records", [])
        last_reply = str(ctx.get("last_reply") or "")
        initial = str(ctx.get("initial_observation") or "")
        current_task = last_reply or initial

        snippets: list[str] = []
        if callable(snippet_provider) and current_task:
            try:
                snippets = list(snippet_provider(current_task) or [])
            except Exception:
                pass

        turn_dicts = pruner.prune(records)
        ctx_obj = pruner.build_context_text(core_identity, turn_dicts, snippets)
        return pruner.format_for_llm(ctx_obj, current_task=current_task)

    return observation_hook


__all__ = [
    "PrunedContext",
    "ContextWindowPruner",
    "build_pruned_observation_hook",
]
