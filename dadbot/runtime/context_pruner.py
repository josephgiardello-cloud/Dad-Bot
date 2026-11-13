"""Context window pruner for the AgentDriverLoop.

Implements the "Core Identity + Last N Turns + Relevant Snippets" strategy.
Prevents context bloat while keeping the full turn history in the durable ledger.

Usage:
    pruner = ContextWindowPruner(max_turns=10, max_chars=12_000)
    pruned_records = pruner.prune(result.records)
    ctx_text = pruner.build_context_text(core_identity, pruned_records, relevant_snippets)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
    dropped_snippets: int = 0


@dataclass(slots=True)
class MemoryRecall:
    snippets: list[str]
    signals: dict[str, Any]


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
        *,
        current_task: str = "",
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

        context = PrunedContext(
            core_identity=str(core_identity or ""),
            turns=kept,
            relevant_snippets=snippets,
            total_chars=identity_chars + chars_used + snippet_chars,
            dropped_turns=dropped,
            strategy=strategy,
            dropped_snippets=len(relevant_snippets or []) - len(snippets),
        )
        rendered = self._render_for_llm(context, current_task=current_task)
        while len(rendered) > self.max_chars:
            if context.turns:
                context.turns.pop(0)
                context.dropped_turns += 1
            elif context.relevant_snippets:
                context.relevant_snippets.pop()
                context.dropped_snippets += 1
            else:
                raise ValueError("Core identity and current task exceed the context character budget")
            rendered = self._render_for_llm(context, current_task=current_task)
        context.total_chars = len(rendered)
        return context

    def format_for_llm(
        self,
        ctx: PrunedContext,
        *,
        current_task: str = "",
    ) -> str:
        """Render a context whose complete identity and task fit the hard budget."""
        text = self._render_for_llm(ctx, current_task=current_task)
        if len(text) > self.max_chars:
            raise ValueError("Rendered context exceeds the context character budget")
        return text

    @staticmethod
    def _render_for_llm(ctx: PrunedContext, *, current_task: str = "") -> str:
        lines: list[str] = []

        # Core identity block
        lines.append("=== CORE IDENTITY ===")
        lines.append(ctx.core_identity.strip())
        lines.append("")

        # Relevant memory snippets
        if ctx.relevant_snippets or ctx.dropped_snippets:
            dropped = f" ({ctx.dropped_snippets} evidence entries dropped)" if ctx.dropped_snippets else ""
            lines.append(f"=== RELEVANT MEMORY{dropped} ===")
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
    strict_snippets: bool = False,
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
        system_observation = str(ctx.get("system_observation") or "").strip()
        current_task = "\n\n".join(
            part for part in dict.fromkeys((initial, system_observation, last_reply)) if part
        )

        snippets: list[str] = []
        ctx["memory_signals"] = {"conflicts": [], "temporal": []}
        if callable(snippet_provider) and current_task:
            if strict_snippets:
                provided = snippet_provider(initial or current_task)
            else:
                try:
                    provided = snippet_provider(initial or current_task)
                except Exception:
                    provided = None

            if isinstance(provided, MemoryRecall):
                snippets = list(provided.snippets)
                ctx["memory_signals"] = dict(provided.signals)
            else:
                snippets = list(provided or [])

        identity = core_identity
        retrieval_status = ctx["memory_signals"].get("retrieval")
        if retrieval_status:
            identity = (
                f"{identity}\n=== MEMORY RETRIEVAL ===\n"
                f"{json.dumps(retrieval_status, sort_keys=True, ensure_ascii=True)}"
            )
        turn_dicts = pruner.prune(records)
        ctx_obj = pruner.build_context_text(identity, turn_dicts, snippets, current_task=current_task)
        evidence = ctx["memory_signals"].get("evidence")
        if isinstance(evidence, list):
            kept_count = len(ctx_obj.relevant_snippets)
            ctx["memory_signals"]["evidence"] = evidence[:kept_count]
            ctx["memory_signals"]["pruning"] = {
                "retrieved": len(evidence), "kept": kept_count, "dropped": len(evidence) - kept_count,
            }
            kept_text = "\n".join(ctx_obj.relevant_snippets)
            ctx["memory_signals"]["temporal"] = [
                item for item in ctx["memory_signals"].get("temporal", [])
                if item.get("summary") and item["summary"] in kept_text
            ]
            ctx["memory_signals"]["conflicts"] = [
                item for item in ctx["memory_signals"].get("conflicts", [])
                if item.get("left") in kept_text and item.get("right") in kept_text
            ]
        return pruner.format_for_llm(ctx_obj, current_task=current_task)

    return observation_hook


__all__ = [
    "ContextWindowPruner",
    "MemoryRecall",
    "PrunedContext",
    "build_pruned_observation_hook",
]
