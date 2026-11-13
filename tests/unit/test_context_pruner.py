"""Unit tests for dadbot.runtime.context_pruner (Task 2)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from dadbot.runtime.context_pruner import (
    ContextWindowPruner,
    MemoryRecall,
    PrunedContext,
    build_pruned_observation_hook,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Minimal LoopTurnRecord stub (avoid importing the real class)
# ---------------------------------------------------------------------------


@dataclass
class _TurnRecord:
    turn_index: int
    observation: str
    reply: str
    commit_status: str = "committed"


def _make_records(n: int, *, committed: bool = True) -> list[_TurnRecord]:
    status = "committed" if committed else "failed"
    return [_TurnRecord(i + 1, f"obs_{i+1}", f"reply_{i+1}", status) for i in range(n)]


# ---------------------------------------------------------------------------
# ContextWindowPruner.prune
# ---------------------------------------------------------------------------


class TestContextWindowPruner:
    def test_prune_returns_last_n_committed(self) -> None:
        records = _make_records(15)
        pruner = ContextWindowPruner(max_turns=10)
        pruned = pruner.prune(records)
        assert len(pruned) == 10
        # Should be the last 10 turns
        assert pruned[0]["turn_index"] == 6
        assert pruned[-1]["turn_index"] == 15

    def test_prune_skips_uncommitted(self) -> None:
        committed = _make_records(5, committed=True)
        failed = _make_records(3, committed=False)
        # Interleave committed and failed
        mixed = committed[:2] + failed + committed[2:]
        pruner = ContextWindowPruner(max_turns=10)
        pruned = pruner.prune(mixed)
        for t in pruned:
            assert t["commit_status"] == "committed"

    def test_prune_fewer_records_than_max(self) -> None:
        records = _make_records(3)
        pruner = ContextWindowPruner(max_turns=10)
        pruned = pruner.prune(records)
        assert len(pruned) == 3

    def test_prune_empty_records(self) -> None:
        pruner = ContextWindowPruner(max_turns=10)
        assert pruner.prune([]) == []

    def test_prune_dict_keys(self) -> None:
        records = _make_records(2)
        pruner = ContextWindowPruner(max_turns=10)
        pruned = pruner.prune(records)
        assert set(pruned[0].keys()) == {"turn_index", "observation", "reply", "commit_status"}

    def test_invalid_max_turns_raises(self) -> None:
        with pytest.raises(ValueError):
            ContextWindowPruner(max_turns=0)

    def test_invalid_max_chars_raises(self) -> None:
        with pytest.raises(ValueError):
            ContextWindowPruner(max_chars=10)


# ---------------------------------------------------------------------------
# ContextWindowPruner.build_context_text
# ---------------------------------------------------------------------------


class TestBuildContextText:
    def _pruner(self) -> ContextWindowPruner:
        return ContextWindowPruner(max_turns=10, max_chars=4000)

    def test_returns_pruned_context(self) -> None:
        records = _make_records(5)
        pruner = self._pruner()
        turn_dicts = pruner.prune(records)
        ctx = pruner.build_context_text("I am Dad Bot.", turn_dicts)
        assert isinstance(ctx, PrunedContext)
        assert ctx.core_identity == "I am Dad Bot."
        assert len(ctx.turns) > 0

    def test_snippets_capped_at_max_snippets(self) -> None:
        pruner = ContextWindowPruner(max_turns=10, max_snippets=3)
        ctx = pruner.build_context_text("identity", [], relevant_snippets=["a", "b", "c", "d"])
        assert len(ctx.relevant_snippets) == 3

    def test_dropped_turns_counted(self) -> None:
        # Very tiny char budget forces dropping
        pruner = ContextWindowPruner(max_turns=10, max_chars=300)
        records = _make_records(10)
        turn_dicts = pruner.prune(records)
        ctx = pruner.build_context_text("id " * 5, turn_dicts)
        # May have dropped some turns; dropped_turns should reflect reality
        assert ctx.dropped_turns == len(turn_dicts) - len(ctx.turns)

    def test_total_chars_is_approximate(self) -> None:
        pruner = ContextWindowPruner(max_turns=5, max_chars=8000)
        records = _make_records(5)
        turn_dicts = pruner.prune(records)
        ctx = pruner.build_context_text("identity", turn_dicts)
        assert ctx.total_chars > 0


# ---------------------------------------------------------------------------
# ContextWindowPruner.format_for_llm
# ---------------------------------------------------------------------------


class TestFormatForLLM:
    def test_contains_identity(self) -> None:
        pruner = ContextWindowPruner()
        ctx = PrunedContext(
            core_identity="I am Dad Bot.",
            turns=[],
            relevant_snippets=[],
            total_chars=100,
            dropped_turns=0,
            strategy="test",
        )
        text = pruner.format_for_llm(ctx)
        assert "I am Dad Bot." in text
        assert "CORE IDENTITY" in text

    def test_contains_history(self) -> None:
        pruner = ContextWindowPruner()
        ctx = PrunedContext(
            core_identity="id",
            turns=[{"turn_index": 1, "observation": "hello", "reply": "world", "commit_status": "committed"}],
            relevant_snippets=[],
            total_chars=100,
            dropped_turns=0,
            strategy="test",
        )
        text = pruner.format_for_llm(ctx)
        assert "hello" in text
        assert "world" in text

    def test_contains_snippets(self) -> None:
        pruner = ContextWindowPruner()
        ctx = PrunedContext(
            core_identity="id",
            turns=[],
            relevant_snippets=["memory note 1"],
            total_chars=100,
            dropped_turns=0,
            strategy="test",
        )
        text = pruner.format_for_llm(ctx)
        assert "memory note 1" in text

    def test_includes_current_task(self) -> None:
        pruner = ContextWindowPruner()
        ctx = PrunedContext("id", [], [], 0, 0, "test")
        text = pruner.format_for_llm(ctx, current_task="find the answer")
        assert "find the answer" in text

    def test_no_history_block_when_no_turns(self) -> None:
        pruner = ContextWindowPruner()
        ctx = PrunedContext("id", [], [], 0, 0, "test")
        text = pruner.format_for_llm(ctx)
        assert "RECENT HISTORY" not in text


# ---------------------------------------------------------------------------
# build_pruned_observation_hook
# ---------------------------------------------------------------------------


class TestBuildPrunedObservationHook:
    def test_returns_non_empty_string(self) -> None:
        pruner = ContextWindowPruner(max_turns=5)
        hook = build_pruned_observation_hook(pruner, core_identity="I am Dad.")
        ctx = {
            "turn_index": 2,
            "last_reply": "I replied this",
            "initial_observation": "do something",
            "records": _make_records(2),
        }
        result = hook(ctx)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_core_identity(self) -> None:
        pruner = ContextWindowPruner()
        hook = build_pruned_observation_hook(pruner, core_identity="DAD IDENTITY")
        ctx = {"turn_index": 1, "last_reply": "", "initial_observation": "go", "records": []}
        result = hook(ctx)
        assert "DAD IDENTITY" in result

    def test_snippet_provider_called(self) -> None:
        calls: list[str] = []

        def provider(query: str) -> list[str]:
            calls.append(query)
            return ["relevant thing"]

        pruner = ContextWindowPruner()
        hook = build_pruned_observation_hook(pruner, snippet_provider=provider)
        ctx = {"turn_index": 1, "last_reply": "the query text", "initial_observation": "go", "records": []}
        result = hook(ctx)
        assert len(calls) == 1
        assert "relevant thing" in result

    def test_snapshot_provider_error_is_swallowed(self) -> None:
        def bad_provider(q: str) -> None:
            raise RuntimeError("db down")

        pruner = ContextWindowPruner()
        hook = build_pruned_observation_hook(pruner, snippet_provider=bad_provider)
        ctx = {"turn_index": 1, "last_reply": "x", "initial_observation": "y", "records": []}
        # Must not raise
        result = hook(ctx)
        assert isinstance(result, str)

    def test_strict_provider_errors_are_visible(self) -> None:
        def bad_provider(_query: str) -> None:
            raise RuntimeError("memory store unavailable")

        hook = build_pruned_observation_hook(
            ContextWindowPruner(),
            snippet_provider=bad_provider,
            strict_snippets=True,
        )
        ctx = {"turn_index": 1, "last_reply": "", "initial_observation": "start", "records": []}

        with pytest.raises(RuntimeError, match="memory store unavailable"):
            hook(ctx)

    def test_system_observation_is_included_in_current_task(self) -> None:
        pruner = ContextWindowPruner()
        hook = build_pruned_observation_hook(pruner)
        ctx = {
            "turn_index": 2,
            "last_reply": "previous reply",
            "initial_observation": "start",
            "system_observation": "Plan verifier rejected the prior plan.",
            "records": [],
        }

        result = hook(ctx)

        assert "Plan verifier rejected the prior plan." in result

    def test_memory_recall_signals_are_attached_to_reflection_context(self) -> None:
        signals = {"conflicts": [{"left": "old", "right": "new"}], "temporal": [{"updated_at": "2026-01-01"}]}
        hook = build_pruned_observation_hook(
            ContextWindowPruner(),
            snippet_provider=lambda _query: MemoryRecall(["[general] remembered"], signals),
        )
        ctx = {"turn_index": 1, "last_reply": "", "initial_observation": "start", "records": []}

        result = hook(ctx)

        assert "[general] remembered" in result
        assert ctx["memory_signals"] == signals

    def test_whole_evidence_and_signals_are_pruned_to_the_rendered_budget(self) -> None:
        first = 'Fact one\nEvidence: {"source_id":"one","provenance":{"trace_id":"trace-one"}}'
        oversized = 'Fact two\nEvidence: {"source_id":"two","provenance":{"evidence":"' + "x" * 2000 + '"}}'
        hook = build_pruned_observation_hook(
            ContextWindowPruner(max_chars=700),
            snippet_provider=lambda _query: MemoryRecall(
                [first, oversized],
                {
                    "evidence": [{"source_id": "one"}, {"source_id": "two"}],
                    "temporal": [{"summary": "Fact one"}, {"summary": "Fact two"}],
                    "conflicts": [],
                    "retrieval": {"status": "degraded", "reason": "stale_index"},
                },
            ),
            strict_snippets=True,
        )
        context = {"initial_observation": "Find the fact", "records": _make_records(3)}
        rendered = hook(context)
        assert len(rendered) <= 700
        assert first in rendered and "Fact two" not in rendered
        assert "stale_index" in rendered and "1 evidence entries dropped" in rendered
        assert context["memory_signals"]["evidence"] == [{"source_id": "one"}]
        assert context["memory_signals"]["temporal"] == [{"summary": "Fact one"}]

    def test_oversized_pinned_task_fails_instead_of_truncating_identity(self) -> None:
        hook = build_pruned_observation_hook(ContextWindowPruner(max_chars=256), core_identity="Pinned identity")
        with pytest.raises(ValueError, match="identity and current task"):
            hook({"initial_observation": "x" * 300})

    def test_recall_keeps_the_initial_goal_after_a_generic_reply(self) -> None:
        queries = []
        hook = build_pruned_observation_hook(
            ContextWindowPruner(), snippet_provider=lambda query: queries.append(query) or [],
        )
        rendered = hook({"initial_observation": "Recall the project owner", "last_reply": "Step completed."})
        assert queries == ["Recall the project owner"]
        assert "Step completed." in rendered
