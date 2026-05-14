"""Unit tests for dadbot.runtime.context_pruner (Task 2)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from dadbot.runtime.context_pruner import ContextWindowPruner, PrunedContext, build_pruned_observation_hook

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
