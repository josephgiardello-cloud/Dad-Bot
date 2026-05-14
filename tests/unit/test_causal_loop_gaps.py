"""Unit tests for the 4 causal loop gap modules.

Covers:
  Gap 1 — MemoryContextInjector (memory_context_injector.py)
  Gap 2 — MemoryCoherenceEngine (memory_coherence_engine.py)
  Gap 3 — CausalDepGraph (causal_dependency_graph.py)
  Gap 4 — MemoryFeedbackPolicy (memory_feedback_policy.py)
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from dadbot.core.tool_memory_causal_contract import CausalMemoryEntry

# ── Gap 1 ──────────────────────────────────────────────────────────────────
from dadbot.core.memory_context_injector import (
    InjectionRankStrategy,
    MemoryContextBudget,
    MemoryContextCandidate,
    MemoryContextInjector,
    build_default_injector,
)

# ── Gap 2 ──────────────────────────────────────────────────────────────────
from dadbot.core.memory_coherence_engine import (
    ConflictRecord,
    ConflictResolutionStrategy,
    CoherentMemoryView,
    DecayRule,
    MemoryCoherenceEngine,
)

# ── Gap 3 ──────────────────────────────────────────────────────────────────
from dadbot.core.causal_dependency_graph import (
    CausalDepGraph,
    CausalEdge,
    CausalNode,
    build_temporal_graph,
)

# ── Gap 4 ──────────────────────────────────────────────────────────────────
from dadbot.core.failure_policy_engine import (
    FailurePolicyEngine,
    PolicyAction,
    PolicyDecision,
)
from dadbot.core.memory_feedback_policy import (
    MemoryAwarePolicyContext,
    MemoryPolicyAdjustment,
    ToolMemoryProfile,
    ToolSelectionAdvisor,
    compute_scheduling_priority,
    suggest_response_tone,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NOW_MS = 1_700_000_000_000   # Fixed epoch for deterministic decay tests


def _entry(
    *,
    tool_name: str = "tool_a",
    status: str = "ok",
    failure_class: str | None = None,
    failure_severity: str | None = None,
    policy_action: str | None = None,
    attempt: int = 1,
    latency_ms: float = 10.0,
    output_preview: str = "",
    error: str = "",
    timestamp_ms: int | None = None,
    causal_key: str | None = None,
    contract_version: str = "1.0",
) -> CausalMemoryEntry:
    ts = timestamp_ms if timestamp_ms is not None else NOW_MS
    key = causal_key or f"{tool_name}:{ts}"
    return CausalMemoryEntry(
        tool_name=tool_name,
        contract_version=contract_version,
        attempt=attempt,
        status=status,
        causal_key=key,
        timestamp_ms=ts,
        latency_ms=latency_ms,
        failure_class=failure_class,
        failure_severity=failure_severity,
        policy_action=policy_action,
        output_preview=output_preview,
        error=error,
    )


# ===========================================================================
# GAP 1: MemoryContextInjector
# ===========================================================================


class TestMemoryContextBudget:
    def test_defaults(self):
        b = MemoryContextBudget()
        assert b.max_slots == 8
        assert b.max_chars == 4000
        assert b.min_relevance_score == 0.0

    def test_invalid_max_slots(self):
        with pytest.raises(ValueError):
            MemoryContextBudget(max_slots=0)

    def test_invalid_max_chars(self):
        with pytest.raises(ValueError):
            MemoryContextBudget(max_chars=-1)

    def test_invalid_min_relevance(self):
        with pytest.raises(ValueError):
            MemoryContextBudget(min_relevance_score=1.5)


class TestInjectionRankStrategy:
    def test_all_strategies_exist(self):
        strategies = {s.value for s in InjectionRankStrategy}
        assert "recency" in strategies
        assert "relevance" in strategies
        assert "severity" in strategies
        assert "severity_then_recency" in strategies
        assert "policy_action_weight" in strategies


class TestMemoryContextInjector:
    def _make_entries(self) -> list[CausalMemoryEntry]:
        return [
            _entry(tool_name="tool_a", status="ok", timestamp_ms=NOW_MS - 5000),
            _entry(
                tool_name="tool_b",
                status="error",
                failure_severity="critical",
                policy_action="escalate",
                timestamp_ms=NOW_MS - 1000,
                causal_key="tool_b:critical",
            ),
            _entry(
                tool_name="tool_c",
                status="error",
                failure_severity="high",
                timestamp_ms=NOW_MS - 2000,
                causal_key="tool_c:high",
            ),
        ]

    def test_select_returns_list_of_strings(self):
        injector = build_default_injector()
        entries = self._make_entries()
        snippets = injector.select(entries)
        assert isinstance(snippets, list)
        assert all(isinstance(s, str) for s in snippets)

    def test_slot_budget_respected(self):
        injector = MemoryContextInjector(
            budget=MemoryContextBudget(max_slots=2),
            strategy=InjectionRankStrategy.RECENCY,
        )
        entries = self._make_entries()
        snippets = injector.select(entries)
        assert len(snippets) <= 2

    def test_char_budget_respected(self):
        injector = MemoryContextInjector(
            budget=MemoryContextBudget(max_slots=100, max_chars=50),
        )
        entries = self._make_entries()
        snippets = injector.select(entries)
        total = sum(len(s) for s in snippets)
        assert total <= 50

    def test_severity_then_recency_strategy_puts_critical_first(self):
        injector = MemoryContextInjector(
            budget=MemoryContextBudget(max_slots=3),
            strategy=InjectionRankStrategy.SEVERITY_THEN_RECENCY,
        )
        entries = self._make_entries()
        snippets = injector.select(entries)
        # Critical (tool_b) should appear in first snippet
        assert len(snippets) >= 1
        assert "tool_b" in snippets[0]

    def test_policy_action_weight_escalate_first(self):
        injector = MemoryContextInjector(
            budget=MemoryContextBudget(max_slots=3),
            strategy=InjectionRankStrategy.POLICY_ACTION_WEIGHT,
        )
        entries = self._make_entries()
        snippets = injector.select(entries)
        assert "escalate" in snippets[0]

    def test_relevance_strategy_scores_matching_entries_higher(self):
        entries = [
            _entry(tool_name="payment_tool", output_preview="payment processed", causal_key="p1"),
            _entry(tool_name="email_tool", output_preview="email sent", causal_key="p2"),
        ]
        injector = MemoryContextInjector(
            strategy=InjectionRankStrategy.RELEVANCE,
        )
        snippets = injector.select(entries, query_context="payment failed")
        assert "payment" in snippets[0]

    def test_min_relevance_filter_drops_non_matching(self):
        entries = [
            _entry(tool_name="unrelated_tool", output_preview="weather report", causal_key="u1"),
        ]
        injector = MemoryContextInjector(
            budget=MemoryContextBudget(min_relevance_score=0.9),
        )
        snippets = injector.select(entries, query_context="payment invoice")
        assert snippets == []

    def test_empty_entries_returns_empty(self):
        injector = build_default_injector()
        assert injector.select([]) == []

    def test_score_returns_candidates(self):
        injector = build_default_injector()
        entries = self._make_entries()
        candidates = injector.score(entries)
        assert len(candidates) == len(entries)
        assert all(isinstance(c, MemoryContextCandidate) for c in candidates)

    def test_candidate_relevance_score_clamped(self):
        c = MemoryContextCandidate(
            entry=_entry(causal_key="k1"),
            relevance_score=5.0,   # Out of range
        )
        assert c.relevance_score == 1.0

    def test_recency_strategy_latest_first(self):
        old = _entry(tool_name="older_tool", timestamp_ms=NOW_MS - 10000, causal_key="old")
        new = _entry(tool_name="newer_tool", timestamp_ms=NOW_MS, causal_key="new")
        injector = MemoryContextInjector(strategy=InjectionRankStrategy.RECENCY)
        snippets = injector.select([old, new])
        assert "newer_tool" in snippets[0]


# ===========================================================================
# GAP 2: MemoryCoherenceEngine
# ===========================================================================


class TestDecayRule:
    def test_no_decay_returns_one(self):
        rule = DecayRule(half_life_seconds=0.0)
        e = _entry(timestamp_ms=NOW_MS - 86_400_000)  # 1 day ago
        assert rule.weight_for(e, now_ms=NOW_MS) == 1.0

    def test_decay_at_half_life(self):
        rule = DecayRule(half_life_seconds=3600.0)
        # Entry written 1 hour ago → weight should be ~0.5
        e = _entry(timestamp_ms=NOW_MS - 3_600_000)
        w = rule.weight_for(e, now_ms=NOW_MS)
        assert 0.45 < w < 0.55

    def test_min_weight_floor(self):
        rule = DecayRule(half_life_seconds=1.0, min_weight=0.1)
        # Entry written 10000 seconds ago → essentially zero without floor
        e = _entry(timestamp_ms=NOW_MS - 10_000_000)
        w = rule.weight_for(e, now_ms=NOW_MS)
        assert w >= 0.1

    def test_invalid_half_life_raises(self):
        with pytest.raises(ValueError):
            DecayRule(half_life_seconds=-1.0)

    def test_invalid_min_weight_raises(self):
        with pytest.raises(ValueError):
            DecayRule(min_weight=0.0)

    def test_future_entry_weight_is_one(self):
        rule = DecayRule(half_life_seconds=3600.0)
        e = _entry(timestamp_ms=NOW_MS + 1000)  # future
        w = rule.weight_for(e, now_ms=NOW_MS)
        assert w == 1.0


class TestMemoryCoherenceEngine:
    def test_empty_returns_empty_view(self):
        engine = MemoryCoherenceEngine()
        view = engine.process([])
        assert view.entries == []
        assert view.conflicts == []

    def test_single_entry_no_conflict(self):
        engine = MemoryCoherenceEngine()
        e = _entry(causal_key="single")
        view = engine.process([e], now_ms=NOW_MS)
        assert len(view.entries) == 1
        assert view.conflicts == []

    def test_conflict_latest_wins(self):
        engine = MemoryCoherenceEngine(
            conflict_strategy=ConflictResolutionStrategy.LATEST_WINS
        )
        old = _entry(status="error", failure_class="network_timeout",
                     timestamp_ms=NOW_MS - 5000, causal_key="k_old")
        new = _entry(status="ok", failure_class="network_timeout",
                     timestamp_ms=NOW_MS, causal_key="k_new")
        view = engine.process([old, new], now_ms=NOW_MS)
        assert len(view.conflicts) == 1
        conflict = view.conflicts[0]
        assert conflict.winner.causal_key == "k_new"
        assert conflict.loser.causal_key == "k_old"

    def test_conflict_highest_severity_wins(self):
        engine = MemoryCoherenceEngine(
            conflict_strategy=ConflictResolutionStrategy.HIGHEST_SEVERITY
        )
        low = _entry(status="error", failure_class="network_timeout",
                     failure_severity="low", timestamp_ms=NOW_MS, causal_key="k_low")
        critical = _entry(status="ok", failure_class="network_timeout",
                          failure_severity="critical",
                          timestamp_ms=NOW_MS - 1000, causal_key="k_crit")
        view = engine.process([low, critical], now_ms=NOW_MS)
        assert view.conflicts[0].winner.causal_key == "k_crit"

    def test_conflict_most_retries_wins(self):
        engine = MemoryCoherenceEngine(
            conflict_strategy=ConflictResolutionStrategy.MOST_RETRIES
        )
        few = _entry(status="error", failure_class="network_timeout",
                     attempt=1, timestamp_ms=NOW_MS, causal_key="k_few")
        many = _entry(status="ok", failure_class="network_timeout",
                      attempt=5, timestamp_ms=NOW_MS - 1000, causal_key="k_many")
        view = engine.process([few, many], now_ms=NOW_MS)
        assert view.conflicts[0].winner.causal_key == "k_many"

    def test_conflict_policy_escalate_wins(self):
        engine = MemoryCoherenceEngine(
            conflict_strategy=ConflictResolutionStrategy.POLICY_ESCALATE_WINS
        )
        normal = _entry(status="error", failure_class="network_timeout",
                        policy_action="retry", timestamp_ms=NOW_MS, causal_key="k_retry")
        urgent = _entry(status="ok", failure_class="network_timeout",
                        policy_action="escalate",
                        timestamp_ms=NOW_MS - 100, causal_key="k_esc")
        view = engine.process([normal, urgent], now_ms=NOW_MS)
        assert view.conflicts[0].winner.causal_key == "k_esc"

    def test_priority_weight_boosts_critical(self):
        engine = MemoryCoherenceEngine()
        low = _entry(failure_severity="low", causal_key="k_low",
                     timestamp_ms=NOW_MS)
        critical = _entry(failure_severity="critical", status="error",
                          failure_class="x", causal_key="k_crit",
                          timestamp_ms=NOW_MS)
        view = engine.process([low, critical], now_ms=NOW_MS)
        weights = {e.causal_key: w for e, w in view.weighted}
        assert weights["k_crit"] > weights["k_low"]

    def test_no_conflict_same_status(self):
        """Two entries with same status should not produce a ConflictRecord."""
        engine = MemoryCoherenceEngine()
        a = _entry(status="ok", causal_key="ka")
        b = _entry(status="ok", causal_key="kb")
        view = engine.process([a, b], now_ms=NOW_MS)
        assert view.conflicts == []

    def test_decay_filters_old_entries(self):
        """Entries decayed below min_weight should appear in forgotten list."""
        rule = DecayRule(half_life_seconds=1.0, min_weight=0.1)
        engine = MemoryCoherenceEngine(decay=rule)
        # Entry written 1000 seconds ago with half-life of 1s → negligible weight
        old = _entry(timestamp_ms=NOW_MS - 1_000_000, causal_key="k_old")
        fresh = _entry(timestamp_ms=NOW_MS, causal_key="k_fresh")
        view = engine.process([old, fresh], now_ms=NOW_MS)
        assert "k_old" not in {e.causal_key for e in view.entries}
        assert old in view.forgotten

    def test_top_n(self):
        engine = MemoryCoherenceEngine()
        # Use distinct tool names so entries don't collapse under conflict resolution
        entries = [_entry(tool_name=f"tool_{i}", causal_key=f"k{i}") for i in range(10)]
        view = engine.process(entries, now_ms=NOW_MS)
        assert len(view.top(3)) == 3
        assert len(view.top(0)) == 0


# ===========================================================================
# GAP 3: CausalDepGraph
# ===========================================================================


class TestCausalDepGraph:
    def _make_chain(self) -> tuple[CausalDepGraph, CausalMemoryEntry, CausalMemoryEntry, CausalMemoryEntry]:
        """Build a simple A → B → C chain."""
        graph = CausalDepGraph()
        a = _entry(tool_name="tool_a", causal_key="a", timestamp_ms=1000)
        b = _entry(tool_name="tool_b", causal_key="b", timestamp_ms=2000)
        c = _entry(tool_name="tool_c", causal_key="c", timestamp_ms=3000)
        graph.add_node(a)
        graph.add_node(b)
        graph.add_node(c)
        graph.add_edge("a", "b", reason="a feeds b")
        graph.add_edge("b", "c", reason="b feeds c")
        return graph, a, b, c

    def test_add_node_and_contains(self):
        graph = CausalDepGraph()
        e = _entry(causal_key="k1")
        graph.add_node(e)
        assert "k1" in graph
        assert len(graph) == 1

    def test_add_edge_updates_in_out_keys(self):
        graph, a, b, c = self._make_chain()
        node_a = graph.node("a")
        node_b = graph.node("b")
        assert "b" in node_a.out_keys
        assert "a" in node_b.in_keys

    def test_self_loop_raises(self):
        graph = CausalDepGraph()
        e = _entry(causal_key="k1")
        graph.add_node(e)
        with pytest.raises(ValueError):
            graph.add_edge("k1", "k1")

    def test_edge_unknown_source_raises(self):
        graph = CausalDepGraph()
        e = _entry(causal_key="k1")
        graph.add_node(e)
        with pytest.raises(KeyError):
            graph.add_edge("unknown", "k1")

    def test_edge_unknown_target_raises(self):
        graph = CausalDepGraph()
        e = _entry(causal_key="k1")
        graph.add_node(e)
        with pytest.raises(KeyError):
            graph.add_edge("k1", "unknown")

    def test_roots(self):
        graph, a, b, c = self._make_chain()
        roots = graph.roots()
        assert len(roots) == 1
        assert roots[0].entry.causal_key == "a"

    def test_leaves(self):
        graph, a, b, c = self._make_chain()
        leaves = graph.leaves()
        assert len(leaves) == 1
        assert leaves[0].entry.causal_key == "c"

    def test_ancestors_of_c(self):
        graph, a, b, c = self._make_chain()
        ancestors = graph.ancestors_of("c")
        keys = {e.causal_key for e in ancestors}
        assert "a" in keys
        assert "b" in keys
        assert "c" not in keys

    def test_ancestors_of_root_is_empty(self):
        graph, a, b, c = self._make_chain()
        assert graph.ancestors_of("a") == []

    def test_descendants_of_a(self):
        graph, a, b, c = self._make_chain()
        descs = graph.descendants_of("a")
        keys = {e.causal_key for e in descs}
        assert "b" in keys
        assert "c" in keys
        assert "a" not in keys

    def test_descendants_of_leaf_is_empty(self):
        graph, a, b, c = self._make_chain()
        assert graph.descendants_of("c") == []

    def test_causal_chain_a_to_c(self):
        graph, a, b, c = self._make_chain()
        chain = graph.causal_chain("c")
        assert [e.causal_key for e in chain] == ["a", "b", "c"]

    def test_causal_chain_root(self):
        graph, a, b, c = self._make_chain()
        chain = graph.causal_chain("a")
        assert [e.causal_key for e in chain] == ["a"]

    def test_causal_chain_unknown_key(self):
        graph = CausalDepGraph()
        assert graph.causal_chain("nonexistent") == []

    def test_detect_cycles_dag_is_empty(self):
        graph, a, b, c = self._make_chain()
        cycles = graph.detect_cycles()
        assert cycles == []

    def test_subgraph_for_tool(self):
        graph, a, b, c = self._make_chain()
        sub = graph.subgraph_for_tool("tool_a")
        assert "a" in sub
        assert "b" not in sub

    def test_empty_graph(self):
        graph = CausalDepGraph()
        assert len(graph) == 0
        assert graph.roots() == []
        assert graph.leaves() == []

    def test_update_existing_node_preserves_edges(self):
        graph = CausalDepGraph()
        a = _entry(causal_key="a")
        b = _entry(causal_key="b")
        graph.add_node(a)
        graph.add_node(b)
        graph.add_edge("a", "b")
        # Replace node a
        a2 = _entry(causal_key="a", tool_name="new_tool")
        graph.add_node(a2)
        assert "b" in graph.node("a").out_keys


class TestBuildTemporalGraph:
    def test_temporal_chain_links_in_order(self):
        entries = [
            _entry(causal_key="t1", timestamp_ms=1000),
            _entry(causal_key="t2", timestamp_ms=2000),
            _entry(causal_key="t3", timestamp_ms=3000),
        ]
        graph = build_temporal_graph(entries)
        assert len(graph) == 3
        chain = graph.causal_chain("t3")
        assert [e.causal_key for e in chain] == ["t1", "t2", "t3"]

    def test_same_tool_only_flag(self):
        entries = [
            _entry(tool_name="tool_a", causal_key="t1", timestamp_ms=1000),
            _entry(tool_name="tool_b", causal_key="t2", timestamp_ms=2000),
        ]
        graph = build_temporal_graph(entries, same_tool_only=True)
        # No edge between different tools
        assert graph.edges == []

    def test_single_entry_no_edges(self):
        entries = [_entry(causal_key="only")]
        graph = build_temporal_graph(entries)
        assert graph.edges == []


# ===========================================================================
# GAP 4: MemoryFeedbackPolicy
# ===========================================================================


class TestToolMemoryProfile:
    def _entries_for_tool(
        self,
        tool_name: str,
        *,
        n_ok: int = 5,
        n_error: int = 2,
        n_escalate: int = 0,
        n_abort: int = 0,
    ) -> list[CausalMemoryEntry]:
        entries = []
        for i in range(n_ok):
            entries.append(_entry(tool_name=tool_name, status="ok",
                                  causal_key=f"{tool_name}:ok:{i}",
                                  latency_ms=20.0))
        for i in range(n_error):
            entries.append(_entry(
                tool_name=tool_name,
                status="error",
                failure_class="network_timeout",
                policy_action="retry",
                causal_key=f"{tool_name}:err:{i}",
                latency_ms=5.0,
            ))
        for i in range(n_escalate):
            entries.append(_entry(
                tool_name=tool_name,
                status="error",
                failure_class="permission_denied",
                policy_action="escalate",
                causal_key=f"{tool_name}:esc:{i}",
                latency_ms=5.0,
            ))
        for i in range(n_abort):
            entries.append(_entry(
                tool_name=tool_name,
                status="error",
                failure_class="tool_internal_error",
                policy_action="abort",
                causal_key=f"{tool_name}:abort:{i}",
                latency_ms=5.0,
            ))
        return entries

    def test_build_profiles_basic(self):
        entries = self._entries_for_tool("tool_a", n_ok=4, n_error=1)
        profiles = ToolMemoryProfile.build_profiles(entries)
        assert "tool_a" in profiles
        p = profiles["tool_a"]
        assert p.total_executions == 5
        assert p.success_count == 4
        assert p.failure_count == 1
        assert p.failure_rate == pytest.approx(0.2)

    def test_multiple_tools(self):
        entries = (
            self._entries_for_tool("tool_a", n_ok=3)
            + self._entries_for_tool("tool_b", n_ok=1, n_error=4)
        )
        profiles = ToolMemoryProfile.build_profiles(entries)
        assert "tool_a" in profiles
        assert "tool_b" in profiles
        assert profiles["tool_a"].failure_rate < profiles["tool_b"].failure_rate

    def test_reliability_score_healthy_tool(self):
        entries = self._entries_for_tool("tool_a", n_ok=10, n_error=0)
        profiles = ToolMemoryProfile.build_profiles(entries)
        assert profiles["tool_a"].reliability_score > 0.8

    def test_reliability_score_failing_tool(self):
        entries = self._entries_for_tool("tool_x", n_ok=1, n_error=9, n_abort=5)
        profiles = ToolMemoryProfile.build_profiles(entries)
        assert profiles["tool_x"].reliability_score < 0.5

    def test_no_history_is_neutral(self):
        p = ToolMemoryProfile(tool_name="new_tool")
        assert p.failure_rate == 0.0
        assert p.reliability_score == 0.5

    def test_most_common_failure_class(self):
        entries = self._entries_for_tool("t", n_ok=0, n_error=5)
        entries += [_entry(tool_name="t", status="error",
                           failure_class="permission_denied",
                           causal_key="t:perm")]
        profiles = ToolMemoryProfile.build_profiles(entries)
        assert profiles["t"].most_common_failure_class == "network_timeout"


class TestToolSelectionAdvisor:
    def test_ranks_healthy_tool_first(self):
        healthy_entries = [
            _entry(tool_name="reliable", status="ok", causal_key=f"r:{i}") for i in range(10)
        ]
        bad_entries = [
            _entry(tool_name="flaky", status="error", failure_class="x",
                   policy_action="abort", causal_key=f"f:{i}") for i in range(10)
        ]
        profiles = ToolMemoryProfile.build_profiles(healthy_entries + bad_entries)
        advisor = ToolSelectionAdvisor(profiles)
        ranked = advisor.rank(["reliable", "flaky"])
        assert ranked[0].tool_name == "reliable"

    def test_unknown_tool_gets_neutral_score(self):
        advisor = ToolSelectionAdvisor({})
        score = advisor.score_tool("unknown")
        assert score.score == 0.5
        assert "no memory history" in score.reason

    def test_best_returns_top_tool(self):
        profiles = ToolMemoryProfile.build_profiles([
            _entry(tool_name="a", status="ok", causal_key="a1"),
            _entry(tool_name="b", status="error", failure_class="x",
                   policy_action="abort", causal_key="b1"),
        ])
        advisor = ToolSelectionAdvisor(profiles)
        best = advisor.best(["a", "b"])
        assert best is not None
        assert best.tool_name == "a"

    def test_best_empty_list_returns_none(self):
        advisor = ToolSelectionAdvisor({})
        assert advisor.best([]) is None

    def test_ranking_single_tool(self):
        profiles = ToolMemoryProfile.build_profiles([
            _entry(tool_name="solo", status="ok", causal_key="s1"),
        ])
        advisor = ToolSelectionAdvisor(profiles)
        ranked = advisor.rank(["solo"])
        assert len(ranked) == 1


class TestMemoryAwarePolicyContext:
    def _mock_engine(self, action: PolicyAction) -> FailurePolicyEngine:
        engine = MagicMock(spec=FailurePolicyEngine)
        engine.decide.return_value = PolicyDecision(
            action=action,
            reason="base reason",
            confidence=1.0,
        )
        return engine

    def _mock_result(self, tool_name: str = "tool_a") -> Any:
        r = MagicMock()
        r.tool_name = tool_name
        return r

    def _mock_contract(self) -> Any:
        c = MagicMock()
        c.max_total_attempts = 3
        return c

    def test_insufficient_history_no_override(self):
        """With < 3 executions, base decision is unchanged."""
        entries = [_entry(tool_name="t", status="error",
                          failure_class="x", causal_key="e1")]
        profiles = ToolMemoryProfile.build_profiles(entries)
        engine = self._mock_engine(PolicyAction.RETRY)
        ctx = MemoryAwarePolicyContext(engine, profiles)
        decision, adj = ctx.decide_with_memory(
            result=self._mock_result("t"),
            contract=self._mock_contract(),
            attempt=1,
        )
        assert not adj.adjusted
        assert "insufficient memory" in adj.adjustment_reason

    def test_abort_override_on_high_abort_rate(self):
        """If abort_rate >= threshold, RETRY → ABORT."""
        entries = (
            [_entry(tool_name="bad", status="error", policy_action="abort",
                    causal_key=f"e{i}", failure_class="x") for i in range(7)]
            + [_entry(tool_name="bad", status="ok", causal_key="ok1")]
        )
        profiles = ToolMemoryProfile.build_profiles(entries)
        engine = self._mock_engine(PolicyAction.RETRY)
        ctx = MemoryAwarePolicyContext(engine, profiles, abort_threshold=0.5)
        decision, adj = ctx.decide_with_memory(
            result=self._mock_result("bad"),
            contract=self._mock_contract(),
            attempt=1,
        )
        assert adj.adjusted
        assert decision.action == PolicyAction.ABORT

    def test_escalation_override_on_high_escalation_rate(self):
        """If escalation_rate >= threshold, RETRY → ESCALATE."""
        entries = (
            [_entry(tool_name="esc", status="error", policy_action="escalate",
                    causal_key=f"e{i}", failure_class="x") for i in range(6)]
            + [_entry(tool_name="esc", status="ok", causal_key="ok1")]
        )
        profiles = ToolMemoryProfile.build_profiles(entries)
        engine = self._mock_engine(PolicyAction.RETRY)
        ctx = MemoryAwarePolicyContext(engine, profiles, escalation_threshold=0.5)
        decision, adj = ctx.decide_with_memory(
            result=self._mock_result("esc"),
            contract=self._mock_contract(),
            attempt=1,
        )
        assert adj.adjusted
        assert decision.action == PolicyAction.ESCALATE

    def test_relaxation_healthy_tool_escalate_to_retry(self):
        """Healthy tool base ESCALATE should be relaxed to RETRY."""
        entries = [
            _entry(tool_name="good", status="ok", causal_key=f"g{i}") for i in range(10)
        ]
        profiles = ToolMemoryProfile.build_profiles(entries)
        engine = self._mock_engine(PolicyAction.ESCALATE)
        ctx = MemoryAwarePolicyContext(engine, profiles)
        decision, adj = ctx.decide_with_memory(
            result=self._mock_result("good"),
            contract=self._mock_contract(),
            attempt=1,
        )
        assert adj.adjusted
        assert decision.action == PolicyAction.RETRY

    def test_no_override_for_unknown_tool(self):
        engine = self._mock_engine(PolicyAction.RETRY)
        ctx = MemoryAwarePolicyContext(engine, {})
        decision, adj = ctx.decide_with_memory(
            result=self._mock_result("unknown_tool"),
            contract=self._mock_contract(),
            attempt=1,
        )
        assert not adj.adjusted

    def test_adjusted_decision_metadata(self):
        entries = (
            [_entry(tool_name="bad", status="error", policy_action="abort",
                    causal_key=f"e{i}", failure_class="x") for i in range(8)]
            + [_entry(tool_name="bad", status="ok", causal_key="ok1")]
        )
        profiles = ToolMemoryProfile.build_profiles(entries)
        engine = self._mock_engine(PolicyAction.RETRY)
        ctx = MemoryAwarePolicyContext(engine, profiles, abort_threshold=0.5)
        decision, adj = ctx.decide_with_memory(
            result=self._mock_result("bad"),
            contract=self._mock_contract(),
            attempt=1,
        )
        assert decision.metadata is not None
        assert decision.metadata.get("memory_adjusted") is True


class TestSchedulingAndTone:
    def test_no_history_neutral_priority(self):
        assert compute_scheduling_priority(None) == 5

    def test_reliable_tool_high_priority(self):
        entries = [_entry(tool_name="t", status="ok", causal_key=f"e{i}") for i in range(10)]
        profiles = ToolMemoryProfile.build_profiles(entries)
        p = compute_scheduling_priority(profiles["t"])
        assert p <= 3  # High reliability → low number = high priority

    def test_failing_tool_low_priority(self):
        entries = [
            _entry(tool_name="t", status="error", policy_action="abort",
                   failure_class="x", causal_key=f"e{i}") for i in range(10)
        ]
        profiles = ToolMemoryProfile.build_profiles(entries)
        p = compute_scheduling_priority(profiles["t"])
        assert p >= 7  # Low reliability → high number = low priority

    def test_response_tone_confident(self):
        entries = [_entry(tool_name="t", status="ok", causal_key=f"e{i}") for i in range(10)]
        profiles = ToolMemoryProfile.build_profiles(entries)
        tone = suggest_response_tone(profiles["t"])
        assert tone == "confident"

    def test_response_tone_no_history(self):
        assert suggest_response_tone(None) == "confident"

    def test_response_tone_warn_on_aborts(self):
        entries = (
            [_entry(tool_name="t", status="error", policy_action="abort",
                    failure_class="x", causal_key=f"a{i}") for i in range(5)]
            + [_entry(tool_name="t", status="ok", causal_key="ok1")]
        )
        profiles = ToolMemoryProfile.build_profiles(entries)
        tone = suggest_response_tone(profiles["t"])
        assert tone in {"warn", "escalate"}
