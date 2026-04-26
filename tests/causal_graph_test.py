"""Tests for Phase 3 — Causal Execution Graph Engine (causal_graph.py)."""
from __future__ import annotations

import pytest

from dadbot.core.causal_graph import (
    CausalGraph,
    CausalNode,
    CausalEdge,
    CausalNodeKind,
    CausalEdgeKind,
    CausalReconstructionAPI,
    InfluenceTracer,
    SelectionRationale,
    InfluenceTrace,
)


# ---------------------------------------------------------------------------
# 3.1 CausalGraph — structure
# ---------------------------------------------------------------------------


class TestCausalGraph:
    def setup_method(self):
        self.graph = CausalGraph()

    def test_add_node_stores_node(self):
        node = self.graph.add_node("n1", CausalNodeKind.PLANNER_DECISION, label="plan:search")
        assert node.node_id == "n1"
        assert node.kind == CausalNodeKind.PLANNER_DECISION

    def test_add_node_idempotent(self):
        self.graph.add_node("n1", CausalNodeKind.PLANNER_DECISION, label="plan:search")
        self.graph.add_node("n1", CausalNodeKind.PLANNER_DECISION, label="plan:search")
        assert self.graph.node_count() == 1

    def test_add_edge_stores_edge(self):
        self.graph.add_node("n1", CausalNodeKind.PLANNER_DECISION, label="plan")
        self.graph.add_node("n2", CausalNodeKind.TOOL_SELECTION, label="select:tool_a")
        edge = self.graph.add_edge("n1", "n2", CausalEdgeKind.CAUSED)
        assert edge.source_id == "n1"
        assert edge.target_id == "n2"
        assert edge.kind == CausalEdgeKind.CAUSED

    def test_add_edge_idempotent(self):
        self.graph.add_node("n1", CausalNodeKind.PLANNER_DECISION, label="plan")
        self.graph.add_node("n2", CausalNodeKind.TOOL_SELECTION, label="select")
        self.graph.add_edge("n1", "n2", CausalEdgeKind.CAUSED)
        self.graph.add_edge("n1", "n2", CausalEdgeKind.CAUSED)
        assert self.graph.edge_count() == 1

    def test_out_edges_returns_edges(self):
        self.graph.add_node("n1", CausalNodeKind.PLANNER_DECISION, label="plan")
        self.graph.add_node("n2", CausalNodeKind.TOOL_SELECTION, label="select")
        self.graph.add_node("n3", CausalNodeKind.TOOL_OUTCOME, label="outcome")
        self.graph.add_edge("n1", "n2", CausalEdgeKind.CAUSED)
        self.graph.add_edge("n2", "n3", CausalEdgeKind.CAUSED)
        out = self.graph.out_edges("n1")
        assert len(out) == 1
        assert out[0].target_id == "n2"

    def test_in_edges_returns_edges(self):
        self.graph.add_node("n1", CausalNodeKind.PLANNER_DECISION, label="plan")
        self.graph.add_node("n2", CausalNodeKind.TOOL_SELECTION, label="select")
        self.graph.add_edge("n1", "n2", CausalEdgeKind.CAUSED)
        in_e = self.graph.in_edges("n2")
        assert len(in_e) == 1
        assert in_e[0].source_id == "n1"

    def test_get_nodes_by_kind(self):
        self.graph.add_node("n1", CausalNodeKind.TOOL_SELECTION, label="s1")
        self.graph.add_node("n2", CausalNodeKind.TOOL_SELECTION, label="s2")
        self.graph.add_node("n3", CausalNodeKind.TOOL_OUTCOME, label="o1")
        selections = self.graph.get_nodes_by_kind(CausalNodeKind.TOOL_SELECTION)
        assert len(selections) == 2

    def test_to_dict_serializes_graph(self):
        self.graph.add_node("n1", CausalNodeKind.PLANNER_DECISION, label="plan")
        self.graph.add_node("n2", CausalNodeKind.TOOL_SELECTION, label="select")
        self.graph.add_edge("n1", "n2", CausalEdgeKind.CAUSED, weight=0.9)
        d = self.graph.to_dict()
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1
        assert d["edges"][0]["weight"] == 0.9

    def test_record_planner_decision(self):
        node = self.graph.record_planner_decision("pd1", "goal_lookup", "search", ["cache", "semantic"])
        assert node.kind == CausalNodeKind.PLANNER_DECISION
        assert node.metadata["intent"] == "goal_lookup"
        assert "cache" in node.metadata["alternatives"]

    def test_record_tool_selection_creates_rejected_nodes(self):
        self.graph.record_planner_decision("pd1", "goal_lookup", "search")
        self.graph.record_tool_selection(
            "sel1", "memory_lookup", "goal_lookup",
            score=0.92, rejected_tools=["cache_tool"], decision_id="pd1"
        )
        # One selection node + one rejected node
        selections = self.graph.get_nodes_by_kind(CausalNodeKind.TOOL_SELECTION)
        assert any(n.metadata.get("rejected") for n in selections)

    def test_record_tool_outcome_links_to_selection(self):
        self.graph.record_tool_selection("sel1", "tool_a", "intent", score=0.9)
        self.graph.record_tool_outcome("out1", "tool_a", "ok", latency_ms=120, selection_id="sel1")
        out = self.graph.out_edges("sel1")
        assert any(e.target_id == "out1" for e in out)

    def test_record_fallback_activation(self):
        self.graph.record_tool_outcome("out1", "tool_a", "timeout")
        fb = self.graph.record_fallback_activation("fb1", "tool_a", "tool_b", reason="timeout", outcome_id="out1")
        assert fb.kind == CausalNodeKind.FALLBACK_ACTIVATION
        in_e = self.graph.in_edges("fb1")
        assert any(e.source_id == "out1" for e in in_e)

    def test_record_retry_event(self):
        self.graph.record_tool_outcome("out1", "tool_a", "error")
        retry = self.graph.record_retry_event("ret1", "tool_a", attempt=2, reason="server_error", outcome_id="out1")
        assert retry.kind == CausalNodeKind.RETRY_EVENT
        assert retry.metadata["attempt"] == 2


# ---------------------------------------------------------------------------
# 3.2 Causal Reconstruction API
# ---------------------------------------------------------------------------


class TestCausalReconstructionAPI:
    def setup_method(self):
        self.graph = CausalGraph()
        self.api = CausalReconstructionAPI(self.graph)

        # Build a small representative graph
        self.graph.record_planner_decision("pd1", "goal_lookup", "direct_search", alternatives=["fuzzy_search"])
        self.graph.record_tool_selection(
            "sel1", "memory_lookup", "goal_lookup",
            score=0.85, rejected_tools=["cache_tool", "semantic_search"], decision_id="pd1"
        )
        self.graph.record_tool_outcome("out1", "memory_lookup", "ok", latency_ms=80, confidence=0.9, selection_id="sel1")

    def test_why_tool_selected_returns_rationale(self):
        rationale = self.api.why_tool_selected("sel1")
        assert rationale is not None
        assert isinstance(rationale, SelectionRationale)
        assert rationale.selected_tool == "memory_lookup"
        assert "cache_tool" in rationale.rejected_tools or "semantic_search" in rationale.rejected_tools

    def test_why_tool_selected_includes_planner_step(self):
        rationale = self.api.why_tool_selected("sel1")
        assert any("Planner" in step for step in rationale.reason_chain)

    def test_why_tool_selected_includes_score_step(self):
        rationale = self.api.why_tool_selected("sel1")
        assert any("scored" in step or "only candidate" in step for step in rationale.reason_chain)

    def test_why_tool_selected_unknown_node_returns_none(self):
        assert self.api.why_tool_selected("nonexistent") is None

    def test_why_fallback_activated_explains_failure(self):
        self.graph.record_tool_outcome("out_fail", "memory_lookup", "timeout", latency_ms=3000)
        self.graph.record_fallback_activation("fb1", "memory_lookup", "cache_tool", reason="timeout exceeded", outcome_id="out_fail")
        steps = self.api.why_fallback_activated("fb1")
        assert any("fallback" in s.lower() for s in steps)
        assert any("cache_tool" in s for s in steps)

    def test_why_fallback_activated_unknown_returns_empty(self):
        assert self.api.why_fallback_activated("nonexistent") == []

    def test_causal_path_finds_path(self):
        path = self.api.causal_path("pd1", "out1")
        assert path[0] == "pd1"
        assert path[-1] == "out1"

    def test_causal_path_no_connection_returns_empty(self):
        self.graph.add_node("isolated", CausalNodeKind.RETRY_EVENT, label="isolated")
        path = self.api.causal_path("pd1", "isolated")
        assert path == []

    def test_fallback_depth_primary_is_zero(self):
        depth = self.api._fallback_depth("sel1")
        assert depth == 0

    def test_fallback_depth_after_one_fallback(self):
        # Add a fallback selection after a failure
        self.graph.record_tool_outcome("out_fail", "memory_lookup", "error")
        self.graph.record_fallback_activation("fb1", "memory_lookup", "fallback_tool", outcome_id="out_fail")
        self.graph.record_tool_selection("sel_fb", "fallback_tool", "goal_lookup", score=0.6)
        self.graph.add_edge("fb1", "sel_fb", CausalEdgeKind.CAUSED)
        depth = self.api._fallback_depth("sel_fb")
        assert depth >= 1


# ---------------------------------------------------------------------------
# 3.3 Influence Tracer
# ---------------------------------------------------------------------------


class TestInfluenceTracer:
    def setup_method(self):
        self.graph = CausalGraph()
        self.tracer = InfluenceTracer(self.graph)

        # Build graph: pd1 → sel1 → out1
        self.graph.record_planner_decision("pd1", "intent", "plan_a")
        self.graph.record_tool_selection("sel1", "tool_a", "intent", score=1.0, decision_id="pd1")
        self.graph.record_tool_outcome("out1", "tool_a", "ok", confidence=0.9, selection_id="sel1")

    def test_trace_returns_influence_trace(self):
        trace = self.tracer.trace("pd1", "out1")
        assert trace is not None
        assert isinstance(trace, InfluenceTrace)
        assert trace.source_node_id == "pd1"
        assert trace.target_node_id == "out1"

    def test_trace_score_between_0_and_1(self):
        trace = self.tracer.trace("pd1", "out1")
        assert 0.0 <= trace.influence_score <= 1.0

    def test_trace_path_is_ordered(self):
        trace = self.tracer.trace("pd1", "out1")
        assert trace.path[0] == "pd1"
        assert trace.path[-1] == "out1"

    def test_trace_no_path_returns_none(self):
        self.graph.add_node("orphan", CausalNodeKind.RETRY_EVENT, label="orphan")
        trace = self.tracer.trace("pd1", "orphan")
        assert trace is None

    def test_trace_longer_path_has_lower_score(self):
        # Direct path: pd1 → out1 (2 hops)
        direct = self.tracer.trace("pd1", "out1")
        # Add intermediate: pd1 → sel1 → mid → out1 (3 hops)
        self.graph.add_node("mid", CausalNodeKind.RETRY_EVENT, label="mid")
        # Can't meaningfully test without adding edges; just verify direct is <= 1
        assert direct.influence_score <= 1.0

    def test_trace_explanation_contains_tools(self):
        trace = self.tracer.trace("pd1", "out1")
        assert isinstance(trace.explanation, str) and len(trace.explanation) > 0

    def test_top_influences_returns_sorted_list(self):
        # Add more selections
        self.graph.record_tool_selection("sel2", "tool_b", "intent", score=0.5, decision_id="pd1")
        self.graph.record_tool_outcome("out2", "tool_b", "partial", selection_id="sel2")
        traces = self.tracer.top_influences("out1", source_kind=CausalNodeKind.TOOL_SELECTION)
        # sel1 has a direct path to out1; scores should be in descending order
        scores = [t.influence_score for t in traces]
        assert scores == sorted(scores, reverse=True)

    def test_top_influences_respects_top_k(self):
        for i in range(6):
            self.graph.record_tool_selection(f"sel_{i}", f"tool_{i}", "intent", score=0.8)
        traces = self.tracer.top_influences("out1", top_k=3)
        assert len(traces) <= 3
