"""ROI #5 — Architecture Graph Cycle Enforcement.

Tests that:
  1. The cycle-detection algorithm (Tarjan SCC) works correctly on known
     synthetic graphs.
  2. The live dadbot import graph is checked against a set of forbidden
     import edges and must have zero violations.
  3. A synthetic forbidden-edge scenario correctly reports a violation.

Running this in CI provides hard enforcement that no new circular
coupling has been introduced into the architecture.
"""

from __future__ import annotations

from dadbot.uril.architecture import (
    FORBIDDEN_IMPORT_EDGES,
    detect_cycles,
    find_forbidden_cycles,
)

# ---------------------------------------------------------------------------
# Unit tests for Tarjan SCC implementation
# ---------------------------------------------------------------------------


class TestDetectCyclesAlgorithm:
    """Verify Tarjan SCC correctness on synthetic graphs."""

    def test_empty_graph_has_no_cycles(self):
        assert detect_cycles({}) == []

    def test_single_node_no_self_loop_has_no_cycles(self):
        graph = {"a": set()}
        assert detect_cycles(graph) == []

    def test_single_self_loop_is_not_a_cycle_for_purposes_of_scc(self):
        # Tarjan returns self-loops as SCCs of size 1 — not flagged
        graph = {"a": {"a"}}
        cycles = detect_cycles(graph)
        # Our implementation: detect_cycles returns SCCs of size > 1 only
        assert cycles == []

    def test_two_node_mutual_import_is_cycle(self):
        graph = {"a": {"b"}, "b": {"a"}}
        cycles = detect_cycles(graph)
        assert len(cycles) == 1
        assert set(cycles[0]) == {"a", "b"}

    def test_three_node_ring_is_one_cycle(self):
        graph = {"a": {"b"}, "b": {"c"}, "c": {"a"}}
        cycles = detect_cycles(graph)
        assert len(cycles) == 1
        assert set(cycles[0]) == {"a", "b", "c"}

    def test_disconnected_cycles_detected_separately(self):
        graph = {
            "a": {"b"},
            "b": {"a"},  # cycle 1
            "c": {"d"},
            "d": {"c"},  # cycle 2
            "e": set(),  # no cycle
        }
        cycles = detect_cycles(graph)
        assert len(cycles) == 2
        cycle_sets = [frozenset(c) for c in cycles]
        assert frozenset({"a", "b"}) in cycle_sets
        assert frozenset({"c", "d"}) in cycle_sets

    def test_dag_has_no_cycles(self):
        graph = {
            "core": {"models"},
            "services": {"core", "models"},
            "api": {"services", "core"},
            "models": set(),
        }
        assert detect_cycles(graph) == []

    def test_complex_graph_with_isolated_cycle(self):
        graph = {
            "a": {"b", "c"},
            "b": {"d"},
            "c": {"d"},
            "d": {"e"},
            "e": {"b"},  # b→d→e→b cycle
            "f": set(),
        }
        cycles = detect_cycles(graph)
        assert len(cycles) == 1
        assert set(cycles[0]) == {"b", "d", "e"}


# ---------------------------------------------------------------------------
# Unit tests for forbidden-edge detection
# ---------------------------------------------------------------------------


class TestFindForbiddenCycles:
    """find_forbidden_cycles must correctly detect forbidden import edges."""

    def test_no_violations_on_clean_graph(self):
        graph = {
            "dadbot.core.dadbot": {"dadbot.models"},
            "dadbot.models": set(),
            "dadbot_system.kernel": {"dadbot.models"},
        }
        forbidden = [("dadbot_system", "dadbot.core.dadbot")]
        assert find_forbidden_cycles(forbidden, graph=graph) == []

    def test_detects_forbidden_edge(self):
        # dadbot_system imports dadbot.core.dadbot → forbidden
        graph = {
            "dadbot_system.kernel": {"dadbot.core.dadbot"},
            "dadbot.core.dadbot": set(),
        }
        forbidden = [("dadbot_system", "dadbot.core.dadbot")]
        violations = find_forbidden_cycles(forbidden, graph=graph)
        assert len(violations) >= 1
        assert any("dadbot_system" in src and "dadbot.core.dadbot" in dst for src, dst in violations)

    def test_empty_forbidden_list_returns_no_violations(self):
        graph = {"a": {"b"}, "b": {"a"}}
        assert find_forbidden_cycles([], graph=graph) == []

    def test_fragment_matching_is_substring_based(self):
        graph = {
            "myapp.dadbot_system.services": {"myapp.dadbot.core.dadbot.main"},
            "myapp.dadbot.core.dadbot.main": set(),
        }
        forbidden = [("dadbot_system", "dadbot.core.dadbot")]
        violations = find_forbidden_cycles(forbidden, graph=graph)
        assert len(violations) >= 1


# ---------------------------------------------------------------------------
# Live architecture enforcement
# ---------------------------------------------------------------------------


class TestLiveArchitectureCycleEnforcement:
    """Hard CI gate: forbidden coupling edges must not exist in the real codebase."""

    def test_no_forbidden_edges_in_live_graph(self):
        """None of the declared FORBIDDEN_IMPORT_EDGES may exist in dadbot/.

        This is a hard-fail enforcement test.  If it fails, an illegal import
        coupling was introduced into the architecture.
        """
        violations = find_forbidden_cycles()  # uses defaults (live graph + FORBIDDEN_IMPORT_EDGES)
        assert violations == [], "Forbidden import edges detected in live architecture:\n" + "\n".join(
            f"  {src} → {dst}" for src, dst in violations
        )

    def test_live_graph_loads_without_error(self):
        """_build_graph must complete without throwing."""
        cycles = detect_cycles()  # triggers _build_graph() internally
        # We don't assert zero cycles (complex software may have allowed ones)
        # We just assert the function completes and returns a list
        assert isinstance(cycles, list)

    def test_forbidden_edge_list_is_defined(self):
        assert isinstance(FORBIDDEN_IMPORT_EDGES, list)
        assert len(FORBIDDEN_IMPORT_EDGES) >= 1
        for edge in FORBIDDEN_IMPORT_EDGES:
            assert isinstance(edge, tuple) and len(edge) == 2
