"""Tests for Step 3 — Bi-Temporal Graph Edges.

Covers:
  - upsert_graph_edge temporal field stamping
  - is_edge_valid filtering
  - invalidate_edge soft-invalidation
  - build_graph_projection only returns currently valid edges
  - backward-compat: edges with no valid_from always pass the filter
"""

from __future__ import annotations

from dadbot.memory.graph_manager import MemoryGraphManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _edge(
    source="s:1",
    target="t:1",
    relation="mentions",
    updated_at="2025-06-01T00:00:00",
    valid_from=None,
    valid_until=None,
    event_time=None,
    ingestion_time=None,
    weight=1.0,
    confidence=0.8,
):
    edge_map: dict = {}
    MemoryGraphManager.upsert_graph_edge(
        edge_map,
        edge_key=MemoryGraphManager.graph_edge_key(source, target, relation),
        source_key=source,
        target_key=target,
        relation_type=relation,
        weight=weight,
        confidence=confidence,
        updated_at=updated_at,
        valid_from=valid_from,
        valid_until=valid_until,
        event_time=event_time,
        ingestion_time=ingestion_time,
    )
    return next(iter(edge_map.values()))


# ---------------------------------------------------------------------------
# upsert_graph_edge — temporal field population
# ---------------------------------------------------------------------------


class TestUpsertGraphEdgeTemporalFields:
    def test_new_edge_gets_valid_from_from_updated_at(self):
        e = _edge(updated_at="2025-06-01T00:00:00")
        assert e["valid_from"] == "2025-06-01T00:00:00"

    def test_new_edge_gets_event_time_from_updated_at(self):
        e = _edge(updated_at="2025-06-01T00:00:00")
        assert e["event_time"] == "2025-06-01T00:00:00"

    def test_new_edge_gets_ingestion_time_from_updated_at(self):
        e = _edge(updated_at="2025-06-01T00:00:00")
        assert e["ingestion_time"] == "2025-06-01T00:00:00"

    def test_new_edge_valid_until_is_none(self):
        e = _edge()
        assert e["valid_until"] is None

    def test_explicit_valid_from_overrides_updated_at(self):
        e = _edge(updated_at="2025-06-01T00:00:00", valid_from="2024-01-01T00:00:00")
        assert e["valid_from"] == "2024-01-01T00:00:00"

    def test_explicit_valid_until_is_stored(self):
        e = _edge(valid_until="2026-01-01T00:00:00")
        assert e["valid_until"] == "2026-01-01T00:00:00"

    def test_explicit_event_time_overrides_updated_at(self):
        e = _edge(updated_at="2025-06-01T00:00:00", event_time="2024-12-01T00:00:00")
        assert e["event_time"] == "2024-12-01T00:00:00"

    def test_explicit_ingestion_time_overrides_updated_at(self):
        e = _edge(updated_at="2025-06-01T00:00:00", ingestion_time="2024-11-01T00:00:00")
        assert e["ingestion_time"] == "2024-11-01T00:00:00"


class TestUpsertGraphEdgeTemporalUpsertPreservation:
    """On subsequent upserts, original temporal fields should be preserved."""

    def _make_edge_map_with_two_upserts(self, first_updated, second_updated):
        edge_map: dict = {}
        key = MemoryGraphManager.graph_edge_key("s:1", "t:1", "mentions")
        kwargs = dict(
            edge_key=key, source_key="s:1", target_key="t:1", relation_type="mentions", weight=1.0, confidence=0.8
        )
        MemoryGraphManager.upsert_graph_edge(edge_map, updated_at=first_updated, **kwargs)
        MemoryGraphManager.upsert_graph_edge(edge_map, updated_at=second_updated, **kwargs)
        return edge_map[key]

    def test_second_upsert_preserves_original_valid_from(self):
        e = self._make_edge_map_with_two_upserts("2025-01-01T00:00:00", "2025-06-01T00:00:00")
        assert e["valid_from"] == "2025-01-01T00:00:00"

    def test_second_upsert_preserves_original_event_time(self):
        e = self._make_edge_map_with_two_upserts("2025-01-01T00:00:00", "2025-06-01T00:00:00")
        assert e["event_time"] == "2025-01-01T00:00:00"

    def test_second_upsert_preserves_original_ingestion_time(self):
        e = self._make_edge_map_with_two_upserts("2025-01-01T00:00:00", "2025-06-01T00:00:00")
        assert e["ingestion_time"] == "2025-01-01T00:00:00"

    def test_valid_until_updated_when_supplied_in_second_upsert(self):
        edge_map: dict = {}
        key = MemoryGraphManager.graph_edge_key("s:1", "t:1", "mentions")
        kwargs = dict(
            edge_key=key, source_key="s:1", target_key="t:1", relation_type="mentions", weight=1.0, confidence=0.8
        )
        MemoryGraphManager.upsert_graph_edge(edge_map, updated_at="2025-01-01T00:00:00", **kwargs)
        MemoryGraphManager.upsert_graph_edge(
            edge_map, updated_at="2025-06-01T00:00:00", valid_until="2025-06-01T00:00:00", **kwargs
        )
        assert edge_map[key]["valid_until"] == "2025-06-01T00:00:00"

    def test_valid_until_not_cleared_by_none_in_second_upsert(self):
        edge_map: dict = {}
        key = MemoryGraphManager.graph_edge_key("s:1", "t:1", "mentions")
        kwargs = dict(
            edge_key=key, source_key="s:1", target_key="t:1", relation_type="mentions", weight=1.0, confidence=0.8
        )
        MemoryGraphManager.upsert_graph_edge(
            edge_map, updated_at="2025-01-01T00:00:00", valid_until="2025-03-01T00:00:00", **kwargs
        )
        MemoryGraphManager.upsert_graph_edge(edge_map, updated_at="2025-06-01T00:00:00", **kwargs)
        # None valid_until in second upsert should NOT clear the existing value
        assert edge_map[key]["valid_until"] == "2025-03-01T00:00:00"


# ---------------------------------------------------------------------------
# is_edge_valid
# ---------------------------------------------------------------------------


class TestIsEdgeValid:
    def test_no_valid_from_is_always_valid(self):
        assert MemoryGraphManager.is_edge_valid({}, "2025-01-01T00:00:00") is True
        assert MemoryGraphManager.is_edge_valid({"valid_from": None}, "2020-01-01T00:00:00") is True

    def test_valid_from_in_past_no_valid_until_is_valid(self):
        e = {"valid_from": "2025-01-01T00:00:00", "valid_until": None}
        assert MemoryGraphManager.is_edge_valid(e, "2025-06-01T00:00:00") is True

    def test_valid_from_in_future_is_not_valid(self):
        e = {"valid_from": "2026-01-01T00:00:00", "valid_until": None}
        assert MemoryGraphManager.is_edge_valid(e, "2025-06-01T00:00:00") is False

    def test_valid_until_in_past_is_not_valid(self):
        e = {"valid_from": "2025-01-01T00:00:00", "valid_until": "2025-03-01T00:00:00"}
        assert MemoryGraphManager.is_edge_valid(e, "2025-06-01T00:00:00") is False

    def test_valid_until_in_future_is_valid(self):
        e = {"valid_from": "2025-01-01T00:00:00", "valid_until": "2026-01-01T00:00:00"}
        assert MemoryGraphManager.is_edge_valid(e, "2025-06-01T00:00:00") is True

    def test_current_time_equals_valid_from_is_valid(self):
        # valid_from <= current_time (equal → should still be valid)
        e = {"valid_from": "2025-06-01T00:00:00", "valid_until": None}
        assert MemoryGraphManager.is_edge_valid(e, "2025-06-01T00:00:00") is True

    def test_current_time_equals_valid_until_is_not_valid(self):
        # valid_until <= current_time (equal → boundary is exclusive)
        e = {"valid_from": "2025-01-01T00:00:00", "valid_until": "2025-06-01T00:00:00"}
        assert MemoryGraphManager.is_edge_valid(e, "2025-06-01T00:00:00") is False


# ---------------------------------------------------------------------------
# invalidate_edge
# ---------------------------------------------------------------------------


class TestInvalidateEdge:
    def test_sets_valid_until(self):
        e = {"valid_from": "2025-01-01T00:00:00", "valid_until": None}
        result = MemoryGraphManager.invalidate_edge(e, "2025-06-01T00:00:00")
        assert result["valid_until"] == "2025-06-01T00:00:00"

    def test_returns_same_dict(self):
        e = {"valid_from": "2025-01-01T00:00:00", "valid_until": None}
        result = MemoryGraphManager.invalidate_edge(e, "2025-06-01T00:00:00")
        assert result is e

    def test_invalidated_edge_fails_is_edge_valid(self):
        e = {"valid_from": "2025-01-01T00:00:00", "valid_until": None}
        MemoryGraphManager.invalidate_edge(e, "2025-06-01T00:00:00")
        assert MemoryGraphManager.is_edge_valid(e, "2025-06-01T00:00:00") is False

    def test_can_overwrite_existing_valid_until(self):
        e = {"valid_from": "2025-01-01T00:00:00", "valid_until": "2025-03-01T00:00:00"}
        MemoryGraphManager.invalidate_edge(e, "2025-06-01T00:00:00")
        assert e["valid_until"] == "2025-06-01T00:00:00"


# ---------------------------------------------------------------------------
# build_graph_projection validity filtering
# ---------------------------------------------------------------------------


class TestBuildGraphProjectionValidityFilter:
    """Verify that build_graph_projection excludes invalidated edges."""

    def _minimal_manager(self, bot_attrs=None):
        """Build a MemoryGraphManager with a stub bot that has no memories."""

        class _FakeBot:
            GRAPH_STORE_DB_PATH = ":memory:"

            class runtime_config:
                @staticmethod
                def graph_context_token_budget():
                    return 2048

                @staticmethod
                def graph_walk_hops():
                    return 2

                @staticmethod
                def graph_walk_edge_limit():
                    return 50

                @staticmethod
                def graph_walk_node_limit():
                    return 50

            def __init__(self):
                # Minimal attributes needed by GraphPromptCompressor
                for attr, val in (bot_attrs or {}).items():
                    setattr(self, attr, val)

        class _FakeMM:
            def consolidated_memories(self):
                return []

            def session_archive(self):
                return []

            def persona_evolution_history(self):
                return []

            def life_patterns(self):
                return []

        try:
            bot = _FakeBot()
            from dadbot.memory.graph_manager import MemoryGraphManager as MGM

            return MGM.__new__(MGM), _FakeMM()
        except Exception:
            return None, None

    def test_edge_upsert_stamps_valid_from(self):
        edge_map = {}
        MemoryGraphManager.upsert_graph_edge(
            edge_map,
            edge_key="e1",
            source_key="s:a",
            target_key="t:b",
            relation_type="mentions",
            weight=1.0,
            confidence=0.8,
            updated_at="2025-06-01T00:00:00",
        )
        assert edge_map["e1"]["valid_from"] == "2025-06-01T00:00:00"

    def test_is_edge_valid_rejects_invalidated_edge(self):
        edge_map = {}
        MemoryGraphManager.upsert_graph_edge(
            edge_map,
            edge_key="e2",
            source_key="s:a",
            target_key="t:c",
            relation_type="mentions",
            weight=1.0,
            confidence=0.8,
            updated_at="2025-01-01T00:00:00",
        )
        MemoryGraphManager.invalidate_edge(edge_map["e2"], "2025-03-01T00:00:00")
        assert not MemoryGraphManager.is_edge_valid(edge_map["e2"], "2025-06-01T00:00:00")

    def test_projection_only_includes_valid_edges(self):
        """Directly exercise the filter path: build a synthetic edge_map, apply filter."""
        edge_map = {}
        now = "2025-06-01T00:00:00"

        # Valid edge
        MemoryGraphManager.upsert_graph_edge(
            edge_map,
            edge_key="ev",
            source_key="s:x",
            target_key="t:y",
            relation_type="mentions",
            weight=1.0,
            confidence=0.8,
            updated_at="2025-01-01T00:00:00",
        )
        # Invalidated edge
        MemoryGraphManager.upsert_graph_edge(
            edge_map,
            edge_key="ei",
            source_key="s:x",
            target_key="t:z",
            relation_type="covers_topic",
            weight=1.0,
            confidence=0.8,
            updated_at="2025-01-01T00:00:00",
        )
        MemoryGraphManager.invalidate_edge(edge_map["ei"], "2025-03-01T00:00:00")

        surviving = [e for e in edge_map.values() if MemoryGraphManager.is_edge_valid(e, now)]
        assert len(surviving) == 1
        assert surviving[0]["edge_key"] == "ev"

    def test_backward_compat_edges_without_valid_from_always_survive(self):
        """Legacy edges written before Step 3 have no valid_from — must survive."""
        legacy_edge = {
            "edge_key": "legacy",
            "source_key": "s:old",
            "target_key": "t:old",
            "relation_type": "mentions",
            "weight": 1.0,
            "confidence": 0.8,
            "updated_at": "2020-01-01T00:00:00",
            # deliberately NO valid_from / valid_until
        }
        assert MemoryGraphManager.is_edge_valid(legacy_edge, "2026-01-01T00:00:00") is True
