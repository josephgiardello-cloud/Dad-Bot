"""Tests for HeritageGraphManager — semantic cross-linking of narrative memories."""
from __future__ import annotations

import unittest
from types import SimpleNamespace


def _make_bot(narratives=None, consolidated=None, archive=None):
    """Minimal bot stub for HeritageGraphManager."""
    return SimpleNamespace(
        narrative_memories=lambda: list(narratives or []),
        consolidated_memories=lambda: list(consolidated or []),
        session_archive=lambda: list(archive or []),
    )


class HeritageGraphCrossLinkTests(unittest.TestCase):
    def setUp(self):
        from dadbot.managers.heritage_graph import HeritageGraphManager
        self.HeritageGraphManager = HeritageGraphManager

    # ── Positive matching (keyword overlap required) ───────────────────────

    def test_cross_link_query_surfaces_overlapping_topic(self):
        """When narratives share keywords with the query they should be returned."""
        bot = _make_bot(narratives=[
            {
                "topic": "woodworking",
                "period": "2026-03",
                "summary": "Tony showed persistence and practice building a shelf despite setbacks.",
                "evidence": "He kept at it through failure.",
            },
            {
                "topic": "fitness",
                "period": "2026-04",
                "summary": "Tony ran a 5k through endurance and daily practice.",
                "evidence": "Daily training.",
            },
        ])
        mgr = self.HeritageGraphManager(bot)
        links = mgr.cross_link_query("I need persistence and practice to get through this math problem")
        self.assertGreater(len(links), 0)
        topics = [link["from_topic"] for link in links]
        self.assertTrue(any(t in ("woodworking", "fitness") for t in topics))

    def test_cross_link_excludes_current_primary_topic(self):
        """Cross-links must exclude narratives whose topic matches the current context's primary topic."""
        bot = _make_bot(narratives=[
            {
                "topic": "math",
                "period": "2026-02",
                "summary": "Tony showed persistence practice with math division problems.",
                "evidence": "Lots of math sessions.",
            },
            {
                "topic": "woodworking",
                "period": "2026-03",
                "summary": "Tony showed persistence and practice building a shelf.",
                "evidence": "workshop sessions",
            },
        ])
        mgr = self.HeritageGraphManager(bot)
        links = mgr.cross_link_query("I need persistence and practice to get through this math division")
        topics = [link["from_topic"] for link in links]
        self.assertNotIn("math", topics)

    def test_cross_link_generates_bridge_phrase(self):
        bot = _make_bot(narratives=[
            {
                "topic": "woodworking",
                "period": "2026-03",
                "summary": "Tony built a shelf and showed tremendous persistence.",
                "evidence": "cabinet project",
            },
        ])
        mgr = self.HeritageGraphManager(bot)
        links = mgr.cross_link_query("I'm not giving up, showing real persistence on this project")
        if links:
            bridge = links[0].get("bridge_phrase", "")
            self.assertIsInstance(bridge, str)
            self.assertGreater(len(bridge), 10)

    def test_cross_link_returns_empty_for_no_narratives(self):
        bot = _make_bot()
        mgr = self.HeritageGraphManager(bot)
        result = mgr.cross_link_query("trying to finish a project")
        self.assertEqual(result, [])

    def test_cross_link_returns_empty_when_no_keyword_overlap(self):
        """When the query has zero keyword overlap with narratives, nothing should surface."""
        bot = _make_bot(narratives=[
            {
                "topic": "woodworking",
                "period": "2026-03",
                "summary": "Tony built a cabinet despite setbacks.",
                "evidence": "workshop",
            },
        ])
        mgr = self.HeritageGraphManager(bot)
        links = mgr.cross_link_query("xyz123 qwerty zorp blorg")
        self.assertEqual(links, [])

    def test_cross_link_respects_max_links(self):
        bot = _make_bot(narratives=[
            {"topic": "woodworking", "period": "2026-01", "summary": "persistence practice built things", "evidence": ""},
            {"topic": "fitness", "period": "2026-02", "summary": "persistence practice ran 5k", "evidence": ""},
            {"topic": "work", "period": "2026-02", "summary": "persistence practice job promotion", "evidence": ""},
            {"topic": "creativity", "period": "2026-03", "summary": "persistence practice painted canvas", "evidence": ""},
        ])
        mgr = self.HeritageGraphManager(bot)
        links = mgr.cross_link_query("I need persistence and practice doing something new today", max_links=2)
        self.assertLessEqual(len(links), 2)

    # ── heritage_context_block ─────────────────────────────────────────────

    def test_heritage_context_block_returns_string_with_links(self):
        bot = _make_bot(narratives=[
            {
                "topic": "woodworking",
                "period": "2026-03",
                "summary": "Tony built a shelf and showed tremendous persistence.",
                "evidence": "cabinet project",
            },
        ])
        mgr = self.HeritageGraphManager(bot)
        block = mgr.heritage_context_block("I'm struggling but showing real persistence on this challenge")
        if block is not None:
            self.assertIn("Heritage Memory", block)

    def test_heritage_context_block_returns_none_for_empty_store(self):
        bot = _make_bot()
        mgr = self.HeritageGraphManager(bot)
        result = mgr.heritage_context_block("some context text")
        self.assertIsNone(result)

    # ── _friendly_period ───────────────────────────────────────────────────

    def test_friendly_period_formats_year_month(self):
        from dadbot.managers.heritage_graph import _friendly_period
        self.assertEqual(_friendly_period("2026-03"), "March 2026")
        self.assertEqual(_friendly_period("2025-12"), "December 2025")

    def test_friendly_period_passes_through_unknown_formats(self):
        from dadbot.managers.heritage_graph import _friendly_period
        self.assertEqual(_friendly_period("recently"), "recently")
        self.assertEqual(_friendly_period(""), "recently")

    # ── Score structure ────────────────────────────────────────────────────

    def test_cross_link_result_has_required_keys(self):
        bot = _make_bot(narratives=[
            {
                "topic": "fitness",
                "period": "2026-01",
                "summary": "Tony trained for a 5k and persisted through persistence setbacks.",
                "evidence": "running logs",
            },
        ])
        mgr = self.HeritageGraphManager(bot)
        links = mgr.cross_link_query("keep going even though it's hard with persistence")
        if links:
            link = links[0]
            for key in ("from_topic", "from_period", "past_summary", "bridge_phrase", "score"):
                self.assertIn(key, link, f"Missing key: {key}")
            self.assertIsInstance(link["score"], float)


if __name__ == "__main__":
    unittest.main()
