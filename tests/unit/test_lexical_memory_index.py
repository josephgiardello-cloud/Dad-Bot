from concurrent.futures import ThreadPoolExecutor

import pytest

from dadbot.memory.lexical_index import LexicalMemoryIndex

pytestmark = pytest.mark.unit


@pytest.fixture
def index():
    instance = LexicalMemoryIndex()
    yield instance
    instance.close()


def test_existing_catalog_backfills_and_refreshes_after_edit_delete_and_clear(index):
    rows = [("a", "mint tea"), ("b", "coffee order")]
    assert index.scores(rows, {"tea"}) == {"a": 1.0}
    rows[0] = ("a", "green coffee")
    assert index.scores(rows, {"tea"}) == {}
    assert set(index.scores(rows, {"coffee"})) == {"a", "b"}
    rows.pop()
    assert index.scores(rows, {"coffee"}) == {"a": 1.0}
    assert index.scores([], {"coffee"}) == {}
    assert index.scores([("c", "mint tea")], {"tea"}) == {"c": 1.0}


def test_bm25_rewards_rare_query_terms_and_returns_normalized_scores(index):
    rows = [("tea", "mint tea preference"), ("coffee", "coffee preference")]
    rows.extend((f"other-{i}", "coffee preference") for i in range(10))
    scores = index.scores(rows, {"tea", "preference"})
    assert scores["tea"] == 1.0
    assert scores["coffee"] < scores["tea"]
    assert all(0.0 < score <= 1.0 for score in scores.values())


def test_fts_query_literals_are_not_query_language(index):
    assert index.scores([("a", "tea"), ("b", "coffee")], {'tea" OR coffee'}) == {}
    assert index.scores([("a", "tea")], set()) == {}


def test_snapshot_and_scores_do_not_depend_on_catalog_order(index):
    rows = [("b", "mint tea"), ("a", "mint tea")]
    assert index.scores(rows, {"tea"}) == index.scores(list(reversed(rows)), {"tea"})
    assert list(index.scores(rows, {"tea"})) == ["a", "b"]


def test_parallel_catalog_reads_remain_isolated(index):
    def query(i):
        key = str(i)
        assert index.scores([(key, "mint tea")], {"tea"}) == {key: 1.0}
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(query, range(20)))


def test_close_is_idempotent(index):
    index.close()
    index.close()


def test_new_instance_backfills_without_saved_index():
    rows = [("a", "mint tea"), ("b", "coffee")]
    first = LexicalMemoryIndex()
    try:
        before = first.scores(rows, {"tea"})
    finally:
        first.close()
    restarted = LexicalMemoryIndex()
    try:
        assert restarted.scores(rows, {"tea"}) == before
    finally:
        restarted.close()


def test_catalogs_are_not_shared_between_tenant_instances(index):
    other = LexicalMemoryIndex()
    try:
        assert index.scores([("same-key", "mint tea")], {"tea"}) == {"same-key": 1.0}
        assert other.scores([("same-key", "coffee")], {"tea"}) == {}
        assert index.scores([("same-key", "mint tea")], {"tea"}) == {"same-key": 1.0}
    finally:
        other.close()


def test_incremental_update_preserves_unchanged_rows_and_limits_candidates(index):
    rows = [("a", "mint tea"), ("b", "coffee tea"), ("c", "green tea")]
    index.scores(rows, {"tea"})
    before = dict(index._connection.execute("SELECT memory_key, rowid FROM memory_fts"))
    updated = [("a", "mint tea"), ("b", "coffee"), ("d", "black tea")]
    scores = index.scores(updated, {"tea"}, limit=1)
    after = dict(index._connection.execute("SELECT memory_key, rowid FROM memory_fts"))
    assert before["a"] == after["a"]
    assert set(after) == {"a", "b", "d"}
    assert len(scores) == 1
    assert "b" not in index.scores(updated, {"tea"})
    with pytest.raises(ValueError, match="positive"):
        index.scores(updated, {"tea"}, limit=0)


def test_large_catalog_keeps_exact_term_and_removes_deleted_conflict(index):
    rows = [(str(i), f"synthetic item number {i} common topic") for i in range(1000)]
    rows.extend([("old", "favorite beverage is green tea"), ("new", "favorite beverage is black tea")])
    assert index.scores(rows, {"black"}) == {"new": 1.0}
    assert index.scores(rows, {"marmalade"}) == {}
    rows = [row for row in rows if row[0] != "old"]
    assert "old" not in index.scores(rows, {"beverage"})
    assert "new" in index.scores(rows, {"beverage"})
