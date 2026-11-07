from __future__ import annotations

import sqlite3
from threading import RLock


class LexicalMemoryIndex:
    """FTS5 snapshot of the caller's current, tenant-scoped memory catalog."""

    def __init__(self) -> None:
        self._connection = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            self._connection.execute(
                "CREATE VIRTUAL TABLE memory_fts USING fts5(memory_key UNINDEXED, summary, tokenize='unicode61')"
            )
        except sqlite3.Error:
            self._connection.close()
            raise
        self._snapshot: tuple[tuple[str, str], ...] = ()
        self._lock = RLock()
        self._closed = False

    def scores(self, rows: list[tuple[str, str]], tokens: set[str]) -> dict[str, float]:
        snapshot = tuple(sorted(set(rows)))
        with self._lock:
            if snapshot != self._snapshot:
                with self._connection:
                    self._connection.execute("DELETE FROM memory_fts")
                    self._connection.executemany(
                        "INSERT INTO memory_fts(memory_key, summary) VALUES (?, ?)", snapshot,
                    )
                self._snapshot = snapshot
            if not tokens:
                return {}
            expression = " OR ".join('"' + token.replace('"', '""') + '"' for token in sorted(tokens))
            matches = self._connection.execute(
                "SELECT memory_key, -bm25(memory_fts) AS score FROM memory_fts "
                "WHERE memory_fts MATCH ? ORDER BY score DESC, memory_key ASC",
                (expression,),
            ).fetchall()
            maximum = max((score for _, score in matches), default=0.0)
            return {key: score / maximum for key, score in matches} if maximum > 0 else {}

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True
