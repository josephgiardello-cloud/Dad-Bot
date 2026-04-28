import json
import math
import sqlite3
import time
from contextlib import closing
from pathlib import Path


def _stable_cache_key(payload):
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)


def _normalize_tokens(tokens):
    return sorted({str(token or "").strip().lower() for token in list(tokens or []) if str(token or "").strip()})


def _stable_row_tiebreaker(row):
    return (
        str(row.get("updated_at") or ""),
        str(row.get("summary_key") or ""),
    )


class SemanticIndexBackend:
    name = "unknown"

    def ensure_storage(self):
        raise NotImplementedError

    def existing_content_hashes(self):
        raise NotImplementedError

    def delete_keys(self, keys):
        raise NotImplementedError

    def upsert_rows(self, rows):
        raise NotImplementedError

    def fetch_recent(self, limit):
        raise NotImplementedError

    def fetch_candidates(self, query_embedding, query_tokens, query_category, query_mood, limit):
        raise NotImplementedError

    def count(self):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError


class SQLiteSemanticIndex(SemanticIndexBackend):
    name = "sqlite"

    def __init__(self, bot, db_path):
        self.bot = bot
        self.db_path = Path(db_path)
        self._cache_epoch = 0
        self._candidate_cache = {}

    def _invalidate_cache(self):
        self._cache_epoch += 1
        self._candidate_cache.clear()

    def _get_cached_candidates(self, key):
        cached = self._candidate_cache.get(key)
        if cached is None:
            return None
        return json.loads(json.dumps(cached))

    def _set_cached_candidates(self, key, rows):
        self._candidate_cache[key] = json.loads(json.dumps(list(rows or [])))

    def _candidate_cache_key(self, query_embedding, query_tokens, query_category, query_mood, limit):
        embedding_key = None
        if query_embedding is not None:
            embedding_key = [round(float(value), 8) for value in list(query_embedding or [])]
        return _stable_cache_key(
            {
                "epoch": int(self._cache_epoch),
                "query_embedding": embedding_key,
                "query_tokens": _normalize_tokens(query_tokens),
                "query_category": str(query_category or ""),
                "query_mood": str(query_mood or ""),
                "limit": int(limit or 0),
            }
        )

    def with_connection(self, operation, write=False):
        last_error = None
        for attempt in range(4):
            try:
                io_lock = getattr(self.bot, "_io_lock", None)
                if io_lock is None:
                    with closing(sqlite3.connect(self.db_path, timeout=5)) as connection:
                        with connection:
                            connection.execute("PRAGMA busy_timeout = 5000")
                            if write:
                                connection.execute("PRAGMA journal_mode=WAL")
                            return operation(connection)

                with io_lock:
                    with closing(sqlite3.connect(self.db_path, timeout=5)) as connection:
                        with connection:
                            connection.execute("PRAGMA busy_timeout = 5000")
                            if write:
                                connection.execute("PRAGMA journal_mode=WAL")
                            return operation(connection)
            except sqlite3.OperationalError as exc:
                last_error = exc
                if "locked" not in str(exc).lower() or attempt == 3:
                    raise
                time.sleep(0.1 * (attempt + 1))
        raise last_error

    def ensure_storage(self):
        self.with_connection(
            lambda connection: connection.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_memories (
                    summary_key TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    category TEXT,
                    mood TEXT,
                    updated_at TEXT,
                    content_hash TEXT NOT NULL,
                    embedding_json TEXT NOT NULL
                )
                """
            ),
            write=True,
        )
        self.with_connection(
            lambda connection: connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_semantic_memories_updated_at ON semantic_memories(updated_at);
                CREATE INDEX IF NOT EXISTS idx_semantic_memories_category ON semantic_memories(category);
                CREATE INDEX IF NOT EXISTS idx_semantic_memories_mood ON semantic_memories(mood);
                CREATE INDEX IF NOT EXISTS idx_semantic_memories_category_updated_at ON semantic_memories(category, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_semantic_memories_mood_updated_at ON semantic_memories(mood, updated_at DESC);
                """
            ),
            write=True,
        )

    def existing_content_hashes(self):
        if not self.db_path.exists():
            return {}
        return self.with_connection(
            lambda connection: {
                row[0]: row[1]
                for row in connection.execute("SELECT summary_key, content_hash FROM semantic_memories")
            }
        )

    def delete_keys(self, keys):
        if not keys or not self.db_path.exists():
            return
        self.with_connection(
            lambda connection: connection.executemany(
                "DELETE FROM semantic_memories WHERE summary_key = ?",
                [(summary_key,) for summary_key in keys],
            ),
            write=True,
        )
        self._invalidate_cache()

    def upsert_rows(self, rows):
        if not rows:
            return
        self.ensure_storage()
        self.with_connection(
            lambda connection: connection.executemany(
                """
                INSERT OR REPLACE INTO semantic_memories (
                    summary_key,
                    summary,
                    category,
                    mood,
                    updated_at,
                    content_hash,
                    embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row["summary_key"],
                        row["summary"],
                        row["category"],
                        row["mood"],
                        row["updated_at"],
                        row["content_hash"],
                        row["embedding_json"],
                    )
                    for row in rows
                ],
            ),
            write=True,
        )
        self._invalidate_cache()

    def _row_dicts(self, rows):
        return [
            {
                "summary_key": row[0],
                "summary": row[1],
                "category": row[2],
                "mood": row[3],
                "updated_at": row[4],
                "embedding_json": row[5],
            }
            for row in rows
        ]

    def fetch_recent(self, limit):
        if not self.db_path.exists():
            return []
        rows = self.with_connection(
            lambda connection: connection.execute(
                """
                SELECT summary_key, summary, category, mood, updated_at, embedding_json
                FROM semantic_memories
                ORDER BY updated_at DESC, summary_key ASC
                LIMIT ?
                """,
                [limit],
            ).fetchall()
        )
        return self._row_dicts(rows)

    def _filtered_rows(self, query_tokens, query_category, query_mood, limit):
        where_clauses = []
        params = []
        normalized_tokens = _normalize_tokens(query_tokens)
        if normalized_tokens:
            where_clauses.append("(" + " OR ".join("LOWER(summary) LIKE ?" for _ in normalized_tokens) + ")")
            params.extend([f"%{token}%" for token in normalized_tokens])
        if query_category not in {None, "", "general"}:
            where_clauses.append("category = ?")
            params.append(query_category)
        if query_mood not in {None, "", "neutral"}:
            where_clauses.append("mood = ?")
            params.append(query_mood)
        if not where_clauses:
            return []
        rows = self.with_connection(
            lambda connection: connection.execute(
                f"""
                SELECT summary_key, summary, category, mood, updated_at, embedding_json
                FROM semantic_memories
                WHERE {' AND '.join(where_clauses)}
                ORDER BY updated_at DESC, summary_key ASC
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
        )
        return self._row_dicts(rows)

    @staticmethod
    def _cosine_similarity(left, right):
        if not left or not right or len(left) != len(right):
            return -1.0
        dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return -1.0
        return dot_product / (left_norm * right_norm)

    def _rank_rows_by_embedding(self, rows, query_embedding, limit):
        ranked = []
        for row in rows:
            try:
                embedding = json.loads(row.get("embedding_json") or "[]")
            except Exception:
                continue
            score = self._cosine_similarity(query_embedding, embedding)
            if score < 0:
                continue
            materialized = dict(row)
            materialized["retrieval_score"] = round(float(score), 8)
            ranked.append((materialized["retrieval_score"], materialized))
        ranked.sort(
            key=lambda item: (
                item[0],
                str(item[1].get("updated_at") or ""),
                str(item[1].get("summary_key") or ""),
            ),
            reverse=True,
        )
        return [row for _score, row in ranked[:limit]]

    def fetch_candidates(self, query_embedding, query_tokens, query_category, query_mood, limit):
        if not self.db_path.exists():
            return []
        safe_limit = max(1, int(limit or 1))
        cache_key = self._candidate_cache_key(query_embedding, query_tokens, query_category, query_mood, safe_limit)
        cached = self._get_cached_candidates(cache_key)
        if cached is not None:
            return cached

        rows = self._filtered_rows(query_tokens, query_category, query_mood, safe_limit)
        selected = []
        if rows and query_embedding is not None:
            ranked = self._rank_rows_by_embedding(rows, query_embedding, safe_limit)
            if ranked:
                selected = ranked
        if not selected and rows:
            selected = rows
        if not selected and query_embedding is not None:
            candidates = self.fetch_recent(max(safe_limit * 20, 100))
            ranked = self._rank_rows_by_embedding(candidates, query_embedding, safe_limit)
            if ranked:
                selected = ranked
        if not selected:
            selected = self.fetch_recent(safe_limit)

        # Normalize final ordering to guarantee stable replay when scores tie.
        selected = sorted(
            list(selected or []),
            key=lambda row: (
                -float(row.get("retrieval_score", -1.0)),
                str(row.get("updated_at") or ""),
                str(row.get("summary_key") or ""),
            ),
        )
        self._set_cached_candidates(cache_key, selected)
        return selected

    def count(self):
        if not self.db_path.exists():
            return 0
        row = self.with_connection(lambda connection: connection.execute("SELECT COUNT(*) FROM semantic_memories").fetchone())
        return int(row[0]) if row else 0

    def clear(self):
        if not self.db_path.exists():
            return
        try:
            self.with_connection(lambda connection: connection.execute("DELETE FROM semantic_memories"), write=True)
            self._invalidate_cache()
        except Exception:
            try:
                self.db_path.unlink()
            except Exception:
                pass


class PGVectorSemanticIndex(SemanticIndexBackend):
    name = "pgvector"

    def __init__(
        self,
        dsn,
        table="semantic_memories",
        vector_dimensions=None,
        ann_index=None,
        distance_metric="cosine",
        hnsw_m=16,
        hnsw_ef_construction=64,
        ivfflat_lists=100,
    ):
        self.dsn = dsn
        self.table = table
        self.vector_dimensions = int(vector_dimensions) if vector_dimensions else None
        self.ann_index = str(ann_index or "").strip().lower() or None
        if self.ann_index not in {None, "hnsw", "ivfflat"}:
            self.ann_index = None
        self.distance_metric = str(distance_metric or "cosine").strip().lower() or "cosine"
        if self.distance_metric not in {"cosine", "l2", "inner_product"}:
            self.distance_metric = "cosine"
        self.hnsw_m = max(4, int(hnsw_m))
        self.hnsw_ef_construction = max(8, int(hnsw_ef_construction))
        self.ivfflat_lists = max(1, int(ivfflat_lists))
        self._cache_epoch = 0
        self._candidate_cache = {}

    def _invalidate_cache(self):
        self._cache_epoch += 1
        self._candidate_cache.clear()

    def _candidate_cache_key(self, query_embedding, query_tokens, query_category, query_mood, limit):
        embedding_key = None
        if query_embedding is not None:
            embedding_key = [round(float(value), 8) for value in list(query_embedding or [])]
        return _stable_cache_key(
            {
                "epoch": int(self._cache_epoch),
                "query_embedding": embedding_key,
                "query_tokens": _normalize_tokens(query_tokens),
                "query_category": str(query_category or ""),
                "query_mood": str(query_mood or ""),
                "limit": int(limit or 0),
            }
        )

    def with_connection(self, operation):
        import psycopg
        from pgvector.psycopg import register_vector

        with psycopg.connect(self.dsn, autocommit=True, connect_timeout=5) as connection:
            register_vector(connection)
            return operation(connection)

    def ensure_storage(self):
        def operation(connection):
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table} (
                        summary_key TEXT PRIMARY KEY,
                        summary TEXT NOT NULL,
                        category TEXT,
                        mood TEXT,
                        updated_at TEXT,
                        content_hash TEXT NOT NULL,
                        embedding vector,
                        embedding_json TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.table}_updated_at ON {self.table}(updated_at)"
                )
                cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.table}_category ON {self.table}(category)"
                )
                cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.table}_mood ON {self.table}(mood)"
                )
                cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.table}_category_updated_at ON {self.table}(category, updated_at DESC)"
                )
                cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.table}_mood_updated_at ON {self.table}(mood, updated_at DESC)"
                )
                ann_index_sql = self._ann_index_sql()
                if ann_index_sql:
                    cursor.execute(ann_index_sql)

        self.with_connection(operation)

    def _distance_operator(self):
        if self.distance_metric == "l2":
            return "<->"
        if self.distance_metric == "inner_product":
            return "<#>"
        return "<=>"

    def _operator_class(self):
        if self.distance_metric == "l2":
            return "vector_l2_ops"
        if self.distance_metric == "inner_product":
            return "vector_ip_ops"
        return "vector_cosine_ops"

    def _embedding_expression(self):
        if not self.vector_dimensions:
            return "embedding"
        return f"(embedding::vector({self.vector_dimensions}))"

    def _query_vector_cast(self):
        if not self.vector_dimensions:
            return "%s::vector"
        return f"%s::vector({self.vector_dimensions})"

    def _dimension_filter_sql(self):
        if not self.vector_dimensions:
            return None
        return f"vector_dims(embedding) = {self.vector_dimensions}"

    def _ann_index_name(self):
        if not self.ann_index or not self.vector_dimensions:
            return None
        return f"idx_{self.table}_embedding_{self.ann_index}"

    def _ann_index_sql(self):
        index_name = self._ann_index_name()
        if index_name is None:
            return None

        with_clause = ""
        if self.ann_index == "hnsw":
            with_clause = f" WITH (m = {self.hnsw_m}, ef_construction = {self.hnsw_ef_construction})"
        elif self.ann_index == "ivfflat":
            with_clause = f" WITH (lists = {self.ivfflat_lists})"

        return (
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON {self.table} USING {self.ann_index} "
            f"({self._embedding_expression()} {self._operator_class()})"
            f"{with_clause} "
            f"WHERE {self._dimension_filter_sql()}"
        )

    def existing_content_hashes(self):
        self.ensure_storage()

        def operation(connection):
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT summary_key, content_hash FROM {self.table}")
                return {row[0]: row[1] for row in cursor.fetchall()}

        return self.with_connection(operation)

    def delete_keys(self, keys):
        if not keys:
            return
        self.ensure_storage()

        def operation(connection):
            with connection.cursor() as cursor:
                cursor.executemany(
                    f"DELETE FROM {self.table} WHERE summary_key = %s",
                    [(summary_key,) for summary_key in keys],
                )

        self.with_connection(operation)
        self._invalidate_cache()

    def upsert_rows(self, rows):
        if not rows:
            return
        self.ensure_storage()

        def operation(connection):
            with connection.cursor() as cursor:
                cursor.executemany(
                    f"""
                    INSERT INTO {self.table} (
                        summary_key,
                        summary,
                        category,
                        mood,
                        updated_at,
                        content_hash,
                        embedding,
                        embedding_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (summary_key) DO UPDATE SET
                        summary = EXCLUDED.summary,
                        category = EXCLUDED.category,
                        mood = EXCLUDED.mood,
                        updated_at = EXCLUDED.updated_at,
                        content_hash = EXCLUDED.content_hash,
                        embedding = EXCLUDED.embedding,
                        embedding_json = EXCLUDED.embedding_json
                    """,
                    [
                        (
                            row["summary_key"],
                            row["summary"],
                            row["category"],
                            row["mood"],
                            row["updated_at"],
                            row["content_hash"],
                            row["embedding"],
                            row["embedding_json"],
                        )
                        for row in rows
                    ],
                )

        self.with_connection(operation)
        self._invalidate_cache()

    @staticmethod
    def _row_dicts(rows):
        return [
            {
                "summary_key": row[0],
                "summary": row[1],
                "category": row[2],
                "mood": row[3],
                "updated_at": row[4],
                "embedding_json": row[5],
            }
            for row in rows
        ]

    @staticmethod
    def _vector_literal(embedding):
        if embedding is None:
            return None
        return "[" + ",".join(str(float(value)) for value in embedding) + "]"

    def fetch_recent(self, limit):
        self.ensure_storage()

        def operation(connection):
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT summary_key, summary, category, mood, updated_at, embedding_json
                    FROM {self.table}
                    ORDER BY updated_at DESC, summary_key ASC
                    LIMIT %s
                    """,
                    (limit,),
                )
                return self._row_dicts(cursor.fetchall())

        return self.with_connection(operation)

    def _filtered_rows(self, query_embedding, query_tokens, query_category, query_mood, limit):
        where_clauses = []
        params = []
        normalized_tokens = _normalize_tokens(query_tokens)
        if normalized_tokens:
            where_clauses.append("(" + " OR ".join("LOWER(summary) LIKE %s" for _ in normalized_tokens) + ")")
            params.extend([f"%{token}%" for token in normalized_tokens])
        if query_category not in {None, "", "general"}:
            where_clauses.append("category = %s")
            params.append(query_category)
        if query_mood not in {None, "", "neutral"}:
            where_clauses.append("mood = %s")
            params.append(query_mood)
        dimension_filter = self._dimension_filter_sql()
        if dimension_filter:
            where_clauses.append(dimension_filter)
        if not where_clauses:
            return []

        order_clause = "updated_at DESC"
        if query_embedding is not None:
            order_clause = (
                f"{self._embedding_expression()} {self._distance_operator()} {self._query_vector_cast()}, "
                "updated_at DESC, summary_key ASC"
            )
            params.append(self._vector_literal(query_embedding))
        else:
            order_clause = "updated_at DESC, summary_key ASC"
        params.append(limit)

        def operation(connection):
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT summary_key, summary, category, mood, updated_at, embedding_json
                    FROM {self.table}
                    WHERE {' AND '.join(where_clauses)}
                    ORDER BY {order_clause}
                    LIMIT %s
                    """,
                    params,
                )
                return self._row_dicts(cursor.fetchall())

        return self.with_connection(operation)

    def fetch_candidates(self, query_embedding, query_tokens, query_category, query_mood, limit):
        safe_limit = max(1, int(limit or 1))
        cache_key = self._candidate_cache_key(query_embedding, query_tokens, query_category, query_mood, safe_limit)
        cached = self._candidate_cache.get(cache_key)
        if cached is not None:
            return json.loads(json.dumps(cached))

        self.ensure_storage()
        rows = self._filtered_rows(query_embedding, query_tokens, query_category, query_mood, safe_limit)
        if rows:
            self._candidate_cache[cache_key] = json.loads(json.dumps(rows))
            return rows

        if query_embedding is None:
            rows = self.fetch_recent(safe_limit)
            self._candidate_cache[cache_key] = json.loads(json.dumps(rows))
            return rows

        dimension_filter = self._dimension_filter_sql()
        where_clause = f"WHERE {dimension_filter}" if dimension_filter else ""

        def operation(connection):
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT summary_key, summary, category, mood, updated_at, embedding_json
                    FROM {self.table}
                    {where_clause}
                    ORDER BY {self._embedding_expression()} {self._distance_operator()} {self._query_vector_cast()}, updated_at DESC, summary_key ASC
                    LIMIT %s
                    """,
                    (self._vector_literal(query_embedding), safe_limit),
                )
                return self._row_dicts(cursor.fetchall())

        rows = self.with_connection(operation)
        self._candidate_cache[cache_key] = json.loads(json.dumps(rows))
        return rows

    def count(self):
        self.ensure_storage()

        def operation(connection):
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM {self.table}")
                row = cursor.fetchone()
                return int(row[0]) if row else 0

        return self.with_connection(operation)

    def clear(self):
        self.ensure_storage()

        def operation(connection):
            with connection.cursor() as cursor:
                cursor.execute(f"TRUNCATE {self.table}")

        self.with_connection(operation)
        self._invalidate_cache()
