"""SemanticIndexManager â€” owns embedding cache, semantic index sync, and semantic memory search.
Extracted from MemoryManager to thin the god class.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import ollama

from dadbot.utils import json_dumps, json_loads
from dadbot_system.semantic_index import PGVectorSemanticIndex, SQLiteSemanticIndex

logger = logging.getLogger(__name__)


class SemanticIndexManager:
    """Owns embedding cache, semantic index persistence, and semantic memory retrieval."""

    def __init__(self, bot) -> None:
        self._bot = bot
        self._embedding_cache: dict = {}
        self._semantic_index_backend = self._build_semantic_index_backend()
        self._embedding_lock_namespace = "semantic-memory"
        self._last_embedding_lock_report: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Backend construction
    # ------------------------------------------------------------------

    def _build_semantic_index_backend(self):
        postgres_dsn = str(os.environ.get("DADBOT_POSTGRES_DSN") or "").strip()
        semantic_table = (
            str(
                os.environ.get("DADBOT_SEMANTIC_INDEX_TABLE") or "semantic_memories",
            ).strip()
            or "semantic_memories"
        )
        vector_dimensions = str(
            os.environ.get("DADBOT_SEMANTIC_VECTOR_DIM") or "",
        ).strip()
        ann_index = str(os.environ.get("DADBOT_SEMANTIC_ANN_INDEX") or "").strip().lower()
        distance_metric = str(os.environ.get("DADBOT_SEMANTIC_DISTANCE_METRIC") or "cosine").strip().lower() or "cosine"
        hnsw_m = str(os.environ.get("DADBOT_SEMANTIC_HNSW_M") or "16").strip() or "16"
        hnsw_ef_construction = str(os.environ.get("DADBOT_SEMANTIC_HNSW_EF_CONSTRUCTION") or "64").strip() or "64"
        ivfflat_lists = str(os.environ.get("DADBOT_SEMANTIC_IVFFLAT_LISTS") or "100").strip() or "100"
        if postgres_dsn:
            try:
                backend = PGVectorSemanticIndex(
                    postgres_dsn,
                    table=semantic_table,
                    vector_dimensions=int(vector_dimensions) if vector_dimensions else None,
                    ann_index=ann_index or None,
                    distance_metric=distance_metric,
                    hnsw_m=int(hnsw_m),
                    hnsw_ef_construction=int(hnsw_ef_construction),
                    ivfflat_lists=int(ivfflat_lists),
                )
                backend.ensure_storage()
                return backend
            except Exception as exc:
                logger.warning(
                    "PGVector semantic index unavailable, falling back to SQLite: %s",
                    exc,
                )
        return SQLiteSemanticIndex(self._bot, self._bot.SEMANTIC_MEMORY_DB_PATH)

    def ensure_semantic_memory_db(self):
        self._semantic_index_backend.ensure_storage()

    def clear_semantic_memory_index(self):
        self._semantic_index_backend.clear()

    def with_semantic_db(self, operation, write=False):
        if isinstance(self._semantic_index_backend, SQLiteSemanticIndex):
            return self._semantic_index_backend.with_connection(operation, write=write)
        raise RuntimeError(
            "Direct semantic DB access is only available for the SQLite fallback backend.",
        )

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    def embedding_cache_db_path(self):
        return Path(self._bot.SEMANTIC_MEMORY_DB_PATH)

    def with_embedding_cache_db(self, operation, write=False):
        db_path = self.embedding_cache_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        io_lock = getattr(self._bot, "_io_lock", None)

        def invoke():
            # sqlite3 connection context managers do not guarantee close on exit;
            # close explicitly so Windows test temp dirs can be removed reliably.
            with closing(sqlite3.connect(db_path, timeout=5)) as connection:
                with connection:
                    connection.execute("PRAGMA busy_timeout = 5000")
                    if write:
                        connection.execute("PRAGMA journal_mode=WAL")
                    return operation(connection)

        if io_lock is None:
            return invoke()
        with io_lock:
            return invoke()

    def ensure_embedding_cache_storage(self):
        self.with_embedding_cache_db(
            lambda connection: connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_embedding_cache_model_hash ON embedding_cache(model_name, text_hash);
                """,
            ),
            write=True,
        )

    def ensure_embedding_version_lock_storage(self):
        self.with_embedding_cache_db(
            lambda connection: connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS embedding_version_lock (
                    namespace TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    vector_size INTEGER NOT NULL,
                    lock_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """,
            ),
            write=True,
        )

    def embedding_version_lock(self) -> dict[str, object]:
        def _get(connection):
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS embedding_version_lock (
                    namespace TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    vector_size INTEGER NOT NULL,
                    lock_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """,
            )
            return connection.execute(
                """
                SELECT namespace, model_name, vector_size, lock_hash, updated_at
                FROM embedding_version_lock
                WHERE namespace = ?
                """,
                (self._embedding_lock_namespace,),
            ).fetchone()

        row = self.with_embedding_cache_db(_get, write=True)
        if not row:
            return {}
        return {
            "namespace": row[0],
            "model_name": row[1],
            "vector_size": int(row[2]),
            "lock_hash": row[3],
            "updated_at": row[4],
        }

    def _lock_embedding_version(
        self,
        *,
        model_name: str,
        vector_size: int,
        enforce: bool = False,
    ) -> dict[str, object]:
        self.ensure_embedding_version_lock_storage()
        normalized_model = str(model_name or "").strip().lower()
        current = {
            "namespace": self._embedding_lock_namespace,
            "model_name": normalized_model,
            "vector_size": int(vector_size),
        }
        current_hash = hashlib.sha256(json_dumps(current).encode("utf-8")).hexdigest()
        row = self.embedding_version_lock()
        initialized = not bool(row)
        drift_detected = False
        drift_reason = ""
        if row:
            if str(row.get("lock_hash") or "") != current_hash:
                drift_detected = True
                drift_reason = (
                    f"embedding_lock_drift:{row.get('model_name')}:{row.get('vector_size')}"
                    f"->{normalized_model}:{int(vector_size)}"
                )

        self.with_embedding_cache_db(
            lambda connection: connection.execute(
                """
                INSERT INTO embedding_version_lock(namespace, model_name, vector_size, lock_hash, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace) DO UPDATE SET
                    model_name = excluded.model_name,
                    vector_size = excluded.vector_size,
                    lock_hash = excluded.lock_hash,
                    updated_at = excluded.updated_at
                """,
                (
                    self._embedding_lock_namespace,
                    normalized_model,
                    int(vector_size),
                    current_hash,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            ),
            write=True,
        )

        report = {
            "initialized": initialized,
            "drift_detected": drift_detected,
            "drift_reason": drift_reason,
            "lock_hash": current_hash,
            "model_name": normalized_model,
            "vector_size": int(vector_size),
            "enforced": bool(enforce),
        }
        self._last_embedding_lock_report = dict(report)
        strict_env = str(
            os.environ.get("DADBOT_STRICT_EMBEDDING_LOCK", ""),
        ).strip().lower() in {"1", "true", "yes"}
        if drift_detected and (bool(enforce) or strict_env):
            raise RuntimeError(f"Embedding lock drift detected: {drift_reason}")
        return report

    @staticmethod
    def embedding_cache_key(text, model_name):
        normalized_model = str(model_name or "").strip().lower()
        normalized_text = str(text or "")
        return hashlib.sha256(
            f"{normalized_model}\x1f{normalized_text}".encode(),
        ).hexdigest()

    def cached_embeddings_for_texts(self, model_name, texts):
        ordered_unique_texts = [text for text in dict.fromkeys(str(text or "") for text in texts) if text]
        if not ordered_unique_texts:
            return {}

        cached = {}
        pending_keys = []
        key_to_text = {}
        for text in ordered_unique_texts:
            cache_key = self.embedding_cache_key(text, model_name)
            key_to_text[cache_key] = text
            embedding = self._embedding_cache.get(cache_key)
            if isinstance(embedding, list) and len(embedding) > 10:
                cached[text] = embedding
                continue
            pending_keys.append(cache_key)

        if not pending_keys:
            return cached

        self.ensure_embedding_cache_storage()
        placeholders = ", ".join("?" for _ in pending_keys)
        rows = self.with_embedding_cache_db(
            lambda connection: connection.execute(
                f"SELECT cache_key, embedding_json FROM embedding_cache WHERE cache_key IN ({placeholders})",
                pending_keys,
            ).fetchall(),
        )
        for cache_key, embedding_json in rows:
            try:
                embedding = json_loads(embedding_json)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(embedding, list) or len(embedding) <= 10:
                continue
            text = key_to_text.get(cache_key)
            if not text:
                continue
            cached[text] = embedding
            if len(self._embedding_cache) >= 1024:
                self._embedding_cache.pop(next(iter(self._embedding_cache)))
            self._embedding_cache[cache_key] = embedding
        return cached

    def store_cached_embeddings(self, model_name, text_to_embedding):
        rows = []
        updated_at = datetime.now().isoformat(timespec="seconds")
        for text, embedding in (text_to_embedding or {}).items():
            if not text or not isinstance(embedding, list) or len(embedding) <= 10:
                continue
            cache_key = self.embedding_cache_key(text, model_name)
            if len(self._embedding_cache) >= 1024:
                self._embedding_cache.pop(next(iter(self._embedding_cache)))
            self._embedding_cache[cache_key] = embedding
            rows.append(
                (
                    cache_key,
                    str(model_name or "").strip().lower(),
                    hashlib.sha256(str(text).encode("utf-8")).hexdigest(),
                    json_dumps(embedding),
                    updated_at,
                ),
            )
        if not rows:
            return
        self.ensure_embedding_cache_storage()
        self.with_embedding_cache_db(
            lambda connection: connection.executemany(
                """
                INSERT INTO embedding_cache (cache_key, model_name, text_hash, embedding_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    model_name = excluded.model_name,
                    text_hash = excluded.text_hash,
                    embedding_json = excluded.embedding_json,
                    updated_at = excluded.updated_at
                """,
                rows,
            ),
            write=True,
        )

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_embeddings_from_response(response):
        if isinstance(response, dict):
            if response.get("embeddings") is not None:
                return response["embeddings"]
            if response.get("embedding") is not None:
                return [response["embedding"]]
        embeddings = getattr(response, "embeddings", None)
        if embeddings is not None:
            return embeddings
        embedding = getattr(response, "embedding", None)
        if embedding is not None:
            return [embedding]
        return None

    def embed_texts(self, texts, purpose="semantic retrieval"):
        # Compatibility: tests and lightweight runtime setups may inject a
        # per-instance embedding function on the bot facade.
        bot_embed_override = self._bot.__dict__.get("embed_texts")
        override_func = getattr(bot_embed_override, "__func__", None)
        if callable(bot_embed_override) and override_func is not type(self._bot).embed_texts:
            try:
                return list(bot_embed_override(texts, purpose=purpose) or [])
            except TypeError:
                return list(bot_embed_override(texts) or [])

        if isinstance(texts, str):
            inputs = [texts] if texts else []
        else:
            inputs = [text for text in texts if text]

        if not inputs:
            return []

        ordered_inputs = [str(text) for text in inputs]
        unique_inputs = list(dict.fromkeys(ordered_inputs))

        for candidate in self._bot.embedding_model_candidates():
            cached_embeddings = self.cached_embeddings_for_texts(
                candidate,
                unique_inputs,
            )
            missing_inputs = [text for text in unique_inputs if text not in cached_embeddings]
            fresh_embeddings = {}

            if missing_inputs:
                try:
                    response = ollama.embed(model=candidate, input=missing_inputs)
                    embeddings = self._extract_embeddings_from_response(response)
                    if embeddings is None:
                        continue
                    if embeddings and isinstance(embeddings[0], (int, float)):
                        embeddings = [embeddings]
                    if len(embeddings) != len(missing_inputs):
                        continue
                    fresh_embeddings = {
                        text: embedding
                        for text, embedding in zip(missing_inputs, embeddings)
                        if embedding and isinstance(embedding, list) and len(embedding) > 10
                    }
                    if len(fresh_embeddings) != len(missing_inputs):
                        continue
                    self.store_cached_embeddings(candidate, fresh_embeddings)
                except self._bot.ollama_retryable_errors():
                    continue

            resolved = []
            for text in ordered_inputs:
                embedding = fresh_embeddings.get(text) or cached_embeddings.get(text)
                if not embedding or not isinstance(embedding, list) or len(embedding) <= 10:
                    resolved = []
                    break
                resolved.append(embedding)

            if resolved:
                self._bot.ACTIVE_EMBEDDING_MODEL = candidate
                self._lock_embedding_version(
                    model_name=candidate,
                    vector_size=len(list(resolved[0] or [])),
                    enforce=True,
                )
                return resolved

        return []

    def embed_text(self, text):
        embeddings = self.embed_texts([text])
        return embeddings[0] if embeddings else None

    # ------------------------------------------------------------------
    # Index status
    # ------------------------------------------------------------------

    def semantic_index_row_count(self):
        try:
            return self._semantic_index_backend.count()
        except Exception:
            return 0

    def semantic_memory_status(self):
        try:
            lock_row = self.embedding_version_lock()
            status_error = ""
        except Exception as exc:  # noqa: BLE001 - status surface must degrade gracefully in UI smoke/runtime sandboxes
            logger.warning("Semantic memory status fallback: %s", exc)
            lock_row = {}
            status_error = str(exc)
        return {
            "indexed_count": self.semantic_index_row_count(),
            "embedding_model": self._bot.ACTIVE_EMBEDDING_MODEL,
            "backend": self._semantic_index_backend.name,
            "ann_index": getattr(self._semantic_index_backend, "ann_index", None),
            "vector_dimensions": getattr(
                self._semantic_index_backend,
                "vector_dimensions",
                None,
            ),
            "distance_metric": getattr(
                self._semantic_index_backend,
                "distance_metric",
                None,
            ),
            "embedding_lock": lock_row,
            "embedding_lock_last_report": dict(self._last_embedding_lock_report),
            "status_error": status_error,
        }

    # ------------------------------------------------------------------
    # Index key / text helpers
    # ------------------------------------------------------------------

    def semantic_memory_key(self, memory):
        return self._bot.normalize_memory_text(memory.get("summary", ""))

    def memory_embedding_text(self, memory):
        category = str(memory.get("category", "general")).strip().lower()
        mood = self._bot.normalize_mood(memory.get("mood"))
        summary = memory.get("summary", "")
        return f"category={category}; mood={mood}; summary={summary}"

    # ------------------------------------------------------------------
    # Index queue / drain / sync
    # ------------------------------------------------------------------

    @staticmethod
    def snapshot_memory_entries(memories):
        return [dict(memory) for memory in memories]

    def queue_semantic_memory_index(self, memories):
        snapshot = self.snapshot_memory_entries(memories)
        with self._bot._semantic_index_lock:
            self._bot._pending_semantic_index_memories = snapshot
            if self._bot._semantic_index_future is None or self._bot._semantic_index_future.done():
                self._bot._semantic_index_future = self._bot.submit_background_task(
                    self._drain_semantic_index_queue,
                    task_kind="semantic-index",
                    metadata={"memory_count": len(snapshot)},
                )
            return self._bot._semantic_index_future

    def _drain_semantic_index_queue(self):
        while True:
            with self._bot._semantic_index_lock:
                snapshot = self._bot._pending_semantic_index_memories
                self._bot._pending_semantic_index_memories = None
            if snapshot is None:
                return
            try:
                self.sync_semantic_memory_index(snapshot)
            except Exception as exc:
                logger.warning("Semantic memory indexing failed: %s", exc)
            with self._bot._semantic_index_lock:
                if self._bot._pending_semantic_index_memories is None:
                    return

    def wait_for_semantic_index_idle(self, timeout=None):
        with self._bot._semantic_index_lock:
            future = self._bot._semantic_index_future
        if future is None:
            return True
        future.result(timeout=timeout)
        with self._bot._semantic_index_lock:
            return self._bot._pending_semantic_index_memories is None and self._bot._semantic_index_future is future

    def sync_semantic_memory_index(self, memories):
        if not memories:
            self.clear_semantic_memory_index()
            return

        self.ensure_semantic_memory_db()
        current_payload = []
        current_keys = set()
        for memory in memories:
            summary_key = self.semantic_memory_key(memory)
            if not summary_key:
                continue
            payload_text = self.memory_embedding_text(memory)
            current_payload.append((summary_key, payload_text, memory))
            current_keys.add(summary_key)

        existing_rows = self._semantic_index_backend.existing_content_hashes()
        stale_keys = set(existing_rows) - current_keys
        if stale_keys:
            self._semantic_index_backend.delete_keys(stale_keys)

        pending = []
        for summary_key, payload_text, memory in current_payload:
            content_hash = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
            if existing_rows.get(summary_key) == content_hash:
                continue
            pending.append((summary_key, payload_text, content_hash, memory))
        if not pending:
            return

        # Use self.embed_texts directly — avoids the round-trip through bot facade
        embeddings = self.embed_texts(
            [payload_text for _, payload_text, _, _ in pending],
            purpose="semantic memory indexing",
        )
        if len(embeddings) != len(pending):
            logger.info(
                "Semantic memory indexing skipped: embedding batch mismatch "
                "(expected %d, got %d). Will retry next cycle.",
                len(pending),
                len(embeddings),
            )
            return

        self._semantic_index_backend.upsert_rows(
            [
                {
                    "summary_key": summary_key,
                    "summary": memory.get("summary", ""),
                    "category": memory.get("category", "general"),
                    "mood": self._bot.normalize_mood(memory.get("mood")),
                    "updated_at": memory.get("updated_at", ""),
                    "content_hash": content_hash,
                    "embedding": embedding,
                    "embedding_json": json_dumps(embedding),
                }
                for (summary_key, _, content_hash, memory), embedding in zip(
                    pending,
                    embeddings,
                )
            ],
        )

    # ------------------------------------------------------------------
    # Lookup / search
    # ------------------------------------------------------------------

    def semantic_memory_lookup(self, memories):
        return {self.semantic_memory_key(memory): memory for memory in memories if self.semantic_memory_key(memory)}

    def semantic_query_context(self, query, limit):
        # Use self.embed_texts directly
        query_embeddings = self.embed_texts(query, purpose="semantic memory search")
        if not query_embeddings:
            return None
        return {
            "query_embedding": query_embeddings[0],
            "query_tokens": list(self._bot.significant_tokens(query))[
                : self._bot.runtime_config.window("semantic_query_tokens", 6)
            ],
            "query_category": self._bot.infer_memory_category(query),
            "query_mood": self._bot.normalize_mood(query),
            "candidate_limit": max(
                limit * self._bot.runtime_config.semantic_candidate_multiplier,
                self._bot.runtime_config.semantic_candidate_minimum,
            ),
        }

    @staticmethod
    def semantic_memory_filters(query_tokens, query_category, query_mood):
        where_clauses = []
        params = []
        if query_tokens:
            where_clauses.append(
                "(" + " OR ".join("LOWER(summary) LIKE ?" for _ in query_tokens) + ")",
            )
            params.extend([f"%{token.lower()}%" for token in query_tokens])
        if query_category != "general":
            where_clauses.append("category = ?")
            params.append(query_category)
        if query_mood != "neutral":
            where_clauses.append("mood = ?")
            params.append(query_mood)
        return where_clauses, params

    def recent_semantic_rows(self, candidate_limit):
        return self._semantic_index_backend.fetch_recent(candidate_limit)

    def filtered_semantic_rows(self, where_clauses, params, candidate_limit):
        if not where_clauses or not isinstance(
            self._semantic_index_backend,
            SQLiteSemanticIndex,
        ):
            return []
        return self.with_semantic_db(
            lambda connection: connection.execute(
                f"""
                SELECT summary_key, summary, category, mood, updated_at, embedding_json
                FROM semantic_memories
                WHERE {" OR ".join(where_clauses)}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                [*params, candidate_limit],
            ).fetchall(),
        )

    def semantic_candidate_rows(
        self,
        where_clauses,
        params,
        candidate_limit,
        query_embedding=None,
        query_tokens=None,
        query_category="general",
        query_mood="neutral",
    ):
        if isinstance(self._semantic_index_backend, SQLiteSemanticIndex):
            rows = self.filtered_semantic_rows(where_clauses, params, candidate_limit)
            if rows:
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
            return self.recent_semantic_rows(candidate_limit)
        return self._semantic_index_backend.fetch_candidates(
            query_embedding,
            query_tokens or [],
            query_category,
            query_mood,
            candidate_limit,
        )

    def score_semantic_rows(self, rows, current_memories, query_embedding):
        scored = []
        for row in rows:
            summary_key = row.get("summary_key")
            memory = current_memories.get(summary_key)
            if memory is None:
                continue
            try:
                embedding = json_loads(row.get("embedding_json", "[]"))
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping semantic memory row with invalid embedding JSON for key %s",
                    summary_key,
                )
                continue
            similarity = self._bot.cosine_similarity(query_embedding, embedding)
            if similarity > 0:
                scored.append((similarity, memory))
        return scored

    def semantic_memory_matches(self, query, memories, limit=3):
        if not memories:
            return []

        self.queue_semantic_memory_index(memories)

        query_context = self.semantic_query_context(query, limit)
        if query_context is None:
            return []

        current_memories = self.semantic_memory_lookup(memories)
        where_clauses, params = self.semantic_memory_filters(
            query_context["query_tokens"],
            query_context["query_category"],
            query_context["query_mood"],
        )

        try:
            rows = self.semantic_candidate_rows(
                where_clauses,
                params,
                query_context["candidate_limit"],
                query_embedding=query_context["query_embedding"],
                query_tokens=query_context["query_tokens"],
                query_category=query_context["query_category"],
                query_mood=query_context["query_mood"],
            )
        except Exception as exc:
            logger.warning("Semantic memory lookup failed for query %r: %s", query, exc)
            return []

        recent_topics = self._bot.recent_memory_topics(limit=4)
        mood_trend = self._bot.current_memory_mood_trend()
        scored = []
        for similarity, memory in self.score_semantic_rows(
            rows,
            current_memories,
            query_context["query_embedding"],
        ):
            freshness = self._bot.semantic_memory_freshness_weight(memory)
            alignment = self._bot.memory_alignment_weight(
                memory,
                query_tokens=query_context["query_tokens"],
                query_category=query_context["query_category"],
                query_mood=query_context["query_mood"],
                recent_topics=recent_topics,
                mood_trend=mood_trend,
            )
            if alignment <= 0:
                continue
            score = similarity * 5.0 * freshness * alignment
            impact_bonus = min(
                1.5,
                max(0.0, self._bot.memory_impact_score(memory)) * 0.35,
            )
            if score > 0:
                score += impact_bonus
            if score > 0.1:
                scored.append((round(score, 4), memory))

        ranked = sorted(
            scored,
            key=lambda item: (
                item[0],
                item[1].get("updated_at", ""),
                item[1].get("summary", ""),
            ),
            reverse=True,
        )
        return self._bot.select_diverse_ranked_memories(ranked, limit)

    def semantic_retrieval_signature(self, query, memories, limit=3) -> str:
        matches = list(self.semantic_memory_matches(query, memories, limit=limit) or [])
        payload = [
            {
                "summary": str(item.get("summary") or ""),
                "category": str(item.get("category") or ""),
                "mood": str(item.get("mood") or ""),
            }
            for item in matches
            if isinstance(item, dict)
        ]
        return hashlib.sha256(json_dumps(payload).encode("utf-8")).hexdigest()
