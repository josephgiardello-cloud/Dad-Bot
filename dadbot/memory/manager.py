from __future__ import annotations

import logging
from datetime import date
from uuid import uuid4

from dadbot.base.memory_base import (
    GraphManagerProtocol,
    MemoryIntegrationMixin,
    MemoryLifecycleMixin,
    MemorySearchMixin,
)
from dadbot.contracts import DadBotContext, SupportsDadBotAccess
from dadbot.core.execution_boundary import MemoryWriteOwnerScope
from dadbot.core.execution_context import ensure_execution_trace_root
from dadbot.core.execution_context import get_active_core_state, replace_active_core_state
from dadbot.core.core_state import CoreState
from dadbot.core.core_state import memory_projection
from dadbot.memory.lifecycle import MemoryLifecycleManager
from dadbot.memory.normalizers import MemoryNormalizer
from dadbot.memory.semantic_manager import SemanticIndexManager
from dadbot.memory.storage import MemoryStorageBackend

logger = logging.getLogger(__name__)



class MemoryManager(MemoryLifecycleMixin, MemorySearchMixin, MemoryIntegrationMixin):
    """Owns memory normalization, catalog access, and semantic-memory persistence."""

    def add_reminder(
        self,
        title: str,
        due_text: str = "",
        *,
        idempotency_key: str = "",
    ) -> dict | None:
        normalized_title = str(title or "").strip()
        if not normalized_title:
            return None

        normalized_idempotency_key = str(idempotency_key or "").strip()
        memory_store = getattr(self.bot, "MEMORY_STORE", {})
        internal_state = dict(memory_store.get("internal_state") or {}) if isinstance(memory_store, dict) else {}
        idempotency_records = dict(internal_state.get("tool_idempotency") or {})
        if normalized_idempotency_key:
            existing_id = str(idempotency_records.get(normalized_idempotency_key) or "")
            existing = next(
                (
                    dict(item)
                    for item in self.reminder_catalog(include_done=True)
                    if str(item.get("id") or "") == existing_id
                ),
                None,
            )
            if existing is not None:
                return existing
            idempotency_records.pop(normalized_idempotency_key, None)

        timestamp = self.bot.runtime_timestamp()
        reminder = {
            "id": uuid4().hex[:12],
            "title": normalized_title,
            "due_text": str(due_text or "").strip(),
            "status": "open",
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        reminders = self.reminder_catalog(include_done=True)
        reminders.append(reminder)
        if normalized_idempotency_key:
            idempotency_records[normalized_idempotency_key] = reminder["id"]
            if len(idempotency_records) > 256:
                for stale_key in list(idempotency_records)[: len(idempotency_records) - 256]:
                    idempotency_records.pop(stale_key, None)
            internal_state["tool_idempotency"] = idempotency_records
            updated_store = self.mutate_memory_store(reminders=reminders, internal_state=internal_state)
        else:
            updated_store = self.mutate_memory_store(reminders=reminders)
        return next(
            (
                dict(item)
                for item in list(updated_store.get("reminders") or [])
                if str(item.get("id") or "") == reminder["id"]
            ),
            None,
        )

    def store(self, key: str, value: object) -> None:
        """Store a value in the memory store under the given key."""
        self.mutate_memory_store(**{key: value})

    def delete(self, key: str) -> None:
        """Delete a value from the memory store by key."""
        if hasattr(self._storage, "delete"):
            self._storage.delete(key)
        else:
            # Fallback: set the key to None (legacy behavior)
            self.mutate_memory_store(**{key: None})

    def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
        # Defer all property access on bot until initialize()
        self.context = None
        self.bot = bot if bot is not None else None
        self._memory_projection_cache: dict = {}
        self._memory_core_state_cache: CoreState = CoreState()
        self._memory_projection_version: int | None = None
        self.cache_invalidation_revision = 0
        self._graph_manager: GraphManagerProtocol = None
        self._semantic_manager = None
        self._normalizer = None
        self._storage = None
        self._lifecycle = None
        # Do not create sub-managers here; always defer to initialize()

    def initialize(self):
        from dadbot.memory.graph_manager import MemoryGraphManager
        from dadbot.memory.semantic_manager import SemanticIndexManager
        from dadbot.contracts import DadBotContext

        self.context = DadBotContext.from_runtime(self.bot)
        self.bot = self.context.bot
        if self.bot is not None:
            self._graph_manager = MemoryGraphManager(self.bot, self)
            if hasattr(self._graph_manager, 'initialize'):
                self._graph_manager.initialize()
            self._semantic_manager = SemanticIndexManager(self.bot)
            self._normalizer = MemoryNormalizer(self.bot)
            self._storage = MemoryStorageBackend(self.bot, self)
            self._lifecycle = MemoryLifecycleManager(self.bot, self)

    def memory_projection(self) -> dict:
        if self._storage is not None:
            self._storage._authorize_tenant_memory()
        state = self._current_memory_core_state()

        if isinstance(state, CoreState):
            version = int(getattr(state, "version", 0) or 0)
            if self._memory_projection_version == version and self._memory_projection_cache:
                return dict(self._memory_projection_cache or {})
            projected = memory_projection(
                state,
                defaults=self.bot.default_memory_store(),
            )
            normalized = self._normalizer.normalize_memory_store(projected)
            self._memory_core_state_cache = state
            self._memory_projection_cache = dict(normalized or {})
            self._memory_projection_version = version
            return dict(self._memory_projection_cache or {})

        return dict(self._memory_projection_cache or {})

    def _current_memory_core_state(self) -> CoreState:
        active = get_active_core_state()
        cached = self._memory_core_state_cache
        if isinstance(active, CoreState):
            if isinstance(cached, CoreState) and cached.version > active.version:
                replace_active_core_state(cached)
                return cached
            return active
        return cached

    @property
    def memory_store(self) -> dict:
        # Compatibility projection surface; authoritative state remains CoreState.
        return self.memory_projection()

    def _set_memory_projection_cache(self, value: dict | None) -> dict:
        self._memory_projection_cache = dict(value or {})
        self._memory_projection_version = int(getattr(self._memory_core_state_cache, "version", 0) or 0)
        return dict(self._memory_projection_cache)

    def memory_core_state(self) -> CoreState:
        if self._storage is not None:
            self._storage._authorize_tenant_memory()
        return self._memory_core_state_cache

    def _set_memory_core_state_cache(self, state: CoreState) -> CoreState:
        self._memory_core_state_cache = state if isinstance(state, CoreState) else CoreState()
        # Invalidate projection cache version so reads re-project from authority.
        self._memory_projection_version = None
        return self._memory_core_state_cache

    def load_memory_store(self, *, allow_tenant_mirror_fallback=True):
        return self._storage.load_memory_store(allow_tenant_mirror_fallback=allow_tenant_mirror_fallback)

    def should_do_daily_checkin(self) -> bool:
        last_date = str(self.memory_projection().get("last_mood_updated_at") or "").strip()
        return last_date != date.today().isoformat()

    def save_mood_state(self, mood):
        normalized_mood = self.bot.normalize_mood(mood)
        today_stamp = date.today().isoformat()
        recent = self.recent_mood_history()
        recent.append({"mood": normalized_mood, "date": today_stamp})
        self.mutate_memory_store(
            last_mood=normalized_mood,
            last_mood_updated_at=today_stamp,
            recent_moods=recent[-12:],
        )

    def consolidate_turn_outcome(self, turn_context=None, result=None):
        """Compatibility hook for SaveNode-owned post-turn memory consolidation."""
        consolidate = getattr(self.bot, "consolidate_memories", None)
        if not callable(consolidate):
            return {"consolidated_count": 0}
        try:
            consolidated = consolidate() or []
            return {"consolidated_count": len(consolidated)}
        except Exception as exc:
            logger.warning(
                "MemoryManager.consolidate_turn_outcome failed (non-fatal): %s",
                exc,
            )
            return {"consolidated_count": 0, "error": str(exc)}

    # ------------------------------------------------------------------ Sub-managers

    @property
    def graph_manager(self) -> GraphManagerProtocol:
        return self._graph_manager

    @property
    def _graph_store_backend(self):
        # Backward-compatible facade for tests/integration code.
        return self._graph_manager._graph_store_backend

    @property
    def _graph_prompt_compressor(self):
        # Backward-compatible facade for tests/integration code.
        return self._graph_manager._graph_prompt_compressor

    @property
    def semantic(self) -> SemanticIndexManager:
        return self._semantic_manager

    def close(self):
        # Best-effort shutdown hook used by DadBot.shutdown.
        self.wait_for_semantic_index_idle(timeout=5)

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
        return self._semantic_manager.semantic_candidate_rows(
            where_clauses,
            params,
            candidate_limit,
            query_embedding=query_embedding,
            query_tokens=query_tokens,
            query_category=query_category,
            query_mood=query_mood,
        )

    def catalog_semantic_rows(self, current_keys):
        return self._semantic_manager.catalog_semantic_rows(current_keys)

    def graph_retrieval_for_input(self, query, limit=3):
        return self._graph_manager.graph_retrieval_for_input(query, limit)

    def build_graph_summary_context(self, limit=3):
        return self._graph_manager.build_graph_summary_context(limit)

    def save_memory_catalog(self, memories):
        cleaned = []
        for memory in memories:
            normalized = self._normalizer.normalize_persisted_memory_entry(memory)
            if normalized is None:
                raise ValueError("Cannot save an invalid memory entry")
            cleaned.append(normalized)
        self.mutate_memory_store(memories=cleaned)
        self.queue_semantic_memory_index(cleaned)
        self.bot.mark_memory_graph_dirty()
        return cleaned

    def restore_archived_memory(self, archive_id: str) -> dict:
        restored = []

        def restore(store):
            entries = list(store.get("archived_memories") or [])
            matches = [entry for entry in entries if entry.get("archive_id") == archive_id]
            if len(matches) != 1:
                raise ValueError("Archived memory not found or archive ID is ambiguous")
            restored.append(dict(matches[0]["memory"]))
            store["archived_memories"] = [entry for entry in entries if entry.get("archive_id") != archive_id]
            store["memories"] = [*store.get("memories", []), restored[0]]

        self.mutate_memory_store(mutator=restore)
        return restored[0]

    def delete_archived_memory(self, archive_id: str) -> None:
        def delete(store):
            entries = list(store.get("archived_memories") or [])
            matches = [entry for entry in entries if entry.get("archive_id") == archive_id]
            if len(matches) != 1:
                raise ValueError("Archived memory not found or archive ID is ambiguous")
            store["archived_memories"] = [entry for entry in entries if entry.get("archive_id") != archive_id]

        self.mutate_memory_store(mutator=delete)

    def mutate_memory_store(self, mutator=None, normalize=True, save=True, **changes):
        with (
            ensure_execution_trace_root(
                operation="memory_write",
                prompt="[memory-manager-mutate]",
                metadata={"source": "MemoryManager.mutate_memory_store"},
                required=True,
            ),
            MemoryWriteOwnerScope.bind("MemoryManager"),
        ):
            return self._storage.mutate_memory_store(
                mutator=mutator,
                normalize=normalize,
                save=save,
                owner="MemoryManager",
                **changes,
            )


__all__ = ["MemoryManager"]
