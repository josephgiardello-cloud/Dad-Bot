from __future__ import annotations

import logging
from datetime import date

from dadbot.base.memory_base import (
    GraphManagerProtocol,
    MemoryIntegrationMixin,
    MemoryLifecycleMixin,
    MemorySearchMixin,
)
from dadbot.contracts import DadBotContext, SupportsDadBotAccess
from dadbot.core.execution_boundary import MemoryWriteOwnerScope
from dadbot.core.execution_context import ensure_execution_trace_root
from dadbot.core.execution_context import get_active_core_state
from dadbot.core.core_state import CoreState
from dadbot.core.core_state import memory_projection
from dadbot.memory.lifecycle import MemoryLifecycleManager
from dadbot.memory.normalizers import MemoryNormalizer
from dadbot.memory.semantic_manager import SemanticIndexManager
from dadbot.memory.storage import MemoryStorageBackend

logger = logging.getLogger(__name__)



class MemoryManager(MemoryLifecycleMixin, MemorySearchMixin, MemoryIntegrationMixin):
    """Owns memory normalization, catalog access, and semantic-memory persistence."""

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
        self._graph_manager: GraphManagerProtocol = None
        self._semantic_manager = None
        self._normalizer = None
        self._storage = None
        self._lifecycle = None
        # Do not create sub-managers here; always defer to initialize()

    def initialize(self):
        from dadbot.memory.graph_manager import MemoryGraphManager
        from dadbot.memory.semantic_manager import SemanticIndexManager
        from dadbot.memory.normalizers import MemoryNormalizer
        from dadbot.memory.storage import MemoryStorageBackend
        from dadbot.memory.lifecycle import MemoryLifecycleManager
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
        active = get_active_core_state()
        if isinstance(active, CoreState):
            state = active
        else:
            state = self._memory_core_state_cache

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

    @property
    def memory_store(self) -> dict:
        # Compatibility projection surface; authoritative state remains CoreState.
        return self.memory_projection()

    def _set_memory_projection_cache(self, value: dict | None) -> dict:
        self._memory_projection_cache = dict(value or {})
        self._memory_projection_version = int(getattr(self._memory_core_state_cache, "version", 0) or 0)
        return dict(self._memory_projection_cache)

    def memory_core_state(self) -> CoreState:
        return self._memory_core_state_cache

    def _set_memory_core_state_cache(self, state: CoreState) -> CoreState:
        self._memory_core_state_cache = state if isinstance(state, CoreState) else CoreState()
        # Invalidate projection cache version so reads re-project from authority.
        self._memory_projection_version = None
        return self._memory_core_state_cache

    def load_memory_store(self):
        return self._storage.load_memory_store()

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

    def graph_retrieval_for_input(self, query, limit=3):
        return self._graph_manager.graph_retrieval_for_input(query, limit)

    def build_graph_summary_context(self, limit=3):
        return self._graph_manager.build_graph_summary_context(limit)

    def save_memory_catalog(self, memories):
        cleaned = self.clean_memory_entries(memories[-50:])
        self.mutate_memory_store(memories=cleaned)
        self.queue_semantic_memory_index(cleaned)
        self.bot.mark_memory_graph_dirty()
        return cleaned

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
