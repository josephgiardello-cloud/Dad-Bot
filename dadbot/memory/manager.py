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
from dadbot.memory.lifecycle import MemoryLifecycleManager
from dadbot.memory.migration import CURRENT_SCHEMA_VERSION
from dadbot.memory.normalizers import MemoryNormalizer
from dadbot.memory.semantic_manager import SemanticIndexManager
from dadbot.memory.storage import MemoryStorageBackend

logger = logging.getLogger(__name__)


class MemoryManager(MemoryLifecycleMixin, MemorySearchMixin, MemoryIntegrationMixin):
    """Owns memory normalization, catalog access, and semantic-memory persistence."""

    def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
        from dadbot.memory.graph_manager import MemoryGraphManager

        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot
        self._memory_store = {}
        self._graph_manager: GraphManagerProtocol = MemoryGraphManager(self.bot, self)
        self._semantic_manager = SemanticIndexManager(self.bot)
        self._normalizer = MemoryNormalizer(self.bot)  # data-shape policy
        self._storage = MemoryStorageBackend(self.bot, self)  # persistence I/O
        self._lifecycle = MemoryLifecycleManager(self.bot, self)  # catalog access

    @property
    def memory_store(self):
        return self._memory_store

    @memory_store.setter
    def memory_store(self, value):
        self._memory_store = value if isinstance(value, dict) else {}

    def initialize_memory_store_defaults(self):
        if "schema_version" not in self._memory_store:
            self._memory_store["schema_version"] = CURRENT_SCHEMA_VERSION
        if "last_scheduled_proactive_at" not in self._memory_store:
            self._memory_store["last_scheduled_proactive_at"] = None
        if "last_memory_compaction_at" not in self._memory_store:
            self._memory_store["last_memory_compaction_at"] = None
        if "last_memory_compaction_summary" not in self._memory_store:
            self._memory_store["last_memory_compaction_summary"] = ""
        if "mcp_local_store" not in self._memory_store:
            self._memory_store["mcp_local_store"] = {}
        if "narrative_memories" not in self._memory_store:
            self._memory_store["narrative_memories"] = []
        if "heritage_cross_links" not in self._memory_store:
            self._memory_store["heritage_cross_links"] = []
        if "advice_audits" not in self._memory_store:
            self._memory_store["advice_audits"] = []
        if "environmental_cues_history" not in self._memory_store:
            self._memory_store["environmental_cues_history"] = []
        if "longitudinal_insights" not in self._memory_store:
            self._memory_store["longitudinal_insights"] = []
        return self._memory_store

    def load_memory_store(self):
        self.memory_store = self._storage.load_memory_store()
        self.initialize_memory_store_defaults()
        return self.memory_store

    def should_do_daily_checkin(self) -> bool:
        last_date = str(self.memory_store.get("last_mood_updated_at") or "").strip()
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
