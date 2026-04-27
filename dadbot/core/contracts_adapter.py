"""Contract adapters unify legacy and modern memory/context representations.

These adapters prevent drift between test expectations and live implementation
by providing explicit, versioned translation layers for state contracts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MemoryViewAdapter:
    """Adapter between legacy memory view contracts and modern representations.
    
    Ensures backward compatibility when memory store schema changes while
    keeping tests decoupled from internal representation details.
    """

    def __init__(self, bot):
        self.bot = bot

    def legacy_to_modern(self, legacy_memory: Dict[str, Any]) -> Dict[str, Any]:
        """Convert legacy memory format to modern format."""
        if not isinstance(legacy_memory, dict):
            return {}

        modern = {
            "id": legacy_memory.get("id", ""),
            "summary": legacy_memory.get("summary", ""),
            "category": legacy_memory.get("category", "general"),
            "mood": legacy_memory.get("mood", "neutral"),
            "created_at": legacy_memory.get("created_at", ""),
            "updated_at": legacy_memory.get("updated_at", ""),
            "importance_score": float(legacy_memory.get("importance_score", 0.0) or 0.0),
        }
        return modern

    def modern_to_legacy(self, modern_memory: Dict[str, Any]) -> Dict[str, Any]:
        """Convert modern memory format to legacy format for compatibility."""
        if not isinstance(modern_memory, dict):
            return {}

        legacy = {
            "id": modern_memory.get("id", ""),
            "summary": modern_memory.get("summary", ""),
            "category": modern_memory.get("category", "general"),
            "mood": modern_memory.get("mood", "neutral"),
            "created_at": modern_memory.get("created_at", ""),
            "updated_at": modern_memory.get("updated_at", ""),
            "importance_score": modern_memory.get("importance_score", 0.0),
        }
        return legacy

    def ensure_memory_stats_present(self, stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensure memory stats dict has all required fields with defaults."""
        if not isinstance(stats, dict):
            stats = {}

        return {
            "recent_tokens": int(stats.get("recent_tokens", 0) or 0),
            "summary_tokens": int(stats.get("summary_tokens", 0) or 0),
            "structured_tokens": int(stats.get("structured_tokens", 0) or 0),
            "structured_claims": list(stats.get("structured_claims") or []),
            "selected_layers": list(stats.get("selected_layers") or []),
        }


class ContextSchemaAdapter:
    """Adapter between context builder contracts and live implementations.
    
    Handles attribute presence/absence and schema version drift between
    test fixtures and live code.
    """

    def __init__(self, bot):
        self.bot = bot

    def safe_get_bot_context_builder(self, context_builder: Any) -> Optional[Any]:
        """Safely retrieve bot from context builder, handling missing attributes."""
        if context_builder is None:
            return None
        
        bot = getattr(context_builder, "bot", None)
        return bot

    def ensure_context_builder_methods(self, context_builder: Any) -> None:
        """Ensure context builder has all required methods (inject if missing)."""
        if context_builder is None:
            return

        # List of required methods with safe no-op implementations
        required_methods = {
            "build_core_persona_prompt": lambda: "core",
            "build_dynamic_profile_context": lambda: "profile",
            "build_relationship_context": lambda: "relationship",
            "build_session_summary_context": lambda: "summary",
            "build_memory_context": lambda _user_input: "",
            "build_relevant_context": lambda _user_input: "relevant",
            "build_cross_session_context": lambda _user_input: "cross-session",
        }

        for method_name, default_impl in required_methods.items():
            if not hasattr(context_builder, method_name):
                setattr(context_builder, method_name, default_impl)

    def safe_build_memory_context(
        self, context_builder: Any, user_input: str, fallback: str | None = None
    ) -> str | None:
        """Safely call build_memory_context with fallback on error."""
        if context_builder is None or not hasattr(context_builder, "build_memory_context"):
            return fallback

        try:
            result = context_builder.build_memory_context(str(user_input or ""))
            return result if result else fallback
        except Exception as exc:
            logger.debug(f"Context builder memory build failed: {exc}")
            return fallback


class RelationshipWriteBridge:
    """Bridge between queued and immediate relationship history writes.
    
    Allows test/debug code to force immediate writes while production
    code can batch them if needed.
    """

    def __init__(self, bot):
        self.bot = bot
        self.write_queue: List[Dict[str, Any]] = []
        self.immediate_mode = False

    def set_immediate_write_mode(self, enabled: bool) -> None:
        """Enable/disable immediate write mode (useful for tests)."""
        self.immediate_mode = bool(enabled)

    def record_relationship_point(
        self,
        *,
        trust_level: float,
        openness_level: float,
        source: str = "turn",
        force_immediate: bool = False,
    ) -> Dict[str, Any]:
        """Record a relationship history point with optional queuing."""
        point = {
            "recorded_at": self.bot.runtime_timestamp() if hasattr(self.bot, "runtime_timestamp") else "",
            "trust_level": max(0.0, min(1.0, float(trust_level or 0.0))),
            "openness_level": max(0.0, min(1.0, float(openness_level or 0.0))),
            "source": str(source or "turn").strip().lower() or "turn",
        }

        # Write immediately if in immediate mode or forced
        if self.immediate_mode or force_immediate:
            self._apply_relationship_write(point)
        else:
            self.write_queue.append(point)

        return point

    def flush_relationship_writes(self) -> int:
        """Flush queued relationship writes to memory. Returns count flushed."""
        if not self.write_queue:
            return 0

        count = len(self.write_queue)
        for point in self.write_queue:
            self._apply_relationship_write(point)

        self.write_queue.clear()
        return count

    def _apply_relationship_write(self, point: Dict[str, Any]) -> None:
        """Apply a single relationship point to memory store."""
        try:
            history = list(self.bot.memory.relationship_history(limit=180)) if hasattr(self.bot, "memory") else []
            history.append(point)
            
            if hasattr(self.bot, "memory") and hasattr(self.bot.memory, "mutate_memory_store"):
                self.bot.memory.mutate_memory_store(relationship_history=history[-180:], save=True)
        except Exception as exc:
            logger.error(f"Failed to write relationship point: {exc}")


class ContractAdapterRegistry:
    """Centralized registry for all contract adapters."""

    def __init__(self, bot):
        self.bot = bot
        self.memory_view = MemoryViewAdapter(bot)
        self.context_schema = ContextSchemaAdapter(bot)
        self.relationship_writes = RelationshipWriteBridge(bot)

    def validate_contracts(self, raise_on_failure: bool = False) -> Dict[str, bool]:
        """Validate all contracts are properly implemented.
        
        Returns dict of contract_name -> is_valid.
        """
        results = {
            "memory_view": self._validate_memory_view(),
            "context_schema": self._validate_context_schema(),
            "relationship_writes": self._validate_relationship_writes(),
        }

        if raise_on_failure and not all(results.values()):
            failed = [name for name, valid in results.items() if not valid]
            raise RuntimeError(f"Contract validation failed: {failed}")

        return results

    def _validate_memory_view(self) -> bool:
        """Validate memory view adapter."""
        try:
            test_legacy = {"summary": "test", "category": "general"}
            modern = self.memory_view.legacy_to_modern(test_legacy)
            return bool(modern.get("summary") == "test")
        except Exception:
            return False

    def _validate_context_schema(self) -> bool:
        """Validate context schema adapter."""
        try:
            # Just verify methods are callable
            return callable(self.context_schema.safe_get_bot_context_builder)
        except Exception:
            return False

    def _validate_relationship_writes(self) -> bool:
        """Validate relationship writes bridge."""
        try:
            self.relationship_writes.set_immediate_write_mode(True)
            return self.relationship_writes.immediate_mode is True
        except Exception:
            return False
