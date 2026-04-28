"""Contract adapters unify legacy and modern memory/context representations.

These adapters prevent drift between test expectations and live implementation
by providing explicit, versioned translation layers for state contracts.

Phase 4.1: All implicit fallbacks must be declared, versioned, and emitted as
first-class events.  Silent substitution (unregistered setattr injection) is
forbidden in strict mode.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fallback contract primitives (Phase 4.1)
# ---------------------------------------------------------------------------

@dataclass
class FallbackRegistration:
    """Declares a single permissible fallback substitution."""

    name: str
    """Unique key identifying the method/attribute being substituted."""

    version: str
    """Semantic version of this fallback declaration."""

    fallback_callable: Callable[..., Any]
    """The safe substitute implementation."""

    contract_description: str
    """Human-readable reason this fallback is safe / expected."""

    substituted_signature: str
    """Textual description of what the real implementation signature should be."""


@dataclass
class FallbackEvent:
    """Structured record emitted every time a declared fallback is used."""

    fallback_source: str
    """Which component issued the fallback (e.g. 'ContextSchemaAdapter')."""

    fallback_reason: str
    """Why the original was absent or unusable."""

    contract_violation_trigger: str
    """The method/attribute that triggered this fallback lookup."""

    substituted_signature: str
    """Expected signature of the real implementation."""

    declared: bool
    """True when the fallback was found in the registry; False = undeclared."""

    version: str
    """Version taken from the FallbackRegistration (empty if undeclared)."""


class FallbackRegistry:
    """Registry of declared, versioned fallback substitutions.

    Usage
    -----
    1. At module/class initialisation, call ``register(FallbackRegistration(...))``
       for every permissible fallback.
    2. When the real implementation is missing, call ``use(name, source, reason)``
       to retrieve the safe substitute and record a ``FallbackEvent``.
    3. Call ``audit()`` at any time to inspect all emitted events.
    """

    def __init__(self) -> None:
        self._declarations: Dict[str, FallbackRegistration] = {}
        self._events: List[FallbackEvent] = []

    def register(self, registration: FallbackRegistration) -> None:
        """Declare a permissible fallback."""
        self._declarations[registration.name] = registration

    def use(
        self,
        name: str,
        *,
        source: str,
        reason: str,
        strict: bool = False,
    ) -> Callable[..., Any]:
        """Retrieve the declared fallback callable and emit a FallbackEvent.

        Parameters
        ----------
        name:    The method/attribute key.
        source:  Component requesting the fallback.
        reason:  Why the original was absent.
        strict:  When True, raises ContractViolationError for undeclared
                 fallbacks (or re-raises if strict mode env is active).

        Raises
        ------
        ContractViolationError  when ``strict=True`` and the fallback is
                                not declared in the registry.
        """
        declaration = self._declarations.get(name)
        declared = declaration is not None

        event = FallbackEvent(
            fallback_source=source,
            fallback_reason=reason,
            contract_violation_trigger=name,
            substituted_signature=declaration.substituted_signature if declaration else "unknown",
            declared=declared,
            version=declaration.version if declaration else "",
        )
        self._events.append(event)

        if not declared:
            msg = (
                f"[Phase 4.1] Undeclared fallback '{name}' requested by '{source}': {reason}. "
                "All fallbacks must be registered before use."
            )
            logger.warning(msg)
            if strict:
                raise ContractViolationError(msg)
            # Provide a transparent no-op so callers don't crash in lenient mode.
            return lambda *a, **kw: None

        logger.debug(
            "[Phase 4.1] Declared fallback used: name=%s source=%s version=%s",
            name, source, declaration.version,
        )
        return declaration.fallback_callable

    def audit(self) -> List[FallbackEvent]:
        """Return a snapshot of all emitted fallback events."""
        return list(self._events)

    def declared_names(self) -> List[str]:
        """Return names of all registered declarations."""
        return list(self._declarations.keys())


class ContractViolationError(RuntimeError):
    """Raised when a strict-mode contract rule is violated."""


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


# ---------------------------------------------------------------------------
# Module-level fallback registry — shared across all adapter instances.
# Populate once at import time so declarations survive object re-creation.
# ---------------------------------------------------------------------------

_CONTEXT_BUILDER_FALLBACK_REGISTRY = FallbackRegistry()

_CONTEXT_BUILDER_FALLBACK_REGISTRY.register(FallbackRegistration(
    name="build_core_persona_prompt",
    version="1.0.0",
    fallback_callable=lambda: "core",
    contract_description="No-op core persona prompt; safe for test stubs.",
    substituted_signature="build_core_persona_prompt(self) -> str",
))
_CONTEXT_BUILDER_FALLBACK_REGISTRY.register(FallbackRegistration(
    name="build_dynamic_profile_context",
    version="1.0.0",
    fallback_callable=lambda: "profile",
    contract_description="No-op profile context; safe for test stubs.",
    substituted_signature="build_dynamic_profile_context(self) -> str",
))
_CONTEXT_BUILDER_FALLBACK_REGISTRY.register(FallbackRegistration(
    name="build_relationship_context",
    version="1.0.0",
    fallback_callable=lambda: "relationship",
    contract_description="No-op relationship context; safe for test stubs.",
    substituted_signature="build_relationship_context(self) -> str",
))
_CONTEXT_BUILDER_FALLBACK_REGISTRY.register(FallbackRegistration(
    name="build_session_summary_context",
    version="1.0.0",
    fallback_callable=lambda: "summary",
    contract_description="No-op session summary; safe for test stubs.",
    substituted_signature="build_session_summary_context(self) -> str",
))
_CONTEXT_BUILDER_FALLBACK_REGISTRY.register(FallbackRegistration(
    name="build_memory_context",
    version="1.0.0",
    fallback_callable=lambda _user_input: "",
    contract_description="No-op memory context; safe for test stubs.",
    substituted_signature="build_memory_context(self, user_input: str) -> str",
))
_CONTEXT_BUILDER_FALLBACK_REGISTRY.register(FallbackRegistration(
    name="build_relevant_context",
    version="1.0.0",
    fallback_callable=lambda _user_input: "relevant",
    contract_description="No-op relevant context; safe for test stubs.",
    substituted_signature="build_relevant_context(self, user_input: str) -> str",
))
_CONTEXT_BUILDER_FALLBACK_REGISTRY.register(FallbackRegistration(
    name="build_cross_session_context",
    version="1.0.0",
    fallback_callable=lambda _user_input: "cross-session",
    contract_description="No-op cross-session context; safe for test stubs.",
    substituted_signature="build_cross_session_context(self, user_input: str) -> str",
))


def _strict_mode() -> bool:
    """Return True when Phase 4 strict mode is active (env var or default False)."""
    return os.environ.get("PHASE4_STRICT", "").strip() == "1"


class ContextSchemaAdapter:
    """Adapter between context builder contracts and live implementations.

    Phase 4.1: Method injection now goes through ``_CONTEXT_BUILDER_FALLBACK_REGISTRY``.
    Every injected method is a declared fallback with version, description, and a
    ``FallbackEvent`` record.  Undeclared injection raises ``ContractViolationError``
    in strict mode.
    """

    def __init__(self, bot, *, fallback_registry: FallbackRegistry | None = None):
        self.bot = bot
        self._registry = fallback_registry or _CONTEXT_BUILDER_FALLBACK_REGISTRY

    def safe_get_bot_context_builder(self, context_builder: Any) -> Optional[Any]:
        """Safely retrieve bot from context builder, handling missing attributes."""
        if context_builder is None:
            return None

        bot = getattr(context_builder, "bot", None)
        return bot

    def ensure_context_builder_methods(self, context_builder: Any) -> List[FallbackEvent]:
        """Ensure context builder has all required methods via the fallback registry.

        Returns the list of ``FallbackEvent`` records for every method that was
        absent and required a declared fallback substitute.

        Raises
        ------
        ContractViolationError
            When strict mode is active and any required method is missing but
            has no declared fallback (should never happen with a fully populated
            registry, but guards against future method additions without a
            corresponding registration).
        """
        if context_builder is None:
            return []

        required_method_names = [
            "build_core_persona_prompt",
            "build_dynamic_profile_context",
            "build_relationship_context",
            "build_session_summary_context",
            "build_memory_context",
            "build_relevant_context",
            "build_cross_session_context",
        ]

        events: List[FallbackEvent] = []
        strict = _strict_mode()
        events_before = len(self._registry.audit())

        for method_name in required_method_names:
            if not hasattr(context_builder, method_name):
                substitute = self._registry.use(
                    method_name,
                    source="ContextSchemaAdapter",
                    reason=f"'{method_name}' absent on context_builder type "
                           f"'{type(context_builder).__name__}'",
                    strict=strict,
                )
                setattr(context_builder, method_name, substitute)

        # Collect only the new events produced by this call.
        all_events = self._registry.audit()
        events = all_events[events_before:]
        return events

    def safe_build_memory_context(
        self, context_builder: Any, user_input: str, fallback: str | None = None
    ) -> str | None:
        """Safely call build_memory_context with a declared fallback on error.

        Phase 4.1: Exceptions no longer silently vanish — they are logged at
        WARNING level with a structured fallback event emitted via the registry.
        The ``fallback`` value is returned as before, but the failure is now
        traceable.
        """
        if context_builder is None or not hasattr(context_builder, "build_memory_context"):
            # Attribute absence is the declared "build_memory_context" fallback.
            self._registry.use(
                "build_memory_context",
                source="ContextSchemaAdapter.safe_build_memory_context",
                reason="context_builder is None or missing build_memory_context",
                strict=False,  # always lenient — caller controls error path via return value
            )
            return fallback

        try:
            result = context_builder.build_memory_context(str(user_input or ""))
            return result if result else fallback
        except Exception as exc:
            logger.warning(
                "[Phase 4.1] build_memory_context raised; applying declared fallback. "
                "error=%s context_builder_type=%s",
                exc, type(context_builder).__name__,
            )
            self._registry.use(
                "build_memory_context",
                source="ContextSchemaAdapter.safe_build_memory_context",
                reason=f"build_memory_context raised: {exc}",
                strict=False,
            )
            return fallback

    def fallback_audit(self) -> List[FallbackEvent]:
        """Return all fallback events emitted through this adapter's registry."""
        return self._registry.audit()


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

    def __init__(self, bot, *, strict_fallback: bool | None = None):
        self.bot = bot
        self.memory_view = MemoryViewAdapter(bot)
        self.context_schema = ContextSchemaAdapter(bot)
        self.relationship_writes = RelationshipWriteBridge(bot)
        # strict_fallback=None → defers to _strict_mode() env check
        self._strict_fallback = strict_fallback

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

    def audit_fallback_usage(self) -> List[FallbackEvent]:
        """Return all fallback events emitted across all adapters in this registry."""
        return self.context_schema.fallback_audit()

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
