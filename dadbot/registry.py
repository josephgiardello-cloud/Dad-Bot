from __future__ import annotations

from functools import cache
from importlib import import_module
from typing import Any
from dadbot.models import BoundaryComplianceDeclaration


@cache
def _invariance_contract_api() -> tuple[Any, Any, Any, Any]:
    module = import_module("dadbot.core.invariance_contract")
    return (
        getattr(module, "build_boundary_compliance"),
        getattr(module, "get_evaluation_contract"),
        getattr(module, "resolve_boundary_declaration"),
        getattr(module, "serialize_boundary_declarations"),
    )

# ---------------------------------------------------------------------------
# Service descriptor + topological wiring
# ---------------------------------------------------------------------------


class ServiceDescriptor:
    """Declarative description of a wirable manager/service.

    ``depends_on`` names must refer to other descriptors in the same wiring
    batch, or to managers already resolved on the bot before this batch runs.
    The topological sort uses this information to detect cycles and validate
    that the declared order is realizable.
    """

    __slots__ = ("depends_on", "factory", "name")

    def __init__(
        self,
        name: str,
        factory: Callable[[], Any],
        depends_on: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.factory = factory
        self.depends_on = depends_on


def _topo_sort_descriptors(
    descriptors: list[ServiceDescriptor],
) -> list[ServiceDescriptor]:
    """Return *descriptors* in a valid topological order based on ``depends_on``.

    Only considers intra-batch dependencies (names in ``depends_on`` that
    refer to other descriptors in the same batch).  External dependencies
    (already-wired managers on the bot) are assumed satisfied.

    Raises ``RuntimeError`` if a cycle is detected.
    """
    batch_names: set[str] = {d.name for d in descriptors}
    index: dict[str, ServiceDescriptor] = {d.name: d for d in descriptors}

    visited: set[str] = set()
    on_stack: set[str] = set()
    order: list[ServiceDescriptor] = []

    def visit(name: str) -> None:
        if name not in batch_names or name in visited:
            return
        if name in on_stack:
            raise RuntimeError(
                f"Cycle detected in service dependency graph involving '{name}'. "
                "Check depends_on declarations in wire_bootstrap_managers / "
                "wire_runtime_managers.",
            )
        on_stack.add(name)
        for dep in index[name].depends_on:
            if dep in batch_names:
                visit(dep)
        on_stack.discard(name)
        visited.add(name)
        order.append(index[name])

    for descriptor in descriptors:
        visit(descriptor.name)

    return order


def _wire_from_descriptors(bot: Any, descriptors: list[ServiceDescriptor]) -> None:
    """Wire *descriptors* onto *bot* in a validated topological order.

    Each descriptor's factory is a zero-argument callable that closes over
    the bot reference.  The resolved instance is both returned from
    ``bot._resolve_dependency`` (for injection-registry overrides) and set
    as a named attribute on bot, matching the behaviour of the old imperative
    ``bot.name = bot._resolve_dependency(name, factory)`` pattern.
    """
    for descriptor in _topo_sort_descriptors(descriptors):
        instance = bot._resolve_dependency(descriptor.name, descriptor.factory)
        setattr(bot, descriptor.name, instance)


_registry_logger = __import__("logging").getLogger(__name__)


@cache
def _resolve_attr(path: str) -> Any:
    module_name, attr_name = path.split(":", 1)
    module = import_module(module_name)
    return getattr(module, attr_name)


def _instantiate(path: str, *args: Any, **kwargs: Any) -> Any:
    return _resolve_attr(path)(*args, **kwargs)


class ServiceRegistry:
    """Dependency-injection registry for DadBot manager instances.

    Maintains two parallel structures:
    - ``_services``     â€” name â†’ manager instance (for direct lookup by manager name)
    - ``_provider_map`` â€” public attr name â†’ manager instance (for O(1) delegation)

    Registration order matters: the *first* manager registered that exposes a
    given attribute name wins, matching the priority order of
    ``DadBot._MANAGER_DELEGATE_CHAIN``.
    """

    def __init__(self):
        self._services: dict[str, Any] = {}
        self._provider_map: dict[str, Any] = {}
        _, get_evaluation_contract, _, _ = _invariance_contract_api()
        self.evaluation_contract = get_evaluation_contract()
        self.boundary_contracts: dict[str, BoundaryComplianceDeclaration] = {}
        build_boundary_compliance, _, _, _ = _invariance_contract_api()
        self.declare_boundary_compliance(build_boundary_compliance("registry"))

    def declare_boundary_compliance(
        self,
        declaration: BoundaryComplianceDeclaration | dict[str, Any],
    ) -> BoundaryComplianceDeclaration:
        _, _, resolve_boundary_declaration, _ = _invariance_contract_api()
        model = BoundaryComplianceDeclaration.model_validate(declaration)
        resolved = resolve_boundary_declaration(model.boundary, model)
        self.boundary_contracts[resolved.boundary] = resolved
        return resolved

    def snapshot_contract_compliance(self) -> dict[str, Any]:
        _, _, _, serialize_boundary_declarations = _invariance_contract_api()
        return {
            "evaluation_contract": self.evaluation_contract.model_dump(mode="json"),
            "boundary_contracts": serialize_boundary_declarations(
                self.boundary_contracts,
            ),
        }

    def register(self, name: str, instance: Any) -> Any:
        self._services[name] = instance
        # Index every public attribute of this service into the provider map.
        # Only write if not already claimed (first-registered manager wins).
        for attr in dir(instance):
            if not attr.startswith("_") and attr not in self._provider_map:
                self._provider_map[attr] = instance
        return instance

    def get(self, name: str, *, optional: bool = False) -> Any:
        if name not in self._services:
            if optional:
                return None
            raise KeyError(f"Service '{name}' is not registered")
        return self._services[name]

    def get_provider(self, name: str) -> Any | None:
        """Return the manager instance that owns attribute *name*, or ``None``.

        O(1) â€” backed by the pre-built ``_provider_map`` index.  The caller is
        expected to do ``getattr(provider, name)`` to obtain the actual value.
        """
        return self._provider_map.get(name)

    def shutdown_all(self) -> None:
        """Gracefully shut down every registered service that exposes ``shutdown()``."""
        for svc_name, instance in list(self._services.items()):
            shutdown_fn = getattr(instance, "shutdown", None)
            if callable(shutdown_fn):
                try:
                    shutdown_fn()
                except Exception as exc:
                    _registry_logger.warning(
                        "Error shutting down service %r: %s",
                        svc_name,
                        exc,
                    )

    @classmethod
    def boot(
        cls,
        config_path: str = "config.yaml",
        *,
        bot: Any | None = None,
    ) -> ServiceRegistry:
        registry = cls()
        config = _resolve_attr("dadbot.config_schema:ConfigSchema").from_file(
            config_path,
        )
        registry.register("config", config)

        telemetry = registry.register(
            "telemetry",
            _instantiate(
                "dadbot.infrastructure.telemetry:Logger",
                "dadbot.orchestration",
            ),
        )
        registry.register(
            "storage_backend",
            _instantiate("dadbot.infrastructure.storage:FileSystemAdapter"),
        )
        registry.register(
            "model_adapter",
            _instantiate("dadbot.infrastructure.llm:OllamaModelAdapter"),
        )

        if bot is not None:
            boot_declaration = getattr(bot, "_boundary_contracts", {}).get("boot")
            if boot_declaration is not None:
                registry.declare_boundary_compliance(boot_declaration)

        if bot is None:
            telemetry.warning(
                "Booted registry without bot runtime; service wrappers disabled",
            )
            registry.register(
                "turn_graph",
                _instantiate("dadbot.core.graph:TurnGraph", registry),
            )
            return registry

        persistence_manager = _instantiate(
            "dadbot.managers.conversation_persistence:ConversationPersistenceManager",
            bot,
        )
        context_builder = _instantiate("dadbot.context:ContextBuilder", bot)
        maintenance_manager = _instantiate(
            "dadbot.managers.maintenance:MaintenanceScheduler",
            bot,
        )
        long_term_manager = _instantiate(
            "dadbot.managers.long_term:LongTermSignalsManager",
            bot,
        )

        # turn_service is already wired by wire_runtime_managers before the registry
        # is first booted (lazy, on the first graph turn). Passing it here enables
        # PersistenceService.finalize_turn to commit history, maintenance, reflection,
        # and health snapshot atomically at the end of every graph turn.
        turn_service = bot.turn_service

        # Semantic index for RAG: surfaces relevant long-term memories in ContextService
        # rather than only the most-recent window, avoiding context bloat as memory grows.
        semantic_index = None
        semantic_db_path = getattr(bot, "SEMANTIC_MEMORY_DB_PATH", None)
        if semantic_db_path is not None:
            try:
                semantic_index = _instantiate(
                    "dadbot_system.semantic_index:SQLiteSemanticIndex",
                    bot,
                    semantic_db_path,
                )
                semantic_index.ensure_storage()
            except Exception:
                pass  # Non-fatal: ContextService degrades to base context without RAG

        registry.register(
            "persistence_service",
            _instantiate(
                "dadbot.services.persistence:PersistenceService",
                persistence_manager,
                turn_service=turn_service,
            ),
        )
        post_commit_worker = getattr(bot, "_post_commit_worker", None)
        if post_commit_worker is None:
            post_commit_worker = _instantiate(
                "dadbot.services.post_commit_worker:PostCommitWorker",
                bot,
            )
            bot._post_commit_worker = post_commit_worker
        registry.register("post_commit_worker", post_commit_worker)
        registry.register(
            "context_service",
            _instantiate(
                "dadbot.services.context_service:ContextService",
                context_builder,
                bot.memory_manager,
                semantic_index=semantic_index,
            ),
        )
        registry.register(
            "runtime_service",
            _instantiate(
                "dadbot.services.runtime_service:RuntimeService",
                bot,
                registry.get("model_adapter"),
            ),
        )
        registry.register(
            "maintenance_service",
            _instantiate(
                "dadbot.services.maintenance_service:MaintenanceService",
                maintenance_manager,
                long_term_manager,
            ),
        )
        registry.register(
            "agent_service",
            _instantiate("dadbot.services.agent_service:AgentService", bot),
        )
        registry.register(
            "safety_service",
            _instantiate("dadbot.services.safety_service:SafetyService", bot),
        )

        # Canonical aliases used by DadBotOrchestrator node lookup
        registry.register("health", registry.get("maintenance_service"))
        registry.register("memory", registry.get("context_service"))
        registry.register("llm", registry.get("agent_service"))
        registry.register("safety", registry.get("safety_service"))
        registry.register("storage", registry.get("persistence_service"))
        reflection_service = getattr(
            bot,
            "internal_state_manager",
            None,
        ) or registry.get("persistence_service")
        registry.register("reflection", reflection_service)
        registry.register(
            "turn_graph",
            _instantiate("dadbot.core.graph:TurnGraph", registry),
        )

        telemetry.info(
            "ServiceRegistry boot complete",
            services=list(registry._services.keys()),
        )
        return registry


def boot_registry(
    config_path: str = "config.yaml",
    *,
    bot: Any | None = None,
) -> ServiceRegistry:
    return ServiceRegistry.boot(config_path=config_path, bot=bot)


def wire_bootstrap_managers(bot: Any) -> None:
    """Attach managers required before profile/memory hydration."""
    descriptors: list[ServiceDescriptor] = [
        ServiceDescriptor(
            "runtime_storage",
            lambda: _instantiate(
                "dadbot.managers.runtime_storage:RuntimeStorageManager",
                bot.bot_context,
            ),
        ),
        ServiceDescriptor(
            "profile_runtime",
            lambda: _instantiate(
                "dadbot.managers.profile_runtime:ProfileRuntimeManager",
                bot,
            ),
            depends_on=("runtime_storage",),
        ),
        ServiceDescriptor(
            "memory_manager",
            lambda: _instantiate(
                "dadbot.memory.manager:MemoryManager",
                bot.bot_context,
            ),
        ),
    ]
    _wire_from_descriptors(bot, descriptors)


def wire_runtime_managers(bot: Any) -> None:
    """Attach all turn/runtime managers after profile/memory hydration.

    The first descriptor batch covers all managers with computable factories.
    ``prompt_composer`` and ``reply_generation`` are pure attribute aliases
    resolved after their respective owners are wired.  The runtime-state
    bundle is resolved last because it has side-effects (setting private
    attrs on bot) that ``runtime_state_manager`` depends on.
    """
    descriptors: list[ServiceDescriptor] = [
        # ── signal / memory helpers ─────────────────────────────────────────
        ServiceDescriptor(
            "long_term_signals",
            lambda: _instantiate(
                "dadbot.managers.long_term:LongTermSignalsManager",
                bot,
            ),
        ),
        ServiceDescriptor(
            "memory_query",
            lambda: _instantiate(
                "dadbot.managers.memory_query:MemoryQueryManager",
                bot.bot_context,
            ),
        ),
        ServiceDescriptor(
            "memory_commands",
            lambda: _instantiate(
                "dadbot.managers.memory_commands:MemoryCommandManager",
                bot.bot_context,
            ),
        ),
        ServiceDescriptor(
            "memory_coordinator",
            lambda: _instantiate(
                "dadbot.managers.memory_coordination:MemoryCoordinator",
                bot,
            ),
        ),
        # ── safety / profile / context ──────────────────────────────────────
        ServiceDescriptor(
            "safety_support",
            lambda: _instantiate("dadbot.managers.safety:SafetySupportManager", bot),
        ),
        ServiceDescriptor(
            "profile_context",
            lambda: _instantiate("dadbot.profile:ProfileContextManager", bot),
        ),
        ServiceDescriptor(
            "context_builder",
            lambda: _instantiate("dadbot.context:ContextBuilder", bot),
        ),
        ServiceDescriptor(
            "tone_context",
            lambda: _instantiate("dadbot.tone:ToneContextBuilder", bot.bot_context),
        ),
        ServiceDescriptor(
            "personality_service",
            lambda: _instantiate(
                "dadbot.managers.personality_service:PersonalityServiceManager",
                bot,
            ),
        ),
        ServiceDescriptor(
            "mood_manager",
            lambda: _instantiate("dadbot.mood:MoodManager", bot),
        ),
        ServiceDescriptor(
            "relationship_manager",
            lambda: _instantiate(
                "dadbot.relationship:RelationshipManager",
                bot.bot_context,
            ),
        ),
        ServiceDescriptor(
            "internal_state_manager",
            lambda: _instantiate(
                "dadbot.managers.internal_state:InternalStateManager",
                bot,
            ),
        ),
        # ── prompt / multimodal / turn ──────────────────────────────────────
        ServiceDescriptor(
            "prompt_assembly",
            lambda: _instantiate(
                "dadbot.managers.prompt_assembly:PromptAssemblyManager",
                bot.bot_context,
            ),
        ),
        ServiceDescriptor(
            "multimodal_handler",
            lambda: _instantiate(
                "dadbot.managers.multimodal:MultimodalManager",
                bot.bot_context,
            ),
        ),
        ServiceDescriptor(
            "turn_service",
            lambda: _instantiate(
                "dadbot.services.turn_service:TurnService",
                bot.bot_context,
            ),
        ),
        # ── reply pipeline ──────────────────────────────────────────────────
        ServiceDescriptor(
            "reply_supervisor",
            lambda: _instantiate(
                "dadbot.managers.reply_supervisor:ReplySupervisorManager",
                bot.bot_context,
            ),
        ),
        ServiceDescriptor(
            "reply_finalization",
            lambda: _instantiate(
                "dadbot.managers.reply_finalization:ReplyFinalizationManager",
                bot,
            ),
        ),
        # ── conversation / orchestration / scheduling ───────────────────────
        ServiceDescriptor(
            "conversation_persistence",
            lambda: _instantiate(
                "dadbot.managers.conversation_persistence:ConversationPersistenceManager",
                bot,
            ),
        ),
        ServiceDescriptor(
            "runtime_orchestration",
            lambda: _instantiate(
                "dadbot.managers.runtime_orchestration:RuntimeOrchestrationManager",
                bot.bot_context,
            ),
        ),
        ServiceDescriptor(
            "status_reporting",
            lambda: _instantiate(
                "dadbot.managers.status_reporting:StatusReportingManager",
                bot.bot_context,
            ),
        ),
        ServiceDescriptor(
            "maintenance_scheduler",
            lambda: _instantiate(
                "dadbot.managers.maintenance:MaintenanceScheduler",
                bot,
            ),
        ),
        # ── interface / tools ───────────────────────────────────────────────
        ServiceDescriptor(
            "runtime_interface",
            lambda: _instantiate(
                "dadbot.managers.runtime_interface:RuntimeInterfaceManager",
                bot.bot_context,
            ),
        ),
        ServiceDescriptor(
            "tool_registry",
            lambda: _instantiate("dadbot.agentic:ToolRegistry", bot),
        ),
        ServiceDescriptor(
            "agentic_handler",
            lambda: _instantiate(
                "dadbot.agentic:AgenticHandler",
                bot,
                bot.tool_registry,
            ),
            depends_on=("tool_registry",),
        ),
        # ── peripheral managers ─────────────────────────────────────────────
        ServiceDescriptor(
            "health_manager",
            lambda: _instantiate("dadbot.managers.health:RuntimeHealthManager", bot),
        ),
        ServiceDescriptor(
            "tts_manager",
            lambda: _instantiate("dadbot.managers.tts:TTSManager", bot),
        ),
        ServiceDescriptor(
            "avatar_manager",
            lambda: _instantiate("dadbot.managers.avatar:AvatarManager", bot),
        ),
        ServiceDescriptor(
            "calendar_manager",
            lambda: _instantiate("dadbot.managers.calendar:CalendarManager", bot),
        ),
        ServiceDescriptor(
            "email_manager",
            lambda: _instantiate("dadbot.managers.email:EmailManager", bot),
        ),
        ServiceDescriptor(
            "model_runtime",
            lambda: _instantiate(
                "dadbot.managers.runtime_model:RuntimeModelManager",
                bot.bot_context,
            ),
        ),
        ServiceDescriptor(
            "runtime_client",
            lambda: _instantiate(
                "dadbot.managers.runtime_client:RuntimeClientManager",
                bot.bot_context,
            ),
        ),
        ServiceDescriptor(
            "session_summary_manager",
            lambda: _instantiate(
                "dadbot.managers.session_summary:SessionSummaryManager",
                bot.bot_context,
            ),
        ),
    ]
    _wire_from_descriptors(bot, descriptors)

    # ── post-wire aliases (sub-attribute reads, not independent managers) ──
    # prompt_composer is a pure alias for prompt_assembly.
    bot.prompt_composer = bot.prompt_assembly
    # reply_generation is a sub-attribute of turn_service.
    bot.reply_generation = bot.turn_service.reply_generation

    # ── runtime-state bundle (side-effect step required before runtime_state_manager) ──
    # _resolve_runtime_state_bundle populates private bot attrs that
    # RuntimeStateManager reads at construction time; it cannot be a descriptor factory.
    runtime_state_bundle = bot._resolve_runtime_state_bundle()
    bot._runtime_state_store = runtime_state_bundle.get("store")
    bot._runtime_event_bus = runtime_state_bundle.get("event_bus")
    bot._runtime_state = runtime_state_bundle["container"]
    bot.runtime_state_manager = bot._resolve_dependency(
        "runtime_state_manager",
        lambda: _instantiate(
            "dadbot.state:RuntimeStateManager",
            bot,
            bot._runtime_state,
        ),
    )


_EXTRA_MANAGER_ATTRS = ("internal_state_manager",)


def build_manager_registry(bot: Any) -> ServiceRegistry:
    """Return a ``ServiceRegistry`` populated with all wired bot manager instances.

    Walks ``DadBot._MANAGER_DELEGATE_CHAIN`` (plus a short list of extra manager
    attributes not in the chain) and registers each instance by name.  Uses
    ``object.__getattribute__`` so that the lookup never recurses through
    ``DadBot.__getattr__``.

    Call this after both ``wire_bootstrap_managers`` and ``wire_runtime_managers``
    have run so that every manager is present on the bot.
    """
    registry = ServiceRegistry()
    boot_declaration = getattr(bot, "_boundary_contracts", {}).get("boot")
    if boot_declaration is not None:
        registry.declare_boundary_compliance(boot_declaration)
    chain = getattr(type(bot), "_MANAGER_DELEGATE_CHAIN", ())
    for manager_name in (*chain, *_EXTRA_MANAGER_ATTRS):
        try:
            manager = object.__getattribute__(bot, manager_name)
        except AttributeError:
            continue
        if manager is not None:
            registry.register(manager_name, manager)
    return registry


def wire_bot_managers(bot: Any) -> None:
    """Compatibility helper for callers that want full manager wiring in one call."""
    wire_bootstrap_managers(bot)
    wire_runtime_managers(bot)


class DadBotServiceContainer:
    """Structured runtime container for DadBot managers and orchestration services.

    This wraps the existing wiring helpers so the facade can depend on a single
    container object without forcing a broad manager-construction rewrite.
    """

    def __init__(self, bot: Any):
        self.bot = bot
        self.registry = ServiceRegistry()
        self._turn_orchestrator = None

    def wire_bootstrap(self) -> None:
        wire_bootstrap_managers(self.bot)
        self.refresh_registry()

    def wire_runtime(self) -> None:
        wire_runtime_managers(self.bot)
        self.refresh_registry()

    def refresh_registry(self) -> ServiceRegistry:
        previous_declarations = dict(
            getattr(self.registry, "boundary_contracts", {}) or {},
        )
        self.registry = build_manager_registry(self.bot)
        for boundary, declaration in previous_declarations.items():
            if boundary != "registry":
                self.registry.declare_boundary_compliance(declaration)
        for name, instance in self.registry._services.items():
            setattr(self, name, instance)
        return self.registry

    def get_provider(self, name: str) -> Any | None:
        return self.registry.get_provider(name)

    def get(self, name: str, *, optional: bool = False) -> Any:
        return self.registry.get(name, optional=optional)

    @property
    def turn_orchestrator(self):
        if self._turn_orchestrator is None:
            orchestrator_cls = _resolve_attr(
                "dadbot.core.orchestrator:DadBotOrchestrator",
            )
            self._turn_orchestrator = orchestrator_cls(
                config_path=self.bot.config.turn_graph_config_path,
                bot=self.bot,
                strict=bool(getattr(self.bot.config, "strict_graph_mode", True)),
            )
            self.bot._turn_orchestrator = self._turn_orchestrator
        return self._turn_orchestrator

    @property
    def service_registry(self) -> ServiceRegistry:
        return self.registry

    def validate_facade(self, *, smoke: bool = False) -> None:
        """Validate DadBot facade wiring using the extracted validator."""
        _resolve_attr("dadbot.core.facade_validator:validate_dadbot_facade")(
            self.bot,
            smoke=smoke,
        )
