"""Phase B: Tool Registry and Executor.

Replaces ad-hoc tool dispatch (_dispatch_builtin_tool) with a declarative,
extensible registry that enables:
- Plugin-based tool registration
- Capability-based resolution
- Determinism/side-effect classification
- Deterministic execution contracts
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from dadbot.core.runtime_types import (
    CanonicalPayload,
    ToolDeterminismClass,
    ToolExecutionStatus,
    ToolInvocation,
    ToolResult,
    ToolSideEffectClass,
    ToolSpec,
)


class ToolExecutor(Protocol):
    """Protocol for executing a tool invocation and returning a result."""

    def __call__(self, invocation: ToolInvocation) -> ToolResult:
        """Execute the tool with given invocation.
        
        Returns:
            ToolResult with status, payload, latency, replay_safe flag.
        """
        ...


class ToolRegistry:
    """Central registry for tool specs and executors.
    
    Enables:
    - Plugin-based tool registration
    - Capability-based discovery
    - Deterministic execution contracts
    - Version negotiation
    """

    def __init__(self) -> None:
        self._specs_by_name: dict[str, dict[str, ToolSpec]] = {}
        self._executors: dict[tuple[str, str], ToolExecutor] = {}

    def register(
        self,
        spec: ToolSpec,
        executor: ToolExecutor,
    ) -> None:
        """Register a tool spec and its executor.
        
        Args:
            spec: ToolSpec defining the tool's contract
            executor: Callable that executes ToolInvocation → ToolResult
            
        Raises:
            ValueError: If spec is invalid or already registered with different executor
        """
        if not spec.name or not spec.version:
            raise ValueError("ToolSpec requires non-empty name and version")

        if not callable(executor):
            raise ValueError(f"executor for {spec.name} must be callable")

        name_lower = str(spec.name).strip().lower()
        version_lower = str(spec.version).strip().lower()

        # Ensure name entry exists
        if name_lower not in self._specs_by_name:
            self._specs_by_name[name_lower] = {}

        # Version collision: update (allow re-registration)
        self._specs_by_name[name_lower][version_lower] = spec
        self._executors[(name_lower, version_lower)] = executor

    def resolve(
        self,
        name: str,
        version: str | None = None,
    ) -> tuple[ToolSpec, ToolExecutor] | None:
        """Resolve a tool spec and executor by name and optional version.
        
        If version is None, returns the highest registered version.
        If version is specified, returns exact match or None.
        
        Args:
            name: Tool name (case-insensitive)
            version: Specific version string, or None for latest
            
        Returns:
            (ToolSpec, ToolExecutor) tuple, or None if not found
        """
        name_lower = str(name).strip().lower()
        if name_lower not in self._specs_by_name:
            return None

        versions = self._specs_by_name[name_lower]
        if not versions:
            return None

        # If version specified, look for exact match
        if version is not None:
            version_lower = str(version).strip().lower()
            if version_lower not in versions:
                return None
            spec = versions[version_lower]
            executor = self._executors.get((name_lower, version_lower))
            return (spec, executor) if executor else None

        # No version specified: return latest (highest semver)
        best_version = self._resolve_best_version(versions.keys())
        if not best_version:
            return None

        spec = versions[best_version]
        executor = self._executors.get((name_lower, best_version))
        return (spec, executor) if executor else None

    @staticmethod
    def _resolve_best_version(versions: Any) -> str | None:
        """Resolve highest semver from version strings."""
        version_list = list(versions or [])
        if not version_list:
            return None
        # Simple sort: assumes semver format (x.y.z)
        # For production, use packaging.version.parse
        try:
            sorted_versions = sorted(
                version_list,
                key=lambda v: tuple(int(x) for x in str(v).split(".")[:3]),
                reverse=True,
            )
            return sorted_versions[0]
        except (ValueError, IndexError):
            # Fallback to lexical sort
            return sorted(version_list, reverse=True)[0]

    def discover(
        self,
        *,
        capability: str | None = None,
        determinism: ToolDeterminismClass | None = None,
        has_side_effects: bool | None = None,
    ) -> list[ToolSpec]:
        """Discover tools by capability or execution properties.
        
        Args:
            capability: Filter by required capability
            determinism: Filter by determinism class
            has_side_effects: Filter by side-effect presence
            
        Returns:
            List of matching ToolSpecs
        """
        results: list[ToolSpec] = []

        for versions in self._specs_by_name.values():
            for spec in versions.values():
                # Capability filter
                if capability is not None:
                    if str(capability).strip().lower() not in {c.lower() for c in spec.capabilities}:
                        continue

                # Determinism filter
                if determinism is not None:
                    if spec.determinism != determinism:
                        continue

                # Side-effects filter
                if has_side_effects is not None:
                    has_effects = spec.side_effect_class != ToolSideEffectClass.PURE
                    if has_effects != has_side_effects:
                        continue

                results.append(spec)

        return results

    def list_registered(self) -> dict[str, list[str]]:
        """List all registered tools and their versions.
        
        Returns:
            {tool_name: [version, ...]}
        """
        return {
            name: sorted(versions.keys())
            for name, versions in self._specs_by_name.items()
        }


class ToolContract:
    """Validates tool invocation against spec."""

    @staticmethod
    def validate(spec: ToolSpec, invocation: ToolInvocation) -> tuple[bool, str]:
        """Validate invocation matches spec contract.
        
        Args:
            spec: Expected tool spec
            invocation: Actual invocation
            
        Returns:
            (is_valid, error_message) tuple
        """
        if spec.name.lower() != invocation.tool_spec.name.lower():
            return False, f"tool name mismatch: expected {spec.name}, got {invocation.tool_spec.name}"

        required_perms = spec.required_permissions
        if required_perms:
            caller_perms = set()
            if invocation.caller:
                # In practice, extract from context; for now assume available
                pass
            # This is a placeholder; real validation would check caller permissions
            # against spec.required_permissions

        return True, "ok"


class ToolExecutionContext:
    """Runtime context for tool execution."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self.execution_count = 0
        self.last_result: ToolResult | None = None

    def execute(self, invocation: ToolInvocation) -> ToolResult:
        """Execute a tool invocation via registry.
        
        Args:
            invocation: Tool invocation request
            
        Returns:
            ToolResult with status, payload, effects
        """
        resolved = self.registry.resolve(
            invocation.tool_spec.name,
            version=invocation.tool_spec.version,
        )
        if not resolved:
            return ToolResult(
                tool_name=invocation.tool_spec.name,
                invocation_id=invocation.invocation_id,
                status=ToolExecutionStatus.ERROR,
                error=f"tool not registered: {invocation.tool_spec.name}",
                replay_safe=False,
            )

        spec, executor = resolved

        # Validate contract
        is_valid, validation_error = ToolContract.validate(spec, invocation)
        if not is_valid:
            return ToolResult(
                tool_name=spec.name,
                invocation_id=invocation.invocation_id,
                status=ToolExecutionStatus.ERROR,
                error=validation_error,
                replay_safe=False,
            )

        # Execute
        try:
            result = executor(invocation)
            # Mark replay_safe based on spec determinism
            if not result.replay_safe:
                result.replay_safe = spec.is_idempotent()
            self.execution_count += 1
            self.last_result = result
            return result
        except Exception as exc:
            return ToolResult(
                tool_name=spec.name,
                invocation_id=invocation.invocation_id,
                status=ToolExecutionStatus.ERROR,
                error=str(exc),
                replay_safe=False,
            )
