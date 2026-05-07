from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dadbot.core.external_tool_runtime import (
    DynamicToolRegistry,
    ExternalToolRuntime,
    ToolCapability,
    ToolExecutionResult,
    ToolExecutionStatus,
)


@dataclass
class ToolRuntimeService:
    """Service-owned facade for dynamic tool registry/runtime execution."""

    registry: DynamicToolRegistry
    runtime: ExternalToolRuntime

    @classmethod
    def build(cls) -> ToolRuntimeService:
        registry = DynamicToolRegistry()
        runtime = ExternalToolRuntime(registry)
        return cls(registry=registry, runtime=runtime)

    def register(self, capability: ToolCapability, handler: Any) -> None:
        self.registry.register(capability, handler)

    def execute(self, tool_name: str, payload: dict[str, Any], **kwargs: Any) -> ToolExecutionResult:
        return self.runtime.execute(tool_name, payload, **kwargs)


__all__ = [
    "DynamicToolRegistry",
    "ExternalToolRuntime",
    "ToolCapability",
    "ToolExecutionResult",
    "ToolExecutionStatus",
    "ToolRuntimeService",
]
