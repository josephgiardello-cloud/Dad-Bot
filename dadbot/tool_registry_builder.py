from typing import Any
from dadbot.tools.executors import ToolRegistry, build_default_tool_registry

def _build_and_register_tool_registry(bot: Any = None) -> ToolRegistry:
    """Build the canonical registry of built-in tool executors."""
    _ = bot
    return build_default_tool_registry()
