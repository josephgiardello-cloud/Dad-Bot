from typing import Any
# --- ToolRegistry builder for Phase B wiring ---
from dadbot.tools.executors import ToolRegistry

def _build_and_register_tool_registry(bot: Any = None) -> ToolRegistry:
    """Instantiate and register all built-in tools in the new ToolRegistry."""
    registry = ToolRegistry()
    # Register built-in tools here as needed
    return registry
