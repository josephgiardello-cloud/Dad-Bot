"""Legacy executor registry and runtime tool definitions."""
from typing import Any, Dict, Protocol


class ToolExecutor(Protocol):
    def execute(self, action: str, **kwargs: Any) -> Any:
        ...


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Any] = {}

    def register(self, name: str, executor: Any):
        self._tools[name] = executor

    def get(self, name: str) -> Any:
        return self._tools.get(name)

    def get_available_tools(self) -> list[dict[str, Any]]:
        # These definitions are dispatched by ToolPipelineMixin, not by this
        # legacy registry. Do not register placeholder executors here.
        definitions = {
            "reminder": {
                "type": "function",
                "function": {
                    "name": "set_reminder",
                    "description": "Create a reminder for the user.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "What to remember."},
                            "due_text": {"type": "string", "description": "When the reminder is due."},
                        },
                        "required": ["title"],
                    },
                },
            },
            "web_search": {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query."},
                        },
                        "required": ["query"],
                    },
                },
            },
        }
        return [definitions[name] for name in ("reminder", "web_search")]

    def describe_all(self) -> str:
        return ', '.join(self._tools.keys())


def build_default_tool_registry() -> ToolRegistry:
    return ToolRegistry()


tool_registry = build_default_tool_registry()

def execute_tool(tool_name: str, action: str, **kwargs: Any) -> Any:
    executor = tool_registry.get(tool_name)
    if executor is None:
        raise ValueError(f"Tool '{tool_name}' not found in registry.")
    return executor.execute(action, **kwargs)
