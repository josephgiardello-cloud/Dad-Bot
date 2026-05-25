"""
Unified tool executor interface for DadBot tools (reminders, calendar, web search, etc.).
Integrates with Phase B tool registry for reliable action execution.
"""
from typing import Any, Dict, Protocol

class ToolExecutor(Protocol):
    def execute(self, action: str, **kwargs: Any) -> Any:
        ...

class ReminderExecutor:
    def execute(self, action: str, **kwargs: Any) -> Any:
        return f"Reminder executed: {action} with {kwargs}"

class CalendarExecutor:
    def execute(self, action: str, **kwargs: Any) -> Any:
        return f"Calendar action: {action} with {kwargs}"

class WebSearchExecutor:
    def execute(self, action: str, **kwargs: Any) -> Any:
        return f"Web search: {action} with {kwargs}"

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Any] = {}
    def register(self, name: str, executor: Any):
        self._tools[name] = executor
    def get(self, name: str) -> Any:
        return self._tools.get(name)
    def describe_all(self) -> str:
        return ', '.join(self._tools.keys())

# Canonical Phase B tool registry instance
tool_registry = ToolRegistry()
tool_registry.register('reminder', ReminderExecutor())
tool_registry.register('calendar', CalendarExecutor())
tool_registry.register('web_search', WebSearchExecutor())

def execute_tool(tool_name: str, action: str, **kwargs: Any) -> Any:
    executor = tool_registry.get(tool_name)
    if executor is None:
        raise ValueError(f"Tool '{tool_name}' not found in registry.")
    return executor.execute(action, **kwargs)
