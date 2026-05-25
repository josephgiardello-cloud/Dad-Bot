from typing import Any, Dict, Callable

class Tool:
    """
    Base class for agent tools. Each tool must implement the `run` method.
    """
    name: str
    description: str
    parameters: Dict[str, Any]

    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters

    def run(self, **kwargs) -> Any:
        raise NotImplementedError("Tool must implement run()")

class ToolRegistryDynamic:
    """
    Dynamic registry for agent tools. Allows registration and invocation by name.
    """
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def available_tools(self):
        return list(self._tools.values())

    def invoke(self, name: str, **kwargs) -> Any:
        tool = self.get(name)
        return tool.run(**kwargs)
