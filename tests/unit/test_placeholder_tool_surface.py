from __future__ import annotations

import asyncio

import pytest

from dadbot.core.services import build_services
from dadbot.services._tool_pipeline_mixin import ToolPipelineMixin
from dadbot.tools.executors import build_default_tool_registry, execute_tool

pytestmark = pytest.mark.unit


LEGACY_DISPATCH_NAMES = ("reminder", "set_reminder", "calendar", "web_search")


def test_default_tool_catalog_only_advertises_supported_runtime_handlers():
    registry = build_default_tool_registry()
    registered = {name for name in LEGACY_DISPATCH_NAMES if registry.get(name) is not None}
    definitions = registry.get_available_tools()
    advertised = {definition["function"]["name"] for definition in definitions}
    runtime_handlers = {
        "set_reminder": ToolPipelineMixin._execute_set_reminder_tool_sync,
        "web_search": ToolPipelineMixin._execute_web_search_tool_sync,
    }

    assert registered == set()
    assert advertised == set(runtime_handlers)
    assert "calendar" not in advertised
    assert all(callable(runtime_handlers[name]) for name in advertised)


@pytest.mark.parametrize("tool_name", LEGACY_DISPATCH_NAMES)
def test_placeholder_selection_cannot_execute_an_effect(tool_name):
    effects = []

    with pytest.raises(ValueError, match=f"Tool '{tool_name}' not found"):
        execute_tool(tool_name, "run", callback=lambda: effects.append(tool_name))

    assert effects == []


@pytest.mark.parametrize("tool_name", LEGACY_DISPATCH_NAMES)
def test_legacy_planner_does_not_execute_selected_placeholder(tool_name):
    async def selected_tool(_prompt):
        return f'Thought: selected\nAction: {{"tool": "{tool_name}", "args": {{}}}}'

    services = build_services(llm_port=selected_tool)
    result = asyncio.run(services["planner"].plan("try the tool", max_steps=1))

    assert result["trace"][0]["observation"] == f"Tool '{tool_name}' not found."
    assert services["tool_registry"].describe_all() == ""
