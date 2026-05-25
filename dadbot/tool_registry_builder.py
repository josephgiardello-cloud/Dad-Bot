from typing import Any
# --- ToolRegistry builder for Phase B wiring ---
def _build_and_register_tool_registry(bot: Any):
    """Instantiate and register all built-in tools in the new ToolRegistry."""
    from dadbot.core import tool_registry as tr
    registry = tr.ToolRegistry()
    # Register built-in ToolSpecs and executors
    registry.register(tr.SET_REMINDER_SPEC, tr.ToolRegistry.make_set_reminder_executor(bot))
    registry.register(tr.WEB_SEARCH_SPEC, tr.ToolRegistry.make_web_search_executor(bot))
    registry.register(tr.CREATE_CALENDAR_EVENT_SPEC, tr.ToolRegistry.make_create_calendar_event_executor(bot, registry))
    registry.register(tr.DRAFT_EMAIL_SPEC, tr.ToolRegistry.make_draft_email_executor(bot))
    return registry
