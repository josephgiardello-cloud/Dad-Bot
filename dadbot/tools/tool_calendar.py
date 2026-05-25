from dadbot.base.tool_base import Tool
from typing import Any
from datetime import datetime

class CalendarTool(Tool):
    def __init__(self):
        super().__init__(
            name="calendar_add_event",
            description="Add an event to the local calendar store.",
            parameters={"title": "Event title (str)", "datetime": "Event datetime (str or datetime)"}
        )

    def run(self, **kwargs) -> Any:
        title = kwargs.get("title")
        dt = kwargs.get("datetime")
        # For demo, just echo back. Real impl would persist to a calendar store.
        if not title or not dt:
            return {"error": "Missing title or datetime"}
        try:
            # Parse datetime if string
            if isinstance(dt, str):
                dt = datetime.fromisoformat(dt)
            # Here you would save to a real calendar
            return {"event": {"title": title, "datetime": dt.isoformat()}}
        except Exception as e:
            return {"error": str(e)}
