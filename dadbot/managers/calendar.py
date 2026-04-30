"""iCal feed fetching and calendar sync background worker."""

from __future__ import annotations

import urllib.request
from datetime import UTC, datetime
from typing import Any


class CalendarManager:
    """Owns iCal feed URL, event fetching, and background calendar sync."""

    def __init__(self, bot):
        self.bot = bot

    def ical_feed_url(self) -> str:
        return str(self.bot.PROFILE.get("ical_feed_url", "")).strip()

    def set_ical_feed_url(self, url: str, save: bool = True) -> None:
        self.bot.PROFILE["ical_feed_url"] = str(url or "").strip()
        if save:
            self.bot.save_profile()

    def fetch_upcoming_ical_events(self, limit: int = 8) -> list[dict]:
        """Fetch upcoming events from the configured iCal feed URL using pure stdlib."""
        url = self.ical_feed_url()
        if not url:
            return []
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            events: list[dict] = []
            current: dict = {}
            for line in raw.splitlines():
                line = line.strip()
                if line == "BEGIN:VEVENT":
                    current = {}
                elif line == "END:VEVENT":
                    if current.get("DTSTART") and current.get("SUMMARY"):
                        events.append(current)
                    current = {}
                elif line.startswith("SUMMARY:"):
                    current["SUMMARY"] = line[8:].strip()
                elif line.startswith("DTSTART"):
                    dt_str = line.split(":", 1)[-1].strip()
                    try:
                        if "T" in dt_str:
                            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                        else:
                            dt = datetime.strptime(dt_str, "%Y%m%d").replace(
                                tzinfo=None,
                            )
                        current["DTSTART"] = dt
                    except Exception:
                        pass
                elif line.startswith("DESCRIPTION:"):
                    current["DESCRIPTION"] = line[12:].strip()
                elif line.startswith("LOCATION:"):
                    current["LOCATION"] = line[9:].strip()
            now = datetime.now(UTC)
            upcoming = []
            for e in events:
                dt = e.get("DTSTART")
                if isinstance(dt, datetime):
                    dt_aware = dt if dt.tzinfo else dt.replace(tzinfo=UTC)
                    if dt_aware > now:
                        upcoming.append(e)
            upcoming.sort(key=lambda e: e["DTSTART"])
            return upcoming[:limit]
        except Exception:
            return []

    def schedule_calendar_sync(self) -> Any:
        """Enqueue a background iCal sync. Returns the task future or None."""
        if not self.ical_feed_url():
            return None
        return self.bot.submit_background_task(
            self._run_ical_sync,
            task_kind="calendar-sync",
            metadata={"feed_url": self.ical_feed_url()},
        )

    def _run_ical_sync(self) -> dict:
        """Background worker: fetch events and queue proactive messages for special dates."""
        events = self.fetch_upcoming_ical_events(limit=10)
        for ev in events[:5]:
            summary = ev.get("SUMMARY", "")
            if "birthday" in summary.lower() or "anniversary" in summary.lower():
                self.bot.queue_proactive_message(
                    f"\U0001f382 {summary} is coming up soon! Want me to remind you closer to the date?",
                    source="calendar",
                )
        self.bot.PROFILE["last_ical_sync"] = self.bot.runtime_timestamp()
        self.bot.save_profile()
        return {"events_synced": len(events)}
