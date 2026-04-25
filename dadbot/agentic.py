from __future__ import annotations

import email.utils
import json
import re
import time
import uuid
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from dateutil import parser as dateutil_parser
from dateutil.relativedelta import relativedelta


class ToolRegistry:
    def __init__(self, bot):
        self.bot = bot

    @staticmethod
    def _parse_toggle_command(stripped, slash, action):
        match = re.match(rf"^{re.escape(slash)}(?:\s+(on|off|status))?$", stripped, flags=re.IGNORECASE)
        if not match:
            return None
        mode = str(match.group(1) or "status").strip().lower() or "status"
        return {"action": action, "args": {"mode": mode}}

    @staticmethod
    def _parse_list_command(lowered, phrases, action):
        if any(phrase in lowered for phrase in phrases):
            return {"action": action, "args": {}}
        return None

    def _parse_command_structure(self, stripped, lowered):
        slash_commands = {
            "/status": "status_snapshot",
            "/dad": "dad_snapshot",
            "/proactive": "proactive_snapshot",
            "/evolve": "force_persona_evolution",
            "/help": "command_help",
        }
        if lowered in slash_commands:
            return {"action": slash_commands[lowered], "args": {}}

        quiet = self._parse_toggle_command(stripped, "/quiet", "quiet_mode_toggle")
        if quiet is not None:
            return quiet

        voice = self._parse_toggle_command(stripped, "/voice", "voice_mode_toggle")
        if voice is not None:
            return voice

        reject_match = re.match(r"^/reject(?:\s+(.*))?$", stripped, flags=re.IGNORECASE)
        if reject_match:
            return {
                "action": "reject_persona_trait",
                "args": {"query": str(reject_match.group(1) or "").strip()},
            }

        reminder_match = re.match(r"^(?:remind me to|set a reminder to|set reminder to)\s+(.+)$", stripped, flags=re.IGNORECASE)
        if reminder_match:
            title, due_text = self.split_reminder_details(reminder_match.group(1).strip())
            return {
                "action": "set_reminder",
                "args": {"title": title, "due_text": due_text},
            }

        calendar_match = re.match(
            r"^(?:add to calendar|schedule event|create calendar event|calendar add)\s+(.+)$",
            stripped,
            flags=re.IGNORECASE,
        )
        if calendar_match:
            title, due_text = self.split_reminder_details(calendar_match.group(1).strip())
            return {
                "action": "create_calendar_event",
                "args": {"title": title, "due_text": due_text},
            }

        reminders = self._parse_list_command(
            lowered,
            ["what reminders do i have", "list my reminders", "show my reminders", "list reminders"],
            "list_reminders",
        )
        if reminders is not None:
            return reminders

        calendars = self._parse_list_command(
            lowered,
            [
                "what's on my calendar",
                "what is on my calendar",
                "show my calendar",
                "check my calendar",
                "list calendar events",
                "show calendar events",
            ],
            "list_calendar_events",
        )
        if calendars is not None:
            return calendars

        email_match = re.match(
            r"^(?:draft (?:an )?email|write (?:an )?email|email draft)\s+to\s+(.+)$",
            stripped,
            flags=re.IGNORECASE,
        )
        if email_match:
            tail = email_match.group(1).strip()
            parts = re.split(r"\s+(?:about|regarding|re:)\s+", tail, maxsplit=1, flags=re.IGNORECASE)
            recipient = str(parts[0] or "").strip()
            subject = str(parts[1] or "") if len(parts) > 1 else ""
            return {
                "action": "draft_email",
                "args": {"recipient": recipient, "subject": subject},
            }

        search_match = re.match(
            r"^(?:search(?: the web)? for|look up|check the web for|find info on|find information on)\s+(.+)$",
            stripped,
            flags=re.IGNORECASE,
        )
        if search_match:
            return {
                "action": "web_lookup",
                "args": {"query": search_match.group(1).strip()},
            }

        return None

    @staticmethod
    def _validate_command(structure):
        if not isinstance(structure, dict):
            return False
        action = str(structure.get("action") or "").strip()
        if not action:
            return False
        args = structure.get("args")
        if args is None:
            return True
        return isinstance(args, dict)

    @staticmethod
    def _resolve_tool(structure):
        return str(structure.get("action") or "").strip()

    @staticmethod
    def _extract_arguments(structure):
        args = structure.get("args")
        if isinstance(args, dict):
            return dict(args)
        return {}

    def parse_tool_command(self, user_input):
        stripped = str(user_input or "").strip()
        lowered = stripped.lower()
        structure = self._parse_command_structure(stripped, lowered)
        if not self._validate_command(structure):
            return None

        action = self._resolve_tool(structure)
        arguments = self._extract_arguments(structure)
        payload = {"action": action}
        payload.update(arguments)
        return payload

    def get_available_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "set_reminder",
                    "description": "Create a reminder for Tony. Use this when he mentions something he needs to remember or do later.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Short clear title of the reminder"},
                            "due_text": {"type": "string", "description": "Optional due date/time like 'tomorrow' or 'next Friday'"},
                        },
                        "required": ["title"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Look up factual information on the web (weather, news, facts, how-to, etc.). Use only when the user asks a question that needs current or external info.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Clear search query"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_calendar_event",
                    "description": "Create a local calendar event in Dad Bot storage. Use for scheduling commitments and appointments.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Event title"},
                            "due_text": {"type": "string", "description": "Optional date/time text like 'tomorrow 2pm'"},
                        },
                        "required": ["title"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "draft_email",
                    "description": "Create a local .eml draft message without sending anything.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "recipient": {"type": "string", "description": "Recipient name or email"},
                            "subject": {"type": "string", "description": "Email subject"},
                            "body": {"type": "string", "description": "Optional draft body"},
                        },
                        "required": ["recipient"],
                    },
                },
            },
        ]

    def infer_agentic_reminder_request(self, user_input):
        patterns = [
            r"^(?:i need to remember to|don't let me forget to|dont let me forget to|can you remind me to)\s+(.+)$",
            r"^(?:i should remember to)\s+(.+)$",
        ]
        stripped = str(user_input or "").strip()
        for pattern in patterns:
            match = re.match(pattern, stripped, flags=re.IGNORECASE)
            if not match:
                continue
            title, due_text = self.split_reminder_details(match.group(1).strip())
            if title:
                return {"title": title, "due_text": due_text}
        return None

    def should_autonomous_web_lookup(self, user_input):
        stripped = str(user_input or "").strip()
        lowered = stripped.lower()
        if not stripped or self.parse_tool_command(stripped) is not None:
            return False
        if self.bot.matching_topics(stripped):
            return False
        if any(term in lowered for term in ["remember", "recall", "before", "previous"]):
            return False

        triggers = [
            r"\b(weather|forecast|temperature|rain|snow)\b",
            r"\b(latest|news|update|release date|price of)\b",
            r"\b(what is|what's|who is|when is|where is)\b",
            r"\b(how do i fix|how to fix|error code|traceback|stack trace)\b",
        ]
        return any(re.search(pattern, lowered) for pattern in triggers)

    @staticmethod
    def normalize_lookup_query(user_input):
        query = str(user_input or "").strip().rstrip("?!. ")
        query = re.sub(r"^(?:dad[, ]+)?", "", query, flags=re.IGNORECASE)
        query = re.sub(r"^(?:can you|could you|would you|please)\s+", "", query, flags=re.IGNORECASE)
        return query.strip()

    @staticmethod
    def reminder_has_date_signal(detail):
        lowered = str(detail or "").strip().lower()
        if not lowered:
            return False

        signal_patterns = [
            r"\b(today|tomorrow|tonight|next week|next month|next year)\b",
            r"\b(mon|monday|tue|tuesday|wed|wednesday|thu|thursday|fri|friday|sat|saturday|sun|sunday)\b",
            r"\b(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\b",
            r"\b\d{1,2}[/:.-]\d{1,2}(?:[/:.-]\d{2,4})?\b",
            r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
            r"\b(noon|midnight)\b",
            r"\b(by|on|at)\b",
        ]
        return any(re.search(pattern, lowered) for pattern in signal_patterns)

    @staticmethod
    def normalize_relative_reminder_phrase(detail, reference):
        replacements = [
            (r"\btomorrow\b", (reference + relativedelta(days=1)).strftime("%Y-%m-%d")),
            (r"\btoday\b", reference.strftime("%Y-%m-%d")),
            (r"\btonight\b", reference.strftime("%Y-%m-%d") + " 8:00 PM"),
            (r"\bnext week\b", (reference + relativedelta(days=7)).strftime("%Y-%m-%d")),
            (r"\bnext month\b", (reference + relativedelta(months=1)).strftime("%Y-%m-%d")),
            (r"\bnext year\b", (reference + relativedelta(years=1)).strftime("%Y-%m-%d")),
        ]

        normalized = detail
        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        return normalized

    @classmethod
    def split_reminder_details(cls, detail):
        reminder_text = str(detail or "").strip()
        if not reminder_text or not cls.reminder_has_date_signal(reminder_text):
            return reminder_text, ""

        reference = datetime.now().replace(second=0, microsecond=0)
        parser_default = reference.replace(hour=0, minute=0)
        normalized_text = cls.normalize_relative_reminder_phrase(reminder_text, reference)

        try:
            parsed_due, leftover_tokens = dateutil_parser.parse(
                normalized_text,
                fuzzy_with_tokens=True,
                default=parser_default,
            )
        except (ValueError, OverflowError, TypeError):
            return reminder_text, ""

        title = re.sub(r"\s+", " ", "".join(leftover_tokens)).strip(" ,.;:-")
        title = re.sub(r"\b(?:at|on|by)\b\s*$", "", title, flags=re.IGNORECASE).strip(" ,.;:-")
        if not title or title == reminder_text:
            return reminder_text, ""

        has_explicit_time = bool(
            re.search(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", normalized_text.lower())
            or re.search(r"\b\d{1,2}:\d{2}\b", normalized_text.lower())
            or re.search(r"\b(noon|midnight|tonight)\b", reminder_text.lower())
        )
        due_text = parsed_due.strftime("%Y-%m-%d %I:%M %p" if has_explicit_time else "%Y-%m-%d")
        due_text = due_text.replace(" 0", " ")
        return title, due_text


class AgenticHandler:
    def __init__(self, bot, tool_registry: ToolRegistry):
        self.bot = bot
        self.tool_registry = tool_registry

    def add_reminder(self, title, due_text=""):
        timestamp = datetime.now().isoformat(timespec="seconds")
        normalized = self.bot.normalize_reminder_entry(
            {
                "title": title,
                "due_text": due_text,
                "status": "open",
                "created_at": timestamp,
                "updated_at": timestamp,
            }
        )
        if normalized is None:
            return None

        reminders = [item for item in self.bot.MEMORY_STORE.get("reminders", []) if item.get("status") != "done"]
        for reminder in reminders:
            if self.bot.normalize_memory_text(reminder.get("title", "")) == self.bot.normalize_memory_text(normalized["title"]):
                previous_due_at = reminder.get("due_at")
                reminder["due_text"] = normalized["due_text"] or reminder.get("due_text", "")
                reminder["due_at"] = normalized.get("due_at") or reminder.get("due_at")
                reminder["updated_at"] = normalized["updated_at"]
                if reminder.get("due_at") != previous_due_at:
                    reminder["last_notified_at"] = None
                    reminder["notification_count"] = 0
                self.bot.mutate_memory_store(reminders=reminders)
                return reminder

        reminders.append(normalized)
        self.bot.mutate_memory_store(reminders=reminders[-50:])
        return normalized

    def _calendar_events_path(self):
        destination = self.bot.env_path(
            "DADBOT_CALENDAR_EVENTS_PATH",
            self.bot.MEMORY_PATH.with_name("dad_calendar_events.json"),
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def _load_calendar_events(self):
        path = self._calendar_events_path()
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(payload, list):
            return []
        events = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            title = str(entry.get("title") or "").strip()
            if not title:
                continue
            events.append(
                {
                    "event_id": str(entry.get("event_id") or uuid.uuid4().hex),
                    "title": title,
                    "due_text": str(entry.get("due_text") or "").strip(),
                    "due_at": str(entry.get("due_at") or "").strip(),
                    "created_at": str(entry.get("created_at") or datetime.now().isoformat(timespec="seconds")),
                }
            )
        return events

    def _save_calendar_events(self, events):
        path = self._calendar_events_path()
        self.bot.write_json_atomically(path, list(events or []), backup=True)

    @staticmethod
    def _coerce_due_iso(due_text):
        cleaned = str(due_text or "").strip()
        if not cleaned:
            return ""
        try:
            parsed = dateutil_parser.parse(cleaned, fuzzy=True, default=datetime.now().replace(second=0, microsecond=0))
            return parsed.isoformat(timespec="seconds")
        except Exception:
            return ""

    def add_calendar_event(self, title, due_text=""):
        normalized_title = str(title or "").strip()
        if not normalized_title:
            return None
        normalized_due = str(due_text or "").strip()
        if not normalized_due:
            normalized_title, normalized_due = self.tool_registry.split_reminder_details(normalized_title)

        due_at = self._coerce_due_iso(normalized_due)
        now = datetime.now().isoformat(timespec="seconds")
        events = self._load_calendar_events()
        created_event = {
            "event_id": uuid.uuid4().hex,
            "title": normalized_title,
            "due_text": normalized_due,
            "due_at": due_at,
            "created_at": now,
        }
        events.append(created_event)
        events.sort(key=lambda item: (str(item.get("due_at") or "9999"), str(item.get("title") or "")))
        self._save_calendar_events(events[-200:])
        return created_event

    def list_calendar_events(self, limit=8):
        events = self._load_calendar_events()
        if not events:
            return []
        events.sort(key=lambda item: (str(item.get("due_at") or "9999"), str(item.get("created_at") or "")))
        return events[: max(1, int(limit or 1))]

    def delete_calendar_event(self, event_id: str) -> bool:
        """Delete a calendar event by ID. Returns True if found and removed."""
        events = self._load_calendar_events()
        filtered = [e for e in events if str(e.get("event_id") or "") != str(event_id or "").strip()]
        if len(filtered) == len(events):
            return False
        self._save_calendar_events(filtered)
        return True

    def draft_email(self, recipient, subject="", body=""):
        to_value = str(recipient or "").strip()
        if not to_value:
            return None
        subject_value = str(subject or "").strip() or "Quick note"
        body_value = str(body or "").strip()
        if not body_value:
            body_value = (
                f"Hi {to_value},\n\n"
                "Wanted to send a quick note.\n\n"
                "Best,\nTony"
            )

        message = EmailMessage()
        message["To"] = to_value
        message["From"] = "tony@local.dadbot"
        message["Date"] = email.utils.format_datetime(datetime.now())
        message["Subject"] = subject_value
        message.set_content(body_value)

        drafts_dir = self.bot.env_path("DADBOT_EMAIL_DRAFT_DIR", self.bot.MEMORY_PATH.with_name("email_drafts"))
        drafts_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", subject_value.lower()).strip("-") or "draft"
        file_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{slug[:40]}.eml"
        destination = Path(drafts_dir) / file_name
        destination.write_bytes(message.as_bytes())
        return {
            "path": str(destination),
            "recipient": to_value,
            "subject": subject_value,
        }

    @staticmethod
    def format_reminder_list(reminders):
        if not reminders:
            return "You don't have any open reminders right now, buddy."

        parts = []
        for reminder in reminders[:5]:
            title = reminder.get("title", "something")
            due_text = reminder.get("due_text")
            if due_text:
                parts.append(f"{title} ({due_text})")
            else:
                parts.append(title)

        joined = "; ".join(parts)
        return f"Here's what I've got on your reminder list: {joined}."

    @staticmethod
    def extract_related_topic_results(related_topics):
        results = []
        for item in related_topics or []:
            if not isinstance(item, dict):
                continue
            if item.get("Text"):
                results.append(item)
                continue
            results.extend(AgenticHandler.extract_related_topic_results(item.get("Topics", [])))
        return results

    def lookup_web(self, query):
        params = urlencode(
            {
                "q": query,
                "format": "json",
                "no_redirect": 1,
                "no_html": 1,
                "skip_disambig": 1,
            }
        )
        url = f"https://api.duckduckgo.com/?{params}"
        request = Request(
            url,
            headers={
                "User-Agent": "DadBot/1.0 (+https://local.dadbot)",
                "Accept": "application/json, text/plain, */*",
            },
        )

        payload = None
        for attempt in range(3):
            try:
                with urlopen(request, timeout=8) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except Exception:
                if attempt == 2:
                    return None
                time.sleep(0.2 * (attempt + 1))

        abstract = str(payload.get("AbstractText") or payload.get("Answer") or payload.get("Definition") or "").strip()
        source_url = str(payload.get("AbstractURL") or payload.get("DefinitionURL") or "").strip()
        heading = str(payload.get("Heading") or query).strip()

        if not abstract:
            related = self.extract_related_topic_results(payload.get("RelatedTopics", []))
            if related:
                abstract = str(related[0].get("Text") or "").strip()
                source_url = str(related[0].get("FirstURL") or source_url).strip()

        if not abstract:
            return None

        source_label = ""
        if source_url:
            try:
                source_label = urlparse(source_url).netloc.replace("www.", "")
            except Exception:
                source_label = ""

        return {
            "heading": heading,
            "summary": abstract,
            "source_url": source_url,
            "source_label": source_label,
        }

    def autonomous_tool_result_for_input(self, user_input, current_mood, attachments=None):
        settings = self.bot.agentic_tool_settings()
        if not settings["enabled"]:
            self.bot.update_planner_debug(
                fallback_status="disabled",
                fallback_reason="Heuristic fallback is disabled because agentic tools are turned off.",
                final_path="disabled",
            )
            return None, None

        reminder_request = self.tool_registry.infer_agentic_reminder_request(user_input) if settings["auto_reminders"] else None
        if reminder_request is not None:
            reminder = self.bot.add_reminder(reminder_request["title"], reminder_request.get("due_text", ""))
            if reminder is not None:
                self.bot.update_planner_debug(
                    fallback_status="used_tool",
                    fallback_reason="Heuristic reminder detection matched the turn.",
                    fallback_tool="set_reminder",
                    final_path="heuristic_tool",
                )
                if reminder.get("due_text"):
                    reply = f"I went ahead and turned that into a reminder for you, Tony: {reminder['title']} ({reminder['due_text']})."
                else:
                    reply = f"I went ahead and turned that into a reminder for you, Tony: {reminder['title']}."
                return self.bot.reply_finalization.finalize(reply, current_mood, user_input), None

        if settings["auto_web_lookup"] and self.tool_registry.should_autonomous_web_lookup(user_input):
            result = self.bot.lookup_web(self.tool_registry.normalize_lookup_query(user_input))
            if result is not None:
                source = f" Source: {result['source_label']}." if result.get("source_label") else ""
                observation = f"{result['heading']}: {result['summary']}{source}"
                self.bot.update_planner_debug(
                    fallback_status="used_tool",
                    fallback_reason="Heuristic web lookup detection matched the turn.",
                    fallback_tool="web_search",
                    fallback_observation=observation,
                    final_path="heuristic_tool",
                )
                return None, observation

        self.bot.update_planner_debug(
            fallback_status="no_tool",
            fallback_reason="Heuristic fallback did not find a tool action.",
            final_path="no_tool",
        )
        return None, None

    # ------------------------------------------------------------------
    # handle_tool_command helpers â€“ one per logical action group
    # ------------------------------------------------------------------

    def _handle_reminder_action(self, command):
        action = command.get("action")
        if action == "set_reminder":
            reminder = self.bot.add_reminder(command.get("title", ""), command.get("due_text", ""))
            if reminder is None:
                return "I couldn't turn that into a reminder cleanly, buddy."
            if reminder.get("due_text"):
                return f"I've got it, Tony. I'll remind you to {reminder['title']} {reminder['due_text']}."
            return f"I've got it, Tony. I'll remind you to {reminder['title']}."
        if action == "list_reminders":
            return self.format_reminder_list(self.bot.reminder_catalog())
        return None

    def _handle_calendar_action(self, command):
        action = command.get("action")
        if action == "create_calendar_event":
            event = self.add_calendar_event(command.get("title", ""), command.get("due_text", ""))
            if event is None:
                return "I couldn't create that calendar event yet, buddy."
            if event.get("due_text"):
                return f"Added to your local calendar, Tony: {event['title']} ({event['due_text']})."
            return f"Added to your local calendar, Tony: {event['title']}."
        if action == "list_calendar_events":
            events = self.list_calendar_events(limit=6)
            if not events:
                return "Your local calendar is clear right now, buddy."
            parts = []
            for event in events:
                title = str(event.get("title") or "something")
                when = str(event.get("due_text") or "")
                parts.append(f"{title} ({when})" if when else title)
            return "Local calendar events: " + "; ".join(parts) + "."
        return None

    def _handle_draft_email(self, command):
        if command.get("action") != "draft_email":
            return None
        draft = self.draft_email(
            command.get("recipient", ""),
            command.get("subject", ""),
            command.get("body", ""),
        )
        if draft is None:
            return "I couldn't draft that email yet, buddy."
        return (
            f"I drafted that email locally, Tony. Subject: {draft['subject']}. "
            f"Saved at: {draft['path']}"
        )

    def _handle_snapshot_action(self, command):
        action = command.get("action")
        if action == "status_snapshot":
            return self.bot.format_status_snapshot()
        if action == "dad_snapshot":
            return self.bot.format_dad_snapshot()
        if action == "proactive_snapshot":
            return self.bot.format_proactive_snapshot()
        return None

    def _handle_persona_action(self, command):
        action = command.get("action")
        if action == "force_persona_evolution":
            entry = self.bot.evolve_persona(force=True)
            if entry is None:
                return "I checked for a real dad-growth shift, buddy, but nothing strong enough earned a permanent trait yet."
            feedback = str(entry.get("critique_feedback") or "").strip()
            if feedback:
                return f"I locked in a small dad evolution: {entry['trait']}. {feedback}"
            return f"I locked in a small dad evolution: {entry['trait']}."
        if action == "reject_persona_trait":
            removed = self.bot.reject_persona_trait(command.get("query", ""))
            if removed is None:
                return "I couldn't find an evolved trait to roll back right now, buddy."
            return f"Alright, buddy. I rolled back this evolved trait: {removed['trait']}."
        if action == "command_help":
            return self.bot.command_help_text()
        return None

    def _handle_settings_toggle(self, command):
        action = command.get("action")
        if action == "quiet_mode_toggle":
            mode = str(command.get("mode") or "status").strip().lower() or "status"
            if mode == "on":
                self.bot.set_health_quiet_mode(True, save=True)
                return "Quiet mode is now on, buddy. Dad will pause proactive nudges while pressure is elevated."
            if mode == "off":
                self.bot.set_health_quiet_mode(False, save=True)
                return "Quiet mode is now off, buddy. Dad can resume proactive nudges when appropriate."
            enabled = self.bot.health_quiet_mode_enabled()
            return f"Quiet mode is currently {'on' if enabled else 'off'}, buddy."
        if action == "voice_mode_toggle":
            mode = str(command.get("mode") or "status").strip().lower() or "status"
            profile = self.bot.PROFILE if isinstance(self.bot.PROFILE, dict) else {}
            voice_config = profile.get("voice") if isinstance(profile.get("voice"), dict) else {}
            if mode == "on":
                voice_config["enabled"] = True
                profile["voice"] = voice_config
                self.bot.save_profile()
                return "Voice mode is now marked on in your profile, buddy. Streamlit voice controls can use it immediately."
            if mode == "off":
                voice_config["enabled"] = False
                profile["voice"] = voice_config
                self.bot.save_profile()
                return "Voice mode is now marked off in your profile, buddy."
            enabled = bool(voice_config.get("enabled", False))
            return f"Voice mode is currently {'on' if enabled else 'off'}, buddy."
        return None

    def _handle_web_lookup(self, command):
        if command.get("action") != "web_lookup":
            return None
        result = self.bot.lookup_web(command.get("query", ""))
        if result is None:
            return "I tried to look that up for you, buddy, but I couldn't get a clean result right now."
        source = f" Source: {result['source_label']}." if result.get("source_label") else ""
        return f"I looked that up for you, Tony. {result['heading']}: {result['summary']}{source}"

    def handle_tool_command(self, user_input):
        command = self.tool_registry.parse_tool_command(user_input)
        if command is None:
            return None

        for handler in (
            self._handle_reminder_action,
            self._handle_calendar_action,
            self._handle_draft_email,
            self._handle_snapshot_action,
            self._handle_persona_action,
            self._handle_settings_toggle,
            self._handle_web_lookup,
        ):
            result = handler(command)
            if result is not None:
                return result
        return None
