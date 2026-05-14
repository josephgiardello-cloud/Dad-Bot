from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from dadbot.notifications import send_local_notification

logger = logging.getLogger(__name__)


class MaintenanceScheduler:
    """Owns post-turn background maintenance and periodic durable synthesis cadence."""

    def __init__(self, bot):
        self.bot = bot

    def _background_memory_store_patch_queue(self) -> list[dict]:
        queue = getattr(self.bot, "_background_memory_store_patch_queue", None)
        if not isinstance(queue, list):
            queue = []
            self.bot._background_memory_store_patch_queue = queue
        return queue

    def _queue_or_apply_memory_store_patch(self, **patch) -> str:
        normalized_patch = dict(patch or {})
        if bool(getattr(self.bot, "_graph_commit_active", False)):
            self.bot.mutate_memory_store(**normalized_patch)
            return "applied"
        queue = self._background_memory_store_patch_queue()
        queue.append(normalized_patch)
        if len(queue) > 128:
            del queue[:-128]
        return "queued"

    def should_run_periodic_durable_synthesis(self, force=False):
        if force:
            return bool(self.bot.conversation_history())
        if self.bot.LIGHT_MODE:
            return False
        if self.bot.session_turn_count() < 3:
            return False
        if len(self.bot.conversation_history()) < 4:
            return False

        last_turn = int(
            self.bot.MEMORY_STORE.get("last_background_synthesis_turn", 0) or 0,
        )
        interval = max(
            3,
            int(getattr(self.bot, "RELATIONSHIP_REFLECTION_INTERVAL", 3) or 3),
        )
        return self.bot.session_turn_count() - last_turn >= interval

    def run_periodic_durable_synthesis(
        self,
        trigger_text="",
        force=False,
        turn_context=None,
    ):
        if not self.should_run_periodic_durable_synthesis(force=force):
            return {
                "ran": False,
                "archived": False,
                "consolidated_count": 0,
                "pattern_count": 0,
                "persona_evolved": False,
            }

        history = self.bot.conversation_history()
        if not history:
            return {
                "ran": False,
                "archived": False,
                "consolidated_count": 0,
                "pattern_count": 0,
                "persona_evolved": False,
            }

        archive_before = len(self.bot.session_archive())
        archive_entry = self.bot.archive_session_context(history)
        if turn_context is None:
            logger.debug(
                "periodic_synthesis_background_path node_id=%s stage=%s execution_path=%s",
                "background:periodic_synthesis",
                "background_maintenance",
                ["archive_session_context", "update_memory_store", "consolidate_memories"],
            )
        self.bot.update_memory_store(history, turn_context=turn_context)
        if turn_context is None:
            consolidated = []
            timeline = ""
            forgetting = {
                "removed": 0,
                "backup_path": "",
                "ran": False,
                "skipped": "missing_turn_context",
            }
        else:
            consolidated = self.bot.consolidate_memories(turn_context=turn_context)
            timeline = self.bot.refresh_relationship_timeline(turn_context=turn_context)
            forgetting = self.bot.apply_controlled_forgetting(turn_context=turn_context)
        patterns = self.bot.detect_life_patterns()
        persona_entry = self.bot.evolve_persona()
        # Background maintenance is non-semantic in Phase 4 strict mode.
        # Graph projection/sync is SaveNode-owned only.

        updated_at = self.bot.runtime_timestamp()
        self._queue_or_apply_memory_store_patch(
            last_background_synthesis_at=updated_at,
            last_background_synthesis_turn=self.bot.session_turn_count(),
        )
        archive_after = len(self.bot.session_archive())
        return {
            "ran": True,
            "archived": archive_entry is not None,
            "archive_count_delta": max(0, archive_after - archive_before),
            "consolidated_count": len(consolidated or []),
            "timeline_updated": bool(timeline),
            "pattern_count": len(patterns or []),
            "persona_evolved": persona_entry is not None,
            "forgotten_count": int(forgetting.get("removed", 0) or 0),
            "forgetting_backup_path": str(forgetting.get("backup_path") or ""),
            "updated_at": updated_at,
            "turn_count": self.bot.session_turn_count(),
            "trigger_text": str(trigger_text or "").strip(),
        }

    def should_run_memory_compaction(self, force=False, reference_time=None):
        if force:
            return True
        if self.bot.LIGHT_MODE:
            return False
        if not self.bot.session_archive() and not self.bot.consolidated_memories():
            return False
        now = self._coerce_reference_time(reference_time)
        last_run = self._parse_iso_datetime(
            self.bot.MEMORY_STORE.get("last_memory_compaction_at"),
        )
        if last_run is None:
            return True
        return now - last_run >= timedelta(hours=24)

    def run_memory_compaction(self, force=False, reference_time=None):
        now = self._coerce_reference_time(reference_time)
        if not self.should_run_memory_compaction(force=force, reference_time=now):
            return {
                "ran": False,
                "summary": str(
                    self.bot.MEMORY_STORE.get("last_memory_compaction_summary") or "",
                ),
                "updated_at": self.bot.MEMORY_STORE.get("last_memory_compaction_at"),
                "narrative_count": len(self.bot.narrative_memories()),
            }

        # Background maintenance is non-semantic in Phase 4 strict mode.
        # Graph projection/sync is SaveNode-owned only.

        narrative_memories = self._distill_narrative_memories(reference_time=now)
        summary = (
            self.bot.build_graph_summary_context(limit=4)
            or self.bot.long_term_signals.summarize_memory_graph()
            or "No strong graph links yet."
        )
        insights = self.bot.long_term_signals.synthesize_longitudinal_insights(
            force=force,
            reference_time=now,
            max_items=12,
        )
        updated_at = now.isoformat(timespec="seconds")
        self._queue_or_apply_memory_store_patch(
            last_memory_compaction_at=updated_at,
            last_memory_compaction_summary=str(summary or "").strip(),
            narrative_memories=narrative_memories,
        )
        return {
            "ran": True,
            "summary": str(summary or "").strip(),
            "updated_at": updated_at,
            "archive_count": len(self.bot.session_archive()),
            "consolidated_count": len(self.bot.consolidated_memories()),
            "narrative_count": len(narrative_memories),
            "insight_count": len(insights or []),
        }

    def _recent_cue_keys(self, history, now):
        cutoff = now - timedelta(days=2)
        keys = set()
        for entry in list(history or []):
            if not isinstance(entry, dict):
                continue
            cue_key = str(entry.get("cue_key") or "").strip()
            detected_at = self._parse_iso_datetime(entry.get("detected_at"))
            if cue_key and detected_at is not None and detected_at >= cutoff:
                keys.add(cue_key)
        return keys

    @staticmethod
    def _cue_key(kind, token):
        return f"{str(kind or 'cue').strip().lower()}::{str(token or '').strip().lower()}"

    @staticmethod
    def _workshop_like(text):
        lowered = str(text or "").lower()
        markers = ("workshop", "wood", "shop", "build", "project", "garage", "bench")
        return any(marker in lowered for marker in markers)

    def _extract_environmental_signals(self):
        signals = {
            "calendar_events": [],
            "ical_events": [],
            "draft_modified_times": [],
        }

        try:
            for event in self.bot.agentic_handler.list_calendar_events(limit=8) or []:
                if isinstance(event, dict):
                    signals["calendar_events"].append(dict(event))
        except Exception:
            pass

        try:
            for event in self.bot.calendar_manager.fetch_upcoming_ical_events(limit=6) or []:
                if isinstance(event, dict):
                    signals["ical_events"].append(dict(event))
        except Exception:
            pass

        try:
            drafts_dir = self.bot.env_path(
                "DADBOT_EMAIL_DRAFT_DIR",
                self.bot.MEMORY_PATH.with_name("email_drafts"),
            )
            if isinstance(drafts_dir, Path) and drafts_dir.exists():
                for item in drafts_dir.glob("*.eml"):
                    if not item.is_file():
                        continue
                    signals["draft_modified_times"].append(
                        datetime.fromtimestamp(item.stat().st_mtime),
                    )
        except Exception:
            pass

        return signals

    def _derive_temporal_features(self, signals, now):
        horizon = {
            "calendar_cutoff": now - timedelta(hours=8),
            "draft_cutoff": now - timedelta(hours=36),
        }
        derived = {
            "calendar_events": [],
            "ical_events": list(signals.get("ical_events") or []),
            "has_recent_drafts": False,
            "recent_draft_count": 0,
        }

        for event in list(signals.get("calendar_events") or []):
            title = str(event.get("title") or "").strip()
            if not title:
                continue
            due_at = self._parse_iso_datetime(event.get("due_at"))
            if due_at is not None and due_at < horizon["calendar_cutoff"]:
                continue
            derived["calendar_events"].append(event)

        recent_count = sum(
            1
            for modified_at in list(signals.get("draft_modified_times") or [])
            if modified_at >= horizon["draft_cutoff"]
        )
        derived["recent_draft_count"] = recent_count
        derived["has_recent_drafts"] = recent_count > 0

        return derived

    def _derive_behavioral_features(self, derived):
        cues = []

        for event in list(derived.get("calendar_events") or []):
            title = str(event.get("title") or "").strip()
            cue_key = self._cue_key("calendar", event.get("event_id") or title)
            if self._workshop_like(title):
                message = f"Workshop radar ping, buddy: {title} is on your calendar. Want me to help break it into a first step?"
            else:
                message = f"Calendar heads-up: {title} is on deck. Need a quick plan so it feels lighter?"
            cues.append({"cue_key": cue_key, "kind": "calendar", "message": message})

        for event in list(derived.get("ical_events") or []):
            summary = str(event.get("SUMMARY") or "").strip()
            if not summary:
                continue
            cue_key = self._cue_key("ical", summary)
            if self._workshop_like(summary):
                message = f"I saw '{summary}' coming up. Want a quick workshop game plan before it sneaks up?"
            else:
                message = f"I noticed '{summary}' coming up on your calendar. Want a quick prep check-in?"
            cues.append({"cue_key": cue_key, "kind": "ical", "message": message})

        if bool(derived.get("has_recent_drafts")):
            cue_key = self._cue_key(
                "email-drafts",
                derived.get("recent_draft_count") or 0,
            )
            cues.append(
                {
                    "cue_key": cue_key,
                    "kind": "email",
                    "message": "I noticed you have a recent email draft hanging out. Want help finishing it in one pass?",
                },
            )

        return cues

    @staticmethod
    def _assemble_environmental_context(cues):
        assembled = []
        for cue in list(cues or []):
            if not isinstance(cue, dict):
                continue
            cue_key = str(cue.get("cue_key") or "").strip()
            kind = str(cue.get("kind") or "").strip()
            message = str(cue.get("message") or "").strip()
            if cue_key and kind and message:
                assembled.append({"cue_key": cue_key, "kind": kind, "message": message})
        return assembled

    def _scan_environmental_cues(self, now):
        signals = self._extract_environmental_signals()
        temporal = self._derive_temporal_features(signals, now)
        behavioral = self._derive_behavioral_features(temporal)
        return self._assemble_environmental_context(behavioral)

    def _queue_shadow_repair_prompt(self, now):
        audits = [
            dict(item) for item in list(self.bot.MEMORY_STORE.get("advice_audits") or []) if isinstance(item, dict)
        ]
        if not audits:
            return 0, audits

        for entry in reversed(audits):
            if not bool(entry.get("needs_repair")):
                continue
            if str(entry.get("repair_sent_at") or "").strip():
                continue
            recorded_at = self._parse_iso_datetime(entry.get("recorded_at"))
            if recorded_at is not None and now - recorded_at < timedelta(minutes=30):
                continue
            self.bot.queue_proactive_message(
                "I looked back at how I came at that earlier, buddy, and I was too hard. I'm sorry. Want a gentler reset together?",
                source="shadow-thread",
            )
            entry["repair_sent_at"] = now.isoformat(timespec="seconds")
            return 1, audits
        return 0, audits

    def _distill_narrative_memories(self, reference_time=None):
        now = self._coerce_reference_time(reference_time)
        archives = list(self.bot.session_archive())
        if len(archives) < 2:
            return list(self.bot.narrative_memories())

        # Skip only the freshest slice when history is deep; smaller archives still
        # need enough evidence to produce a usable narrative compression.
        source_archives = archives[:-4] if len(archives) > 8 else archives
        clusters = defaultdict(list)
        for entry in source_archives:
            topics = [str(topic).strip().lower() for topic in entry.get("topics", []) if str(topic).strip()]
            if not topics:
                topics = ["general"]
            primary_topic = topics[0]
            created_at = str(
                entry.get("created_at") or now.isoformat(timespec="seconds"),
            )
            month_key = created_at[:7] if len(created_at) >= 7 else now.strftime("%Y-%m")
            clusters[(primary_topic, month_key)].append(dict(entry))

        narratives = []
        for (topic, month_key), entries in sorted(clusters.items()):
            if len(entries) < 2:
                continue
            moods = [self.bot.normalize_mood(item.get("dominant_mood")) for item in entries]
            mood_counts = defaultdict(int)
            for mood in moods:
                mood_counts[mood] += 1
            dominant_mood = sorted(
                mood_counts.items(),
                key=lambda item: (item[1], item[0]),
                reverse=True,
            )[0][0]
            summaries = [
                str(item.get("summary") or "").strip() for item in entries if str(item.get("summary") or "").strip()
            ]
            evidence = "; ".join(summaries[:3])
            period_start = str(entries[0].get("created_at") or month_key)
            period_end = str(entries[-1].get("created_at") or month_key)
            narrative_summary = (
                f"Tony's {topic} arc during {month_key} centered on {dominant_mood} momentum across {len(entries)} sessions. "
                f"Key throughline: {summaries[0] if summaries else 'the theme repeated enough to form a durable narrative.'}"
            )
            narratives.append(
                {
                    "topic": topic,
                    "period": month_key,
                    "dominant_mood": dominant_mood,
                    "summary": narrative_summary,
                    "evidence": evidence,
                    "source_count": len(entries),
                    "period_start": period_start,
                    "period_end": period_end,
                    "updated_at": now.isoformat(timespec="seconds"),
                },
            )

        return narratives[-24:]

    @staticmethod
    def _coerce_reference_time(reference_time=None):
        if isinstance(reference_time, datetime):
            return reference_time.replace(second=0, microsecond=0)
        raw = str(reference_time or "").strip()
        if not raw:
            return datetime.now().replace(second=0, microsecond=0)
        try:
            return datetime.fromisoformat(raw).replace(second=0, microsecond=0)
        except ValueError:
            return datetime.now().replace(second=0, microsecond=0)

    @staticmethod
    def _parse_iso_datetime(value):
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    @staticmethod
    def _due_phrase(reminder, due_at):
        due_text = str(reminder.get("due_text") or "").strip()
        if due_text:
            return due_text
        return due_at.strftime("%Y-%m-%d %I:%M %p").replace(" 0", " ")

    def _build_due_reminder_message(self, reminder, due_at, now):
        title = str(reminder.get("title") or "something").strip() or "something"
        due_phrase = self._due_phrase(reminder, due_at)
        if due_at <= now:
            return f"Quick heads-up, buddy: your reminder to {title} was due {due_phrase}. Want to tackle it now?"
        if due_at.date() == now.date():
            return f"Quick heads-up, buddy: your reminder to {title} is coming up today at {due_at.strftime('%I:%M %p').lstrip('0')}."
        return f"Quick heads-up, buddy: don't forget to {title} by {due_phrase}."

    def notification_settings(self):
        configured = self.bot.PROFILE.get("notifications", {}) if isinstance(self.bot.PROFILE, dict) else {}
        if not isinstance(configured, dict):
            configured = {}
        backend = str(configured.get("backend") or "auto").strip().lower() or "auto"
        if backend not in {"auto", "notifypy", "plyer"}:
            backend = "auto"
        try:
            quiet_start = int(configured.get("quiet_hours_start", 23))
        except (TypeError, ValueError):
            quiet_start = 23
        try:
            quiet_end = int(configured.get("quiet_hours_end", 7))
        except (TypeError, ValueError):
            quiet_end = 7
        return {
            "enabled": bool(configured.get("enabled", False)),
            "backend": backend,
            "quiet_hours_start": max(0, min(23, quiet_start)),
            "quiet_hours_end": max(0, min(23, quiet_end)),
            "notify_patterns": bool(configured.get("notify_patterns", True)),
            "notify_reminders": bool(configured.get("notify_reminders", True)),
        }

    @staticmethod
    def _within_quiet_hours(now, *, start_hour, end_hour):
        hour = int(now.hour)
        if start_hour == end_hour:
            return False
        if start_hour < end_hour:
            return start_hour <= hour < end_hour
        return hour >= start_hour or hour < end_hour

    def maybe_send_proactive_notification(self, message, *, source, now):
        settings = self.notification_settings()
        if not settings.get("enabled"):
            return False, "disabled"
        if source == "scheduled-pattern" and not settings.get("notify_patterns", True):
            return False, "patterns-disabled"
        if source == "scheduled-reminder" and not settings.get(
            "notify_reminders",
            True,
        ):
            return False, "reminders-disabled"
        if self._within_quiet_hours(
            now,
            start_hour=int(settings.get("quiet_hours_start", 23) or 23),
            end_hour=int(settings.get("quiet_hours_end", 7) or 7),
        ):
            return False, "quiet-hours"

        title = "Dad Bot reminder" if source == "scheduled-reminder" else "Dad Bot check-in"
        return send_local_notification(
            title,
            str(message or "").strip(),
            backend=str(settings.get("backend") or "auto"),
        )

    def _process_due_reminders(
        self,
        reminders,
        now,
        lead_window,
        repeat_window,
        *,
        force=False,
    ):
        """Process due/upcoming reminders. Mutates reminder dicts in-place. Returns (queued, notif_count)."""
        queued = 0
        notifications_sent = 0
        for reminder in reminders:
            if reminder.get("status") == "done":
                continue
            due_at = self._parse_iso_datetime(reminder.get("due_at"))
            if due_at is None:
                continue
            if not force and due_at > now + lead_window:
                continue
            last_notified_at = self._parse_iso_datetime(
                reminder.get("last_notified_at"),
            )
            if not force and last_notified_at is not None and now - last_notified_at < repeat_window:
                continue
            message = self._build_due_reminder_message(reminder, due_at, now)
            self.bot.queue_proactive_message(message, source="scheduled-reminder")
            sent, _backend = self.maybe_send_proactive_notification(
                message,
                source="scheduled-reminder",
                now=now,
            )
            if sent:
                notifications_sent += 1
            reminder["last_notified_at"] = now.isoformat(timespec="seconds")
            reminder["notification_count"] = int(reminder.get("notification_count", 0) or 0) + 1
            queued += 1
        return queued, notifications_sent

    def _process_pattern_nudges(
        self,
        patterns,
        now,
        weekday_name,
        pattern_hour,
        pattern_confidence,
        *,
        force=False,
    ):
        """Process scheduled pattern nudges. Mutates pattern dicts in-place. Returns (queued, notif_count)."""
        queued = 0
        notifications_sent = 0
        for pattern in patterns:
            day_hint = str(pattern.get("day_hint") or "").strip().lower()
            if not day_hint or day_hint != weekday_name:
                continue
            try:
                confidence = int(pattern.get("confidence", 0) or 0)
            except (TypeError, ValueError):
                confidence = 0
            if not force and confidence < pattern_confidence:
                continue
            if not force and now.hour < pattern_hour:
                continue
            last_proactive_at = self._parse_iso_datetime(
                pattern.get("last_proactive_at"),
            )
            if not force and last_proactive_at is not None and last_proactive_at.date() == now.date():
                continue
            pattern_message = self.bot.build_pattern_message(pattern)
            self.bot.queue_proactive_message(
                pattern_message,
                source="scheduled-pattern",
            )
            sent, _backend = self.maybe_send_proactive_notification(
                pattern_message,
                source="scheduled-pattern",
                now=now,
            )
            if sent:
                notifications_sent += 1
            pattern["last_proactive_at"] = now.isoformat(timespec="seconds")
            queued += 1
        return queued, notifications_sent

    def _process_environmental_cues(self, cue_history, now, *, force=False):
        """Scan and queue environmental cues. Mutates cue_history in-place. Returns (queued, notif_count)."""
        recent_cue_keys = self._recent_cue_keys(cue_history, now)
        queued = 0
        notifications_sent = 0
        for cue in self._scan_environmental_cues(now):
            cue_key = str(cue.get("cue_key") or "").strip()
            message = str(cue.get("message") or "").strip()
            if not cue_key or not message or cue_key in recent_cue_keys:
                continue
            self.bot.queue_proactive_message(message, source="environmental-cue")
            sent, _backend = self.maybe_send_proactive_notification(
                message,
                source="environmental-cue",
                now=now,
            )
            if sent:
                notifications_sent += 1
            cue_history.append(
                {
                    "cue_key": cue_key,
                    "kind": str(cue.get("kind") or "environment").strip().lower() or "environment",
                    "message": message,
                    "detected_at": now.isoformat(timespec="seconds"),
                },
            )
            recent_cue_keys.add(cue_key)
            queued += 1
        return queued, notifications_sent

    def run_scheduled_proactive_jobs(self, force=False, reference_time=None):
        health = self.bot.current_runtime_health_snapshot(
            log_warnings=False,
            persist=False,
            max_age_seconds=120,
        )
        health_level = str(health.get("level") or "green").strip().lower() or "green"
        if self.bot.health_quiet_mode_enabled() and health_level in {"yellow", "red"} and not force:
            return {
                "queued_reminders": 0,
                "queued_patterns": 0,
                "queued_environmental": 0,
                "queued_shadow_repairs": 0,
                "queued_total": 0,
                "checked_at": self.bot.runtime_timestamp(),
                "suppressed": True,
                "suppressed_reason": "quiet mode during runtime pressure",
            }

        if self.bot.LIGHT_MODE and not force:
            return {
                "queued_reminders": 0,
                "queued_patterns": 0,
                "queued_environmental": 0,
                "queued_shadow_repairs": 0,
                "queued_total": 0,
                "checked_at": self.bot.runtime_timestamp(),
            }

        now = self._coerce_reference_time(reference_time)
        cadence = self.bot.cadence_settings()
        lead_window = timedelta(
            minutes=max(
                5,
                int(cadence.get("scheduled_reminder_lead_minutes", 45) or 45),
            ),
        )
        repeat_window = timedelta(
            hours=max(1, int(cadence.get("scheduled_reminder_repeat_hours", 12) or 12)),
        )
        pattern_hour = max(0, int(cadence.get("scheduled_pattern_hour", 8) or 8))
        pattern_confidence = max(
            1,
            int(cadence.get("scheduled_pattern_min_confidence", 80) or 80),
        )

        reminders = [dict(item) for item in self.bot.reminder_catalog(include_done=True)]
        patterns = [dict(item) for item in self.bot.life_patterns()]
        cue_history = [
            dict(item)
            for item in list(
                self.bot.MEMORY_STORE.get("environmental_cues_history") or [],
            )
            if isinstance(item, dict)
        ]

        queued_reminders, notif_reminders = self._process_due_reminders(
            reminders,
            now,
            lead_window,
            repeat_window,
            force=force,
        )
        weekday_name = now.strftime("%A").lower()
        queued_patterns, notif_patterns = self._process_pattern_nudges(
            patterns,
            now,
            weekday_name,
            pattern_hour,
            pattern_confidence,
            force=force,
        )
        queued_environmental, notif_env = self._process_environmental_cues(
            cue_history,
            now,
            force=force,
        )
        queued_shadow_repairs, audits = self._queue_shadow_repair_prompt(now)
        notifications_sent = notif_reminders + notif_patterns + notif_env

        self._queue_or_apply_memory_store_patch(
            reminders=reminders,
            life_patterns=patterns,
            advice_audits=audits,
            environmental_cues_history=cue_history[-240:],
            last_scheduled_proactive_at=now.isoformat(timespec="seconds"),
        )
        return {
            "queued_reminders": queued_reminders,
            "queued_patterns": queued_patterns,
            "queued_environmental": queued_environmental,
            "queued_shadow_repairs": queued_shadow_repairs,
            "queued_total": queued_reminders + queued_patterns + queued_environmental + queued_shadow_repairs,
            "notifications_sent": notifications_sent,
            "checked_at": now.isoformat(timespec="seconds"),
        }

    def run_proactive_heartbeat(self, force=False, reference_time=None):
        now = self._coerce_reference_time(reference_time)
        result = dict(
            self.run_scheduled_proactive_jobs(force=force, reference_time=now) or {},
        )
        compaction = self.run_memory_compaction(force=force, reference_time=now)

        queued_daily_checkin = 0
        notifications_sent = int(result.get("notifications_sent", 0) or 0)
        last_daily_checkin_at = self._parse_iso_datetime(
            self.bot.MEMORY_STORE.get("last_daily_checkin_at"),
        )
        already_queued_today = last_daily_checkin_at is not None and last_daily_checkin_at.date() == now.date()

        if (
            not result.get("suppressed")
            and (force or self.bot.memory.should_do_daily_checkin())
            and not already_queued_today
        ):
            message = self.bot.daily_checkin_greeting()
            queued = self.bot.queue_proactive_message(message, source="daily-checkin")
            if queued is not None:
                sent, _backend = self.maybe_send_proactive_notification(
                    message,
                    source="daily-checkin",
                    now=now,
                )
                if sent:
                    notifications_sent += 1
                queued_daily_checkin = 1
            self._queue_or_apply_memory_store_patch(
                last_daily_checkin_at=now.isoformat(timespec="seconds"),
            )

        result["queued_daily_checkin"] = queued_daily_checkin
        result["queued_total"] = int(result.get("queued_total", 0) or 0) + queued_daily_checkin
        result["notifications_sent"] = notifications_sent
        result["memory_compaction"] = compaction
        result["heartbeat_checked_at"] = now.isoformat(timespec="seconds")
        return result

    def maintenance_snapshot(self):
        background = self.bot.background_task_snapshot(limit=8)
        latest_task = next(
            (item for item in background.get("recent", []) if item.get("task_kind") == "post-turn-maintenance"),
            None,
        )
        return {
            "last_background_synthesis_at": self.bot.MEMORY_STORE.get(
                "last_background_synthesis_at",
            ),
            "last_background_synthesis_turn": int(
                self.bot.MEMORY_STORE.get("last_background_synthesis_turn", 0) or 0,
            ),
            "last_memory_compaction_at": self.bot.MEMORY_STORE.get(
                "last_memory_compaction_at",
            ),
            "last_memory_compaction_summary": str(
                self.bot.MEMORY_STORE.get("last_memory_compaction_summary") or "",
            ),
            "last_scheduled_proactive_at": self.bot.MEMORY_STORE.get(
                "last_scheduled_proactive_at",
            ),
            "latest_task": dict(latest_task or {}),
        }

    def run_post_turn_maintenance(self, user_input, current_mood):
        if self.bot.LIGHT_MODE:
            self.bot.current_runtime_health_snapshot(
                force=True,
                log_warnings=False,
                persist=False,
            )
            return {
                "summary_refreshed": False,
                "relationship_reflected": False,
                "wisdom_generated": False,
                "periodic_synthesis": False,
                "memory_graph_refreshed": False,
            }

        health = self.bot.current_runtime_health_snapshot(
            log_warnings=False,
            persist=False,
            max_age_seconds=120,
        )
        health_level = str(health.get("level") or "green").strip().lower() or "green"
        if self.bot.should_delay_noncritical_maintenance(health):
            summary_before = str(self.bot.session_summary or "")
            summary_after = self.bot.refresh_session_summary()
            self.bot.current_runtime_health_snapshot(
                force=True,
                log_warnings=False,
                persist=False,
            )
            return {
                "summary_refreshed": str(summary_after or "") != summary_before,
                "scheduled_proactive": False,
                "scheduled_proactive_count": 0,
                "relationship_reflected": False,
                "wisdom_generated": False,
                "periodic_synthesis": False,
                "periodic_archive_delta": 0,
                "persona_evolved": False,
                "memory_graph_refreshed": not bool(
                    getattr(self.bot, "_memory_graph_dirty", False),
                ),
                "delayed_noncritical": True,
                "health_level": health_level,
            }

        summary_before = str(self.bot.session_summary or "")
        summary_after = self.bot.refresh_session_summary()
        scheduled_proactive = self.run_scheduled_proactive_jobs()
        relationship = None
        wisdom = None
        synthesis = self.run_periodic_durable_synthesis(user_input)
        compaction = self.run_memory_compaction()
        # Background maintenance is non-semantic in Phase 4 strict mode.
        # Graph projection/sync is SaveNode-owned only.
        self.bot.current_runtime_health_snapshot(
            force=True,
            log_warnings=False,
            persist=False,
        )
        # Continuous learning / RLHF — non-critical, runs in background
        self.bot.schedule_continuous_learning()
        return {
            "summary_refreshed": str(summary_after or "") != summary_before,
            "scheduled_proactive": bool(scheduled_proactive.get("queued_total")),
            "scheduled_proactive_count": int(
                scheduled_proactive.get("queued_total", 0) or 0,
            ),
            "relationship_reflected": relationship is not None,
            "wisdom_generated": wisdom is not None,
            "periodic_synthesis": bool(synthesis.get("ran")),
            "periodic_archive_delta": int(synthesis.get("archive_count_delta", 0) or 0),
            "persona_evolved": bool(synthesis.get("persona_evolved")),
            "memory_compaction": bool(compaction.get("ran")),
            "memory_compaction_updated_at": compaction.get("updated_at"),
            "memory_graph_refreshed": not bool(
                getattr(self.bot, "_memory_graph_dirty", False),
            ),
        }

    def schedule_post_turn_maintenance(self, user_input, current_mood):
        if self.bot.LIGHT_MODE:
            return None
        normalized_mood = self.bot.normalize_mood(current_mood)
        return self.bot.submit_background_task(
            self.run_post_turn_maintenance,
            user_input,
            normalized_mood,
            task_kind="post-turn-maintenance",
            metadata={
                "current_mood": normalized_mood,
                "active_hypothesis": self.bot.relationship.snapshot().get(
                    "active_hypothesis",
                )
                or "supportive_baseline",
                "history_messages": len(self.bot.conversation_history()),
                "turn_count": self.bot.session_turn_count(),
            },
        )


__all__ = ["MaintenanceScheduler"]
