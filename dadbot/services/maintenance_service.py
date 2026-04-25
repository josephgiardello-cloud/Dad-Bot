from __future__ import annotations

import logging
from typing import Any

from dadbot.managers.long_term import LongTermSignalsManager
from dadbot.managers.maintenance import MaintenanceScheduler

logger = logging.getLogger(__name__)


class MaintenanceService:
    """Service wrapper for maintenance cadence, long-term signals, and proactive engagement.

    ``tick()`` is called at the start of every turn by the HealthNode.  It:

    1. Runs periodic durable synthesis (archive, consolidate, evolve persona â€¦)
       when the cadence threshold is met.
    2. Triggers scheduled proactive jobs (due reminders, life-pattern check-ins)
       so the bot can *initiate* contact rather than only ever react.  Proactive
       messages are queued via ``bot.queue_proactive_message`` and surfaced to the
       UI at next startup or via OS notifications when enabled.
    """

    def __init__(self, maintenance: MaintenanceScheduler, long_term: LongTermSignalsManager):
        self.maintenance = maintenance
        self.long_term = long_term

    def tick(self, turn_context: Any) -> dict[str, Any]:
        trigger_text = str(getattr(turn_context, "user_input", "") or "")

        # 1. Periodic durable synthesis (memory archive, consolidation, etc.)
        synthesis_result = self.maintenance.run_periodic_durable_synthesis(trigger_text=trigger_text)

        # 2. Proactive engagement: fire any due reminders or life-pattern check-ins.
        #    run_scheduled_proactive_jobs queues messages via bot.queue_proactive_message
        #    and optionally sends OS notifications.  Errors are non-fatal.
        try:
            self.maintenance.run_scheduled_proactive_jobs()
        except Exception as exc:
            logger.debug("MaintenanceService.tick: proactive jobs failed (non-fatal): %s", exc)

        return synthesis_result
