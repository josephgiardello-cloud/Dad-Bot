from __future__ import annotations

import logging
from typing import Any

from dadbot.managers.long_term import LongTermSignalsManager
from dadbot.managers.maintenance import MaintenanceScheduler

logger = logging.getLogger(__name__)


class MaintenanceService:
    """Service wrapper for maintenance cadence, long-term signals, and proactive engagement.

    ``tick()`` is called at the start of every turn by the HealthNode.  It fires
    scheduled proactive jobs (due reminders, life-pattern check-ins) so the bot
    can *initiate* contact rather than only ever react.  Proactive messages are
    queued via ``bot.queue_proactive_message`` and surfaced to the UI at next
    startup or via OS notifications when enabled.

    Durable synthesis (archive, consolidate, evolve persona, forgetting) is
    **post-commit only** — it runs via SaveNode → ``schedule_post_turn_maintenance``
    → background task.  It must not execute in the execution-path hot path.
    """

    def __init__(
        self,
        maintenance: MaintenanceScheduler,
        long_term: LongTermSignalsManager,
    ):
        self.maintenance = maintenance
        self.long_term = long_term

    def tick(self, turn_context: Any) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "ok",
            "ran": False,
            "archived": False,
            "consolidated_count": 0,
            "pattern_count": 0,
            "persona_evolved": False,
        }
        try:
            result["proactive"] = self.maintenance.run_scheduled_proactive_jobs()
        except Exception as exc:
            logger.warning(
                "MaintenanceService.tick: proactive jobs failed (non-fatal): %s",
                exc,
                exc_info=True,
            )
            result.update(status="degraded", error=str(exc))
        return result
