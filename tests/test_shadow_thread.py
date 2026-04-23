from datetime import datetime


def test_finalize_records_shadow_audit(bot):
    bot.MEMORY_STORE["advice_audits"] = []

    reply = bot.reply_finalization.finalize(
        "You need to stop avoiding this and take one small step.",
        current_mood="stressed",
        user_input="I keep putting this off.",
    )

    assert isinstance(reply, str)
    audits = bot.MEMORY_STORE.get("advice_audits") or []
    assert len(audits) == 1
    assert "goal_alignment_score" in audits[0]
    assert "needs_repair" in audits[0]


def test_maintenance_queues_environmental_and_shadow_repair(bot, mocker):
    now = datetime(2026, 4, 21, 9, 0, 0)
    bot.MEMORY_STORE["advice_audits"] = [
        {
            "recorded_at": "2026-04-20T19:00:00",
            "needs_repair": True,
            "hard_tone": True,
            "goal_alignment_score": 38,
            "reply_excerpt": "you should have done better",
        }
    ]

    mocker.patch.object(
        bot.agentic_handler,
        "list_calendar_events",
        return_value=[
            {
                "event_id": "evt-1",
                "title": "Workshop: finish shelf",
                "due_at": "2026-04-21T18:00:00",
            }
        ],
    )
    mocker.patch.object(bot.calendar_manager, "fetch_upcoming_ical_events", return_value=[])

    result = bot.maintenance_scheduler.run_scheduled_proactive_jobs(force=True, reference_time=now)

    assert result["queued_environmental"] >= 1
    assert result["queued_shadow_repairs"] == 1
    sources = [item.get("source") for item in bot.pending_proactive_messages()]
    assert "environmental-cue" in sources
    assert "shadow-thread" in sources

    audits = bot.MEMORY_STORE.get("advice_audits") or []
    assert audits and str(audits[-1].get("repair_sent_at") or "").strip()


def test_longitudinal_synthesis_persists_insights(bot):
    bot.MEMORY_STORE["session_archive"] = [
        {
            "id": "a",
            "summary": "Electronics project got overwhelming again.",
            "created_at": "2026-04-10T20:00:00",
            "topics": ["electronics"],
            "dominant_mood": "stressed",
            "turn_count": 4,
        },
        {
            "id": "b",
            "summary": "Electronics troubleshooting felt heavy.",
            "created_at": "2026-04-12T20:00:00",
            "topics": ["electronics"],
            "dominant_mood": "stressed",
            "turn_count": 4,
        },
        {
            "id": "c",
            "summary": "Woodworking practice felt focused.",
            "created_at": "2026-04-13T20:00:00",
            "topics": ["woodworking"],
            "dominant_mood": "positive",
            "turn_count": 4,
        },
        {
            "id": "d",
            "summary": "Electronics got easier after slowing down.",
            "created_at": "2026-04-14T20:00:00",
            "topics": ["electronics"],
            "dominant_mood": "stressed",
            "turn_count": 4,
        },
    ]

    insights = bot.long_term_signals.synthesize_longitudinal_insights(force=True, reference_time="2026-04-21T09:00:00")

    assert insights
    assert any(item.get("topic") == "electronics" for item in insights)
    assert all("confidence" in item for item in insights)
    assert all("evidence_count" in item for item in insights)
    assert bot.MEMORY_STORE.get("longitudinal_insights")
