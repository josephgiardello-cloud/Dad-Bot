def test_tool_registry_parses_calendar_and_email_commands(bot):
    calendar_command = bot.parse_tool_command("add to calendar project sync tomorrow 2pm")
    email_command = bot.parse_tool_command("draft email to alex@example.com about sprint update")

    assert calendar_command is not None
    assert calendar_command["action"] == "create_calendar_event"
    assert "project sync" in calendar_command["title"].lower()

    assert email_command is not None
    assert email_command["action"] == "draft_email"
    assert email_command["recipient"] == "alex@example.com"
    assert "sprint update" in email_command["subject"].lower()


def test_calendar_event_and_email_draft_use_local_paths(bot, tmp_path, monkeypatch):
    calendar_path = tmp_path / "calendar_events.json"
    drafts_dir = tmp_path / "email_drafts"
    monkeypatch.setenv("DADBOT_CALENDAR_EVENTS_PATH", str(calendar_path))
    monkeypatch.setenv("DADBOT_EMAIL_DRAFT_DIR", str(drafts_dir))

    calendar_reply = bot.handle_tool_command("add to calendar dentist appointment on 2030-03-03 9:30 am")
    list_reply = bot.handle_tool_command("list calendar events")
    email_reply = bot.handle_tool_command("draft email to coach@example.com about game plan")

    assert "local calendar" in str(calendar_reply).lower()
    assert "calendar events" in str(list_reply).lower()
    assert "saved at:" in str(email_reply).lower()

    assert calendar_path.exists()
    assert drafts_dir.exists()
    assert any(path.suffix == ".eml" for path in drafts_dir.glob("*.eml"))


def test_voice_toggle_command_updates_profile_state(bot, monkeypatch):
    monkeypatch.setattr(bot, "save_profile", lambda: None)

    on_reply = bot.handle_tool_command("/voice on")
    status_reply = bot.handle_tool_command("/voice status")
    off_reply = bot.handle_tool_command("/voice off")

    assert "on" in str(on_reply).lower()
    assert "currently on" in str(status_reply).lower()
    assert "off" in str(off_reply).lower()
    assert bool(bot.PROFILE.get("voice", {}).get("enabled", True)) is False
