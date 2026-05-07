from __future__ import annotations

from types import SimpleNamespace

import pytest

from dadbot.managers import runtime_interface as runtime_interface_module
from dadbot.managers.runtime_interface import ChatInsight, ChatTurn, RuntimeInterfaceManager


pytestmark = pytest.mark.unit


def _manager_with_bot(bot) -> RuntimeInterfaceManager:
    manager = RuntimeInterfaceManager.__new__(RuntimeInterfaceManager)
    manager.bot = bot
    manager.ui_mode = "chat"
    manager.story_mode_password = ""
    manager.story_mode_failed_attempts = 0
    manager.story_mode_locked_until_ts = 0.0
    return manager


def test_runtime_interface_defaults_to_chat_mode(monkeypatch) -> None:
    monkeypatch.setenv("DADBOT_UI_MODE", "")
    monkeypatch.setattr(
        runtime_interface_module.DadBotContext,
        "from_runtime",
        staticmethod(lambda bot: SimpleNamespace(bot=bot)),
    )

    bot = SimpleNamespace(UI_MODE="")
    manager = RuntimeInterfaceManager(bot)

    assert manager.ui_mode == "chat"


def test_render_turn_chat_mode_hides_hud_and_shows_soft_hint(capsys) -> None:
    spoken: list[str] = []
    bot = SimpleNamespace(print_speaker_message=lambda _speaker, text: spoken.append(text))
    manager = _manager_with_bot(bot)

    hud_calls: list[bool] = []
    manager._render_audit_hud = lambda: hud_calls.append(True)  # type: ignore[method-assign]

    chat_turn = ChatTurn(
        assistant_text="Lets slow down and reassess.",
        overlays=[ChatInsight(title="System Insight", lines=["risk=moderate"])],
    )
    manager._render_turn(chat_turn)

    output = capsys.readouterr().out
    assert spoken == ["Lets slow down and reassess."]
    assert "system insight available" in output.lower()
    assert hud_calls == []


def test_render_turn_debug_mode_renders_hud(capsys) -> None:
    spoken: list[str] = []
    bot = SimpleNamespace(print_speaker_message=lambda _speaker, text: spoken.append(text))
    manager = _manager_with_bot(bot)
    manager.ui_mode = "debug"

    hud_calls: list[bool] = []
    manager._render_audit_hud = lambda: hud_calls.append(True)  # type: ignore[method-assign]

    chat_turn = ChatTurn(assistant_text="Debug response", overlays=[ChatInsight(title="System", lines=["x"])])
    manager._render_turn(chat_turn)

    output = capsys.readouterr().out
    assert spoken == ["Debug response"]
    assert "system insight available" in output.lower()
    assert hud_calls == [True]


def test_ui_commands_support_mode_switch_and_insight(capsys) -> None:
    messages: list[str] = []
    bot = SimpleNamespace(print_system_message=lambda text: messages.append(text))
    manager = _manager_with_bot(bot)

    turn = ChatTurn(assistant_text="text", overlays=[ChatInsight(title="System", lines=["cause chain"] )])

    assert manager._handle_ui_command("/mode operator") is True
    assert manager.ui_mode == "operator"
    assert manager._handle_ui_command("/mode invalid") is True
    assert manager._handle_ui_command("/insight", chat_turn=turn) is True

    output = capsys.readouterr().out
    assert any("UI mode switched" in message for message in messages)
    assert any("Unknown UI mode" in message for message in messages)
    assert "[Why I said this]" in output


def test_mode_command_accepts_story_mode() -> None:
    messages: list[str] = []
    bot = SimpleNamespace(print_system_message=lambda text: messages.append(text))
    manager = _manager_with_bot(bot)
    manager.story_mode_password = "secret"

    assert manager._handle_ui_command("/mode story secret") is True
    assert manager.ui_mode == "story"
    assert any("UI mode switched" in message for message in messages)


def test_story_mode_requires_password() -> None:
    messages: list[str] = []
    bot = SimpleNamespace(print_system_message=lambda text: messages.append(text))
    manager = _manager_with_bot(bot)
    manager.story_mode_password = "secret"

    assert manager._handle_ui_command("/mode story") is True
    assert manager.ui_mode == "chat"
    assert any("Incorrect story mode password" in message for message in messages)


def test_story_mode_password_is_case_sensitive() -> None:
    messages: list[str] = []
    bot = SimpleNamespace(print_system_message=lambda text: messages.append(text))
    manager = _manager_with_bot(bot)
    manager.story_mode_password = "CaseSensitive!"

    assert manager._handle_ui_command("/mode story casesensitive!") is True
    assert manager.ui_mode == "chat"
    assert any("Incorrect story mode password" in message for message in messages)


def test_story_mode_lockout_backoff_after_failed_attempts(monkeypatch) -> None:
    messages: list[str] = []
    bot = SimpleNamespace(print_system_message=lambda text: messages.append(text))
    manager = _manager_with_bot(bot)
    manager.story_mode_password = "secret"

    monotonic_state = {"value": 100.0}
    monkeypatch.setattr(runtime_interface_module.time, "monotonic", lambda: monotonic_state["value"])

    assert manager._handle_ui_command("/mode story wrong") is True
    assert manager._handle_ui_command("/mode story wrong") is True
    assert manager._handle_ui_command("/mode story wrong") is True

    assert manager.story_mode_failed_attempts == 3
    assert manager.story_mode_locked_until_ts > monotonic_state["value"]
    assert any("locked for" in message for message in messages)

    messages.clear()
    assert manager._handle_ui_command("/mode story secret") is True
    assert manager.ui_mode == "chat"
    assert any("temporarily locked" in message for message in messages)

    monotonic_state["value"] = manager.story_mode_locked_until_ts + 1.0
    messages.clear()
    assert manager._handle_ui_command("/mode story secret") is True
    assert manager.ui_mode == "story"
    assert any("UI mode switched" in message for message in messages)


def test_help_ui_includes_story_mode_setup_guidance() -> None:
    messages: list[str] = []
    bot = SimpleNamespace(print_system_message=lambda text: messages.append(text))
    manager = _manager_with_bot(bot)

    assert manager._handle_ui_command("/help ui") is True
    assert any("Story mode setup" in message for message in messages)


def test_contextual_learning_keyword_detection() -> None:
    assert RuntimeInterfaceManager._should_trigger_contextual_learning("My mom is in the hospital") is True
    assert RuntimeInterfaceManager._should_trigger_contextual_learning("I like pizza") is False


def test_insight_command_falls_back_to_last_turn_context(capsys) -> None:
    messages: list[str] = []
    bot = SimpleNamespace(
        print_system_message=lambda text: messages.append(text),
        _last_turn_context=SimpleNamespace(
            safe_result=("reply", False),
            state={
                "reflection_summary": {
                    "current_risk_level": "moderate",
                    "predicted_drift_probability": 0.5,
                    "likely_trigger_category": "fatigue",
                },
                "ux_feedback": {
                    "memory_message": "prior turn context",
                },
            },
        ),
    )
    manager = _manager_with_bot(bot)

    assert manager._handle_ui_command("/insight") is True
    output = capsys.readouterr().out
    assert "[Why I said this]" in output
    assert "No turn insight available yet." not in messages


def test_unknown_slash_command_is_handled_as_ui_feedback() -> None:
    messages: list[str] = []
    bot = SimpleNamespace(print_system_message=lambda text: messages.append(text))
    manager = _manager_with_bot(bot)

    assert manager._handle_ui_command("/not-real") is True
    assert any("Unknown command" in message for message in messages)
