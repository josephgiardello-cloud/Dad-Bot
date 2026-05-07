from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dadbot import app_runtime


@pytest.mark.unit
def test_parse_args_status_command() -> None:
    args = app_runtime.parse_args(["status", "--json"])
    assert args.command == "status"
    assert bool(args.json) is True


@pytest.mark.unit
def test_parse_args_trace_command() -> None:
    args = app_runtime.parse_args(["trace", "turn-123", "--limit", "7"])
    assert args.command == "trace"
    assert args.turn_id == "turn-123"
    assert int(args.limit) == 7


@pytest.mark.unit
def test_parse_args_doctor_command() -> None:
    args = app_runtime.parse_args(["doctor", "--json"])
    assert args.command == "doctor"
    assert bool(args.json) is True


@pytest.mark.unit
def test_parse_args_restart_command() -> None:
    args = app_runtime.parse_args(["restart", "--light"])
    assert args.command == "restart"
    assert bool(args.light) is True


@pytest.mark.unit
def test_parse_args_no_command_returns_none_command() -> None:
    args = app_runtime.parse_args([])
    assert args.command is None


@pytest.mark.unit
def test_run_operator_command_unknown_returns_error() -> None:
    class _Args:
        command = "unknown"

    code = app_runtime._run_operator_command(
        _Args(),
        dadbot_cls=object,
        script_path=Path("Dad.py"),
    )
    assert code == 2


@pytest.mark.unit
def test_run_operator_command_no_command_returns_none() -> None:
    class _Args:
        command = ""

    result = app_runtime._run_operator_command(
        _Args(),
        dadbot_cls=object,
        script_path=Path("Dad.py"),
    )
    assert result is None


@pytest.mark.unit
def test_cli_status_emits_json(capsys) -> None:
    mock_supervisor = MagicMock()
    mock_supervisor.get_status.return_value = {"status": "no_lock", "pid": None}

    with patch("dadbot.app_runtime.get_runtime_supervisor", return_value=mock_supervisor):
        code = app_runtime._cli_status(as_json=True)

    assert code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["command"] == "status"
    assert "runtime" in payload


@pytest.mark.unit
def test_cli_status_emits_text(capsys) -> None:
    mock_supervisor = MagicMock()
    mock_supervisor.get_status.return_value = {"status": "no_lock"}

    with patch("dadbot.app_runtime.get_runtime_supervisor", return_value=mock_supervisor):
        code = app_runtime._cli_status(as_json=False)

    assert code == 0
    captured = capsys.readouterr()
    assert "command" in captured.out


@pytest.mark.unit
def test_cli_trace_empty_turn_id_returns_error(capsys) -> None:
    code = app_runtime._cli_trace(object, turn_id="", limit=10, as_json=False)
    assert code == 2


@pytest.mark.unit
def test_cli_trace_returns_events(capsys) -> None:
    mock_bot = MagicMock()
    mock_bot.list_turn_events.return_value = [{"type": "TURN_START"}]
    mock_bot.replay_turn_events.return_value = {"status": "ok"}
    mock_dadbot_cls = MagicMock(return_value=mock_bot)

    code = app_runtime._cli_trace(mock_dadbot_cls, turn_id="tr-abc", limit=5, as_json=True)

    assert code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["trace_id"] == "tr-abc"
    assert payload["event_count"] == 1
    assert len(payload["events"]) == 1


@pytest.mark.unit
def test_cli_doctor_ok_returns_zero(capsys) -> None:
    mock_supervisor = MagicMock()
    mock_supervisor.preflight_check.return_value = (True, [])
    mock_supervisor.get_status.return_value = {"status": "no_lock"}

    mock_bot = MagicMock()
    mock_bot.current_runtime_health_snapshot.return_value = {"healthy": True}
    mock_dadbot_cls = MagicMock(return_value=mock_bot)

    with patch("dadbot.app_runtime.get_runtime_supervisor", return_value=mock_supervisor):
        code = app_runtime._cli_doctor(mock_dadbot_cls, as_json=True)

    assert code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["command"] == "doctor"
    assert payload["preflight_ok"] is True


@pytest.mark.unit
def test_cli_doctor_preflight_failure_returns_one(capsys) -> None:
    mock_supervisor = MagicMock()
    mock_supervisor.preflight_check.return_value = (False, ["Stale lock detected"])
    mock_supervisor.get_status.return_value = {"status": "locked"}

    mock_bot = MagicMock()
    mock_bot.current_runtime_health_snapshot.return_value = {}
    mock_dadbot_cls = MagicMock(return_value=mock_bot)

    with patch("dadbot.app_runtime.get_runtime_supervisor", return_value=mock_supervisor):
        code = app_runtime._cli_doctor(mock_dadbot_cls, as_json=False)

    assert code == 1


@pytest.mark.unit
def test_run_operator_command_dispatches_status() -> None:
    class _Args:
        command = "status"
        json = True

    mock_supervisor = MagicMock()
    mock_supervisor.get_status.return_value = {}

    with patch("dadbot.app_runtime.get_runtime_supervisor", return_value=mock_supervisor):
        code = app_runtime._run_operator_command(
            _Args(), dadbot_cls=object, script_path=Path("Dad.py")
        )

    assert code == 0


@pytest.mark.unit
def test_run_operator_command_dispatches_trace() -> None:
    class _Args:
        command = "trace"
        turn_id = "tr-test"
        limit = 5
        json = True

    mock_bot = MagicMock()
    mock_bot.list_turn_events.return_value = []
    mock_bot.replay_turn_events.return_value = {}
    mock_dadbot_cls = MagicMock(return_value=mock_bot)

    code = app_runtime._run_operator_command(
        _Args(), dadbot_cls=mock_dadbot_cls, script_path=Path("Dad.py")
    )
    assert code == 0


@pytest.mark.unit
def test_run_operator_command_restart_stops_on_stop_failure() -> None:
    class _Args:
        command = "restart"
        no_signoff = False
        light = False

    with (
        patch("dadbot.app_runtime.stop_streamlit_app", return_value=3) as stop_mock,
        patch("dadbot.app_runtime.launch_streamlit_app") as launch_mock,
    ):
        code = app_runtime._run_operator_command(
            _Args(), dadbot_cls=object, script_path=Path("Dad.py")
        )

    assert code == 3
    stop_mock.assert_called_once()
    launch_mock.assert_not_called()


@pytest.mark.unit
def test_run_operator_command_restart_launches_with_expected_flags() -> None:
    class _Args:
        command = "restart"
        no_signoff = True
        light = True

    with (
        patch("dadbot.app_runtime.stop_streamlit_app", return_value=0) as stop_mock,
        patch("dadbot.app_runtime.launch_streamlit_app", return_value=0) as launch_mock,
    ):
        code = app_runtime._run_operator_command(
            _Args(), dadbot_cls=object, script_path=Path("Dad.py")
        )

    assert code == 0
    stop_mock.assert_called_once_with(script_path=Path("Dad.py"))
    launch_mock.assert_called_once_with(
        append_signoff=False,
        light_mode=True,
        script_path=Path("Dad.py"),
    )
