from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import dadbot.app_runtime as app_runtime


pytestmark = pytest.mark.unit


class _FakeSocket:
    def setsockopt(self, *args, **kwargs):
        return None

    def bind(self, *args, **kwargs):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_launch_streamlit_app_self_heals_preflight_before_launch(tmp_path: Path) -> None:
    script_path = tmp_path / "Dad.py"
    script_path.write_text("print('dad')\n", encoding="utf-8")

    mock_supervisor = MagicMock()
    mock_supervisor.preflight_check.side_effect = [
        (False, ["Orphaned lock detected: PID 101 on port 8501"]),
        (True, []),
    ]
    mock_supervisor.attempt_self_heal.return_value = (
        True,
        ["Removed orphaned lock: PID 101 on port 8501"],
    )
    mock_supervisor.acquire_lock.return_value = (True, "ok")

    process = MagicMock()
    process.wait.return_value = 0
    process.poll.return_value = None

    response = MagicMock()
    response.__enter__.return_value = object()
    response.__exit__.return_value = False

    with (
        patch("dadbot.app_runtime.get_runtime_supervisor", return_value=mock_supervisor),
        patch("dadbot.app_runtime.socket", side_effect=lambda *args, **kwargs: _FakeSocket()),
        patch("dadbot.app_runtime.subprocess.Popen", return_value=process),
        patch("dadbot.app_runtime.urlopen", return_value=response),
        patch("dadbot.app_runtime.webbrowser.open", return_value=None),
    ):
        result = app_runtime.launch_streamlit_app(script_path=script_path)

    assert result == 0
    mock_supervisor.attempt_self_heal.assert_called_once()
    mock_supervisor.release_lock.assert_called_once()


def test_launch_streamlit_app_retries_once_after_early_boot_failure(tmp_path: Path) -> None:
    script_path = tmp_path / "Dad.py"
    script_path.write_text("print('dad')\n", encoding="utf-8")

    mock_supervisor = MagicMock()
    mock_supervisor.preflight_check.return_value = (True, [])
    mock_supervisor.attempt_self_heal.return_value = (False, [])
    mock_supervisor.acquire_lock.return_value = (True, "ok")

    failed_process = MagicMock()
    failed_process.poll.return_value = 1
    failed_process.wait.return_value = 1

    healthy_process = MagicMock()
    healthy_process.poll.return_value = None
    healthy_process.wait.return_value = 0

    response = MagicMock()
    response.__enter__.return_value = object()
    response.__exit__.return_value = False

    urlopen_calls = iter([RuntimeError("boot failed"), response])

    def _urlopen_side_effect(*args, **kwargs):
        value = next(urlopen_calls)
        if isinstance(value, Exception):
            raise value
        return value

    with (
        patch("dadbot.app_runtime.get_runtime_supervisor", return_value=mock_supervisor),
        patch("dadbot.app_runtime.socket", side_effect=lambda *args, **kwargs: _FakeSocket()),
        patch("dadbot.app_runtime.subprocess.Popen", side_effect=[failed_process, healthy_process]) as popen_mock,
        patch("dadbot.app_runtime.urlopen", side_effect=_urlopen_side_effect),
        patch("dadbot.app_runtime.webbrowser.open", return_value=None),
    ):
        result = app_runtime.launch_streamlit_app(script_path=script_path)

    assert result == 0
    assert popen_mock.call_count == 2
    mock_supervisor.set_state.assert_any_call("DEGRADED")
    mock_supervisor.set_state.assert_any_call("RUNNING")
    mock_supervisor.release_lock.assert_called_once()