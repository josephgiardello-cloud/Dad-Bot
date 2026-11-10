from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


def test_snapshot_session_state_only_advances_thread_timestamp_on_change(bot):
    manager = bot.runtime_state_manager
    manager.sync_active_thread_snapshot()
    state = manager._container.state
    original_updated_at = state.chat_threads[0]["updated_at"]
    bot.history.append({"role": "user", "content": "A real thread update."})
    timestamps = iter(f"2099-01-01T00:00:{second:02d}" for second in range(1, 10))

    with patch.object(manager, "thread_timestamp", side_effect=lambda: next(timestamps)):
        changed_snapshot = bot.snapshot_session_state()
        unchanged_snapshot = bot.snapshot_session_state()

    assert changed_snapshot["chat_threads"][0]["updated_at"] != original_updated_at
    assert unchanged_snapshot["chat_threads"][0]["updated_at"] == changed_snapshot["chat_threads"][0]["updated_at"]
