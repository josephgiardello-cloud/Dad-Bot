from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


def test_snapshot_session_state_only_advances_thread_timestamp_on_change(bot):
    manager = bot.runtime_state_manager
    manager.sync_active_thread_snapshot()
    state = manager._container.state
    original_updated_at = state.chat_threads[0]["updated_at"]
    bot.history.append({"role": "user", "content": "A real thread update."})

    with patch.object(manager, "thread_timestamp", return_value="2099-01-01T00:00:00"):
        changed_snapshot = bot.snapshot_session_state()
        unchanged_snapshot = bot.snapshot_session_state()

    assert changed_snapshot["chat_threads"][0]["updated_at"] == "2099-01-01T00:00:00"
    assert unchanged_snapshot["chat_threads"][0]["updated_at"] == "2099-01-01T00:00:00"
    assert original_updated_at != changed_snapshot["chat_threads"][0]["updated_at"]
