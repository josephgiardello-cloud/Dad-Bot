from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dadbot.ui import data


@pytest.mark.unit
@pytest.mark.parametrize(
    ("selection", "expected"),
    [("Keep A", "left"), ("Keep B", "right"), ("Both could be true", "both"), ("Dismiss both", "none")],
)
def test_conflict_ui_passes_selected_resolution(monkeypatch, selection, expected):
    ui = MagicMock()
    ui.columns.return_value = (MagicMock(), MagicMock())
    ui.radio.return_value = selection
    ui.button.return_value = True
    monkeypatch.setattr(data, "st", ui)
    bot = MagicMock()
    bot.consolidated_contradictions.return_value = [{"left_summary": "A", "right_summary": "B"}]
    bot.resolve_consolidated_contradiction.return_value = {"resolution": expected}
    data._render_conflict_resolution_section(bot)
    bot.resolve_consolidated_contradiction.assert_called_once_with("A", "B", keep=expected, reason="user_review")
    ui.success.assert_called_once()


@pytest.mark.unit
def test_conflict_ui_does_not_report_success_for_failed_resolution(monkeypatch):
    ui = MagicMock()
    ui.columns.return_value = (MagicMock(), MagicMock())
    ui.radio.return_value = "Keep A"
    ui.button.return_value = True
    monkeypatch.setattr(data, "st", ui)
    bot = MagicMock()
    bot.consolidated_contradictions.return_value = [{"left_summary": "A", "right_summary": "B"}]
    bot.resolve_consolidated_contradiction.return_value = None
    data._render_conflict_resolution_section(bot)
    ui.success.assert_not_called()
    ui.warning.assert_called_once()


@pytest.mark.unit
def test_manual_forgetting_runs_explicit_manual_path(monkeypatch):
    ui = MagicMock()
    ui.button.side_effect = [False, False, True]
    monkeypatch.setattr(data, "st", ui)
    bot = MagicMock()
    bot.memory_catalog.return_value = []
    bot.memory_manager.memory_projection.return_value = {"archived_memories": []}
    bot.memory_coordinator.apply_manual_controlled_forgetting.return_value = {"removed": 0, "ran": True}
    data._render_memory_management_section(bot)
    bot.memory_coordinator.apply_manual_controlled_forgetting.assert_called_once_with()
    ui.info.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize("operation", ["restore", "delete"])
def test_archived_memory_ui_executes_selected_operation(monkeypatch, operation):
    ui = MagicMock()
    ui.button.return_value = False
    ui.selectbox.side_effect = ["", "archive-1"]
    restore_column, delete_column = MagicMock(), MagicMock()
    restore_column.button.return_value = operation == "restore"
    delete_column.button.return_value = operation == "delete"
    ui.columns.return_value = (restore_column, delete_column)
    monkeypatch.setattr(data, "st", ui)
    bot = MagicMock()
    bot.memory_catalog.return_value = []
    bot.memory_manager.memory_projection.return_value = {
        "archived_memories": [{
            "archive_id": "archive-1", "archived_at": "2026-01-01",
            "memory": {"summary": "A stored memory."},
        }],
    }
    data._render_memory_management_section(bot)
    expected = getattr(bot.memory_manager, f"{operation}_archived_memory")
    expected.assert_called_once_with("archive-1")
