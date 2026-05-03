from __future__ import annotations

from pathlib import Path

import pytest

from tools.million_event_replay_stressor import run_stressor

pytestmark = pytest.mark.unit


def test_million_event_replay_stressor_smoke(tmp_path: Path) -> None:
    db = tmp_path / "stress.sqlite3"
    summary = tmp_path / "summary.json"

    report = run_stressor(
        db_path=db,
        event_count=5000,
        batch_size=1000,
        summary_out=summary,
    )

    assert report["event_count_observed"] == 5000
    assert report["elapsed_seconds"] >= 0.0
    assert report["snapshot_summary_seconds"] >= 0.0
    assert report["db_size_after_bytes"] >= report["db_size_before_bytes"]
    assert summary.exists()
