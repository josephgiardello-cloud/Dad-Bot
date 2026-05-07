from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from dadbot.managers.runtime_storage import RuntimeStorageManager

pytestmark = pytest.mark.unit


def _build_runtime(tmp_path: Path) -> RuntimeStorageManager:
    bot = SimpleNamespace(
        SESSION_LOG_DIR=tmp_path,
        config=SimpleNamespace(primary_identity_log_filenames=("relational_ledger.jsonl",)),
    )
    return RuntimeStorageManager(bot)


def test_rotation_candidates_exclude_relational_ledger(tmp_path: Path) -> None:
    runtime_storage = _build_runtime(tmp_path)

    protected = tmp_path / "relational_ledger.jsonl"
    rotatable = tmp_path / "runtime.log"
    protected.write_text("{}\n", encoding="utf-8")
    rotatable.write_text("line\n", encoding="utf-8")

    names = [path.name for path in runtime_storage.session_log_rotation_candidates()]
    assert "runtime.log" in names
    assert "relational_ledger.jsonl" not in names


def test_prune_session_logs_preserves_primary_identity_log(tmp_path: Path) -> None:
    runtime_storage = _build_runtime(tmp_path)

    protected = tmp_path / "relational_ledger.jsonl"
    old_log = tmp_path / "old.log"
    new_log = tmp_path / "new.log"

    protected.write_text("ledger\n", encoding="utf-8")
    old_log.write_text("old\n", encoding="utf-8")
    new_log.write_text("new\n", encoding="utf-8")

    # Oldest first so prune(max_files=1) drops old.log and keeps new.log + protected ledger.
    now = time.time()
    os.utime(old_log, (now - 120.0, now - 120.0))
    os.utime(new_log, (now - 10.0, now - 10.0))

    removed = runtime_storage.prune_session_logs(max_files=1)
    removed_names = {path.name for path in removed}

    assert "old.log" in removed_names
    assert protected.exists()
    assert new_log.exists()
