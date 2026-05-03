from __future__ import annotations

import json
from pathlib import Path

import pytest

from dadbot.core.canonical_event import validate_trace
from dadbot.core.execution_schema import EXECUTION_TRACE_CONTRACT_SCHEMA_VERSION
from dadbot.core.kernel_locks import KernelReplaySequenceLock
from dadbot.core.legacy_upcaster import upcast_event_log, upcast_trace_contract

pytestmark = pytest.mark.unit

_FIXTURE_DIR = Path(__file__).parent / "legacy_payloads"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_legacy_trace_contract_migrates_to_current_schema() -> None:
    payload = _load_json(_FIXTURE_DIR / "trace_contract_v0.json")
    migrated = upcast_trace_contract(payload)

    assert migrated.get("schema_version") == EXECUTION_TRACE_CONTRACT_SCHEMA_VERSION
    assert migrated.get("version") == "1.0"
    assert migrated.get("event_count") == 3


def test_legacy_event_log_upcasts_and_replays_under_current_kernel_lock() -> None:
    payload = _load_json(_FIXTURE_DIR / "event_log_v0.json")
    upgraded = upcast_event_log(payload)

    validate_trace(upgraded)
    digest, canonical = KernelReplaySequenceLock.strict_hash(
        trace_id="legacy-trace",
        events=upgraded,
    )
    assert len(canonical) == 3
    assert isinstance(digest, str) and len(digest) == 64
