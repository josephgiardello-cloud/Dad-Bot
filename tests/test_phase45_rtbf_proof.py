from __future__ import annotations

import json
from pathlib import Path

from dadbot.core.rtbf_proof import record_rtbf_receipt


def test_record_rtbf_receipt_writes_jsonl(tmp_path: Path) -> None:
    receipt_path = tmp_path / "rtbf_receipts.jsonl"
    marker = record_rtbf_receipt(
        receipt_path=receipt_path,
        actor="user",
        reason="gdpr_delete_command",
        before_snapshot={"memory_count": 12},
        after_snapshot={"memory_count": 0},
    )
    assert marker["event_type"] == "rtbf_delete_marker"
    assert marker["proof_hash"]
    lines = receipt_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["receipt_id"] == marker["receipt_id"]
    assert parsed["before_hash"] != parsed["after_hash"]
