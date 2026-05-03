#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ROOT = ROOT / "SYSTEM_SNAPSHOT"
LATEST_INDEX = SNAPSHOT_ROOT / "LATEST_SNAPSHOT.json"
LATEST_SUMMARY = SNAPSHOT_ROOT / "snapshot_summary_latest.txt"
LATEST_README = SNAPSHOT_ROOT / "README_latest.md"


def _require(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing required canonical artifact: {path}")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_snapshot_id(summary_text: str) -> str:
    for line in summary_text.splitlines():
        if "Snapshot :" in line:
            return line.split("Snapshot :", 1)[1].strip()
    return ""


def verify() -> None:
    _require(SNAPSHOT_ROOT)
    _require(LATEST_INDEX)
    _require(LATEST_SUMMARY)
    _require(LATEST_README)

    index = json.loads(_read_text(LATEST_INDEX))
    if str(index.get("status") or "") != "canonical_latest":
        raise RuntimeError("LATEST_SNAPSHOT.json status must be 'canonical_latest'")

    summary_path = Path(str(index.get("snapshot_summary") or "").strip())
    readme_path = Path(str(index.get("snapshot_readme") or "").strip())
    snapshot_dir = Path(str(index.get("snapshot_dir") or "").strip())
    snapshot_id = str(index.get("snapshot_id") or "").strip()

    if not snapshot_id:
        raise RuntimeError("LATEST_SNAPSHOT.json snapshot_id must be non-empty")

    for artifact_path in (summary_path, readme_path, snapshot_dir):
        if not artifact_path.exists():
            raise RuntimeError(f"LATEST_SNAPSHOT.json references missing path: {artifact_path}")

    latest_summary_text = _read_text(LATEST_SUMMARY)
    latest_readme_text = _read_text(LATEST_README)
    canonical_summary_text = _read_text(summary_path)
    canonical_readme_text = _read_text(readme_path)

    if latest_summary_text != canonical_summary_text:
        raise RuntimeError("snapshot_summary_latest.txt does not match referenced canonical snapshot_summary")
    if latest_readme_text != canonical_readme_text:
        raise RuntimeError("README_latest.md does not match referenced canonical README")

    extracted_snapshot_id = _extract_snapshot_id(latest_summary_text)
    if extracted_snapshot_id != snapshot_id:
        raise RuntimeError(
            "Snapshot id mismatch between LATEST_SNAPSHOT.json and snapshot_summary_latest.txt",
        )

    expected_dir_name = f"snapshot_{snapshot_id}"
    if snapshot_dir.name != expected_dir_name:
        raise RuntimeError(
            f"snapshot_dir name mismatch: expected {expected_dir_name}, got {snapshot_dir.name}",
        )

    print("Canonical snapshot guard PASSED")


if __name__ == "__main__":
    try:
        verify()
    except Exception as exc:  # noqa: BLE001
        print(f"Canonical snapshot guard FAILED: {exc}")
        sys.exit(1)
