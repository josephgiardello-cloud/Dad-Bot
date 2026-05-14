"""Shared helpers for Phase 4 certification and baseline tracking."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest


def utc_now_iso() -> str:
    """Return a stable UTC timestamp for run metadata."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@pytest.fixture
def phase4a_db_path(monkeypatch: pytest.MonkeyPatch) -> str:
    """Provide a SQLite path for Phase 4 tests.

    Opt-in in-memory mode:
    - Set DADBOT_PHASE4_DB_MODE=memory
    - Uses shared-cache URI plus a keep-alive connection so database state
      survives across checkpointer open/close cycles in a single test.

    Default mode uses a temporary on-disk DB and deletes it after test.
    """
    mode = str(os.environ.get("DADBOT_PHASE4_DB_MODE", "file")).strip().lower()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as runtime_dir:
        runtime_root = Path(runtime_dir)
        profile_path = runtime_root / "dad_profile.json"
        memory_path = runtime_root / "dad_memory.json"

        template_path = Path(__file__).resolve().parents[1] / "dad_profile.template.json"
        try:
            profile_payload = json.loads(template_path.read_text(encoding="utf-8"))
            if not isinstance(profile_payload, dict) or not profile_payload:
                raise ValueError("invalid template payload")
        except Exception:
            profile_payload = {
                "name": "Dad",
                "relationship": "father",
                "llm": {"provider": "ollama", "model": "llama3.2"},
            }
        profile_path.write_text(json.dumps(profile_payload, indent=2, ensure_ascii=True), encoding="utf-8")
        memory_path.write_text(json.dumps({"memories": []}, ensure_ascii=True), encoding="utf-8")

        monkeypatch.setenv("DADBOT_PROFILE_PATH", str(profile_path))
        monkeypatch.setenv("DADBOT_MEMORY_PATH", str(memory_path))
        monkeypatch.setenv("DADBOT_AUTO_INIT_PROFILE", "1")

        if mode == "memory":
            db_uri = f"file:dadbot_phase4_{uuid.uuid4().hex}?mode=memory&cache=shared"
            keepalive = sqlite3.connect(db_uri, uri=True)
            try:
                yield db_uri
            finally:
                keepalive.close()
            return

        path = str(runtime_root / f"phase4a_{uuid.uuid4().hex}.db")
        try:
            yield path
        finally:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass


def load_baseline_store(store_path: Path) -> dict:
    """Load baseline store payload or initialize a valid empty schema."""
    if not store_path.exists():
        return {"schema_version": 1, "records": []}
    try:
        payload = json.loads(store_path.read_text(encoding="utf-8"))
    except Exception:
        return {"schema_version": 1, "records": []}
    if not isinstance(payload, dict):
        return {"schema_version": 1, "records": []}
    payload.setdefault("schema_version", 1)
    payload.setdefault("records", [])
    if not isinstance(payload["records"], list):
        payload["records"] = []
    return payload


def write_baseline_store(store_path: Path, payload: dict) -> None:
    """Write the baseline JSON store in a deterministic format."""
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )


def append_baseline_record(store_path: Path, record: dict, *, keep_last: int = 30) -> tuple[dict, dict | None]:
    """Append a new baseline record and return (store, previous_record)."""
    store = load_baseline_store(store_path)
    previous = store["records"][-1] if store["records"] else None
    store["records"].append(record)
    if keep_last > 0 and len(store["records"]) > keep_last:
        store["records"] = store["records"][-keep_last:]
    write_baseline_store(store_path, store)
    return store, previous


def _lane_elapsed(lane_payload: dict) -> float | None:
    elapsed = lane_payload.get("elapsed_s")
    if elapsed is None:
        return None
    try:
        return float(elapsed)
    except Exception:
        return None


def compute_lane_deltas(current: dict, previous: dict | None) -> list[dict]:
    """Compare lane elapsed time and return a normalized delta table."""
    rows: list[dict] = []
    prev_lanes = (previous or {}).get("lanes", {})
    cur_lanes = current.get("lanes", {})
    for lane_name, lane_data in cur_lanes.items():
        cur_elapsed = _lane_elapsed(lane_data)
        prev_elapsed = _lane_elapsed(prev_lanes.get(lane_name, {})) if isinstance(prev_lanes, dict) else None
        delta_pct = None
        if cur_elapsed is not None and prev_elapsed and prev_elapsed > 0:
            delta_pct = ((cur_elapsed - prev_elapsed) / prev_elapsed) * 100.0
        rows.append(
            {
                "lane": lane_name,
                "current_s": cur_elapsed,
                "baseline_s": prev_elapsed,
                "delta_pct": delta_pct,
            }
        )
    return rows


def evaluate_regressions(current: dict, previous: dict | None, *, threshold_pct: float = 15.0) -> list[str]:
    """Return lane names whose elapsed time regressed more than threshold."""
    regressions: list[str] = []
    for row in compute_lane_deltas(current, previous):
        delta = row.get("delta_pct")
        if delta is not None and delta > threshold_pct:
            regressions.append(str(row["lane"]))
    return regressions


def render_capability_record_markdown(current: dict, previous: dict | None) -> str:
    """Render the official capability record markdown with baseline deltas."""
    lane_delta_rows = compute_lane_deltas(current, previous)
    regressions = set(evaluate_regressions(current, previous))

    lines: list[str] = []
    lines.append("# DAD-BOT OFFICIAL CAPABILITY RECORD")
    lines.append("")
    lines.append(f"Generated: {current.get('generated_at', utc_now_iso())}")
    lines.append("")
    lines.append("## Baseline Summary")
    lines.append(f"- Baseline source: {current.get('baseline_store', 'tests/phase4_baselines.json')}")
    lines.append(f"- Previous baseline available: {'yes' if previous else 'no'}")
    lines.append("")
    lines.append("## Certification Matrix")
    lines.append("")
    lines.append(
        "| Section / Subsection | Claimed Feature | Test Coverage | Current Benchmark / Metric | Change from Baseline | Status | Notes |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    cold = current.get("cold_start", {})
    total_cold = cold.get("total_s")
    prev_total_cold = (previous or {}).get("cold_start", {}).get("total_s") if previous else None
    cold_delta = "n/a"
    if isinstance(total_cold, (int, float)) and isinstance(prev_total_cold, (int, float)) and prev_total_cold > 0:
        cold_delta = f"{((float(total_cold) - float(prev_total_cold)) / float(prev_total_cold)) * 100.0:+.1f}%"

    lines.append(
        f"| Startup / Cold Start | DadBot cold-start envelope | startup import timing probe | {total_cold:.3f}s total"
        if isinstance(total_cold, (int, float))
        else "| Startup / Cold Start | DadBot cold-start envelope | startup import timing probe | n/a"
    )
    if isinstance(total_cold, (int, float)):
        lines[-1] += f" | {cold_delta} | {'Proven' if total_cold <= 1.8 else 'Partial'} | Target <= 1.8s guard |"
    else:
        lines[-1] += " | n/a | Gap | cold-start probe unavailable |"

    for row in lane_delta_rows:
        lane = row["lane"]
        lane_data = current.get("lanes", {}).get(lane, {})
        tests = lane_data.get("tests", "?")
        result = lane_data.get("result", "unknown")
        elapsed = row.get("current_s")
        baseline_elapsed = row.get("baseline_s")
        delta_pct = row.get("delta_pct")
        elapsed_text = f"{elapsed:.2f}s" if isinstance(elapsed, float) else "n/a"
        baseline_text = f"{delta_pct:+.1f}%" if isinstance(delta_pct, float) else "n/a"
        status = "Gap" if result != "pass" else ("Partial" if lane in regressions else "Proven")
        lines.append(
            f"| Lanes / {lane} | Lane stability and speed | pytest marker lane | {tests} tests, {elapsed_text}, {result} | {baseline_text} | {status} | Baseline={baseline_elapsed if baseline_elapsed is not None else 'n/a'}s |"
        )

    db_sizes = current.get("db_sizes_bytes", {})
    total_db = sum(v for v in db_sizes.values() if isinstance(v, int))
    prev_db_sizes = (previous or {}).get("db_sizes_bytes", {}) if previous else {}
    prev_total_db = sum(v for v in prev_db_sizes.values() if isinstance(v, int)) if prev_db_sizes else 0
    db_delta = "n/a"
    if prev_total_db > 0:
        db_delta = f"{((total_db - prev_total_db) / prev_total_db) * 100.0:+.1f}%"

    lines.append(
        f"| Persistence / DB Footprint | Soak-related DB growth visibility | sqlite size snapshot | {total_db} bytes | {db_delta} | Proven | Aggregated from root SQLite files |"
    )

    lines.append("")
    lines.append("## Top Slow Tests (Global)")
    lines.append("")
    for idx, item in enumerate(current.get("top_slowest_tests", [])[:10], start=1):
        lines.append(f"{idx}. {item}")

    return "\n".join(lines) + "\n"
