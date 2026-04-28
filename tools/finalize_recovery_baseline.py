"""Finalize certification baseline artifacts from per-lane recovery outputs."""
from __future__ import annotations

import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.phase4_helpers import append_baseline_record, render_capability_record_markdown, utc_now_iso

LANES: list[tuple[str, str]] = [
    ("DEV", "lane_dev"),
    ("INTEGRATION", "lane_integration"),
    ("DURABILITY_P4", "lane_durability_p4"),
    ("SOAK", "lane_soak"),
    ("UI", "lane_ui"),
    ("FULL_CERT", "lane_full_cert"),
]


def _parse_junit_tests(path: Path) -> int:
    if not path.exists():
        return 0
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    if root.tag == "testsuite":
        return int(root.attrib.get("tests", "0") or 0)
    if root.tag == "testsuites":
        total_attr = root.attrib.get("tests")
        if total_attr:
            return int(total_attr or 0)
        total = 0
        for suite in root.findall("testsuite"):
            total += int(suite.attrib.get("tests", "0") or 0)
        return total
    return 0


def _parse_meta(path: Path) -> tuple[str, int, float]:
    lane, code, elapsed = path.read_text(encoding="utf-8").strip().split("\t")
    return lane, int(code), float(elapsed)


def _parse_top_slowest(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    items: list[str] = []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^\d+\.\d+s\s+", stripped) and (
            " call " in f" {stripped} " or stripped.endswith(" call")
        ):
            items.append(stripped)
    return items[:10]


def _collect_cold_start() -> dict:
    t0 = time.perf_counter()
    import dadbot  # noqa: F401

    t1 = time.perf_counter()
    from dadbot.core.dadbot import DadBot

    t2 = time.perf_counter()
    DadBot()
    t3 = time.perf_counter()
    return {
        "dadbot_import_s": round(t1 - t0, 6),
        "dadbot_class_import_s": round(t2 - t1, 6),
        "dadbot_init_s": round(t3 - t2, 6),
        "total_s": round(t3 - t0, 6),
    }


def _collect_db_sizes() -> dict[str, int]:
    sizes: dict[str, int] = {}
    for pattern in ("*.db", "*.sqlite", "*.sqlite3"):
        for path in ROOT.glob(pattern):
            try:
                sizes[path.name] = int(path.stat().st_size)
            except Exception:
                continue
    return dict(sorted(sizes.items()))


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: finalize_recovery_baseline.py <run_dir_relative_to_repo>")
        return 2

    run_dir = (ROOT / sys.argv[1]).resolve()
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return 2

    record: dict = {
        "generated_at": utc_now_iso(),
        "baseline_store": "tests/phase4_baselines.json",
        "lanes": {},
        "cold_start": _collect_cold_start(),
        "db_sizes_bytes": _collect_db_sizes(),
        "top_slowest_tests": [],
    }

    for lane_name, base in LANES:
        _, exit_code, elapsed_s = _parse_meta(run_dir / f"{base}.meta.tsv")
        tests = _parse_junit_tests(run_dir / f"{base}.junit.xml")
        top_slowest = _parse_top_slowest(run_dir / f"{base}.txt")
        record["lanes"][lane_name] = {
            "command": f"pytest lane {lane_name}",
            "exit_code": exit_code,
            "result": "pass" if exit_code == 0 else "fail",
            "elapsed_s": round(elapsed_s, 3),
            "tests": tests,
            "summary": {},
            "top_slowest": top_slowest,
            "timed_out": False,
        }

    flat: list[str] = []
    for lane_name, lane_data in record["lanes"].items():
        for item in lane_data.get("top_slowest", []):
            flat.append(f"[{lane_name}] {item}")
    record["top_slowest_tests"] = flat[:10]

    baseline_record = run_dir / "baseline_record.json"
    baseline_record.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

    baseline_store = ROOT / "tests" / "phase4_baselines.json"
    _, previous = append_baseline_record(baseline_store, record, keep_last=50)

    report = render_capability_record_markdown(record, previous)
    run_report = run_dir / "dad_capability_record.md"
    run_report.write_text(report, encoding="utf-8")
    (ROOT / "dad_capability_record.md").write_text(report, encoding="utf-8")

    print(f"WROTE_RECORD={baseline_record}")
    print(f"WROTE_REPORT={run_report}")
    print(f"UPDATED_STORE={baseline_store}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
