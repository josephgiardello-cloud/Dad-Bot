"""Capture lane baselines and generate capability record artifacts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.phase4_helpers import (
    append_baseline_record,
    render_capability_record_markdown,
    utc_now_iso,
)

BASELINE_STORE = ROOT / "tests" / "phase4_baselines.json"

LANES: list[tuple[str, list[str]]] = [
    ("DEV", ["-m", "unit"]),
    ("INTEGRATION", ["-m", "integration"]),
    ("DURABILITY_P4", ["-m", "durability or phase4_cert"]),
    ("SOAK", ["-m", "soak"]),
    ("UI", ["-m", "ui"]),
    ("FULL_CERT", []),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture certification baseline metrics.")
    parser.add_argument("--durations", type=int, default=20, help="Pytest durations count per lane")
    parser.add_argument(
        "--lane-timeout",
        type=int,
        default=None,
        help="Fixed max seconds per lane before timeout (overrides baseline-derived timeout)",
    )
    parser.add_argument(
        "--timeout-multiplier",
        type=float,
        default=1.35,
        help="Multiplier for baseline lane elapsed when deriving timeout",
    )
    parser.add_argument(
        "--timeout-buffer-s",
        type=int,
        default=30,
        help="Additional seconds added to baseline-derived timeout",
    )
    parser.add_argument(
        "--timeout-min-s",
        type=int,
        default=90,
        help="Minimum lane timeout when deriving from baselines",
    )
    parser.add_argument(
        "--timeout-max-s",
        type=int,
        default=1200,
        help="Maximum lane timeout when deriving from baselines",
    )
    parser.add_argument(
        "--lanes",
        default="all",
        help="Comma-separated lane names to run (default: all)",
    )
    parser.add_argument(
        "--cold-start-timeout-s",
        type=int,
        default=45,
        help="Max seconds for cold-start probe before timing out",
    )
    parser.add_argument(
        "--write-baseline", action="store_true", help="Append run metrics to tests/phase4_baselines.json"
    )
    parser.add_argument(
        "--report-markdown", action="store_true", help="Generate DAD-BOT OFFICIAL CAPABILITY RECORD markdown"
    )
    parser.add_argument("--artifact-dir", default="session_logs", help="Parent directory for run artifacts")
    return parser.parse_args()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _extract_summary_counts(output: str) -> dict:
    summary = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return summary

    candidate = ""
    for line in reversed(lines):
        if " in " in line and any(
            token in line for token in ("passed", "failed", "error", "skipped", "xfailed", "xpassed")
        ):
            candidate = line
            break

    for key in list(summary.keys()):
        m = re.search(rf"(\d+)\s+{key}", candidate)
        if m:
            summary[key] = int(m.group(1))

    # Fallback for runs where the final short summary line is not emitted
    # but individual status counts still appear somewhere in output.
    if not any(summary.values()):
        for key in list(summary.keys()):
            all_matches = re.findall(rf"(\d+)\s+{key}", output)
            if all_matches:
                summary[key] = int(all_matches[-1])

    return summary


def _extract_top_slowest(output: str, *, max_items: int = 10) -> list[str]:
    slow_lines: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if re.match(r"^\d+\.\d+s\s+", stripped) and (" call " in f" {stripped} " or stripped.endswith(" call")):
            slow_lines.append(stripped)
    return slow_lines[:max_items]


def _count_tests_from_junit_xml(xml_path: Path) -> int | None:
    if not xml_path.exists():
        return None
    try:
        root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if root.tag == "testsuite":
        value = root.attrib.get("tests")
        if value and value.isdigit():
            return int(value)
    if root.tag == "testsuites":
        value = root.attrib.get("tests")
        if value and value.isdigit():
            return int(value)
        total = 0
        for suite in root.findall("testsuite"):
            tests_value = suite.attrib.get("tests")
            if tests_value and tests_value.isdigit():
                total += int(tests_value)
        if total > 0:
            return total
    return None


def _run_lane(
    python_exe: str,
    lane_name: str,
    lane_args: list[str],
    durations: int,
    lane_timeout: int,
    junit_xml_path: Path,
    lane_log_path: Path,
) -> tuple[dict, str]:
    cmd = [
        python_exe,
        "-m",
        "pytest",
        *lane_args,
        f"--durations={durations}",
        "-q",
        "-p",
        "no:randomly",
        f"--junitxml={junit_xml_path}",
    ]
    start = time.perf_counter()
    timed_out = False
    lane_log_path.parent.mkdir(parents=True, exist_ok=True)
    with lane_log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        try:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=lane_timeout,
            )
            returncode = int(proc.returncode)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = 124
            log_file.write(f"\n\n[cert_baseline_capture] LANE TIMEOUT after {lane_timeout}s: {lane_name}\n")
    elapsed = time.perf_counter() - start
    output = lane_log_path.read_text(encoding="utf-8", errors="replace")
    summary = _extract_summary_counts(output)
    tests = sum(summary.values())
    if tests == 0:
        tests = summary.get("passed", 0) + summary.get("failed", 0) + summary.get("errors", 0)
    if tests == 0:
        junit_tests = _count_tests_from_junit_xml(junit_xml_path)
        if junit_tests is not None:
            tests = junit_tests

    lane_payload = {
        "command": " ".join(cmd),
        "exit_code": returncode,
        "result": "pass" if returncode == 0 else "fail",
        "elapsed_s": round(elapsed, 3),
        "tests": tests,
        "summary": summary,
        "top_slowest": _extract_top_slowest(output, max_items=10),
        "timed_out": timed_out,
    }
    return lane_payload, output


def _run_cold_start_probe(python_exe: str, timeout_s: int) -> dict:
    snippet = (
        "import json,os,time;"
        "os.environ.setdefault('PYTEST_CURRENT_TEST','cold_start_probe');"
        "t0=time.perf_counter();"
        "import dadbot;"
        "t1=time.perf_counter();"
        "from dadbot.core.dadbot import DadBot;"
        "t2=time.perf_counter();"
        "bot=DadBot();"
        "t3=time.perf_counter();"
        "bot.shutdown();"
        "print(json.dumps({'dadbot_import_s':round(t1-t0,6),'dadbot_class_import_s':round(t2-t1,6),'dadbot_init_s':round(t3-t2,6),'total_s':round(t3-t0,6)}))"
    )
    try:
        proc = subprocess.run(
            [python_exe, "-c", snippet],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_s)),
        )
    except subprocess.TimeoutExpired:
        return {
            "error": f"cold-start probe timed out after {int(timeout_s)}s",
            "timed_out": True,
        }
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        return {"error": output.strip()[-800:]}

    for line in reversed([x.strip() for x in output.splitlines() if x.strip()]):
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except Exception:
                continue
    return {"error": "cold-start probe returned no JSON payload"}


def _collect_db_sizes() -> dict[str, int]:
    patterns = ("*.db", "*.sqlite", "*.sqlite3")
    sizes: dict[str, int] = {}
    for pattern in patterns:
        for path in ROOT.glob(pattern):
            try:
                sizes[path.name] = int(path.stat().st_size)
            except Exception:
                continue
    return dict(sorted(sizes.items()))


def _flatten_slowest(lanes: dict[str, dict], *, top_n: int = 10) -> list[str]:
    all_items: list[str] = []
    for lane_name, lane_data in lanes.items():
        for entry in lane_data.get("top_slowest", []):
            all_items.append(f"[{lane_name}] {entry}")
    return all_items[:top_n]


def _lane_order() -> list[str]:
    return [name for name, _ in LANES]


def _select_lanes(lanes_arg: str) -> list[tuple[str, list[str]]]:
    requested = [x.strip().upper() for x in lanes_arg.split(",") if x.strip()]
    if not requested or requested == ["ALL"]:
        return LANES
    available = {name: args for name, args in LANES}
    invalid = [name for name in requested if name not in available]
    if invalid:
        choices = ", ".join(_lane_order())
        raise ValueError(f"Unknown lane(s): {', '.join(invalid)}. Available: {choices}")
    return [(name, available[name]) for name in requested]


def _latest_baseline_record() -> dict | None:
    if not BASELINE_STORE.exists():
        return None
    try:
        payload = json.loads(BASELINE_STORE.read_text(encoding="utf-8"))
    except Exception:
        return None
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        return None
    latest = records[-1]
    return latest if isinstance(latest, dict) else None


def _derive_lane_timeout(
    lane_name: str,
    latest_baseline: dict | None,
    *,
    fixed_timeout: int | None,
    multiplier: float,
    buffer_s: int,
    min_s: int,
    max_s: int,
) -> tuple[int, str]:
    if fixed_timeout is not None and fixed_timeout > 0:
        return int(fixed_timeout), "fixed"
    baseline_elapsed = None
    if isinstance(latest_baseline, dict):
        lanes = latest_baseline.get("lanes")
        if isinstance(lanes, dict):
            lane_info = lanes.get(lane_name)
            if isinstance(lane_info, dict):
                elapsed = lane_info.get("elapsed_s")
                if isinstance(elapsed, (int, float)):
                    baseline_elapsed = float(elapsed)
    if baseline_elapsed is None:
        return min_s, "fallback-min"
    derived = int(round((baseline_elapsed * multiplier) + buffer_s))
    bounded = max(min_s, min(max_s, derived))
    return bounded, f"baseline({baseline_elapsed:.3f}s)"


def main() -> int:
    args = parse_args()
    try:
        selected_lanes = _select_lanes(args.lanes)
    except ValueError as exc:
        print(str(exc))
        return 2
    generated_at = utc_now_iso()
    run_dir = ROOT / args.artifact_dir / f"cert_baseline_{generated_at.replace(':', '').replace('-', '')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable
    record: dict = {
        "generated_at": generated_at,
        "baseline_store": str(BASELINE_STORE.relative_to(ROOT)).replace("\\", "/"),
        "lanes": {},
        "cold_start": {},
        "db_sizes_bytes": {},
        "top_slowest_tests": [],
    }

    latest_baseline = _latest_baseline_record()
    if latest_baseline is None:
        print("No prior baseline record found; using minimum/fixed timeouts.")
    else:
        print("Loaded prior baseline record for timeout derivation.")

    for lane_name, lane_args in selected_lanes:
        lane_timeout, timeout_source = _derive_lane_timeout(
            lane_name,
            latest_baseline,
            fixed_timeout=args.lane_timeout,
            multiplier=args.timeout_multiplier,
            buffer_s=args.timeout_buffer_s,
            min_s=args.timeout_min_s,
            max_s=args.timeout_max_s,
        )
        print(f"[{lane_name}] starting... timeout={lane_timeout}s source={timeout_source}", flush=True)
        junit_xml_path = run_dir / f"lane_{_slug(lane_name)}.junit.xml"
        lane_log = run_dir / f"lane_{_slug(lane_name)}.txt"
        payload, output = _run_lane(
            python_exe,
            lane_name,
            lane_args,
            args.durations,
            lane_timeout,
            junit_xml_path,
            lane_log,
        )
        record["lanes"][lane_name] = payload
        timeout_tag = " timeout" if payload.get("timed_out") else ""
        print(
            f"[{lane_name}] exit={payload['exit_code']} elapsed={payload['elapsed_s']:.2f}s tests={payload['tests']}{timeout_tag}"
        )

    record["cold_start"] = _run_cold_start_probe(python_exe, args.cold_start_timeout_s)
    record["db_sizes_bytes"] = _collect_db_sizes()
    record["top_slowest_tests"] = _flatten_slowest(record["lanes"], top_n=10)

    raw_json_path = run_dir / "baseline_record.json"
    raw_json_path.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")

    previous: dict | None = None
    if args.write_baseline:
        _, previous = append_baseline_record(BASELINE_STORE, record, keep_last=50)
        print(f"Updated baseline store: {BASELINE_STORE}")

    if args.report_markdown:
        report_md = render_capability_record_markdown(record, previous)
        report_path = run_dir / "dad_capability_record.md"
        report_path.write_text(report_md, encoding="utf-8")
        latest_report = ROOT / "dad_capability_record.md"
        latest_report.write_text(report_md, encoding="utf-8")
        print(f"Wrote capability report: {report_path}")

    print(f"Artifact directory: {run_dir}")
    failures = [name for name, data in record["lanes"].items() if data.get("exit_code", 1) != 0]
    if failures:
        print(f"FAILED LANES: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
