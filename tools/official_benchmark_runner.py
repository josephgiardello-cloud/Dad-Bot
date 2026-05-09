from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "official_benchmarks"


@dataclass
class ProbeResult:
    benchmark: str
    available: bool
    command: str
    returncode: int
    output: str


@dataclass
class RunResult:
    benchmark: str
    attempted: bool
    succeeded: bool
    command: str
    returncode: int
    output: str


def _now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _run(cmd: list[str], *, timeout_s: int = 45) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_s)),
        )
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return int(proc.returncode), out.strip()
    except subprocess.TimeoutExpired as exc:
        out = ((exc.stdout or "") + ("\n" + (exc.stderr or ""))).strip()
        return 124, f"timeout after {timeout_s}s\n{out}".strip()
    except Exception as exc:  # pragma: no cover - defensive
        return 1, str(exc)


def _command_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _python_cmd(preferred: str) -> str:
    token = str(preferred or "").strip()
    chosen = token if token else sys.executable
    if sys.platform.startswith("win"):
        return chosen.replace("\\", "/")
    return chosen


def _probe_candidates(benchmark: str, candidates: list[list[str]]) -> ProbeResult:
    last_attempt: ProbeResult | None = None
    for cmd in candidates:
        if not cmd:
            continue
        first = cmd[0]
        if first != sys.executable and not _command_exists(first):
            continue
        rc, out = _run(cmd, timeout_s=30)
        ok = rc == 0
        if ok:
            return ProbeResult(
                benchmark=benchmark,
                available=True,
                command=shlex.join(cmd),
                returncode=rc,
                output=out[-4000:],
            )
        last_attempt = ProbeResult(
            benchmark=benchmark,
            available=False,
            command=shlex.join(cmd),
            returncode=rc,
            output=out[-4000:],
        )
    if last_attempt is not None:
        return last_attempt
    return ProbeResult(
        benchmark=benchmark,
        available=False,
        command="",
        returncode=127,
        output="No known harness command detected in current environment.",
    )


def _default_probe_matrix(
    *,
    swebench_python: str,
    bfcl_python: str,
    osworld_python: str,
    osworld_root: str,
) -> dict[str, list[list[str]]]:
    swe_py = _python_cmd(swebench_python)
    bfcl_py = _python_cmd(bfcl_python)
    os_py = _python_cmd(osworld_python)
    os_root = Path(str(osworld_root).strip()) if str(osworld_root).strip() else None

    osworld_candidates: list[list[str]] = [
        [os_py, "-m", "desktop_env", "--help"],
        [os_py, "-m", "mm_agents", "--help"],
    ]
    if os_root:
        osworld_candidates = [
            [os_py, str(os_root / "run.py"), "--help"],
            [os_py, str(os_root / "show_result.py"), "--help"],
            [os_py, str(os_root / "quickstart.py"), "--help"],
        ] + osworld_candidates

    return {
        "swebench": [
            ["sb-cli", "--help"],
            [swe_py, "-c", "import swebench; print('swebench-import-ok')"],
            [swe_py, "-m", "swebench.harness.run_evaluation", "--help"],
            [swe_py, "-m", "swebench", "--help"],
        ],
        "bfcl": [
            ["bfcl-eval", "--help"],
            [bfcl_py, "-c", "import bfcl_eval; print('bfcl-import-ok')"],
            [bfcl_py, "-m", "bfcl_eval", "--help"],
            [bfcl_py, "-m", "bfcl", "--help"],
        ],
        "osworld": osworld_candidates,
    }


def _is_placeholder_osworld_package(python_cmd: str) -> bool:
    rc, out = _run(
        [
            _python_cmd(python_cmd),
            "-c",
            (
                "import importlib.metadata as m; "
                "dist = m.distribution('osworld'); "
                "meta = dist.metadata; "
                "summary = (meta.get('Summary') or '').lower(); "
                "print('placeholder' if ('placeholder' in summary or 'reserve the name' in summary) else 'ok')"
            ),
        ],
        timeout_s=20,
    )
    return rc == 0 and out.strip().lower() == "placeholder"


def _run_if_requested(
    benchmark: str,
    requested_cmd: str,
    *,
    fallback_probe: ProbeResult,
    timeout_s: int,
) -> RunResult:
    if requested_cmd.strip():
        cmd = shlex.split(requested_cmd)
        rc, out = _run(cmd, timeout_s=timeout_s)
        return RunResult(
            benchmark=benchmark,
            attempted=True,
            succeeded=rc == 0,
            command=shlex.join(cmd),
            returncode=rc,
            output=out[-12000:],
        )

    return RunResult(
        benchmark=benchmark,
        attempted=False,
        succeeded=False,
        command="",
        returncode=127,
        output=(
            "No explicit harness execution command provided. "
            "Provide --{benchmark}-cmd with the official evaluator invocation."
        ).format(benchmark=benchmark),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Probe and run official benchmark harnesses (SWE-bench, BFCL, OSWorld) in the local environment, "
            "then emit a machine-readable report."
        ),
    )
    p.add_argument("--run", default="all", help="Comma list: swebench,bfcl,osworld or all")
    p.add_argument("--execute", action="store_true", help="Execute discovered/requested harness commands")
    p.add_argument("--timeout-s", type=int, default=1800, help="Timeout for each harness execution")
    p.add_argument("--swebench-cmd", default="", help="Explicit SWE-bench command to execute")
    p.add_argument("--bfcl-cmd", default="", help="Explicit BFCL command to execute")
    p.add_argument("--osworld-cmd", default="", help="Explicit OSWorld command to execute")
    p.add_argument(
        "--bfcl-python",
        default="",
        help="Python executable to use for BFCL probing/execution (recommended: Python 3.12 env)",
    )
    p.add_argument(
        "--osworld-root",
        default="",
        help="Path to cloned official OSWorld repository for run.py/show_result.py probing",
    )
    p.add_argument(
        "--osworld-python",
        default="",
        help="Python executable to use for OSWorld probing/execution (defaults to current interpreter)",
    )
    p.add_argument("--output", default="", help="Optional output JSON path")
    return p.parse_args()


def _selected(raw: str) -> set[str]:
    token = str(raw or "").strip().lower()
    if token in {"", "all"}:
        return {"swebench", "bfcl", "osworld"}
    selected = {part.strip() for part in token.split(",") if part.strip()}
    allowed = {"swebench", "bfcl", "osworld"}
    invalid = sorted(selected - allowed)
    if invalid:
        raise ValueError(f"Unknown --run values: {', '.join(invalid)}")
    return selected


def main() -> int:
    args = parse_args()
    try:
        selected = _selected(args.run)
    except ValueError as exc:
        print(str(exc))
        return 2

    probes: list[ProbeResult] = []
    runs: list[RunResult] = []
    matrix = _default_probe_matrix(
        swebench_python=sys.executable,
        bfcl_python=args.bfcl_python,
        osworld_python=args.osworld_python,
        osworld_root=args.osworld_root,
    )

    for bench in ["swebench", "bfcl", "osworld"]:
        if bench not in selected:
            continue
        probe = _probe_candidates(bench, matrix[bench])
        if bench == "osworld" and probe.available:
            command_text = str(probe.command or "").lower()
            is_package_probe = "import osworld" in command_text or " -m osworld" in command_text
            if is_package_probe and _is_placeholder_osworld_package(args.osworld_python):
                probe = ProbeResult(
                    benchmark="osworld",
                    available=False,
                    command=probe.command,
                    returncode=2,
                    output=(
                        "Detected placeholder PyPI package 'osworld' (name-reservation package), "
                        "not the official xlang-ai/OSWorld harness. Provide --osworld-root and "
                        "optionally --osworld-python for official local evaluation commands."
                    ),
                )
        probes.append(probe)

        if args.execute:
            explicit_cmd = {
                "swebench": args.swebench_cmd,
                "bfcl": args.bfcl_cmd,
                "osworld": args.osworld_cmd,
            }[bench]
            if not explicit_cmd.strip() and bench == "bfcl" and str(args.bfcl_python).strip():
                explicit_cmd = f"{_python_cmd(args.bfcl_python)} -m bfcl_eval --help"
            if not explicit_cmd.strip() and bench == "osworld" and str(args.osworld_root).strip():
                os_py = _python_cmd(args.osworld_python)
                run_py = Path(str(args.osworld_root).strip()) / "run.py"
                run_py_str = str(run_py).replace("\\", "/") if sys.platform.startswith("win") else str(run_py)
                explicit_cmd = f"{os_py} {run_py_str} --help"
            runs.append(
                _run_if_requested(
                    bench,
                    explicit_cmd,
                    fallback_probe=probe,
                    timeout_s=int(args.timeout_s),
                )
            )

    stamp = _now_stamp()
    default_out = ARTIFACT_ROOT / f"official_benchmark_report_{stamp}.json"
    out_path = Path(args.output) if str(args.output).strip() else default_out
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "workspace_root": str(ROOT),
        "selected_benchmarks": sorted(selected),
        "execute": bool(args.execute),
        "timeout_s": int(args.timeout_s),
        "probes": [asdict(p) for p in probes],
        "runs": [asdict(r) for r in runs],
    }

    if runs:
        completed = [r for r in runs if r.attempted]
        passed = [r for r in completed if r.succeeded]
        payload["summary"] = {
            "attempted": len(completed),
            "succeeded": len(passed),
            "failed": len(completed) - len(passed),
        }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"WROTE_OFFICIAL_REPORT={out_path}")

    for probe in probes:
        status = "available" if probe.available else "missing"
        print(f"PROBE {probe.benchmark}: {status} ({probe.command or 'none'})")

    for run in runs:
        status = "ok" if run.succeeded else ("skipped" if not run.attempted else "failed")
        print(f"RUN {run.benchmark}: {status} (rc={run.returncode})")

    if args.execute and runs and any(r.attempted and not r.succeeded for r in runs):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
