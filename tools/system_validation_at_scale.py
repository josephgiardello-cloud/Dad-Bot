from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CommandResult:
    name: str
    command: list[str]
    passed: bool
    exit_code: int
    duration_seconds: float
    stdout_tail: list[str]


def _run_pytest(
    *,
    name: str,
    args: list[str],
    env: dict[str, str] | None = None,
    tail_lines: int = 40,
) -> CommandResult:
    command = [sys.executable, "-m", "pytest", *args]
    merged_env = dict(os.environ)
    if env:
        merged_env.update({k: str(v) for k, v in env.items()})

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=merged_env,
    )
    elapsed = time.perf_counter() - started
    stdout = str(completed.stdout or "")
    stderr = str(completed.stderr or "")
    joined = (stdout + "\n" + stderr).strip().splitlines()
    tail = joined[-tail_lines:] if joined else []
    return CommandResult(
        name=name,
        command=command,
        passed=completed.returncode == 0,
        exit_code=int(completed.returncode),
        duration_seconds=round(float(elapsed), 3),
        stdout_tail=tail,
    )


def _long_horizon_suite(turns: int) -> list[CommandResult]:
    return [
        _run_pytest(
            name="long_horizon_chaos_loop",
            args=[
                "tests/chaos/test_adversarial_turn_loop.py::TestAdversarialTurnLoop::test_n_turn_loop_all_invariants_hold",
                "-q",
            ],
            env={"DADBOT_CHAOS_TURNS": str(turns)},
        ),
    ]


def _adversarial_fuzz_suite() -> list[CommandResult]:
    return [
        _run_pytest(
            name="adversarial_determinism_fuzzing",
            args=["tests/adversarial/test_determinism_fuzzing.py", "-q"],
        ),
        _run_pytest(
            name="mutation_attacks",
            args=["tests/adversarial/test_mutation_attacks.py", "-q"],
        ),
        _run_pytest(
            name="impossible_state_tripwire",
            args=["tests/unit/test_impossible_state_tripwire.py", "-q"],
        ),
    ]


def _replay_chaos_suite(*, include_unstable_recovery: bool = False) -> list[CommandResult]:
    results = [
        _run_pytest(
            name="checkpoint_chaos",
            args=["tests/chaos/test_checkpoint_chaos.py", "-q"],
        ),
        _run_pytest(
            name="replay_equivalence_harness",
            args=["tests/test_replay_equivalence_harness.py", "-q"],
        ),
        _run_pytest(
            name="execution_replay_suite",
            args=["tests/execution_replay_test.py", "-q"],
        ),
    ]
    if include_unstable_recovery:
        results.append(
            _run_pytest(
                name="recovery_and_idempotency",
                args=["tests/test_recovery_and_idempotency.py", "-q"],
            ),
        )
    return results


def _report_payload(*, turns: int, results: list[CommandResult], long_horizon_skipped: bool) -> dict[str, Any]:
    groups = {
        "long_horizon_simulation": ["long_horizon_chaos_loop"],
        "adversarial_fuzzing": [
            "adversarial_determinism_fuzzing",
            "mutation_attacks",
            "impossible_state_tripwire",
        ],
        "replay_chaos": [
            "checkpoint_chaos",
            "replay_equivalence_harness",
            "execution_replay_suite",
            "recovery_and_idempotency",
        ],
    }

    index = {item.name: item for item in results}
    group_summary: dict[str, Any] = {}
    for group_name, names in groups.items():
        subset = [index[n] for n in names if n in index]
        group_summary[group_name] = {
            "passed": all(item.passed for item in subset),
            "skipped": bool(group_name == "long_horizon_simulation" and long_horizon_skipped),
            "tests": [
                {
                    "name": item.name,
                    "passed": item.passed,
                    "exit_code": item.exit_code,
                    "duration_seconds": item.duration_seconds,
                }
                for item in subset
            ],
        }

    return {
        "schema_version": "dadbot-system-validation-at-scale.v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "turns_requested": int(turns),
        "overall_passed": all(item.passed for item in results),
        "groups": group_summary,
        "results": [
            {
                "name": item.name,
                "command": item.command,
                "passed": item.passed,
                "exit_code": item.exit_code,
                "duration_seconds": item.duration_seconds,
                "stdout_tail": item.stdout_tail,
            }
            for item in results
        ],
        "known_open_gaps": [
            "Distributed multi-node correctness (concurrent writers / network partitions) is not validated by this local runner.",
            "Operational telemetry thresholds and alert routing are not asserted here; this runner validates test behavior, not production monitoring pipelines.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run system validation at scale (chaos + fuzz + replay).")
    parser.add_argument("--turns", type=int, default=1000, help="Long-horizon chaos turn count (default: 1000).")
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/validation_scale_report.json",
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--skip-long-horizon",
        action="store_true",
        help="Skip long-horizon chaos loop phase.",
    )
    parser.add_argument(
        "--include-unstable-recovery",
        action="store_true",
        help="Include tests/test_recovery_and_idempotency.py in replay chaos phase.",
    )
    args = parser.parse_args()

    if args.turns < 1:
        raise SystemExit("--turns must be >= 1")

    results: list[CommandResult] = []
    if not args.skip_long_horizon:
        results.extend(_long_horizon_suite(int(args.turns)))
    results.extend(_adversarial_fuzz_suite())
    results.extend(_replay_chaos_suite(include_unstable_recovery=bool(args.include_unstable_recovery)))

    payload = _report_payload(turns=int(args.turns), results=results, long_horizon_skipped=bool(args.skip_long_horizon))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote report: {output_path.as_posix()}")
    print(f"Overall passed: {payload['overall_passed']}")
    for item in results:
        print(
            f"- {item.name}: {'PASS' if item.passed else 'FAIL'} "
            f"(exit={item.exit_code}, {item.duration_seconds:.3f}s)",
        )

    return 0 if payload["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
