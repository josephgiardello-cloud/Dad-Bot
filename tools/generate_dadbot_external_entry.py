from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _get_git_commit_sha() -> str:
    """Get current git HEAD commit SHA, or empty string if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def _map_result_to_response(result: dict[str, Any]) -> dict[str, Any]:
    trace = dict(result.get("trace") or {})
    execution = dict(result.get("execution") or {})
    raw_state = trace.get("raw_state")
    if not isinstance(raw_state, dict):
        raw_state = {}

    tools = trace.get("tools_executed")
    if not isinstance(tools, list):
        tools = []
    memory = trace.get("memory_accessed")
    if not isinstance(memory, list):
        memory = []

    planner_output = trace.get("planner_output")
    if not isinstance(planner_output, dict):
        planner_output = None

    return {
        "scenario": str(result.get("scenario") or ""),
        "response": str(trace.get("final_response") or ""),
        "completed": bool(execution.get("completed", False)),
        "error": str(execution.get("error")) if execution.get("error") else None,
        "planner_output": planner_output,
        "tools_executed": [str(v) for v in tools],
        "memory_accessed": [str(v) for v in memory],
        "raw_state": raw_state,
    }


def _run_mock() -> list[dict[str, Any]]:
    from tests.benchmark_runner import BenchmarkRunner
    from tests.scenario_suite import SCENARIOS

    runner = BenchmarkRunner(strict=False, mode="mock")
    return runner.run_all_scenarios(SCENARIOS)


def _run_orchestrator(*, offline_llm_stub: bool = False) -> list[dict[str, Any]]:
    from dadbot.core.dadbot import DadBot
    from tests.benchmark_runner import BenchmarkRunner
    from tests.scenario_suite import SCENARIOS

    bot = DadBot()
    runner = None
    try:
        orchestrator = getattr(bot, "turn_orchestrator", None)
        if orchestrator is None:
            raise RuntimeError("DadBot.turn_orchestrator is not available")

        runner = BenchmarkRunner(
            strict=False,
            mode="orchestrator",
            orchestrator=orchestrator,
            use_offline_llm_stub=bool(offline_llm_stub),
        )
        return runner.run_all_scenarios(SCENARIOS)
    finally:
        if runner is not None:
            try:
                runner.close()
            except Exception:
                pass
        try:
            bot.shutdown()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Dad-Bot entrant artifact for external benchmark loop.",
    )
    parser.add_argument(
        "--mode",
        choices=("orchestrator", "mock"),
        default="orchestrator",
        help="Execution backend for scenario runs.",
    )
    parser.add_argument(
        "--agent-name",
        default="dadbot",
        help="Agent label to embed in artifact.",
    )
    parser.add_argument(
        "--model",
        default="llama3.2:latest",
        help="Model label to embed in artifact metadata.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/external_benchmark_examples/dadbot.live.json",
        help="Output artifact path.",
    )
    parser.add_argument(
        "--offline-llm-stub",
        action="store_true",
        help="Patch runtime LLM calls with a deterministic offline benchmark stub.",
    )
    parser.add_argument(
        "--cert",
        action="store_true",
        help="Certification mode: reject offline stubs, capture git SHA, emit strict schema.",
    )
    parser.add_argument(
        "--require-completed",
        action="store_true",
        help="In normal mode, require explicit 'completed' field in all responses (no default).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # Cert mode: reject offline stub.
    if bool(args.cert) and bool(args.offline_llm_stub):
        print("ERROR: --cert and --offline-llm-stub are mutually exclusive.")
        return 1
    
    out_path = Path(str(args.output).strip())
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    try:
        if args.mode == "orchestrator":
            results = _run_orchestrator(offline_llm_stub=bool(args.offline_llm_stub))
        else:
            results = _run_mock()

        responses = [_map_result_to_response(result) for result in results]
        payload = {
            "agent": str(args.agent_name),
            "model": str(args.model),
            "generated_at": _utc_now_iso(),
            "mode": str(args.mode),
            "responses": responses,
        }

        # In cert mode, include git SHA and offline stub flag.
        if bool(args.cert):
            payload["git_commit_sha"] = _get_git_commit_sha()
            payload["offline_llm_stub"] = False  # Explicit: not using stub.
        elif bool(args.offline_llm_stub):
            payload["offline_llm_stub"] = True

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        rel_path = out_path
        try:
            rel_path = out_path.relative_to(ROOT)
        except ValueError:
            pass
        print(f"WROTE_DADBOT_ENTRY={str(rel_path).replace('\\', '/')}")
        if bool(args.cert):
            print(f"CERT_MODE=true GIT_SHA={payload.get('git_commit_sha', 'unknown')}")
        return 0
    except Exception as exc:
        print(f"ERROR: failed to generate Dad-Bot benchmark artifact: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
