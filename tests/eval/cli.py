from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from Dad import DadBot
from dadbot.core.orchestrator import DadBotOrchestrator
from tests.eval.datasets import default_eval_cases, full_benchmark_cases
from tests.eval.runner import RunnerConfig, run_eval_suite


REQUIRED_RUNTIME_SERVICES: tuple[str, ...] = (
    "maintenance_service",
    "context_service",
    "agent_service",
    "safety_service",
    "persistence_service",
    "runtime_service",
    "turn_graph",
)


def production_bootstrap() -> tuple[DadBotOrchestrator, Any | None, dict[str, Any]]:
    """Build runtime exactly as production does: real DadBot + orchestrator(bot=...)."""
    bot = DadBot()
    runtime = DadBotOrchestrator(bot=bot, strict=False)
    registry = getattr(runtime, "registry", None)

    tool_registry = getattr(bot, "tool_registry", None)
    services: dict[str, Any] = {}
    if registry is not None:
        services = dict(getattr(registry, "_services", {}) or {})
        # Missing binding fix: ensure runtime registry has the instantiated tool registry.
        if tool_registry is not None and services.get("tool_registry") is None:
            try:
                registry.register("tool_registry", tool_registry)
            except Exception:
                services = dict(getattr(registry, "_services", {}) or {})
            services = dict(getattr(registry, "_services", {}) or {})
        if tool_registry is None:
            try:
                tool_registry = registry.get("tool_registry", optional=True)
            except Exception:
                tool_registry = None

    if tool_registry is not None:
        services.setdefault("tool_registry", tool_registry)
    planner_service = getattr(bot, "turn_service", None)
    if planner_service is not None:
        services.setdefault("planner_service", planner_service)

    return runtime, tool_registry, services


def validate_runtime(runtime: Any, *, tool_registry: Any | None, mode: str) -> None:
    registry = getattr(runtime, "registry", None) or getattr(runtime, "service_registry", None)
    if registry is None:
        raise RuntimeError("Runtime does not expose a service registry")

    get_fn = getattr(registry, "get", None)
    if not callable(get_fn):
        raise RuntimeError("Runtime registry does not expose get(name)")

    missing: list[str] = []
    for service_name in REQUIRED_RUNTIME_SERVICES:
        instance = None
        try:
            instance = get_fn(service_name, optional=True)
        except TypeError:
            try:
                instance = get_fn(service_name)
            except Exception:
                instance = None
        except Exception:
            instance = None

        if instance is None:
            missing.append(service_name)

    if missing:
        raise RuntimeError(f"Missing required service(s): {', '.join(missing)}")

    registry_tool_registry = None
    try:
        registry_tool_registry = get_fn("tool_registry", optional=True)
    except Exception:
        registry_tool_registry = None

    if tool_registry is None:
        raise RuntimeError("Missing required service: tool_registry")
    if registry_tool_registry is None:
        raise RuntimeError("Missing runtime binding: registry.tool_registry is not registered")

    planner_service = getattr(getattr(runtime, "bot", None), "turn_service", None)
    if planner_service is None:
        raise RuntimeError("Missing required planner service: bot.turn_service")

    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode not in {"production-safe", "evaluation-strict"}:
        raise RuntimeError(f"Unknown eval mode: {mode}")


def _build_runtime() -> tuple[DadBotOrchestrator, Any | None, dict[str, Any]]:
    return production_bootstrap()


def main() -> int:
    parser = argparse.ArgumentParser(description="Dad-Bot additive eval harness runner")
    parser.add_argument("--full", action="store_true", help="Run full benchmark case set from tests.scenario_suite")
    parser.add_argument("--output", type=str, default="", help="Optional JSON summary output path")
    parser.add_argument("--langsmith", action="store_true", help="Force-enable LangSmith adapter")
    parser.add_argument("--wandb", action="store_true", help="Force-enable W&B adapter")
    parser.add_argument("--openai-evals", action="store_true", help="Force-enable OpenAI-evals row export")
    parser.add_argument("--openai-evals-output", type=str, default="", help="Optional JSONL output for OpenAI eval rows")
    parser.add_argument(
        "--mode",
        type=str,
        default="evaluation-strict",
        choices=["production-safe", "evaluation-strict"],
        help="Eval mode: production-safe allows fallback, evaluation-strict fails fast on fallback/missing service.",
    )
    args = parser.parse_args()

    cfg = RunnerConfig.from_env()
    cfg.mode = str(args.mode)
    if args.langsmith:
        cfg.langsmith = True
        if cfg.langsmith_client is None:
            try:
                from langsmith import Client  # type: ignore

                cfg.langsmith_client = Client()
            except Exception:
                cfg.langsmith = False
    if args.wandb:
        cfg.wandb = True
    if args.openai_evals:
        cfg.openai_evals = True
    if args.openai_evals_output:
        cfg.openai_evals_output_path = args.openai_evals_output

    cases = full_benchmark_cases() if args.full else default_eval_cases()

    runtime, tool_registry, services = _build_runtime()
    validate_runtime(runtime, tool_registry=tool_registry, mode=cfg.mode)

    summary = run_eval_suite(cases, runtime, tool_registry=tool_registry, config=cfg)

    printable = {
        "success_rate": summary["success_rate"],
        "irrelevance_rate": summary["irrelevance_rate"],
        "avg_tool_calls": summary["avg_tool_calls"],
        "efficiency": summary["efficiency"],
        "tool_execution_rate": summary.get("tool_execution_rate", 0.0),
        "no_tool_rate": summary.get("no_tool_rate", 0.0),
        "robustness_suppression_rate": summary.get("robustness_suppression_rate", 0.0),
        "coherence_violations": int(summary.get("coherence_violations_count", 0)),
        "planner_execution_mismatches": int(summary.get("planner_execution_mismatches", 0)),
        "runs": len(summary["runs"]),
        "strict_failures": len(summary.get("strict_failures", [])),
        "openai_eval_rows": len(summary["openai_evals"]),
        "services_registered": len(services),
        "mode": cfg.mode,
        "langsmith_enabled": bool(cfg.langsmith),
        "wandb_enabled": bool(cfg.wandb),
        "openai_evals_enabled": bool(cfg.openai_evals),
    }
    print(json.dumps(printable, indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, default=str, indent=2), encoding="utf-8")

    if str(cfg.mode).strip().lower() == "evaluation-strict" and len(summary.get("strict_failures", [])) > 0:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
